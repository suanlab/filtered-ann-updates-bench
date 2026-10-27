#!/usr/bin/env python3
"""Concurrent maintenance and search (follow-up 2, I18c; paper_audit/followup2_prereg.md).
  taskset -c <cores> <python> bench/stream_concurrent.py <name> <ivf|ivf_part|postfilter|ema>
Every round's deletions and insertions are applied by one writer in alternating batches of BATCH points while READERS
threads issue the stream's 500 queries in a fixed shuffled cycle until the writer is done; then each query runs once
more with no writer (quiet phase). The search parameter is fixed at round 0 on the quiet index: the smallest grid value
with recall >= 0.95.
  ivf, ivf_part  src/cc_ivf (C++; Faiss 1.7 of the ACORN fork): one reader-writer lock, or one per label
  postfilter     hnswlib (M=32, efC=200) with its own concurrency (add_items and mark_deleted during knn_query; no slot
                 reuse), k' = ef, result filtered by label
  ema            EMA (hashannlib) with no lock, as its library exposes no concurrency control; tested for crashes and for
                 dead or inadmissible results
Per round: writer seconds; concurrent query count and throughput; latency p50/p95/p99 (concurrent and quiet); the share
of returned ids deleted before the query started (stale) or while it ran; recall against the ground truth at round start
and at round end (concurrent), and at round end (quiet). Python readers (postfilter, ema) rely on the bindings releasing
the GIL during search (checked: 4 threads take the time of 1)."""
import sys, os, json, time, threading, subprocess, tempfile
sys.path.insert(0, 'bench')
import numpy as np
from common import ROOT, RES, log, K, write_fbin, recall_vec

name, method = sys.argv[1], sys.argv[2]; D = f'{ROOT}/stream_{name}'; meta = json.load(open(f'{D}/meta.json')); R, N0 = meta['R'], meta['N0']
R = min(R, int(os.environ.get('MAX_ROUND', R)))        # smoke tests only
READERS, WT, BATCH = int(os.environ.get('READERS', 8)), int(os.environ.get('WRITER_THREADS', 4)), int(os.environ.get('BATCH', 1000))
X = np.load(f'{D}/Xall.npy', mmap_mode='r'); Q = np.ascontiguousarray(np.load(f'{D}/Q.npy'), np.float32)
attr = np.load(f'{D}/attr.npy').astype(np.int64); qlab = np.load(f'{D}/qlab.npy').astype(np.int64); gt = np.load(f'{D}/gt.npy'); sch = np.load(f'{D}/sched.npz')
nq = len(Q); n = len(X)
REC = np.dtype([('q', '<i4'), ('phase', '<i4'), ('t0', '<f8'), ('t1', '<f8'), ('ids', '<i4', (K,))])
DT = np.dtype([('id', '<i4'), ('t', '<f8')])

def stats(recs, dtime, r, writer_s):
    """recs: REC array of one round; dtime: DT array (deleted id, time its deletion became visible)."""
    dt = np.full(n, np.inf); dt[dtime['id']] = dtime['t']
    out = {'round': r, 'writer_s': writer_s}
    for ph, lab in ((0, 'concurrent'), (1, 'quiet')):
        x = recs[recs['phase'] == ph]
        if not len(x): out[lab] = {'queries': 0}; continue
        lat = (x['t1'] - x['t0']) * 1e3; ids = x['ids']; valid = ids >= 0
        tdel = np.where(valid, dt[np.where(valid, ids, 0)], np.inf)
        stale = (tdel < x['t0'][:, None]) & valid; during = (tdel >= x['t0'][:, None]) & (tdel < x['t1'][:, None]) & valid
        inadm = valid & (attr[np.where(valid, ids, 0)] != qlab[x['q']][:, None])
        o = {'queries': int(len(x)), 'p50_ms': float(np.percentile(lat, 50)), 'p95_ms': float(np.percentile(lat, 95)), 'p99_ms': float(np.percentile(lat, 99)),
             'stale_share': float(stale.sum() / max(1, valid.sum())), 'deleted_during_share': float(during.sum() / max(1, valid.sum())),
             'inadmissible_share': float(inadm.sum() / max(1, valid.sum())),
             'recall_end': float(recall_vec(ids, gt[r][x['q']]).mean())}
        if ph == 0:
            rs_, re_ = recall_vec(ids, gt[r - 1][x['q']]), recall_vec(ids, gt[r][x['q']])
            o['recall_start'] = float(rs_.mean()); o['recall_best'] = float(np.maximum(rs_, re_).mean())   # mid-round, neither endpoint is exact
            o['qps'] = float(len(x) / writer_s) if writer_s > 0 else None
        out[lab] = o
    return out

res = {'method': method, 'readers': READERS, 'writer_threads': WT, 'batch': BATCH, 'rounds': []}
if method in ('ivf', 'ivf_part'):
    tmp = tempfile.mkdtemp(dir=ROOT); write_fbin(f'{tmp}/X.fbin', X); write_fbin(f'{tmp}/Q.fbin', Q)
    attr.astype(np.int32).tofile(f'{tmp}/attr.i32'); qlab.astype(np.int32).tofile(f'{tmp}/qlab.i32'); np.ascontiguousarray(gt[0], np.int32).tofile(f'{tmp}/gt0.i32')
    for r in range(R): sch[f'del{r}'].astype(np.int32).tofile(f'{tmp}/del{r}.i32'); sch[f'ins{r}'].astype(np.int32).tofile(f'{tmp}/ins{r}.i32')
    p = subprocess.run(['./src/cc_ivf', tmp, method, str(N0), str(R), str(READERS), str(WT), str(BATCH)], capture_output=True, text=True)
    if p.returncode != 0: log('cc_ivf failed', p.returncode, p.stderr[-1500:]); sys.exit(1)
    for line in p.stdout.splitlines():
        ev = json.loads(line)
        if ev['event'] == 'build': res['build_s'] = ev['build_s']
        elif ev['event'] == 'calibrate': res['param'] = f"nprobe={ev['nprobe']}"
        elif ev['event'] == 'round':
            r = ev['round']; recs = np.fromfile(f'{tmp}/out/cc_r{r}.bin', REC); dtime = np.fromfile(f'{tmp}/out/del_time_r{r}.bin', DT)
            res['rounds'].append(stats(recs, dtime, r, ev['writer_s'])); log(name, method, 'round', r, res['rounds'][-1]['concurrent'])
    subprocess.run(['rm', '-rf', tmp])
else:
    if method == 'postfilter':
        import hnswlib
        NI = len(sch['ins0']); h = hnswlib.Index(space='l2', dim=X.shape[1]); h.init_index(max_elements=N0 + R * NI, ef_construction=200, M=32)
        h.set_num_threads(16); t = time.time(); h.add_items(np.ascontiguousarray(X[:N0], np.float32), np.arange(N0)); res['build_s'] = time.time() - t; h.set_num_threads(1)
        def do_search(q, prm):
            I, _ = h.knn_query(Q[q:q + 1], k=prm, num_threads=1); I = I[0]; return I[attr[I] == qlab[q]][:K]
        def set_param(prm): h.set_ef(prm)
        grid = [100, 200, 400, 800, 1600, 3200, 6400]
        def apply_delete(b):
            for x in b: h.mark_deleted(int(x))
        def apply_insert(b): h.add_items(np.ascontiguousarray(X[b], np.float32), b, num_threads=WT)
        def round_end(): pass
    else:
        import faiss, hashannlib
        L = int(attr.max()) + 1; X0 = np.ascontiguousarray(X[:N0], np.float32); kk = int(4 * np.sqrt(N0)); faiss.omp_set_num_threads(16)
        km = faiss.Kmeans(X.shape[1], kk, niter=10, seed=1, max_points_per_centroid=50); km.train(X0)
        fl = faiss.IndexFlatL2(X.shape[1]); fl.add(X0); _, Ic = fl.search(km.centroids, 1); levels = np.zeros(N0, np.int32); levels[np.unique(Ic[:, 0])] = 1
        h = hashannlib.Index(space='l2', dim=X.shape[1]); rec = lambda ids: [[[0], [int(attr[i])]] for i in ids]
        h.init_index(max_elements=n, top_elements=int(levels.sum()), M=40, ef_construction=300, random_seed=100, ft_bits=128,
                     attr_type=[0, 1], max_cate_size=L, allow_replace_deleted=False, edge_level_ft=True)
        h.initAttrMapping([[[0], sorted({int(attr[i]), i % L})] for i in range(N0)]); t = time.time()
        for s in range(0, N0, 100_000): e = min(N0, s + 100_000); h.add_items(X0[s:e], rec(range(s, e)), np.arange(s, e, dtype=np.uint64), num_threads=16, levels=levels[s:e], replace_deleted=False)
        del X0
        def configure():
            h.generateAttrIndexes(); h.set_thresholds(0.0001, 0.0001, 0.0001); h.set_ef_top(1); h.set_ft_flag(True)
            h.set_ft_routing_flag(True); h.set_ft_routing_min_deg(8); h.set_ft_routing_backfill_tail(False); h.set_num_threads(1)
        configure(); res['build_s'] = time.time() - t
        def do_search(q, prm):
            lab_, _ = h.hybrid_knn_query_dnf(Q[q:q + 1], [[[[], [int(qlab[q])]]]], k=K, num_threads=1); return np.asarray(lab_[0]).astype(np.int64)
        def set_param(prm): h.set_ef(prm)
        grid = [16, 32, 64, 128, 256, 512, 1024, 2048]
        def apply_delete(b): h.delete_items(np.asarray(b, np.uint64), num_threads=WT)
        def apply_insert(b): h.add_items(np.ascontiguousarray(X[b], np.float32), rec(b), np.asarray(b, np.uint64), num_threads=WT, levels=np.zeros(len(b), np.int32), replace_deleted=False)
        def round_end(): configure()      # attribute indexes regenerated as in stream_ema.py, after the writer, before the quiet phase
    def run_queries(qs, prm):
        out = np.zeros(len(qs), REC)
        for j, q in enumerate(qs):
            t0 = time.perf_counter(); ids = do_search(q, prm); t1 = time.perf_counter()
            out[j]['q'] = q; out[j]['t0'] = t0; out[j]['t1'] = t1; row = np.full(K, -1, np.int32); row[:len(ids)] = ids[:K]; out[j]['ids'] = row
        return out
    prm = grid[-1]
    for g in grid:
        set_param(g); o = run_queries(range(nq), g)
        if recall_vec(o['ids'], gt[0]).mean() >= 0.95: prm = g; break
    set_param(prm); res['param'] = str(prm); cycle = np.random.default_rng(7).permutation(nq)
    for r in range(1, R + 1):
        de, ins = sch[f'del{r-1}'].astype(np.int64), sch[f'ins{r-1}'].astype(np.int64); dtl = []; done = threading.Event(); counter = [0]; lock = threading.Lock(); recs = []
        def reader():
            mine = []
            while not done.is_set():
                with lock: c = counter[0]; counter[0] += 1
                mine.append(run_queries([int(cycle[c % nq])], prm))
            with lock: recs.extend(mine)
        def writer():
            di = ii = 0
            while di < len(de) or ii < len(ins):
                if di < len(de): b = de[di:di + BATCH]; apply_delete(b); t = time.perf_counter(); dtl.extend((int(x), t) for x in b); di += BATCH
                if ii < len(ins): apply_insert(ins[ii:ii + BATCH]); ii += BATCH
        th = [threading.Thread(target=reader) for _ in range(READERS)]; tw = time.perf_counter()
        for t_ in th: t_.start()
        writer(); ws = time.perf_counter() - tw; done.set()
        for t_ in th: t_.join()
        round_end()
        qparts = np.array_split(np.arange(nq), READERS)
        outs = [None] * READERS
        def qrun(i): outs[i] = run_queries(qparts[i], prm)
        th = [threading.Thread(target=qrun, args=(i,)) for i in range(READERS)]
        for t_ in th: t_.start()
        for t_ in th: t_.join()
        allr = np.concatenate([np.concatenate(recs) if recs else np.zeros(0, REC)] + outs)
        allr['phase'][len(allr) - nq:] = 1
        dtime = np.array(dtl, DT) if dtl else np.zeros(0, DT)
        res['rounds'].append(stats(allr, dtime, r, ws)); log(name, method, 'round', r, res['rounds'][-1]['concurrent'])
os.makedirs(f'{RES}/stream_{name}', exist_ok=True)
json.dump(res, open(f'{RES}/stream_{name}/cc_{method}.json', 'w'), indent=1, default=float); log('DONE', name, 'cc', method)

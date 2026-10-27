#!/usr/bin/env python3
"""Multi-label AND stream arms (follow-up 2, I18b; stream yfcc_stream from bench/stream_prep.py).
  taskset -c <cores> <python> bench/stream_multi.py <name> <scan|ivf|postfilter|ema|acorn>
Points carry tag sets, queries ask for the AND of their 1-2 tags; a point is admissible if live and carrying every tag.
The admissible ids of each query (the intersection of its tags' posting lists) are computed once outside the timers, as
for the static YFCC workloads (predicate evaluation is costed separately in the paper); liveness is applied per round.
  scan        exact scan of the live admissible rows of each query (numpy, one thread)
  ivf         IVF-Flat (nlist = 4 sqrt(N0), trained once on the initial rows), remove_ids + add_with_ids every round,
              per-query IDSelectorBitmap (.venv-faiss)
  postfilter  hnswlib HNSW (M=32, efC=200, deleted slots reused), k' sweep, results filtered by the predicate (main env)
  ema         EMA with the points' tags as one categorical attribute and the query's tags as one DNF clause (AND), one call per query,
              synchronous deletion, as bench/stream_ema.py (.venv-ema)
  acorn       ACORN-gamma via src/acorn_stream_mech_avx2 with ATTR_CSR (main env)
Maintenance 16 threads; search 1 thread; measured rounds 0, 5, 10."""
import sys, os, json, time, subprocess, tempfile
sys.path.insert(0, 'bench')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
from common import ROOT, RES, recall, recall_vec, timed_search, log, K, REPS, ci, write_fbin

name, method = sys.argv[1], sys.argv[2]; D = f'{ROOT}/stream_{name}'; meta = json.load(open(f'{D}/meta.json')); R, N0 = meta['R'], meta['N0']
X = np.load(f'{D}/Xall.npy', mmap_mode='r'); Q = np.ascontiguousarray(np.load(f'{D}/Q.npy'), np.float32); gt = np.load(f'{D}/gt.npy'); sch = np.load(f'{D}/sched.npz')
P = np.load(f'{D}/pred.npz'); pptr, pidx, qptr, qidx = P['ptag_ptr'], P['ptag_idx'], P['qtag_ptr'], P['qtag_idx']
n, d = X.shape; nq = len(Q); MR = meta['measure_rounds']
if os.environ.get('MAX_ROUND'): R = min(R, int(os.environ['MAX_ROUND'])); MR = [0, R]      # smoke tests only
loc = np.repeat(np.arange(n, dtype=np.int64), np.diff(pptr)); order = np.argsort(pidx, kind='stable')
ntag = int(pidx.max()) + 1; tptr = np.zeros(ntag + 1, np.int64); np.cumsum(np.bincount(pidx, minlength=ntag), out=tptr[1:]); tids = loc[order]; del loc, order
qtags = [qidx[qptr[i]:qptr[i + 1]] for i in range(nq)]
def adm_ids(tags):
    ids = tids[tptr[tags[0]]:tptr[tags[0] + 1]]
    for t in tags[1:]: ids = np.intersect1d(ids, tids[tptr[t]:tptr[t + 1]], assume_unique=True)
    return ids
QADM = [adm_ids(t) for t in qtags]
alive = np.zeros(n, bool); alive[:N0] = True
out = {'method': method, 'stream': name, 'rounds': []}; cum = 0.0

if method == 'acorn':
    tmp = tempfile.mkdtemp(dir=ROOT); os.makedirs(f'{tmp}/sched'); os.makedirs(f'{tmp}/out'); os.makedirs(f'{tmp}/csr')
    write_fbin(f'{tmp}/X.fbin', X); write_fbin(f'{tmp}/Q.fbin', Q)
    np.zeros(n, np.int32).tofile(f'{tmp}/attr.i32'); np.zeros(nq, np.int32).tofile(f'{tmp}/qlab.i32')
    for r in range(R): sch[f'del{r}'].astype(np.int32).tofile(f'{tmp}/sched/del{r}.i32'); sch[f'ins{r}'].astype(np.int32).tofile(f'{tmp}/sched/ins{r}.i32')
    pptr.astype(np.int64).tofile(f'{tmp}/csr/ptag_ptr.i64'); pidx.astype(np.int32).tofile(f'{tmp}/csr/ptag_idx.i32')
    qptr.astype(np.int64).tofile(f'{tmp}/csr/qtag_ptr.i64'); qidx.astype(np.int32).tofile(f'{tmp}/csr/qtag_idx.i32')
    efs = [16, 32, 64, 128, 256, 512, 1024, 2048]
    cmd = ['./src/acorn_stream_mech_avx2', f'{tmp}/X.fbin', f'{tmp}/Q.fbin', f'{tmp}/attr.i32', f'{tmp}/qlab.i32', f'{tmp}/sched', str(N0), str(R),
           '32', '12', '64', '16', '1', ','.join(map(str, efs)), f'{tmp}/out']
    p = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, 'ACORN_REPS': str(REPS), 'MEASURE': ','.join(map(str, MR)), 'ATTR_CSR': f'{tmp}/csr'})
    if p.returncode != 0: log('acorn_stream failed', p.returncode, p.stderr[-1500:]); sys.exit(1)
    for line in p.stdout.splitlines():
        if not line.startswith('{'): continue
        ev = json.loads(line)
        if ev['event'] == 'build': out['build_s'] = ev['build_s']; continue
        if ev['event'] != 'round': continue
        r = ev['round']; row = {k: ev[k] for k in ('round', 't_delete', 't_insert', 'cum_maint_s') if k in ev}
        if 'graph_bytes' in ev:
            row['sweep'] = []
            for ef in efs:
                o = np.fromfile(f'{tmp}/out/acorn_r{r}_ef{ef}.i32', np.int32).reshape(nq, K)
                row['sweep'].append({'param': f'ef={ef}', 'recall': recall(o, gt[r]), 'qps': ci(ev[f'qps_ef{ef}'])})
            log(name, 'acorn round', r, [(s['param'], round(s['recall'], 4), round(s['qps']['mean'], 1)) for s in row['sweep']])
        out['rounds'].append(row)
    subprocess.run(['rm', '-rf', tmp])
else:
    t0 = time.time()
    if method == 'ivf':
        import faiss
        faiss.omp_set_num_threads(16); nlist = int(4 * np.sqrt(N0)); quant = faiss.IndexFlatL2(d); ivf = faiss.IndexIVFFlat(quant, d, nlist)
        ivf.train(np.ascontiguousarray(X[np.random.default_rng(0).choice(N0, min(N0, 50 * nlist), replace=False)], np.float32))
        ivf.add_with_ids(np.ascontiguousarray(X[:N0], np.float32), np.arange(N0, dtype=np.int64))
    elif method == 'scan':
        Xf = np.ascontiguousarray(X, np.float32); SQ = (Xf * Xf).sum(1)
    elif method == 'postfilter':
        import hnswlib
        NI = len(sch['ins0']); h = hnswlib.Index(space='l2', dim=d); h.init_index(max_elements=N0 + NI, ef_construction=200, M=32, allow_replace_deleted=True)
        h.set_num_threads(16); h.add_items(np.ascontiguousarray(X[:N0], np.float32), np.arange(N0))
    elif method == 'ema':
        import faiss, hashannlib
        X0 = np.ascontiguousarray(X[:N0], np.float32); kk = int(4 * np.sqrt(N0)); faiss.omp_set_num_threads(16)
        km = faiss.Kmeans(d, kk, niter=10, seed=1, max_points_per_centroid=50); km.train(X0)
        fl = faiss.IndexFlatL2(d); fl.add(X0); _, Ic = fl.search(km.centroids, 1); levels = np.zeros(N0, np.int32); levels[np.unique(Ic[:, 0])] = 1
        rec = lambda ids: [[[0], pidx[pptr[i]:pptr[i + 1]].tolist() or [0]] for i in ids]
        h = hashannlib.Index(space='l2', dim=d)
        h.init_index(max_elements=n, top_elements=int(levels.sum()), M=40, ef_construction=300, random_seed=100, ft_bits=128,
                     attr_type=[0, 1], max_cate_size=ntag, allow_replace_deleted=False, edge_level_ft=True)
        h.initAttrMapping(rec(range(N0)))
        for s in range(0, N0, 100_000):
            e = min(N0, s + 100_000); h.add_items(X0[s:e], rec(range(s, e)), np.arange(s, e, dtype=np.uint64), num_threads=16, levels=levels[s:e], replace_deleted=False)
        del X0
        def configure():
            t = time.time(); h.generateAttrIndexes(); h.set_thresholds(0.0001, 0.0001, 0.0001); h.set_ef_top(1); h.set_ft_flag(True)
            h.set_ft_routing_flag(True); h.set_ft_routing_min_deg(8); h.set_ft_routing_backfill_tail(False); h.set_num_threads(1)
            return time.time() - t
        configure(); preds = [[[[], qtags[i].tolist()]] for i in range(nq)]
    out['build_s'] = time.time() - t0
    for r in range(R + 1):
        row = {'round': r, 't_delete': 0.0, 't_insert': 0.0}
        if r > 0:
            de, ins = sch[f'del{r-1}'].astype(np.int64), sch[f'ins{r-1}'].astype(np.int64)
            if method == 'ivf':
                faiss.omp_set_num_threads(16); t = time.time(); ivf.remove_ids(de); row['t_delete'] = time.time() - t
                t = time.time(); ivf.add_with_ids(np.ascontiguousarray(X[ins], np.float32), ins); row['t_insert'] = time.time() - t
            elif method == 'postfilter':
                t = time.time()
                for x in de: h.mark_deleted(int(x))
                row['t_delete'] = time.time() - t; h.set_num_threads(16)
                t = time.time(); h.add_items(np.ascontiguousarray(X[ins], np.float32), ins, replace_deleted=True); row['t_insert'] = time.time() - t
            elif method == 'ema':
                h.set_num_threads(16); t = time.time(); h.delete_items(de.astype(np.uint64), num_threads=16); row['t_delete'] = time.time() - t; t = time.time()
                for s in range(0, len(ins), 100_000):
                    b = ins[s:s + 100_000]; h.add_items(np.ascontiguousarray(X[b], np.float32), rec(b), b.astype(np.uint64), num_threads=16, levels=np.zeros(len(b), np.int32), replace_deleted=False)
                row['t_insert'] = time.time() - t
                if r in MR: row['t_configure'] = configure()
            alive[de] = False; alive[ins] = True
            cum += row['t_delete'] + row['t_insert'] + row.get('t_configure', 0.0)
        row['cum_maint_s'] = cum
        if r in MR:
            live_adm = [a[alive[a]] for a in QADM]; sweep = []; row['adm_live_median'] = float(np.median([len(a) for a in live_adm]))
            if method == 'scan':
                def run():
                    o = np.full((nq, K), -1, np.int32); tot = 0.0
                    for i in range(nq):
                        t = time.perf_counter(); c = live_adm[i]
                        if len(c):
                            dd = SQ[c] - 2 * Xf[c] @ Q[i]; kk = min(K, len(c)); p_ = np.argpartition(dd, kk - 1)[:kk]; top = c[p_[np.argsort(dd[p_])]]
                        else: top = c
                        tot += time.perf_counter() - t; o[i, :len(top)] = top
                    return o, tot
                o, q = timed_search(run, nq); sweep.append({'param': 'exact', 'recall': recall(o, gt[r]), 'qps': q})
            elif method == 'ivf':
                faiss.omp_set_num_threads(1); keep = []; sels = []
                for c in live_adm:
                    m = np.zeros(n, np.uint8); m[c] = 1; bits = np.packbits(m, bitorder='little'); keep.append(bits); sels.append(faiss.IDSelectorBitmap(n, faiss.swig_ptr(bits)))
                for npb in (8, 16, 32, 64, 128, 256, 512, 1024, 2048):
                    if npb > nlist // 2: break
                    ps = [faiss.SearchParametersIVF(sel=sels[i], nprobe=npb) for i in range(nq)]
                    def run():
                        o = np.full((nq, K), -1, np.int32); tot = 0.0
                        for i in range(nq):
                            t = time.perf_counter(); _, I = ivf.search(Q[i:i + 1], K, params=ps[i]); tot += time.perf_counter() - t; o[i] = I[0]
                        return o, tot
                    o, q = timed_search(run, nq); sweep.append({'param': f'nprobe={npb}', 'recall': recall(o, gt[r]), 'qps': q})
                    if sweep[-1]['recall'] > 0.999: break
            elif method == 'postfilter':
                h.set_num_threads(1); masks = []
                for c in live_adm: m = np.zeros(n, bool); m[c] = True; masks.append(m)
                for kp in (400, 1600, 6400):
                    h.set_ef(kp)
                    def run():
                        o = np.full((nq, K), -1, np.int32); tot = 0.0
                        for i in range(nq):
                            t = time.perf_counter()
                            try: I = h.knn_query(Q[i:i + 1], k=kp)[0][0]
                            except RuntimeError: I = h.knn_query(Q[i:i + 1], k=kp // 2)[0][0]
                            tot += time.perf_counter() - t; keep_ = I[masks[i][I]][:K]; o[i, :len(keep_)] = keep_
                        return o, tot
                    o, q = timed_search(run, nq); sweep.append({'param': f"k'={kp}", 'recall': recall(o, gt[r]), 'qps': q})
            elif method == 'ema':
                for ef in (16, 32, 64, 128, 256, 512, 1024, 2048):
                    h.set_ef(ef)
                    def run():   # one call per query, as the other arms here; a query EMA cannot fill to k asks for fewer (timed)
                        o = np.full((nq, K), -1, np.int64); tot = 0.0
                        for i in range(nq):
                            kk = min(K, len(live_adm[i])); t = time.perf_counter()
                            while kk > 0:
                                try: lb, _ = h.hybrid_knn_query_dnf(Q[i:i + 1], [preds[i]], k=kk, num_threads=1); o[i, :kk] = np.asarray(lb)[0]; break
                                except RuntimeError: kk //= 2
                            tot += time.perf_counter() - t
                        return o, tot
                    o, q = timed_search(run, nq)
                    ok = [(not len(row_[row_ >= 0])) or (alive[row_[row_ >= 0]].all() and all(np.isin(qtags[i], pidx[pptr[x]:pptr[x + 1]]).all() for x in row_[row_ >= 0])) for i, row_ in enumerate(o)]
                    sweep.append({'param': f'ef={ef}', 'recall': recall(o, gt[r]), 'qps': q, 'invalid_queries': int(nq - sum(ok))})
                    if sweep[-1]['recall'] > 0.999: break
            row['sweep'] = sweep
            log(name, method, 'round', r, 'cum', round(cum, 1), [(s['param'], round(s['recall'], 4), round(s['qps']['mean'], 1)) for s in sweep])
        out['rounds'].append(row)
os.makedirs(f'{RES}/stream_{name}', exist_ok=True)
json.dump(out, open(f'{RES}/stream_{name}/{method}.json', 'w'), indent=1, default=float); log('DONE', name, method)

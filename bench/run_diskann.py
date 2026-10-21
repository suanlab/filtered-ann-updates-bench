#!/usr/bin/env python3
"""Filtered-DiskANN arm (.venv-diskann; diskannpy 0.7 StaticMemoryIndex with filter labels).
  taskset -c <cores> .venv-diskann/bin/python bench/run_diskann.py <ds> <schema>
Label schemas: one label per point; each query filters on its label (native).
csr schemas (yfcc tags): points carry their tag sets; diskannpy filters on one label per query, so a
query with two tags (AND) searches its rarer tag with k'=L (swept) and keeps results that also carry the
other tag (marked 'rarest-tag+check'). diskannpy pads results with uninitialized ids when it finds fewer than k
matches, so every returned id is validated against the query's predicate. Build 32 threads (R=64, L=100, alpha=1.2, filter L=100);
search 1 thread, one query per call, L sweep."""
import sys, os, time, shutil
sys.path.insert(0, 'bench')
import numpy as np, diskannpy as dp
from common import load_base, load_schema, recall, per_query, timed_search, save, log, ds_dir, K, THREADS, sweep

ds, sc = sys.argv[1], sys.argv[2]
X, Q, meta = load_base(ds); pred, gt = load_schema(ds, sc); n, d = X.shape; nq = len(Q)
is_label = str(pred['kind']) == 'label'
if is_label:
    labels = [[str(int(a))] for a in pred['attr']]; qlab = [[str(int(b))] for b in pred['qlab']]
else:   # tag schemas: point tag sets and query tags (AND) stored in pred.npz (ptag_*, qtag_*)
    pp, pi = pred['ptag_ptr'], pred['ptag_idx']; qp, qi_ = pred['qtag_ptr'], pred['qtag_idx']
    labels = [[str(t) for t in pi[pp[i]:pp[i + 1]]] or ['none'] for i in range(n)]
    freq = np.bincount(pi, minlength=int(max(pi.max(), qi_.max())) + 1)
    qlab = [[str(t) for t in sorted(qi_[qp[i]:qp[i + 1]].tolist(), key=lambda t: freq[t])] for i in range(nq)]
idir = f'{ds_dir(ds)}/fdiskann_{sc}'; shutil.rmtree(idir, ignore_errors=True); os.makedirs(idir)
t0 = time.time()
dp.build_memory_index(np.ascontiguousarray(X, np.float32), 'l2', idir, complexity=100, graph_degree=64, num_threads=32,
                      alpha=1.2, filter_labels=labels, filter_complexity=100)
build = time.time() - t0; nbytes = sum(os.path.getsize(f'{idir}/{f}') for f in os.listdir(idir))
del labels
log(ds, sc, 'fdiskann built', round(build, 1), 's', round(nbytes / 1e9, 2), 'GB')
idx = dp.StaticMemoryIndex(idir, num_threads=THREADS, initial_search_complexity=1024, enable_filters=True)
if not is_label:
    qadm = [set(pred['qids'][pred['qoff'][i]:pred['qoff'][i + 1]].tolist()) for i in range(nq)]
recs = []
for L in sweep((16, 32, 64, 128, 256, 512, 1024)):
    def run():
        o = np.full((nq, K), -1, np.int32); tot = 0.0; tq = np.zeros(nq)
        for i in range(nq):
            if is_label:
                t = time.perf_counter(); r = idx.search(Q[i], K, L, filter_label=qlab[i][0]); tq[i] = time.perf_counter() - t; tot += tq[i]
                ids = [x for x in r.identifiers[:K].tolist() if x < n and pred['attr'][x] == pred['qlab'][i]]
            else:
                kk = K if len(qlab[i]) == 1 else L      # AND query: overfetch k' = L on the rarer tag, swept with L
                t = time.perf_counter(); r = idx.search(Q[i], kk, max(L, kk), filter_label=qlab[i][0]); tq[i] = time.perf_counter() - t; tot += tq[i]
                ids = [x for x in r.identifiers.tolist() if x in qadm[i]][:K]   # also drops diskannpy's
                # uninitialized padding when fewer than k matches are found
            ids = np.asarray(ids, np.int64); ids = ids[ids < n]; o[i, :len(ids)] = ids
        return o, tot, tq
    def run_pool():     # multi-threaded mode: the binding has no batched filtered search, so a thread pool of
        from concurrent.futures import ThreadPoolExecutor   # per-query calls (scales only if the binding releases the GIL)
        o = np.full((nq, K), -1, np.int32)
        def one(i):
            r = idx.search(Q[i], K, L, filter_label=qlab[i][0])
            return [x for x in r.identifiers[:K].tolist() if x < n and pred['attr'][x] == pred['qlab'][i]]
        t = time.perf_counter()
        with ThreadPoolExecutor(THREADS) as ex: res = list(ex.map(one, range(nq)))
        tot = time.perf_counter() - t
        for i, ids in enumerate(res): o[i, :len(ids)] = ids
        return o, tot
    if THREADS > 1 and is_label: (o, q), tq = timed_search(run_pool, nq), None
    else: o, q, tq = timed_search(run, nq)
    recs.append({'method': 'fdiskann', 'param': f'L={L}', 'recall': recall(o, gt), 'qps': q, 'threads': THREADS, 'build_s': build,
                 'bytes': nbytes, 'extra': {} if is_label else {'and_queries': 'rarest-tag+check'}, **per_query(o, gt, None if is_label else tq)})
    log(ds, sc, 'fdiskann', L, round(recs[-1]['recall'], 4), round(q['mean'], 1))
    if recs[-1]['recall'] > 0.999: break
save(ds, sc, 'fdiskann', recs); del idx; shutil.rmtree(idir, ignore_errors=True)
log('DONE', ds, sc)

#!/usr/bin/env python3
"""faiss-based arms (.venv-faiss).  OPENBLAS_NUM_THREADS=1 taskset -c <cores> .venv-faiss/bin/python bench/run_faiss.py <ds> <schema> [arms]
  scan_part   exact scan over the admissible points only (pre-filter). Label schemas: per-label
              partitions materialized at build time, queries batched per label (BLAS) and also one
              query per call; csr schemas: per-query gather of the admissible ids + scan.
  scan_mask   exact scan over all points with the predicate as a bitmap (IDSelectorBitmap).
  ivf         IVF-Flat (nlist = 4 sqrt n), predicate as IDSelectorBitmap, one query per call, nprobe sweep.
Predicates are handed to every method precomputed (admissible set / bitmap); predicate evaluation is
not timed for any method. Builds use 32 threads, searches 1 thread."""
import sys, time
sys.path.insert(0, 'bench')
import numpy as np, faiss
from common import load_base, load_schema, groups, admissible, admissible_ids, recall, per_query, timed_search, save, log, K, THREADS, MT_MODE, pool_map, faiss_bitmap

ds, sc = sys.argv[1], sys.argv[2]
arms = sys.argv[3].split(',') if len(sys.argv) > 3 else ['scan_part', 'scan_mask', 'ivf']
X, Q, meta = load_base(ds); pred, gt = load_schema(ds, sc); n, d = X.shape; nq = len(Q)
X = np.ascontiguousarray(X); G = groups(pred, nq); is_label = str(pred['kind']) == 'label'
KEEP = []
def bitmap(mask): return faiss_bitmap(mask, KEEP)

if 'scan_part' in arms:
    faiss.omp_set_num_threads(32); t0 = time.time()
    parts = {}
    if is_label:
        for key, _ in G:
            ids = admissible_ids(pred, key); ix = faiss.IndexFlatL2(d); ix.add(X[ids]); parts[key] = (ix, ids)
    build = time.time() - t0; faiss.omp_set_num_threads(THREADS); recs = []
    def run(batched):
        o = np.full((nq, K), -1, np.int32); tot = 0.0; tq = np.zeros(nq)
        for key, qi in G:
            if is_label:
                ix, ids = parts[key]
                if batched:
                    t = time.perf_counter(); _, I = ix.search(Q[qi], K); tot += time.perf_counter() - t
                    o[qi] = np.where(I >= 0, ids[np.maximum(I, 0)], -1)
                else:
                    for i in qi:
                        t = time.perf_counter(); _, I = ix.search(Q[i:i + 1], K); tq[i] = time.perf_counter() - t; tot += tq[i]
                        o[i] = np.where(I[0] >= 0, ids[np.maximum(I[0], 0)], -1)
            else:
                ids = admissible_ids(pred, key); i = qi[0]
                if len(ids) == 0: continue
                t = time.perf_counter(); _, I = faiss.knn(Q[i:i + 1], X[ids], min(K, len(ids))); tq[i] = time.perf_counter() - t; tot += tq[i]
                o[i, :I.shape[1]] = ids[I[0]]
        return (o, tot) if batched else (o, tot, tq)
    qkey = {int(i): key for key, qi in G for i in qi}
    def run_pool():                                       # query-parallel: single-threaded per-query calls
        faiss.omp_set_num_threads(1)
        def one(i):
            ix, ids = parts[qkey[i]]; _, I = ix.search(Q[i:i + 1], K); return np.where(I[0] >= 0, ids[np.maximum(I[0], 0)], -1)
        res, wall = pool_map(one, nq, THREADS); return np.stack(res).astype(np.int32), wall
    modes = ['pool'] if (THREADS > 1 and MT_MODE == 'pool' and is_label) else (([True] if THREADS > 1 else [True, False]) if is_label else [False])
    for batched in modes:
        res = timed_search(run_pool if batched == 'pool' else (lambda: run(batched)), nq); o, q = res[0], res[1]
        recs.append({'method': 'scan_part', 'param': 'pool' if batched == 'pool' else ('batched' if batched else 'per_query'), 'recall': recall(o, gt), 'qps': q,
                     'threads': THREADS, 'build_s': build, 'bytes': int(n * d * 4 + n * 8), **per_query(o, gt, res[2] if len(res) > 2 else None)})
        log(ds, sc, recs[-1]['method'], recs[-1]['param'], round(recs[-1]['recall'], 4), round(q['mean'], 1))
    save(ds, sc, 'scan_part', recs); del parts

if 'scan_mask' in arms or 'ivf' in arms:
    sels = {key: bitmap(admissible(pred, key, n)) for key, _ in G}

if 'scan_mask' in arms:
    full = faiss.IndexFlatL2(d); full.add(X); faiss.omp_set_num_threads(THREADS)
    def run():
        o = np.full((nq, K), -1, np.int32); tot = 0.0; tq = np.zeros(nq)
        for key, qi in G:
            t = time.perf_counter(); _, I = full.search(Q[qi], K, params=faiss.SearchParameters(sel=sels[key])); dt = time.perf_counter() - t
            tot += dt; tq[qi] = dt / len(qi); o[qi] = I
        return (o, tot) if is_label else (o, tot, tq)
    res = timed_search(run, nq); o, q = res[0], res[1]
    rec = {'method': 'scan_mask', 'param': 'bitmap', 'recall': recall(o, gt), 'qps': q, 'threads': THREADS, 'build_s': 0.0, 'bytes': int(n * d * 4),
           **per_query(o, gt, res[2] if len(res) > 2 else None)}
    log(ds, sc, 'scan_mask', round(rec['recall'], 4), round(q['mean'], 1)); save(ds, sc, 'scan_mask', [rec]); del full

if 'ivf' in arms:
    nlist = int(4 * np.sqrt(n)); faiss.omp_set_num_threads(32); t0 = time.time()
    quant = faiss.IndexFlatL2(d); ivf = faiss.IndexIVFFlat(quant, d, nlist)
    ivf.train(X[np.random.default_rng(0).choice(n, min(n, 50 * nlist), replace=False)]); ivf.add(X)
    build = time.time() - t0; faiss.omp_set_num_threads(THREADS); recs = []
    for npb in [p for p in (8, 16, 32, 64, 128, 256, 512, 1024, 2048) if p <= nlist // 2]:
        def run_batched():      # multi-threaded mode: one call per label group, Faiss parallelizes over queries
            o = np.full((nq, K), -1, np.int32); tot = 0.0
            for key, qi in G:
                p = faiss.SearchParametersIVF(sel=sels[key], nprobe=npb)
                t = time.perf_counter(); _, I = ivf.search(Q[qi], K, params=p); tot += time.perf_counter() - t; o[qi] = I
            return o, tot
        def run():
            o = np.full((nq, K), -1, np.int32); tot = 0.0; tq = np.zeros(nq)
            for key, qi in G:
                p = faiss.SearchParametersIVF(sel=sels[key], nprobe=npb)
                for i in qi:
                    t = time.perf_counter(); _, I = ivf.search(Q[i:i + 1], K, params=p); tq[i] = time.perf_counter() - t; tot += tq[i]
                    o[i] = I[0]
            return o, tot, tq
        def run_pool():                                   # query-parallel: single-threaded per-query calls
            faiss.omp_set_num_threads(1)
            ps = {key: faiss.SearchParametersIVF(sel=sels[key], nprobe=npb) for key, _ in G}; qk = {int(i): key for key, qi in G for i in qi}
            res, wall = pool_map(lambda i: ivf.search(Q[i:i + 1], K, params=ps[qk[i]])[1][0], nq, THREADS)
            return np.stack(res).astype(np.int32), wall
        if THREADS > 1: (o, q), tq = timed_search(run_pool if MT_MODE == 'pool' else run_batched, nq), None
        else: o, q, tq = timed_search(run, nq)
        recs.append({'method': 'ivf', 'param': f'nprobe={npb}', 'recall': recall(o, gt), 'qps': q, 'threads': THREADS, 'build_s': build,
                     'bytes': int(n * d * 4 + n * 8 + nlist * d * 4), 'extra': {'nlist': nlist}, **per_query(o, gt, tq)})
        log(ds, sc, 'ivf', npb, round(recs[-1]['recall'], 4), round(q['mean'], 1))
        if recs[-1]['recall'] > 0.999: break
    save(ds, sc, 'ivf', recs)
log('DONE', ds, sc)

#!/usr/bin/env python3
"""Per-call overhead of each method's search binding and the cost of evaluating YFCC predicates.
Run in each environment (whatever is importable is measured); writes results/bench/overhead_<env>.json.
  call overhead: median wall time of a one-query search call on a 2,000-point index whose search does
    almost no work (IVF nprobe=1, HNSW ef=10, DiskANN L=10, RIG ef=1), i.e. the binding's fixed cost
    that every per-query-timed method pays inside the timer
  predicate cost (YFCC, 500 queries): intersecting the tag posting lists of a query, building its
    10M-entry boolean mask, and packing it into a bitmap -- the work every method is handed for free"""
import sys, os, json, time
sys.path.insert(0, 'experiments'); sys.path.insert(0, 'bench')
import numpy as np
env = os.path.basename(os.path.dirname(os.path.dirname(sys.executable))) if '.venv' in sys.executable else 'main'
rng = np.random.default_rng(0); X = rng.standard_normal((2000, 64)).astype(np.float32); Q = rng.standard_normal((200, 64)).astype(np.float32)
out = {}
def med(fn, reps=200):
    ts = []
    for i in range(reps):
        t = time.perf_counter(); fn(i % len(Q)); ts.append(time.perf_counter() - t)
    return float(np.median(ts))
try:
    import faiss
    faiss.omp_set_num_threads(1); q = faiss.IndexFlatL2(64); ivf = faiss.IndexIVFFlat(q, 64, 16); ivf.train(X); ivf.add(X)
    bits = np.packbits(np.ones(2000, np.uint8), bitorder='little'); sel = faiss.IDSelectorBitmap(2000, faiss.swig_ptr(bits))
    p = faiss.SearchParametersIVF(sel=sel, nprobe=1)
    out['faiss_ivf_call_s'] = med(lambda i: ivf.search(Q[i:i + 1], 10, params=p))
except ImportError: pass
try:
    import diskannpy as dp, shutil
    d = 'data/bench/_ovh_fd'; shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    dp.build_memory_index(X, 'l2', d, complexity=32, graph_degree=16, num_threads=1, alpha=1.2, filter_labels=[['a']] * 2000, filter_complexity=32)
    ix = dp.StaticMemoryIndex(d, num_threads=1, initial_search_complexity=16, enable_filters=True)
    out['diskannpy_call_s'] = med(lambda i: ix.search(Q[i], 10, 10, filter_label='a')); shutil.rmtree(d)
except ImportError: pass
if env == 'main':
    import hnswlib, rigx
    h = hnswlib.Index(space='l2', dim=64); h.init_index(max_elements=2000, ef_construction=40, M=8); h.add_items(X); h.set_ef(10); h.set_num_threads(1)
    out['hnswlib_call_s'] = med(lambda i: h.knn_query(Q[i:i + 1], k=10))
    part = rigx.carve(X, threads=1); idx = rigx.RigIndex(X, part); m = np.ones(2000, bool); E = idx.entry_set(0.01)
    out['rig_ctypes_call_wall_s'] = med(lambda i: idx.search(Q[i:i + 1], mask=m, ef=1, threads=1, filtered_nav=True, entries=E))
    # YFCC predicate evaluation
    z = np.load('data/bench/yfcc10m/tags/pred.npz'); pp, pi = z['ptag_ptr'], z['ptag_idx']; qp, qi = z['qtag_ptr'], z['qtag_idx']; n = len(pp) - 1
    rows = np.repeat(np.arange(n, dtype=np.int32), np.diff(pp)); order = np.argsort(pi, kind='stable'); ntag = int(pi.max()) + 1
    tptr = np.zeros(ntag + 1, np.int64); np.cumsum(np.bincount(pi, minlength=ntag), out=tptr[1:]); tids = rows[order]; del rows, order
    t_int, t_mask, t_bits = [], [], []
    for i in range(len(qp) - 1):
        tags = qi[qp[i]:qp[i + 1]]
        t = time.perf_counter(); ids = tids[tptr[tags[0]]:tptr[tags[0] + 1]]
        for tg in tags[1:]: ids = np.intersect1d(ids, tids[tptr[tg]:tptr[tg + 1]], assume_unique=True)
        t_int.append(time.perf_counter() - t)
        t = time.perf_counter(); mk = np.zeros(n, bool); mk[ids] = True; t_mask.append(time.perf_counter() - t)
        t = time.perf_counter(); np.packbits(mk, bitorder='little'); t_bits.append(time.perf_counter() - t)
    for k, v in (('yfcc_intersect_s', t_int), ('yfcc_mask_s', t_mask), ('yfcc_bitmap_pack_s', t_bits)):
        out[k] = {'median': float(np.median(v)), 'mean': float(np.mean(v)), 'p99': float(np.quantile(v, 0.99))}
if env == 'main':
    # method-specific work outside the timer: post-filtering's selection of admissible candidates (k'=6400, the
    # largest sweep value, on a 10M-row label mask) and Filtered-DiskANN's check of the second tag of an AND query
    # (membership of L=1024 returned ids in the admissible set)
    n = 10_000_000; m = np.random.default_rng(1).random(n) < 0.1; I = np.random.default_rng(2).integers(0, n, (200, 6400))
    out['postfilter_select_kp6400_s'] = med(lambda i: I[i][m[I[i]]][:10])
    adm = set(np.random.default_rng(3).choice(n, 100_000, replace=False).tolist()); ids = np.random.default_rng(4).integers(0, n, (200, 1024))
    out['fdiskann_and_check_L1024_s'] = med(lambda i: [x for x in ids[i].tolist() if x in adm][:10])
os.makedirs('results/bench', exist_ok=True); json.dump(out, open(f'results/bench/overhead_{env}.json', 'w'), indent=1); print(env, out)

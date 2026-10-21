#!/usr/bin/env python3
"""Main-env arms.  taskset -c <cores> python3 bench/run_main.py <ds> <schema> [arms]
  rig         regenerable pool index (C++ carver, int8 leaf-major pools, filtered navigation with
              predicate-first scans, 1% hash-priority entry sample), ef sweep
  postfilter  hnswlib HNSW (M=32, efC=200) returning k' candidates (ef = k'), then filtered
Builds use 32 threads; searches 1 thread. Index structures are built once per dataset and reused
across schemas when the runner is given several schemas separated by '+'."""
import sys, time, os
sys.path.insert(0, 'experiments'); sys.path.insert(0, 'bench')
import numpy as np
from common import load_base, load_schema, groups, admissible, recall, per_query, timed_search, save, log, K, THREADS, MT_MODE, pool_map, sweep

ds = sys.argv[1]; schemas = sys.argv[2].split('+')
arms = sys.argv[3].split(',') if len(sys.argv) > 3 else ['rig', 'postfilter']
X, Q, meta = load_base(ds); X = np.ascontiguousarray(X); n, d = X.shape; nq = len(Q)

if 'rig' in arms:
    import rigx
    t0 = time.time(); part = rigx.carve(X, threads=32); idx = rigx.RigIndex(X, part); build = time.time() - t0
    E = idx.entry_set(0.01); nbytes = idx.nbytes()
    log(ds, 'rig built', round(build, 1), 's', round(nbytes / 1e9, 2), 'GB')
    for sc in schemas:
        pred, gt = load_schema(ds, sc); G = groups(pred, nq); masks = {key: admissible(pred, key, n) for key, _ in G}
        is_label = str(pred['kind']) == 'label'; recs = []
        for ef in (16, 32, 64, 128, 256, 512):
            def run():
                o = np.full((nq, K), -1, np.int32); tot = 0.0; tq = np.zeros(nq)
                for key, qi in G:
                    oo, _, dt = idx.search(Q[qi], mask=masks[key], ef=ef, threads=THREADS, filtered_nav=True, entries=E, t_entry=8)
                    o[qi] = oo; tot += dt; tq[qi] = dt / len(qi)
                return o, tot, tq
            mu = {key: m.astype(np.uint8) for key, m in masks.items()}; qk = {int(i): key for key, qi in G for i in qi}
            def run_pool():                               # query-parallel: single-threaded per-query kernel calls
                res, wall = pool_map(lambda i: idx.search(Q[i:i + 1], mask=mu[qk[i]], ef=ef, threads=1, filtered_nav=True, entries=E, t_entry=8)[0][0], nq, THREADS)
                return np.stack(res).astype(np.int32), wall
            if THREADS > 1 and MT_MODE == 'pool': (o, q), tq = timed_search(run_pool, nq), None
            else: o, q, tq = timed_search(run, nq)
            recs.append({'method': 'rig', 'param': f'ef={ef}', 'recall': recall(o, gt), 'qps': q, 'threads': THREADS,
                         'build_s': build, 'bytes': nbytes, **per_query(o, gt, tq if not is_label else None)})
            log(ds, sc, 'rig', ef, round(recs[-1]['recall'], 4), round(q['mean'], 1))
            if recs[-1]['recall'] > 0.999 or q['mean'] < 2: break
        save(ds, sc, 'rig', recs)
    del idx, part

if 'postfilter' in arms:
    import hnswlib
    t0 = time.time(); h = hnswlib.Index(space='l2', dim=d); h.init_index(max_elements=n, ef_construction=200, M=32)
    h.set_num_threads(32); h.add_items(X, np.arange(n)); build = time.time() - t0; h.set_num_threads(THREADS)
    nbytes = int(n * (d * 4 + 2 * 32 * 4 + 4 + 8 + 8))
    log(ds, 'hnsw built', round(build, 1), 's')
    for sc in schemas:
        pred, gt = load_schema(ds, sc); G = groups(pred, nq); masks = {key: admissible(pred, key, n) for key, _ in G}
        is_label = str(pred['kind']) == 'label'; recs = []
        for kp in sweep((100, 400, 1600, 6400)):
            h.set_ef(kp)
            # multi-threaded mode: one call over all queries (post-filtering needs no per-label batches, and
            # hnswlib's binding runs single-threaded when a batch has at most 4x as many rows as threads)
            GG = [(None, np.arange(nq))] if THREADS > 1 else G
            def run():
                o = np.full((nq, K), -1, np.int32); tot = 0.0; tq = np.zeros(nq)
                for key, qi in GG:
                    t = time.perf_counter(); I, _ = h.knn_query(Q[qi], k=kp); dt = time.perf_counter() - t; tot += dt; tq[qi] = dt / len(qi)
                    for r, i in enumerate(qi):
                        mk = masks[key] if key is not None else masks[[k_ for k_, q_ in G if i in q_][0]]
                        keep = I[r][mk[I[r]]][:K]; o[i, :len(keep)] = keep
                return o, tot, tq
            o, q, tq = timed_search(run, nq)
            recs.append({'method': 'postfilter', 'param': f"k'={kp}", 'recall': recall(o, gt), 'qps': q, 'threads': THREADS,
                         'build_s': build, 'bytes': nbytes, **per_query(o, gt, tq if not is_label else None)})
            log(ds, sc, 'postfilter', kp, round(recs[-1]['recall'], 4), round(q['mean'], 1))
            if recs[-1]['recall'] > 0.999: break
        save(ds, sc, 'postfilter', recs)
log('DONE', ds, schemas)

#!/usr/bin/env python3
"""Stream arms in the main env.  taskset -c <cores> python3 bench/stream_main.py <name> <rig|postfilter>
  rig         pivot-retaining dynamic path: insertions routed down the fixed leader tree, deletions
              removed from their leaves, compaction (live-only CSR + leaf-major layout) every round;
              filtered navigation from live hash-priority entries
  postfilter  hnswlib HNSW (M=32, efC=200, allow_replace_deleted: deleted slots are reused), k' sweep
Maintenance on 16 threads, every round; search sweeps (1 thread, REPS reps) at the measured rounds.
Output results/bench/stream_<name>/<method>.json."""
import sys, os, json, time
sys.path.insert(0, 'experiments'); sys.path.insert(0, 'bench')
import numpy as np
from common import ROOT, RES, recall, timed_search, log, K

name, method = sys.argv[1], sys.argv[2]; D = f'{ROOT}/stream_{name}'
meta = json.load(open(f'{D}/meta.json')); R, N0 = meta['R'], meta['N0']
X = np.load(f'{D}/Xall.npy'); Q = np.load(f'{D}/Q.npy'); attr = np.load(f'{D}/attr.npy'); qlab = np.load(f'{D}/qlab.npy')
gt = np.load(f'{D}/gt.npy'); sch = np.load(f'{D}/sched.npz'); n, d = X.shape; nq = len(Q); TH = 16
G = [(b, np.where(qlab == b)[0]) for b in range(10)]
out = {'method': method, 'rounds': []}; t0 = time.time()
if method == 'rig':
    import rigx
    dr = rigx.DynamicRig(X, np.arange(N0), threads=TH); idx, _ = dr.refresh()
    alive = dr.alive
else:
    import hnswlib
    NI = len(sch['ins0'])
    h = hnswlib.Index(space='l2', dim=d); h.init_index(max_elements=N0 + NI, ef_construction=200, M=32, allow_replace_deleted=True)
    h.set_num_threads(TH); h.add_items(X[:N0], np.arange(N0)); alive = np.zeros(n, bool); alive[:N0] = True
out['build_s'] = time.time() - t0; cum = 0.0; log(name, method, 'built', round(out['build_s'], 1))
for r in range(R + 1):
    row = {'round': r, 't_delete': 0.0, 't_insert': 0.0, 't_compact': 0.0}
    if r > 0:
        de, ins = sch[f'del{r-1}'], sch[f'ins{r-1}']
        if method == 'rig':
            t = time.time(); dr.delete(de); row['t_delete'] = time.time() - t
            t = time.time(); dr.insert(ins); row['t_insert'] = time.time() - t
            idx, row['t_compact'] = dr.refresh()
        else:
            t = time.time()
            for x in de: h.mark_deleted(int(x))
            row['t_delete'] = time.time() - t; alive[de] = False
            t = time.time(); h.set_num_threads(TH); h.add_items(X[ins], ins, replace_deleted=True); row['t_insert'] = time.time() - t
            alive[ins] = True
        cum += row['t_delete'] + row['t_insert'] + row['t_compact']
    row['cum_maint_s'] = cum
    if r in meta['measure_rounds']:
        masks = {b: alive & (attr == b) for b, _ in G}; sweep = []
        if method == 'rig':
            E = dr.live_entries(0.01); row['bytes'] = idx.nbytes(int(alive.sum())); row['resident'] = int(alive.sum())
            row['ghost_pivots'] = int(dr.ghost_pivots())
            for ef in (16, 32, 64, 128, 256, 512):
                def run():
                    o = np.full((nq, K), -1, np.int32); tot = 0.0
                    for b, qi in G:
                        oo, _, dt = idx.search(Q[qi], mask=masks[b], ef=ef, threads=1, filtered_nav=True, entries=E, t_entry=8)
                        o[qi] = oo; tot += dt
                    return o, tot
                o, q = timed_search(run, nq); sweep.append({'param': f'ef={ef}', 'recall': recall(o, gt[r]), 'qps': q})
                if sweep[-1]['recall'] > 0.999: break
        else:
            h.set_num_threads(1); row['resident'] = int(h.get_current_count())
            row['bytes'] = int(h.get_max_elements() * (d * 4 + 2 * 32 * 4 + 4 + 8 + 8))
            for kp in (400, 1600, 6400):
                h.set_ef(kp)
                short = [0]
                def knn(qs, k):
                    # after many replaced deletions hnswlib may be unable to return k results for some
                    # queries ("Cannot return the results in a contiguous 2D array"); fall back per query
                    # to the largest k it can return, and count those queries
                    try: return [row for row in h.knn_query(qs, k=k)[0]]
                    except RuntimeError:
                        out = []
                        for x in qs:
                            kk = k
                            while True:
                                try: out.append(h.knn_query(x[None], k=kk)[0][0]); break
                                except RuntimeError: kk //= 2
                            if kk < k: short[0] += 1
                        return out
                def run():
                    o = np.full((nq, K), -1, np.int32); tot = 0.0; short[0] = 0
                    for b, qi in G:
                        t = time.perf_counter(); I = knn(Q[qi], kp); tot += time.perf_counter() - t
                        for j, i in enumerate(qi):
                            keep = I[j][masks[b][I[j]]][:K]; o[i, :len(keep)] = keep
                    return o, tot
                o, q = timed_search(run, nq)
                sweep.append({'param': f"k'={kp}", 'recall': recall(o, gt[r]), 'qps': q, 'short_queries': short[0]})
        row['sweep'] = sweep
        log(name, method, 'round', r, 'cum', round(cum, 1), [(s['param'], round(s['recall'], 4), round(s['qps']['mean'], 1)) for s in sweep])
    out['rounds'].append(row)
os.makedirs(f'{RES}/stream_{name}', exist_ok=True)
json.dump(out, open(f'{RES}/stream_{name}/{method}.json', 'w'), indent=1, default=float); log('DONE', name, method)

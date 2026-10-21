#!/usr/bin/env python3
"""H6: Filtered-DiskANN with a stale main index and an exact per-label delta (.venv-diskann).
  OPENBLAS_NUM_THREADS=1 taskset -c <cores> .venv-diskann/bin/python bench/stream_diskann_delta.py <name> [age,age,...]
The main index is built once over the live set of round 0 (16 threads; its build time is the rebuild cost B).
At round a (the index's age), a query searches the stale index, drops results deleted since round 0 (asking for
more results in proportion to the share of round-0 points that died), scans exactly the points of its label inserted
since round 0 and still live (the delta), and merges the two by distance. Both parts are timed, 1 thread, L sweep.
The analysis turns QPS-by-age into the total cost of rebuilding every P rounds, B/P + Q * mean_{a<P} 1/QPS(a),
and compares the best P with P* = sqrt(2B / (Q c s u)) (economic order quantity)."""
import sys, os, json, time, shutil
sys.path.insert(0, 'bench')
import numpy as np, diskannpy as dp
from common import ROOT, RES, recall, timed_search, log, K

name = sys.argv[1]; D = f'{ROOT}/stream_{name}'; meta = json.load(open(f'{D}/meta.json')); R, N0 = meta['R'], meta['N0']
AGES = [int(a) for a in sys.argv[2].split(',')] if len(sys.argv) > 2 else [0, 1, 2, 4, 8]
X = np.load(f'{D}/Xall.npy', mmap_mode='r'); Q = np.load(f'{D}/Q.npy'); attr = np.load(f'{D}/attr.npy'); qlab = np.load(f'{D}/qlab.npy')
gt = np.load(f'{D}/gt.npy'); sch = np.load(f'{D}/sched.npz'); nq = len(Q)
alive = np.zeros(len(X), bool); alive[:N0] = True
base = np.where(alive)[0]; idir = f'{D}/fdd_r0'; shutil.rmtree(idir, ignore_errors=True); os.makedirs(idir)
t0 = time.time()
dp.build_memory_index(np.ascontiguousarray(X[base], np.float32), 'l2', idir, complexity=100, graph_degree=64, num_threads=16,
                      alpha=1.2, filter_labels=[[str(int(attr[x]))] for x in base], filter_complexity=100)
B = time.time() - t0
idx = dp.StaticMemoryIndex(idir, num_threads=1, initial_search_complexity=1024, enable_filters=True)
out = {'method': 'fdiskann_delta', 'rebuild_s': B, 'ages': []}
r = 0
for age in AGES:
    while r < age: alive[sch[f'del{r}']] = False; alive[sch[f'ins{r}']] = True; r += 1
    surv = float(alive[base].mean())
    inserted = np.zeros(len(X), bool); inserted[N0:] = True     # rows after N0 are inserted after round 0
    delta = {int(b): np.where(alive & inserted & (attr == b))[0] for b in np.unique(qlab)}
    DX = {b: np.ascontiguousarray(X[ids], np.float32) for b, ids in delta.items()}; DN = {b: (v * v).sum(1) for b, v in DX.items()}
    sweep = []
    for L in (16, 32, 64, 128, 256, 512, 1024):
        kq = min(L, int(np.ceil(K / max(surv, 1e-3))))
        def run():
            o = np.full((nq, K), -1, np.int32); tot = 0.0
            for i in range(nq):
                b = int(qlab[i]); t = time.perf_counter()
                res = idx.search(Q[i], kq, L, filter_label=str(b))
                ids = res.identifiers[:kq].astype(np.int64); ds = res.distances[:kq]
                ok = (ids >= 0) & (ids < len(base)); ids, ds = base[ids[ok]], ds[ok]
                ok = alive[ids] & (attr[ids] == b); ids, ds = ids[ok], ds[ok]          # deleted since round 0, padding
                if len(delta[b]):
                    dd = DN[b] - 2 * DX[b] @ Q[i]                                         # + |q|^2, constant per query
                    kk = min(K, len(dd)); p = np.argpartition(dd, kk - 1)[:kk]
                    ids = np.concatenate([ids, delta[b][p]]); ds = np.concatenate([ds, dd[p] + float(Q[i] @ Q[i])])
                top = ids[np.argsort(ds, kind='stable')[:K]]; tot += time.perf_counter() - t; o[i, :len(top)] = top
            return o, tot
        o, q = timed_search(run, nq); sweep.append({'param': f'L={L}', 'k_search': kq, 'recall': recall(o, gt[age]), 'qps': q})
        if sweep[-1]['recall'] > 0.999: break
    out['ages'].append({'age': age, 'survival': surv, 'delta_points': int(sum(len(v) for v in delta.values())),
                        'delta_per_label_mean': float(np.mean([len(v) for v in delta.values()])), 'sweep': sweep})
    log(name, 'fdiskann+delta age', age, 'survival', round(surv, 3), [(s['param'], round(s['recall'], 4), round(s['qps']['mean'], 1)) for s in sweep])
del idx; shutil.rmtree(idir, ignore_errors=True)
os.makedirs(f'{RES}/stream_{name}', exist_ok=True)
json.dump(out, open(f'{RES}/stream_{name}/fdiskann_delta.json', 'w'), indent=1, default=float); log('DONE', name, 'fdiskann_delta')

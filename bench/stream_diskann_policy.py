#!/usr/bin/env python3
"""Review experiment E5: run a rebuild policy instead of fitting one (.venv-diskann).
  OPENBLAS_NUM_THREADS=1 taskset -c <cores> .venv-diskann/bin/python bench/stream_diskann_policy.py <name> <P>
Filtered-DiskANN is built over the live set of round 0 and rebuilt over the live set every P rounds (16 threads; each
rebuild's time is recorded); between rebuilds, a query searches the stale index, drops results deleted since its
build, scans exactly the points of its label inserted since the build and still live, and merges the two by distance
(as bench/stream_diskann_delta.py). Unlike that runner, which measures one index at several ages and lets the analysis
fit the period, this one executes the policy over the whole stream and measures every round (1 thread, L sweep), so
the executed total cost of any query rate can be computed from the same run: sum of rebuild and delta-preparation
times + Q * sum over rounds of 1/QPS at the recall target. The stale index is asked for K / s_l results, s_l being
the surviving share of the query's label among the build's points, and asked again with twice as many (up to L,
inside the timer) when fewer than K live results come back; queries still short after that are counted per sweep
point. P >= R means the index built at round 0 is never rebuilt within the stream."""
import sys, os, json, time, shutil
sys.path.insert(0, 'bench')
import numpy as np, diskannpy as dp
from common import ROOT, RES, recall, timed_search, log, K

name, P = sys.argv[1], int(sys.argv[2]); D = f'{ROOT}/stream_{name}'; meta = json.load(open(f'{D}/meta.json')); R, N0 = meta['R'], meta['N0']
X = np.load(f'{D}/Xall.npy', mmap_mode='r'); Q = np.load(f'{D}/Q.npy'); attr = np.load(f'{D}/attr.npy'); qlab = np.load(f'{D}/qlab.npy')
gt = np.load(f'{D}/gt.npy'); sch = np.load(f'{D}/sched.npz'); nq = len(Q)
alive = np.zeros(len(X), bool); alive[:N0] = True
idir = f'{D}/fdp_{P}_{os.getpid()}'

def build():
    base = np.where(alive)[0]; shutil.rmtree(idir, ignore_errors=True); os.makedirs(idir); t0 = time.time()
    dp.build_memory_index(np.ascontiguousarray(X[base], np.float32), 'l2', idir, complexity=100, graph_degree=64, num_threads=16,
                          alpha=1.2, filter_labels=[[str(int(attr[x]))] for x in base], filter_complexity=100)
    b = time.time() - t0
    return base, dp.StaticMemoryIndex(idir, num_threads=1, initial_search_complexity=1024, enable_filters=True), b, alive.copy()

base, idx, B0, alive_at_build = build()
out = {'method': f'fdiskann_policy_P{P}', 'P': P, 'build_s': B0, 'rounds': []}
for r in range(min(R, int(os.environ.get('MAX_ROUND', R))) + 1):     # MAX_ROUND: smoke tests only
    row = {'round': r, 'rebuilt': False, 'rebuild_s': 0.0}
    if r > 0:
        alive[sch[f'del{r-1}']] = False; alive[sch[f'ins{r-1}']] = True
        if r % P == 0 and r < R + 1 and P < R:
            del idx; base, idx, b, alive_at_build = build(); row['rebuilt'] = True; row['rebuild_s'] = b
    surv = float(alive[base].mean())
    t = time.time()                                    # delta preparation is maintenance (it was outside every timer)
    delta = {int(c): np.where(alive & ~alive_at_build & (attr == c))[0] for c in np.unique(qlab)}   # live, inserted since the build
    DX = {c: np.ascontiguousarray(X[ids], np.float32) for c, ids in delta.items()}; DN = {c: (v * v).sum(1) for c, v in DX.items()}
    row['t_delta'] = time.time() - t
    # survival of the build's points per label: the stale index is asked for K / survival results of the query's label
    surv_l = {int(c): max(float(alive[base[attr[base] == c]].mean()), 1e-3) for c in np.unique(qlab)}
    sweep = []
    for L in (16, 32, 64, 128, 256, 512, 1024):
        kq0 = {c: min(L, int(np.ceil(K / v))) for c, v in surv_l.items()}
        def run():
            o = np.full((nq, K), -1, np.int32); tot = 0.0; short = 0
            for i in range(nq):
                c = int(qlab[i]); t = time.perf_counter(); kq = kq0[c]
                while True:      # too few live results from the stale index: ask for more, up to L (timed)
                    res = idx.search(Q[i], kq, L, filter_label=str(c))
                    ids = res.identifiers[:kq].astype(np.int64); ds = res.distances[:kq]
                    ok = (ids >= 0) & (ids < len(base)); ids, ds = base[ids[ok]], ds[ok]
                    ok = alive[ids] & (attr[ids] == c); ids, ds = ids[ok], ds[ok]
                    if len(ids) >= K or kq >= L: break
                    kq = min(L, 2 * kq)
                short += len(ids) < K
                if len(delta[c]):
                    dd = DN[c] - 2 * DX[c] @ Q[i]; kk = min(K, len(dd)); p = np.argpartition(dd, kk - 1)[:kk]
                    ids = np.concatenate([ids, delta[c][p]]); ds = np.concatenate([ds, dd[p] + float(Q[i] @ Q[i])])
                top = ids[np.argsort(ds, kind='stable')[:K]]; tot += time.perf_counter() - t; o[i, :len(top)] = top
            run.short = short
            return o, tot
        o, q = timed_search(run, nq)
        sweep.append({'param': f'L={L}', 'k_search_initial_mean': float(np.mean([kq0[int(c)] for c in qlab])), 'stale_short_queries': run.short,
                      'recall': recall(o, gt[r]), 'qps': q})
        if sweep[-1]['recall'] > 0.999: break
    row.update(survival=surv, delta_points=int(sum(len(v) for v in delta.values())), sweep=sweep); out['rounds'].append(row)
    log(name, f'policy P={P} round', r, 'rebuilt' if row['rebuilt'] else '', round(row['rebuild_s'], 1), 'delta', row['delta_points'],
        [(s['param'], round(s['recall'], 4), round(s['qps']['mean'], 1)) for s in sweep])
del idx; shutil.rmtree(idir, ignore_errors=True)
os.makedirs(f'{RES}/stream_{name}', exist_ok=True)
json.dump(out, open(f'{RES}/stream_{name}/fdiskann_policy_P{P}.json', 'w'), indent=1, default=float); log('DONE', name, f'fdiskann_policy P={P}')

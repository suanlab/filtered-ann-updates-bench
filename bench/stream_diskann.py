#!/usr/bin/env python3
"""Filtered-DiskANN stream arm (.venv-diskann). diskannpy exposes no dynamic filtered index, so the
index is rebuilt over the live set at each measured round (16 threads); the rebuild time is the
maintenance cost of keeping it fresh at that cadence (reported per rebuild). L sweep, 1 thread.
The cost analysis estimates fresh-every-round maintenance as the mean measured rebuild times R
(maintenance_source 'estimated'). REBUILD_ALL=1 rebuilds at every round and sweeps at the measured
rounds only, so the estimate can be checked against the executed total (follow-up 4; 'measured')."""
import sys, os, json, time, shutil
sys.path.insert(0, 'bench')
import numpy as np, diskannpy as dp
from common import ROOT, RES, recall, timed_search, log, K, E2E, latency_pct, peak_rss_mb

name = sys.argv[1]; D = f'{ROOT}/stream_{name}'; meta = json.load(open(f'{D}/meta.json')); R, N0 = meta['R'], meta['N0']
X = np.load(f'{D}/Xall.npy', mmap_mode='r'); Q = np.load(f'{D}/Q.npy'); attr = np.load(f'{D}/attr.npy'); qlab = np.load(f'{D}/qlab.npy')
gt = np.load(f'{D}/gt.npy'); sch = np.load(f'{D}/sched.npz'); nq = len(Q); d = X.shape[1]
REBUILD_ALL = os.environ.get('REBUILD_ALL') == '1'
alive = np.zeros(len(X), bool); alive[:N0] = True
out = {'method': 'fdiskann_rebuild', 'maintenance_source': 'measured' if REBUILD_ALL else 'estimated', 'rounds': []}
for r in range(R + 1):
    if r > 0: alive[sch[f'del{r-1}']] = False; alive[sch[f'ins{r-1}']] = True
    sweep_here = r in meta['measure_rounds']
    if not (sweep_here or REBUILD_ALL): continue
    live = np.where(alive)[0]; idir = f'{D}/fd_r{r}'; shutil.rmtree(idir, ignore_errors=True); os.makedirs(idir)
    t0 = time.time()
    dp.build_memory_index(np.ascontiguousarray(X[live], np.float32), 'l2', idir, complexity=100, graph_degree=64, num_threads=16,
                          alpha=1.2, filter_labels=[[str(int(attr[x]))] for x in live], filter_complexity=100)
    tb = time.time() - t0; nbytes = sum(os.path.getsize(f'{idir}/{f}') for f in os.listdir(idir))
    if not sweep_here:
        out['rounds'].append({'round': r, 'rebuild_s': tb, 'resident': int(len(live)), 'bytes': nbytes}); log(name, 'fdiskann round', r, 'rebuild', round(tb, 1))
        shutil.rmtree(idir, ignore_errors=True); continue
    idx = dp.StaticMemoryIndex(idir, num_threads=1, initial_search_complexity=1024, enable_filters=True); sweep = []
    for L in (16, 32, 64, 128, 256, 512, 1024):
        def run():
            o = np.full((nq, K), -1, np.int32); tot = 0.0; per = np.zeros(nq)
            for i in range(nq):
                t = time.perf_counter(); res = idx.search(Q[i], K, L, filter_label=str(int(qlab[i]))); t2 = time.perf_counter()
                if not E2E: tot += t2 - t
                ids = res.identifiers[:K].astype(np.int64); ids = ids[(ids >= 0) & (ids < len(live))]; ids = live[ids]
                ids = ids[attr[ids] == qlab[i]]; o[i, :len(ids)] = ids        # drop diskannpy's uninitialized padding
                if E2E: dt = time.perf_counter() - t; tot += dt; per[i] = dt     # id mapping and validation are part of the query
            return (o, tot, per) if E2E else (o, tot)
        res_ = timed_search(run, nq); o, q = res_[0], res_[1]; sweep.append({'param': f'L={L}', 'recall': recall(o, gt[r]), 'qps': q})
        if E2E: sweep[-1]['latency'] = latency_pct(res_[2])
        if sweep[-1]['recall'] > 0.999: break
    out['rounds'].append({'round': r, 'rebuild_s': tb, 'resident': int(len(live)), 'bytes': nbytes, 'sweep': sweep})
    log(name, 'fdiskann round', r, 'rebuild', round(tb, 1), [(s['param'], round(s['recall'], 4), round(s['qps']['mean'], 1)) for s in sweep])
    del idx; shutil.rmtree(idir, ignore_errors=True)
os.makedirs(f'{RES}/stream_{name}', exist_ok=True)
if E2E: out['e2e'] = True; out['peak_rss_mb'] = peak_rss_mb()
json.dump(out, open(f'{RES}/stream_{name}/fdiskann_rebuild.json', 'w'), indent=1, default=float); log('DONE', name, 'fdiskann')

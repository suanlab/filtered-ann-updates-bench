#!/usr/bin/env python3
"""Cost of the per-label survival that stream_diskann_policy.py computes outside its timers (REPRODUCE.md).
  taskset -c <core> .venv-faiss/bin/python bench/survival_cost.py      -> results/bench_mb/survival.json
  (OUT_DIR=<dir>, or a BENCH_TAG, redirects the output: common.output_dir; a tagged re-execution keeps the original)
Replays each SIFT stream's schedule against the initial build (the largest stale set) and times the same expression
as the runner, once per round."""
import sys, os, json, time
sys.path.insert(0, 'bench')
import numpy as np
from common import ROOT, output_dir

out = {}
for nm in ('sift500k', 'sift500k_adv'):
    D = f'{ROOT}/stream_{nm}'; meta = json.load(open(f'{D}/meta.json')); R, N0 = meta['R'], meta['N0']
    attr = np.load(f'{D}/attr.npy'); qlab = np.load(f'{D}/qlab.npy'); sch = np.load(f'{D}/sched.npz')
    alive = np.zeros(len(attr), bool); alive[:N0] = True; base = np.where(alive)[0]; ts = []
    for r in range(R + 1):
        if r > 0: alive[sch[f'del{r-1}']] = False; alive[sch[f'ins{r-1}']] = True
        t = time.perf_counter(); _ = {int(c): max(float(alive[base[attr[base] == c]].mean()), 1e-3) for c in np.unique(qlab)}
        ts.append(time.perf_counter() - t)
    out[nm] = {'rounds': R + 1, 'ms_median': 1e3 * float(np.median(ts)), 'ms_max': 1e3 * float(max(ts))}
    print(nm, out[nm], flush=True)
json.dump(out, open(f"{output_dir('results/bench_mb')}/survival.json", 'w'), indent=1)

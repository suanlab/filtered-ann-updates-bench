#!/usr/bin/env python3
"""Workload-independent calibration of the check/distance cost ratio (review: IDEA_ICDE.md, idea 2).
  taskset -c <cores> .venv-faiss/bin/python bench/microbench_cost.py      -> results/bench_mb/cost.json
For d in (96, 4096): random float32 vectors (n rows), Faiss IndexFlatL2 searched one query at a time on one thread
with an IDSelectorBitmap. With an all-false bitmap every row is only checked (c_chk per row); with an all-true bitmap
every row is checked and its distance computed (c_chk + c_dist per row). No dataset, query set, or index of the study
is used, so the ratio can calibrate a dispatch threshold before any workload is measured."""
import sys, os, json, time
sys.path.insert(0, 'bench')
import numpy as np, faiss
from common import faiss_bitmap

faiss.omp_set_num_threads(1); out = {}
for d, n in ((96, 1_000_000), (4096, 100_000)):
    rng = np.random.default_rng(0); X = rng.standard_normal((n, d), dtype=np.float32); Qs = rng.standard_normal((50, d), dtype=np.float32)
    ix = faiss.IndexFlatL2(d); ix.add(X); keep = []
    res = {}
    for name, mask in (('none', np.zeros(n, bool)), ('all', np.ones(n, bool))):
        p = faiss.SearchParameters(sel=faiss_bitmap(mask, keep)); ts = []
        for rep in range(5):
            t = time.perf_counter()
            for q in Qs: ix.search(q[None], 10, params=p)
            ts.append((time.perf_counter() - t) / len(Qs))
        res[name] = float(np.median(ts))
    c_chk = res['none'] / n; c_dist = (res['all'] - res['none']) / n
    out[str(d)] = {'n': n, 'ns_per_check': 1e9 * c_chk, 'ns_per_distance': 1e9 * c_dist, 'chk_over_dist': c_chk / c_dist}
    print(d, out[str(d)], flush=True)
os.makedirs('results/bench_mb', exist_ok=True); json.dump(out, open('results/bench_mb/cost.json', 'w'), indent=1)

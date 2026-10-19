#!/usr/bin/env python3
"""Calibration probe: a fixed single-thread, memory+compute-bound kernel (numpy matrix-vector scan of
2^16 x 128 float32 against 64 queries, 5 runs). Logged before/after every benchmark job so that
runs taken under abnormal machine contention can be detected and repeated."""
import os, time, json, sys
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
rng = np.random.default_rng(0); X = rng.standard_normal((1 << 16, 128), np.float32); xn = (X * X).sum(1); Q = rng.standard_normal((64, 128), np.float32)
ts = []
for _ in range(5):
    t = time.perf_counter()
    for q in Q: np.argpartition(xn - 2 * (X @ q), 10)[:10]
    ts.append(time.perf_counter() - t)
print(json.dumps({'probe': sys.argv[1] if len(sys.argv) > 1 else '', 'time': time.strftime('%F %T'), 'qps': 64 / float(np.median(ts))}), flush=True)

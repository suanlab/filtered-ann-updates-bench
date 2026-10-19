#!/usr/bin/env python3
"""Verify the YFCC-10M workload: (1) our exact filtered ground truth equals the official Big-ANN'23
public ground truth for the 500 selected queries; (2) regenerating the 'tags' schema with bench/prep.py
reproduces data/bench/yfcc10m/tags/pred.npz and gt.npy exactly (written to a scratch directory)."""
import sys, tempfile, shutil
sys.path.insert(0, 'bench'); sys.path.insert(0, 'experiments')
import numpy as np
from prep import schema
qsel = np.load('data/yfcc/qsel.npy'); gt = np.load('data/bench/yfcc10m/tags/gt.npy')
with open('data/yfcc/GT.public.ibin', 'rb') as f:
    n, k = np.fromfile(f, np.int32, 2); off = np.fromfile(f, np.int32, n * k).reshape(n, k)[qsel]
agree = [len(set(a[a >= 0][:10]) & set(b[b >= 0][:10])) / max(1, min(10, (b >= 0).sum())) for a, b in zip(gt, off)]
print(f'official GT agreement: mean {np.mean(agree):.4f}, queries below 1.0: {sum(x < 1 for x in agree)}')
tmp = tempfile.mkdtemp(dir='data/bench')
try:
    schema('yfcc10m', 'tags', out=tmp)
    old, new = np.load('data/bench/yfcc10m/tags/pred.npz'), np.load(f'{tmp}/pred.npz')
    for key in old.files:
        same = key in new.files and np.array_equal(old[key], new[key])
        print(f'pred.npz[{key}]: {"identical" if same else "DIFFERENT"}')
    print('gt.npy:', 'identical' if np.array_equal(gt, np.load(f'{tmp}/gt.npy')) else 'DIFFERENT')
finally:
    shutil.rmtree(tmp)

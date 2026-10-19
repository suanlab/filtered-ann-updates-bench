#!/usr/bin/env python3
"""Prepare update streams with filtered queries (main env).  python3 bench/stream_prep.py <sift500k|deep5m>

Layout data/bench/stream_<name>/: Xall.npy (rows in insertion order: the N0 initial rows, then each
round's insert block, so ids are row numbers and inserts are contiguous), Q.npy, attr.npy (label of
every row; beta=0.9, s=0.1), qlab.npy, sched.npz (del<r>, ins<r>), gt.npy ((R+1) x nq x 10, exact
filtered ground truth over the live set after each round), meta.json.
  sift500k: SIFT descriptors (first 1.5M BIGANN rows, all distinct), 500K live, 20 rounds of 10% uniform
            deletes + 50K inserts of unseen rows (200% turnover)
  sift500k_adv: as sift500k, but deletions are the 10% of live points with the largest projection on the
            label direction (correlation x churn: high labels' regions are emptied progressively)
  deep5m:   DEEP-10M (normalized), 5M live, 10 rounds of 10% deletes + 499,000 inserts of unseen rows
            (100% turnover), labels recomputed on the stream's rows with the same construction as the
            b0.9_s0.1 schema (prep.labels, beta=0.9, 10 labels, seed 11), so they differ from deep10m's labels."""
import sys, os, json
sys.path.insert(0, 'experiments'); sys.path.insert(0, 'bench')
import numpy as np
import rigx
from common import ROOT, log, read_u8bin
from prep import labels

name = sys.argv[1]; D = f'{ROOT}/stream_{name}'; os.makedirs(D, exist_ok=True)
if name in ('sift500k', 'sift500k_adv'):
    # SIFT descriptors without duplicates: the first 1.5M rows of BIGANN (SIFT1B) and its public queries.
    u8 = read_u8bin
    R, N0, NI = 20, 500_000, 50_000
    base = next(f for f in ('data/bigann/base.1.5M.u8bin', 'data/bigann/base.100M.u8bin') if os.path.exists(f))   # any prefix of base.1B
    Xall = u8(base, N0 + R * NI).astype(np.float32); Q = u8('data/bigann/query.public.10K.u8bin', 500).astype(np.float32)
    attr = labels(Xall, 0.9, 10); qlab = np.random.default_rng(7).integers(0, 10, len(Q)).astype(np.int32)
    w = np.random.default_rng(11).standard_normal(Xall.shape[1]).astype(np.float32); w /= np.linalg.norm(w)   # label direction
    score = Xall @ w; score = (score - score.mean()) / score.std() + 0.3 * np.random.default_rng(25).standard_normal(len(Xall)).astype(np.float32)
    rng = np.random.default_rng(21); alive = np.zeros(len(Xall), bool); alive[:N0] = True; dels, inss, gts = [], [], []
    def gt_now():
        g = np.full((len(Q), 10), -1, np.int32)
        for b in range(10):
            qi = np.where(qlab == b)[0]; m = alive & (attr == b)
            if m.sum() >= 10: g[qi] = rigx.brute(Xall, Q[qi], mask=m, k=10, threads=64)[0]
            elif m.any():
                for i in qi: r_ = rigx.brute(Xall, Q[i:i + 1], mask=m, k=10, threads=64)[0][0]; g[i, :m.sum()] = r_[:m.sum()]
        return g
    gts.append(gt_now())
    for r in range(R):
        cur = np.where(alive)[0]; k = int(0.1 * len(cur))
        if name == 'sift500k': de = np.sort(rng.choice(cur, k, replace=False)).astype(np.int32)
        else:   # correlation x churn: delete the live points with the largest projection on the label direction
            de = np.sort(cur[np.argpartition(-score[cur], k)[:k]]).astype(np.int32)
        ins = np.arange(N0 + r * NI, N0 + (r + 1) * NI, dtype=np.int32)
        alive[de] = False; alive[ins] = True; dels.append(de); inss.append(ins); gts.append(gt_now()); log(name, 'gt round', r + 1)
    gt = np.stack(gts)
else:
    X = np.load(f'{ROOT}/deep10m/X.npy', mmap_mode='r'); Q = np.load(f'{ROOT}/deep10m/Q.npy')
    n = len(X); R, N0 = 10, 5_000_000; NI = (n - N0) // R
    perm = np.random.default_rng(23).permutation(n)
    Xall = np.ascontiguousarray(X[perm[:N0 + R * NI]]); attr = labels(Xall, 0.9, 10)
    qlab = np.random.default_rng(7).integers(0, 10, len(Q)).astype(np.int32)
    rng = np.random.default_rng(24); alive = np.zeros(len(Xall), bool); alive[:N0] = True; nxt = N0
    dels, inss = [], []
    for r in range(R):
        cur = np.where(alive)[0]
        de = np.sort(rng.choice(cur, int(0.1 * len(cur)), replace=False)).astype(np.int32)
        ins = np.arange(nxt, nxt + NI, dtype=np.int32); nxt += NI
        alive[de] = False; alive[ins] = True; dels.append(de); inss.append(ins)
    alive = np.zeros(len(Xall), bool); alive[:N0] = True; gts = []
    def gt_now():
        g = np.full((len(Q), 10), -1, np.int32)
        for b in range(10):
            qi = np.where(qlab == b)[0]; g[qi] = rigx.brute(Xall, Q[qi], mask=alive & (attr == b), k=10, threads=64)[0]
        return g
    gts.append(gt_now())
    for r in range(R):
        alive[dels[r]] = False; alive[inss[r]] = True; gts.append(gt_now()); log('gt round', r + 1)
    gt = np.stack(gts)
np.save(f'{D}/Xall.npy', np.ascontiguousarray(Xall, np.float32)); np.save(f'{D}/Q.npy', Q)
np.save(f'{D}/attr.npy', attr.astype(np.int32)); np.save(f'{D}/qlab.npy', qlab.astype(np.int32)); np.save(f'{D}/gt.npy', gt)
np.savez(f'{D}/sched.npz', **{f'del{r}': dels[r] for r in range(R)}, **{f'ins{r}': inss[r] for r in range(R)})
measure = list(range(0, R + 1, 4)) if R == 20 else list(range(0, R + 1, 5))
json.dump({'name': name, 'N0': N0, 'R': R, 'n_rows': int(len(Xall)), 'd': int(Xall.shape[1]), 'nq': int(len(Q)),
           'beta': 0.9, 's': 0.1, 'measure_rounds': measure}, open(f'{D}/meta.json', 'w'), indent=1)
log(name, 'stream prepared', len(Xall), 'rows; measure rounds', measure)

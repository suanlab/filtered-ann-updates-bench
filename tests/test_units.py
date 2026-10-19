#!/usr/bin/env python3
"""Minimal unit tests (audit L14). Run: python3 tests/test_units.py"""
import sys
sys.path.insert(0, 'experiments')
import os
import numpy as np
import pytest
from exp0_regen_gap import robust_prune
from exp5b_dynamics_exact import ExactStableCarve

def test_robust_prune_occlusion():
    # u at origin; y=(1,0) occludes c=(2,0) at alpha=1.2 (1.44*d(y,c)=1.44 < d(u,c)=4);
    # c2=(0,2) is not occluded and must be kept.
    X = np.array([[0,0],[1,0],[2,0],[0,2]], dtype=np.float32)
    ids = np.array([1,2,3], dtype=np.int32)
    ds = ((X[ids] - X[0])**2).sum(1)
    kept = robust_prune(0, ids, ds, X, 1.2, 8)
    assert 1 in kept and 3 in kept and 2 not in kept, kept

def test_robust_prune_keeps_nearest():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 8)).astype(np.float32)
    ids = np.arange(1, 200, dtype=np.int32)
    ds = ((X[ids] - X[0])**2).sum(1)
    kept = robust_prune(0, ids, ds, X, 1.2, 16)
    assert kept[0] == ids[np.argmin(ds)]

def test_csd_state_identity():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((1500, 16)).astype(np.float32)
    L0, f0 = 64, 4
    full = np.arange(1500, dtype=np.int64)
    batch = ExactStableCarve(X, full, L0, f0).state_hash()
    for seed in (0, 1):
        sc = ExactStableCarve(X, np.arange(1000, dtype=np.int64), L0, f0)
        ops = [('i', int(i)) for i in range(1000, 1500)]
        kill = np.random.default_rng(2).choice(1000, 200, replace=False)
        ops += [('d', int(p)) for p in kill]
        np.random.default_rng(seed).shuffle(ops)
        for op, pid in ops:
            (sc.insert if op == 'i' else sc.delete)(pid)
        # rebuild kill set into final alive for batch comparison
        final = np.array(sorted(set(range(1500)) - set(kill.tolist())), dtype=np.int64)
        want = ExactStableCarve(X, final, L0, f0).state_hash()
        assert sc.state_hash() == want, f"seed {seed}: streamed != batch"

def test_int8_distance_error():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((1000, 128)).astype(np.float32)
    q = rng.standard_normal(128).astype(np.float32)
    sx = np.abs(X).max(1) / 127.0; sq = np.abs(q).max() / 127.0
    X8 = np.rint(X / sx[:, None]); q8 = np.rint(q / sq)
    dot8 = (X8 * q8).sum(1) * sx * sq
    d8 = (q*q).sum() + (X*X).sum(1) - 2*dot8
    d = ((X - q)**2).sum(1)
    rel = np.abs(d8 - d) / np.maximum(d, 1e-6)
    assert np.percentile(rel, 99) < 2e-2, np.percentile(rel, 99)
    # exact top-1 must sit inside int8 top-5 (rerank window safety)
    assert np.argmin(d) in np.argsort(d8)[:5]

def test_kernel_leafmajor_matches_gather():
    # skip only when the kernel has not been built; any other import error must fail the test
    if not os.path.exists(os.path.join(os.path.dirname(__file__), '..', 'src', 'librig.so')):
        pytest.skip('src/librig.so not built (scripts/build_kernels.sh)')
    import ctypes
    from exp6_cpp import run, recall, lib, P, f32p
    from exp0_regen_gap import make_gmm, exact_topk, rbc_carve, build_membership
    i8p = ctypes.POINTER(ctypes.c_int8); i64p = ctypes.POINTER(ctypes.c_int64)
    lib.rig_set_quant.argtypes = [i8p, f32p, f32p]
    lib.rig_set_leafmajor.argtypes = [i8p, f32p, f32p, ctypes.POINTER(ctypes.c_int32), i64p]
    n, d = 5000, 32
    X, C = make_gmm(n, d, k=8, rng=np.random.default_rng(4), spread=1.5)
    Q, _ = make_gmm(50, d, k=8, rng=np.random.default_rng(5), spread=1.5, centers=C)
    gt, _ = exact_topk(Q, X, 10)
    leaves, leaders = rbc_carve(X, np.random.default_rng(6)); mem = build_membership(leaves, n)
    G = [[] for _ in range(n)]
    sx = (np.abs(X).max(1)/127.0).astype(np.float32); sx[sx==0]=1
    X8 = np.ascontiguousarray(np.rint(X/sx[:,None]).astype(np.int8)); xn = np.ascontiguousarray((X*X).sum(1).astype(np.float32))
    lib.rig_set_quant(P(X8, i8p), P(sx, f32p), P(xn, f32p))
    order = np.concatenate(leaves).astype(np.int64)
    loff8 = np.zeros(len(leaves)+1, dtype=np.int64); loff8[1:] = np.cumsum([len(l) for l in leaves])
    LX8 = np.ascontiguousarray(X8[order]); Lsx = np.ascontiguousarray(sx[order]); Lxn = np.ascontiguousarray(xn[order])
    Lid = np.ascontiguousarray(order.astype(np.int32)); loff8 = np.ascontiguousarray(loff8)
    import ctypes as ct
    lib.rig_set_leafmajor(P(LX8, i8p), P(Lsx, f32p), P(Lxn, f32p), P(Lid, ct.POINTER(ct.c_int32)), P(loff8, i64p))
    mask = np.ones(n, dtype=bool)
    r5 = recall(run(X, Q, leaves, mem, G, leaders, mask, 5, 1.2, 32, 256, 32, 10, 4, 8, False)[0], gt)
    r7 = recall(run(X, Q, leaves, mem, G, leaders, mask, 7, 1.2, 32, 256, 32, 10, 4, 8, False)[0], gt)
    assert abs(r5 - r7) < 0.06 and r7 > 0.9, (r5, r7)

def test_dynamic_rig_recall_stable_under_churn():
    """Regression: the pivot-retaining path once used deleted leaders as entry points and recall
    collapsed as (1-0.9^r)^t_entry. Entries must be live; recall must stay high after 15 rounds."""
    import rigx
    from exp0_regen_gap import make_gmm, exact_topk
    Xall, C = make_gmm(40000, 32, 16, np.random.default_rng(1), spread=1.5)
    Q, _ = make_gmm(100, 32, 16, np.random.default_rng(2), spread=1.5, centers=C)
    dr = rigx.DynamicRig(Xall, np.arange(10000), threads=4)
    rng = np.random.default_rng(3); nxt = 10000
    for r in range(15):
        al = np.where(dr.alive)[0]
        dr.delete(rng.choice(al, 1000, replace=False)); dr.insert(np.arange(nxt, nxt + 1000)); nxt += 1000
    idx, _ = dr.refresh()
    al = np.where(dr.alive)[0]; g, _ = exact_topk(Q, Xall[al], 10); gt = al[g]
    o, _, _ = idx.search(Q, mask=dr.alive, ef=32, threads=4, entries=dr.live_entries(0.01))
    rec = rigx.recall(o, gt)
    assert rec > 0.95, rec

if __name__ == '__main__':
    import traceback
    fails = 0
    for name, fn in sorted({k: v for k, v in globals().items() if k.startswith('test_')}.items()):
        try:
            fn(); print(f"PASS {name}")
        except Exception:
            fails += 1; print(f"FAIL {name}"); traceback.print_exc()
    sys.exit(1 if fails else 0)

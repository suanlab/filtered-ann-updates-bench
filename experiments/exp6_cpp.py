#!/usr/bin/env python3
"""exp6: C++ kernel wall-clock — explicit in-filter vs RIG masked vs route-harvest vs brute-force filtered.
Reports recall@10 and QPS (nthreads) per selectivity, on synthetic 50K and fashion-mnist."""
import sys, json, time, ctypes, argparse
sys.path.insert(0, 'experiments')
import numpy as np
from exp0_regen_gap import make_gmm, exact_topk, rbc_carve, build_membership, hashprune_graph, log

lib = ctypes.CDLL('src/librig.so')
i32p = ctypes.POINTER(ctypes.c_int32); f32p = ctypes.POINTER(ctypes.c_float)
u8p = ctypes.POINTER(ctypes.c_uint8); i64p = ctypes.POINTER(ctypes.c_int64)
i64p_ = ctypes.POINTER(ctypes.c_int64)
lib.rig_search64.argtypes = [f32p, ctypes.c_int, ctypes.c_int, i64p_, i32p, i64p_, i32p, i64p_, i32p, i32p, ctypes.c_int,
                           u8p, ctypes.c_int, f32p, ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                           ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, i32p, i64p]
lib.rig_set_sketch.argtypes = [f32p, ctypes.c_int, ctypes.c_int]
lib.brute_masked.argtypes = [f32p, ctypes.c_int, ctypes.c_int, u8p, f32p, ctypes.c_int, ctypes.c_int, ctypes.c_int, i32p]

def csr(lists, n):
    off = np.zeros(n + 1, dtype=np.int64); off[1:] = np.cumsum([len(l) for l in lists])
    ids = np.concatenate([np.asarray(l, dtype=np.int32) for l in lists]) if off[-1] > 0 else np.zeros(0, np.int32)
    return np.ascontiguousarray(off), np.ascontiguousarray(ids)

def P(a, t): return a.ctypes.data_as(t)

def run(X, Q, leaves, mem, G, leaders, mask, mode, alpha, R, topC, ef, k, t_entry, nthreads, use_mask):
    n, d = X.shape; nq = Q.shape[0]
    loff, lids = csr(leaves, len(leaves)); moff, mids = csr(mem, n); goff, gids = csr(G, n)
    out = np.zeros((nq, k), dtype=np.int32); nc = np.zeros(nq, dtype=np.int64)
    L = np.ascontiguousarray(leaders.astype(np.int32)); M = np.ascontiguousarray(mask.astype(np.uint8))
    t0 = time.perf_counter()
    lib.rig_search64(P(X, f32p), n, d, P(loff, i64p_), P(lids, i32p), P(moff, i64p_), P(mids, i32p), P(goff, i64p_), P(gids, i32p),
                   P(L, i32p), len(L), P(M, u8p), int(use_mask), P(Q, f32p), nq, alpha, R, topC, ef, k, t_entry, mode, nthreads,
                   P(out, i32p), P(nc, i64p))
    dt = time.perf_counter() - t0
    return out, nc, dt

def recall(out, gt):
    return float(np.mean([len(set(out[i].tolist()) & set(gt[i].tolist())) / gt.shape[1] for i in range(len(gt))]))

def evaluate(name, X, Q, gt_all, leaves, mem, G, leaders, alpha, R, topC, nthreads, sels, out, S=None):
    n, d = X.shape; nq = Q.shape[0]
    rng = np.random.default_rng(7)
    res = {}
    for s in sels:
        if s >= 1.0: mask = np.ones(n, dtype=bool); gt = gt_all
        else:
            mask = rng.random(n) < s; midx = np.where(mask)[0]
            g, _ = exact_topk(Q, X[midx], 10); gt = midx[g]
        row = {}
        # brute force filtered (C++)
        M = np.ascontiguousarray(mask.astype(np.uint8)); ob = np.zeros((nq, 10), dtype=np.int32)
        t0 = time.perf_counter(); lib.brute_masked(P(X, f32p), n, d, P(M, u8p), P(Q, f32p), nq, 10, nthreads, P(ob, i32p)); dt = time.perf_counter() - t0
        row['brute'] = {'recall': recall(ob, gt), 'qps': nq / dt}
        cfgs = [('explicit', 0, 64, 0), ('rig_unmasked_robust', 3, 64, 0), ('rig_unmasked_cone', 3, 64, 1),
                ('route_harvest_ef8_robust', 2, 8, 0), ('route_harvest_ef8_cone', 2, 8, 1),
                ('pool_greedy_ef8', 4, 8, 1), ('pool_greedy_ef16', 4, 16, 1), ('pool_greedy_ef32', 4, 32, 1)]
        for label, mode, ef, pm in cfgs:
            if s >= 1.0 and mode == 2: continue
            if s < 1.0 and mode == 3: continue
            if S is not None: lib.rig_set_sketch(P(S, f32p), S.shape[1], pm)
            o, nc, dt = run(X, Q, leaves, mem, G, leaders, mask, mode, alpha, R, topC, ef, 10, 4, nthreads, s < 1.0)
            row[label] = {'recall': recall(o, gt), 'qps': nq / dt, 'ncomp': float(nc.mean())}
        log(f"[{name}] s={s}: " + " | ".join(f"{k_}: r={v['recall']:.3f} qps={v['qps']:.0f}" for k_, v in row.items()))
        res[str(s)] = row
    out[name] = res

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--nq', type=int, default=500); ap.add_argument('--threads', type=int, default=64)
    ap.add_argument('--datasets', type=str, default='synth,fashion'); args = ap.parse_args()
    alpha, R, topC = 1.2, 32, 256
    out = {'threads': args.threads}
    sels = (1.0, 0.2, 0.1, 0.05, 0.01)
    if 'synth' in args.datasets:
        rng = np.random.default_rng(42)
        X, C = make_gmm(50000, 128, k=64, rng=rng, spread=1.5)
        Q, _ = make_gmm(args.nq, 128, k=64, rng=np.random.default_rng(43), spread=1.5, centers=C)
        gt, _ = exact_topk(Q, X, 10)
        log("synth build..."); leaves, leaders = rbc_carve(X, np.random.default_rng(44)); mem = build_membership(leaves, 50000)
        G = hashprune_graph(X, leaves, 12, 64, np.random.default_rng(45), alpha, R)
        H = np.random.default_rng(45).standard_normal((12, 128)).astype(np.float32); S = np.ascontiguousarray(X @ H.T)
        evaluate('synth50k', X, Q, gt, leaves, mem, G, leaders, alpha, R, topC, args.threads, sels, out, S)
        import hnswlib
        idx = hnswlib.Index(space='l2', dim=128); idx.init_index(max_elements=50000, ef_construction=200, M=16); idx.set_num_threads(args.threads)
        idx.add_items(X); idx.set_ef(64); t0 = time.perf_counter(); I, _ = idx.knn_query(Q, k=10); dt = time.perf_counter() - t0
        out['synth50k']['hnswlib_ef64_unfiltered'] = {'recall': recall(I.astype(np.int32), gt), 'qps': args.nq / dt}
        log(f"[synth50k] hnswlib ef64 unfiltered: r={out['synth50k']['hnswlib_ef64_unfiltered']['recall']:.3f} qps={args.nq/dt:.0f}")
    if 'fashion' in args.datasets:
        import h5py
        with h5py.File('data/fashion-mnist-784-euclidean.hdf5', 'r') as f:
            X = np.ascontiguousarray(np.array(f['train'], dtype=np.float32)); Q = np.ascontiguousarray(np.array(f['test'][:args.nq], dtype=np.float32))
            gt = np.array(f['neighbors'][:args.nq, :10])
        n = X.shape[0]
        log("fashion build..."); leaves, leaders = rbc_carve(X, np.random.default_rng(44)); mem = build_membership(leaves, n)
        G = hashprune_graph(X, leaves, 12, 64, np.random.default_rng(45), alpha, R)
        H = np.random.default_rng(45).standard_normal((12, 784)).astype(np.float32); S = np.ascontiguousarray(X @ H.T)
        evaluate('fashion60k', X, Q, gt, leaves, mem, G, leaders, alpha, R, topC, args.threads, sels, out, S)
        import hnswlib
        idx = hnswlib.Index(space='l2', dim=784); idx.init_index(max_elements=n, ef_construction=200, M=16); idx.set_num_threads(args.threads)
        idx.add_items(X); idx.set_ef(64); t0 = time.perf_counter(); I, _ = idx.knn_query(Q, k=10); dt = time.perf_counter() - t0
        out['fashion60k']['hnswlib_ef64_unfiltered'] = {'recall': recall(I.astype(np.int32), gt), 'qps': args.nq / dt}
        log(f"[fashion60k] hnswlib ef64 unfiltered: r={out['fashion60k']['hnswlib_ef64_unfiltered']['recall']:.3f} qps={args.nq/dt:.0f}")
    json.dump(out, open('results/exp6_cpp.json', 'w'), indent=2); log("DONE")

if __name__ == '__main__':
    main()

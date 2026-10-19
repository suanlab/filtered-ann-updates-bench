#!/usr/bin/env python3
"""exp0: RIG regenerability gap (F1) + search recall + percolation cliff (F3).

Pipeline:
  1. Synthetic GMM data (n, d), held-out queries from same mixture.
  2. Exact kNN ground truth (blocked GEMM).
  3. Reference graph N_ref = RobustPrune(exact top-C candidates).
  4. RBC carving (hash-priority leaders, 2-level fanout) -> leaves.
  5. PiPNN-lite explicit graph: leaf 2NN candidates -> HashPrune -> final prune.
  6. RIG: N_RIG(u) = RobustPrune(top-C of Pool(u)).  Jaccard vs N_ref / explicit.
  7. Beam-search recall@10 vs dist-comps: explicit vs RIG vs hnswlib.
  8. F3: strict in-filter beam under random predicate, selectivity sweep -> cliff.
"""
import argparse, json, time, sys
import numpy as np

def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

# ---------- data ----------
def make_gmm(n, d, k, rng, spread=6.0, centers=None):
    if centers is None:
        centers = rng.standard_normal((k, d)).astype(np.float32) * spread
    assign = rng.integers(0, k, n)
    X = centers[assign] + rng.standard_normal((n, d)).astype(np.float32)
    return X.astype(np.float32), centers

def sqdist_blocks(Q, X, block=4096):
    """Return generator of (start, D_block) squared L2 distances Q_block x X."""
    xn = (X * X).sum(1)
    for s in range(0, Q.shape[0], block):
        q = Q[s:s+block]
        qn = (q * q).sum(1)[:, None]
        D = qn + xn[None, :] - 2.0 * (q @ X.T)
        yield s, D

def exact_topk(Q, X, k, block=2048):
    n = Q.shape[0]
    I = np.empty((n, k), dtype=np.int32)
    Dv = np.empty((n, k), dtype=np.float32)
    for s, D in sqdist_blocks(Q, X, block):
        idx = np.argpartition(D, k, axis=1)[:, :k]
        dv = np.take_along_axis(D, idx, 1)
        o = np.argsort(dv, axis=1)
        I[s:s+D.shape[0]] = np.take_along_axis(idx, o, 1)
        Dv[s:s+D.shape[0]] = np.take_along_axis(dv, o, 1)
    return I, Dv

# ---------- robust prune ----------
def robust_prune(p, cand_ids, cand_d, X, alpha, R):
    order = np.argsort(cand_d)
    ids = cand_ids[order]; ds = cand_d[order]
    keep = []
    kept_vecs = []
    for cid, cd in zip(ids, ds):
        if cid == p: continue
        ok = True
        if kept_vecs:
            kv = np.stack(kept_vecs)
            dd = ((kv - X[cid]) ** 2).sum(1)
            if (alpha * alpha * dd < cd).any():
                ok = False
        if ok:
            keep.append(int(cid)); kept_vecs.append(X[cid])
            if len(keep) >= R: break
    return keep

# ---------- carving ----------
def rbc_carve(X, rng_priority, f0=10, f1=3, L0=None, Cmax=1024, sub_frac=0.05):
    """Vectorized 2-level randomized ball carving with fixed priority order (deterministic given rng)."""
    n = X.shape[0]
    if L0 is None: L0 = max(64, min(1000, n // 100))
    pri = rng_priority.permutation(n)
    rank = np.empty(n, dtype=np.int64); rank[pri] = np.arange(n)
    leaders0 = pri[:L0]
    Lx = X[leaders0]
    near = np.empty((n, f0), dtype=np.int32)
    for s, D in sqdist_blocks(X, Lx, 8192):
        near[s:s+D.shape[0]] = np.argpartition(D, f0, axis=1)[:, :f0]
    pts = np.repeat(np.arange(n, dtype=np.int32), f0); bk = near.ravel()
    order = np.argsort(bk, kind='stable'); pts = pts[order]; bk = bk[order]
    bounds = np.searchsorted(bk, np.arange(L0 + 1))
    leaves = []
    for j in range(L0):
        b = pts[bounds[j]:bounds[j+1]]
        if len(b) <= Cmax:
            if len(b) >= 8: leaves.append(np.array(b, dtype=np.int32))
            continue
        bpri = b[np.argsort(rank[b])]
        L1 = max(2, int(len(b) * sub_frac)); lead1 = bpri[:L1]
        Sx = X[b]; Lx1 = X[lead1]; f = min(f1, L1)
        near1 = np.empty((len(b), f), dtype=np.int32)
        for s, D in sqdist_blocks(Sx, Lx1, 8192):
            near1[s:s+D.shape[0]] = np.argpartition(D, f, axis=1)[:, :f] if L1 > f else np.tile(np.arange(L1), (D.shape[0], 1))[:, :f]
        p1 = np.repeat(b.astype(np.int32), f); k1 = near1.ravel()
        o1 = np.argsort(k1, kind='stable'); p1 = p1[o1]; k1 = k1[o1]
        bd = np.searchsorted(k1, np.arange(L1 + 1))
        for jj in range(L1):
            sb = p1[bd[jj]:bd[jj+1]]
            if len(sb) >= 8:
                for chunk in range(0, len(sb), Cmax):
                    leaves.append(np.array(sb[chunk:chunk+Cmax], dtype=np.int32))
    return leaves, leaders0

def build_membership(leaves, n):
    mem = [[] for _ in range(n)]
    for li, leaf in enumerate(leaves):
        for p in leaf: mem[p].append(li)
    return mem

# ---------- hashprune ----------
def hashprune_graph(X, leaves, m, lmax, rng, alpha, R):
    n, d = X.shape
    H = rng.standard_normal((m, d)).astype(np.float32)
    S = X @ H.T                                    # sketches n x m
    res = [dict() for _ in range(n)]               # hash -> (dist, id)
    def insert(p, c, dist):
        h = int(np.packbits((S[c] > S[p]).astype(np.uint8), bitorder='little')[:2].view(np.uint16)[0]) if m <= 16 else hash(tuple(S[c] > S[p]))
        r = res[p]
        if h in r:
            if dist < r[h][0]: r[h] = (dist, c)
        elif len(r) < lmax:
            r[h] = (dist, c)
        else:
            wk = max(r, key=lambda k: r[k][0])
            if dist < r[wk][0]:
                del r[wk]; r[h] = (dist, c)
    for leaf in leaves:
        Lx = X[leaf]
        D = ((Lx[:, None, :] - Lx[None, :, :]) ** 2).sum(-1) if len(leaf) <= 512 else None
        if D is None:
            ln = (Lx * Lx).sum(1)
            D = ln[:, None] + ln[None, :] - 2.0 * (Lx @ Lx.T)
        np.fill_diagonal(D, np.inf)
        k2 = min(2, len(leaf) - 1)
        nn = np.argpartition(D, k2, axis=1)[:, :k2]
        for i in range(len(leaf)):
            for j in nn[i]:
                a, b_, dv = int(leaf[i]), int(leaf[j]), float(D[i, j])
                insert(a, b_, dv); insert(b_, a, dv)
    # final prune
    G = []
    for p in range(n):
        vals = list(res[p].values())
        if not vals: G.append([]); continue
        ds = np.array([v[0] for v in vals], dtype=np.float32)
        ids = np.array([v[1] for v in vals], dtype=np.int32)
        G.append(robust_prune(p, ids, ds, X, alpha, R))
    return G

# ---------- RIG ----------
def rig_neighbors(p, X, mem, leaves, alpha, R, topC=256):
    pool = np.unique(np.concatenate([leaves[li] for li in mem[p]]))
    pool = pool[pool != p]
    d = ((X[pool] - X[p]) ** 2).sum(1)
    if len(pool) > topC:
        idx = np.argpartition(d, topC)[:topC]
        pool, d = pool[idx], d[idx]
    return robust_prune(p, pool, d, X, alpha, R), len(pool)

# ---------- beam search ----------
def beam_search(q, X, get_neighbors, entry, ef, k):
    import heapq
    dq = lambda i: float(((X[i] - q) ** 2).sum())
    visited = set(); ncomp = 0
    cand = []; top = []
    for e in entry:
        d = dq(e); ncomp += 1
        heapq.heappush(cand, (d, e)); heapq.heappush(top, (-d, e)); visited.add(e)
    while cand:
        d, u = heapq.heappop(cand)
        if top and d > -top[0][0] and len(top) >= ef: break
        for v in get_neighbors(u):
            if v in visited: continue
            visited.add(v)
            dv = dq(v); ncomp += 1
            if len(top) < ef or dv < -top[0][0]:
                heapq.heappush(cand, (dv, v)); heapq.heappush(top, (-dv, v))
                if len(top) > ef: heapq.heappop(top)
    res = sorted([(-d, i) for d, i in top])[:k]
    return [i for _, i in res], ncomp

def recall_at_k(returned, gt, k=10):
    return len(set(returned[:k]) & set(gt[:k].tolist())) / k

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=50000)
    ap.add_argument('--d', type=int, default=128)
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--nq', type=int, default=500)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--spread', type=float, default=1.5)
    ap.add_argument('--alpha', type=float, default=1.2)
    ap.add_argument('--R', type=int, default=32)
    ap.add_argument('--m', type=int, default=12)
    ap.add_argument('--topC', type=int, default=256)
    ap.add_argument('--out', type=str, default='results/exp0.json')
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    out = {'args': vars(args)}

    log(f"data: n={args.n} d={args.d}")
    X, C = make_gmm(args.n, args.d, k=64, rng=rng, spread=args.spread)
    Qrng = np.random.default_rng(args.seed + 1)
    Q, _ = make_gmm(args.nq, args.d, k=64, rng=Qrng, spread=args.spread, centers=C)

    log("exact GT for queries...")
    gtI, _ = exact_topk(Q, X, 10)
    out['gt_done'] = True

    log("reference graph: exact topC + RobustPrune (sampled 2000 nodes for Jaccard)...")
    sample = rng.choice(args.n, 2000, replace=False)
    refI, refD = exact_topk(X[sample], X, 65)
    N_ref = {}
    for si, p in enumerate(sample):
        ids, ds = refI[si], refD[si]
        mask = ids != p
        N_ref[int(p)] = set(robust_prune(int(p), ids[mask], ds[mask], X, args.alpha, args.R))

    log("RBC carving...")
    t0 = time.time()
    leaves, leaders0 = rbc_carve(X, np.random.default_rng(args.seed + 2))
    mem = build_membership(leaves, args.n)
    sizes = np.array([len(l) for l in leaves])
    memcnt = np.array([len(m) for m in mem])
    log(f"  leaves={len(leaves)} size p50={np.median(sizes):.0f} p95={np.percentile(sizes,95):.0f} "
        f"membership p50={np.median(memcnt):.0f} carve_time={time.time()-t0:.1f}s")
    out['carve'] = {'leaves': len(leaves), 'size_p50': float(np.median(sizes)),
                    'mem_p50': float(np.median(memcnt)), 'time_s': time.time() - t0}

    log("PiPNN-lite explicit graph (leaf 2NN -> HashPrune -> final prune)...")
    t0 = time.time()
    G = hashprune_graph(X, leaves, args.m, 64, np.random.default_rng(args.seed + 3), args.alpha, args.R)
    deg = np.array([len(g) for g in G])
    log(f"  build={time.time()-t0:.1f}s avg_deg={deg.mean():.1f}")
    out['pipnn'] = {'build_s': time.time() - t0, 'avg_deg': float(deg.mean())}

    log("RIG regeneration + Jaccard (F1 core measurement)...")
    jac_ref, jac_exp, pools, cover = [], [], [], []
    for p in sample:
        nr, psz = rig_neighbors(int(p), X, mem, leaves, args.alpha, args.R, args.topC)
        s_r = set(nr); pools.append(psz)
        ref = N_ref[int(p)]; exp = set(G[int(p)])
        pool_all = set(np.unique(np.concatenate([leaves[li] for li in mem[int(p)]])).tolist())
        if ref: cover.append(len(pool_all & ref) / len(ref))
        if ref: jac_ref.append(len(s_r & ref) / len(s_r | ref))
        if exp: jac_exp.append(len(s_r & exp) / len(s_r | exp))
    out['jaccard'] = {'vs_ref_mean': float(np.mean(jac_ref)), 'vs_ref_p10': float(np.percentile(jac_ref, 10)),
                      'vs_explicit_mean': float(np.mean(jac_exp)), 'pool_p50': float(np.median(pools)),
                      'pool_covers_ref': float(np.mean(cover))}
    log(f"  Jaccard vs N_ref: mean={np.mean(jac_ref):.3f} p10={np.percentile(jac_ref,10):.3f} | "
        f"vs explicit: {np.mean(jac_exp):.3f} | pool_p50={np.median(pools):.0f} | pool_covers_ref={np.mean(cover):.3f}")

    log("beam search recall (explicit vs RIG vs hnswlib)...")
    Lx0 = X[leaders0]
    def entries_for(q, t=4):
        dl = ((Lx0 - q) ** 2).sum(1)
        return [int(leaders0[i]) for i in np.argpartition(dl, t)[:t]], len(leaders0)
    rig_cache = {}
    def nb_explicit(u): return G[u]
    def nb_rig(u):
        if u not in rig_cache:
            rig_cache[u], _ = rig_neighbors(u, X, mem, leaves, args.alpha, args.R, args.topC)
        return rig_cache[u]
    res_search = {}
    for name, fn in [('explicit', nb_explicit), ('rig', nb_rig)]:
        for ef in (10, 32, 64, 128):
            rs, cs = [], []
            for qi in range(args.nq):
                ent, ec = entries_for(Q[qi])
                ids, nc = beam_search(Q[qi], X, fn, ent, ef, 10)
                nc += ec
                rs.append(recall_at_k(ids, gtI[qi])); cs.append(nc)
            res_search[f'{name}_ef{ef}'] = {'recall': float(np.mean(rs)), 'dist': float(np.mean(cs))}
            log(f"  {name} ef={ef}: recall@10={np.mean(rs):.3f} dist={np.mean(cs):.0f}")
    import hnswlib
    idx = hnswlib.Index(space='l2', dim=args.d); idx.init_index(max_elements=args.n, ef_construction=200, M=16)
    idx.add_items(X, np.arange(args.n))
    for ef in (10, 32, 64, 128):
        idx.set_ef(ef)
        I, _ = idx.knn_query(Q, k=10)
        r = np.mean([recall_at_k(I[i].tolist(), gtI[i]) for i in range(args.nq)])
        res_search[f'hnsw_ef{ef}'] = {'recall': float(r)}
        log(f"  hnswlib ef={ef}: recall@10={r:.3f}")
    out['search'] = res_search

    log("F3 percolation cliff (strict in-filter on explicit graph)...")
    perc = {}
    for s_sel in (0.5, 0.2, 0.1, 0.05, 0.03, 0.02, 0.01):
        mask = rng.random(args.n) < s_sel
        midx = np.where(mask)[0]
        if len(midx) < 50: continue
        nqf = min(200, args.nq)
        gtF, _ = exact_topk(Q[:nqf], X[midx], 10)
        gtF = midx[gtF]
        def nb_filt(u): return [v for v in G[u] if mask[v]]
        rs = []
        for qi in range(nqf):
            ent = list(rng.choice(midx, min(4, len(midx)), replace=False))
            ids, _ = beam_search(Q[qi], X, nb_filt, ent, 64, 10)
            rs.append(len(set(ids) & set(gtF[qi].tolist())) / 10)
        perc[str(s_sel)] = float(np.mean(rs))
        log(f"  selectivity={s_sel}: filtered recall@10={np.mean(rs):.3f}")
    out['percolation'] = perc

    with open(args.out, 'w') as f: json.dump(out, f, indent=2)
    log(f"DONE -> {args.out}")

if __name__ == '__main__':
    main()

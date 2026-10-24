#!/usr/bin/env python3
"""Prepare benchmark workloads (main env).  python3 bench/prep.py <dataset> [<schema> ...]

Datasets: sift1m (ann-benchmarks SIFT, 500 queries), deep10m (ann-benchmarks DEEP, normalized),
deep1m / deep3m (fixed random subsets of deep10m: same distribution, scaling axis), yfcc10m
(BigANN'23 filtered track: CLIP-like uint8 d=192, real tag sets; 500 fixed random public queries).

Synthetic label schemas 'b{beta}_s{s}': each point gets one of L=round(1/s) labels; the label is the
rank-quantile of  beta*proj(x) + (1-beta)*noise  for a fixed random direction, with proj and noise
standardized (beta is a mixing weight, not a correlation coefficient: the Pearson correlation of the score with
proj is beta/sqrt(beta^2+(1-beta)^2), e.g. 0.99 at beta=0.9, and beta=0/1 give independent/fully determined labels); each query asks for one uniformly random label.  For yfcc the schema
'tags' uses the real query predicates (AND of 1-2 tags) as admissible-id lists (csr)."""
import sys, os, json
sys.path.insert(0, 'experiments'); sys.path.insert(0, 'bench')
import numpy as np
from common import ROOT, ds_dir, log, read_u8bin

def _rigx():
    import rigx      # exact filtered ground truth uses the C++ brute-force kernel (src/librig.so); loaded only when needed
    return rigx

NQ = 500

def save_base(ds, X, Q, meta):
    os.makedirs(ds_dir(ds), exist_ok=True)
    np.save(f'{ds_dir(ds)}/X.npy', np.ascontiguousarray(X, np.float32)); np.save(f'{ds_dir(ds)}/Q.npy', np.ascontiguousarray(Q, np.float32))
    json.dump({**meta, 'n': int(len(X)), 'd': int(X.shape[1]), 'nq': int(len(Q))}, open(f'{ds_dir(ds)}/meta.json', 'w'), indent=1)

def base(ds):
    import h5py
    if ds == 'sift1m':
        with h5py.File('data/sift-128-euclidean.hdf5', 'r') as f:
            X = np.array(f['train'], np.float32); Q = np.array(f['test'][:NQ], np.float32)
        save_base(ds, X, Q, {'source': 'ann-benchmarks sift-128-euclidean', 'metric': 'l2'})
    elif ds == 'deep10m':
        with h5py.File('data/deep-image-96-angular.hdf5', 'r') as f:
            X = np.array(f['train'], np.float32); Q = np.array(f['test'][:NQ], np.float32)
        X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9; Q /= np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9
        save_base(ds, X, Q, {'source': 'ann-benchmarks deep-image-96-angular (normalized; L2 = angular order)', 'metric': 'l2'})
    elif ds in ('deep300k', 'deep1m', 'deep3m'):
        X = np.load(f'{ds_dir("deep10m")}/X.npy', mmap_mode='r'); Q = np.load(f'{ds_dir("deep10m")}/Q.npy')
        m = {'deep300k': 316_228, 'deep1m': 1_000_000, 'deep3m': 3_162_278}[ds]
        idx = np.sort(np.random.default_rng(13).choice(len(X), m, replace=False))
        save_base(ds, X[idx], Q, {'source': f'random subset (seed 13) of deep10m, m={m}', 'metric': 'l2'})
    elif ds in ('arxiv1m', 'arxiv300k', 'arxiv100k'):
        # arxiv-for-fanns (Iff et al.): 4096-d stella embeddings of arXiv abstracts; bench/fetch_arxiv.py samples ~1e6 rows of
        # the large set. arxiv1m is that sample, arxiv300k/arxiv100k random subsets of it (seed 13). Vectors stay float16 on
        # disk (upcast when loaded); 500 queries are drawn from the 10,000 public ones (seed 17).
        S_ = 'data/arxiv/large_sample'; Xs = np.load(f'{S_}/X.f16.npy', mmap_mode='r'); Qa = np.load(f'{S_}/Q.npy')
        qsel = np.sort(np.random.default_rng(17).choice(len(Qa), NQ, replace=False))
        m = {'arxiv1m': len(Xs), 'arxiv300k': 316_228, 'arxiv100k': 100_000}[ds]
        idx = np.arange(len(Xs)) if m == len(Xs) else np.sort(np.random.default_rng(13).choice(len(Xs), m, replace=False))
        os.makedirs(ds_dir(ds), exist_ok=True)
        if m == len(Xs):                               # the whole sample: link instead of copying 8 GB
            if not os.path.exists(f'{ds_dir(ds)}/X.npy'): os.symlink(os.path.abspath(f'{S_}/X.f16.npy'), f'{ds_dir(ds)}/X.npy')
        else: np.save(f'{ds_dir(ds)}/X.npy', np.ascontiguousarray(Xs[idx]))
        np.save(f'{ds_dir(ds)}/Q.npy', Qa[qsel])
        np.save(f'{ds_dir(ds)}/rows.npy', idx); np.save(f'{ds_dir(ds)}/qsel.npy', qsel)
        json.dump({'source': f'arxiv-for-fanns-large sample, {m} rows (float16 storage)', 'metric': 'l2', 'n': int(m), 'd': int(Xs.shape[1]), 'nq': NQ},
                  open(f'{ds_dir(ds)}/meta.json', 'w'), indent=1)
    elif ds in ('yfcc10m', 'yfcc10m_test'):
        # yfcc10m: 500 fixed random public queries (seed 17). yfcc10m_test: 500 further public queries disjoint from
        # them (seed 18), used only to validate choices tuned on yfcc10m (the dispatcher); same base vectors.
        u8 = read_u8bin
        Qa = u8('data/yfcc/query.public.100K.u8bin').astype(np.float32)
        qsel = np.sort(np.random.default_rng(17).choice(len(Qa), NQ, replace=False))
        if ds == 'yfcc10m':
            X = u8('data/yfcc/base.10M.u8bin').astype(np.float32); np.save('data/yfcc/qsel.npy', qsel)
            save_base(ds, X, Qa[qsel], {'source': 'BigANN-23 filtered track YFCC-10M', 'metric': 'l2', 'qsel_seed': 17})
        else:
            rest = np.setdiff1d(np.arange(len(Qa)), qsel); qt = np.sort(np.random.default_rng(18).choice(rest, NQ, replace=False))
            np.save('data/yfcc/qsel_test.npy', qt); os.makedirs(ds_dir(ds), exist_ok=True)
            if not os.path.exists(f'{ds_dir(ds)}/X.npy'): os.symlink(os.path.abspath(f'{ds_dir("yfcc10m")}/X.npy'), f'{ds_dir(ds)}/X.npy')
            np.save(f'{ds_dir(ds)}/Q.npy', Qa[qt])
            json.dump({'source': 'BigANN-23 filtered track YFCC-10M, held-out queries', 'metric': 'l2', 'qsel_seed': 18,
                       'n': 10_000_000, 'd': 192, 'nq': NQ}, open(f'{ds_dir(ds)}/meta.json', 'w'), indent=1)
    log(f'{ds}: base written')

def labels(X, beta, L, seed=11):
    n, d = X.shape; rng = np.random.default_rng(seed)
    w = rng.standard_normal(d).astype(np.float32); w /= np.linalg.norm(w)
    proj = np.empty(n, np.float32)
    for i in range(0, n, 1_000_000): proj[i:i + 1_000_000] = np.asarray(X[i:i + 1_000_000]) @ w
    proj = (proj - proj.mean()) / (proj.std() + 1e-9)
    a = beta * proj + (1 - beta) * rng.standard_normal(n).astype(np.float32)
    r = np.empty(n, np.int64); r[np.argsort(a, kind='stable')] = np.arange(n)
    return np.minimum(r * L // n, L - 1).astype(np.int32)

def schema(ds, sc, out=None):
    X = np.load(f'{ds_dir(ds)}/X.npy', mmap_mode='r'); Q = np.load(f'{ds_dir(ds)}/Q.npy'); n = len(X)
    out = out or f'{ds_dir(ds)}/{sc}'; os.makedirs(out, exist_ok=True)
    Xc = np.ascontiguousarray(X, np.float32)
    if sc == 'cat':
        # arXiv: a query asks for the papers whose main categories include its category (the dataset's
        # exact-match-in-set queries); admissible sets as csr, like the YFCC tags
        rows = np.load(f'{ds_dir(ds)}/rows.npy'); qsel = np.load(f'{ds_dir(ds)}/qsel.npy'); S_ = 'data/arxiv/large_sample'
        cats = [json.loads(l)['main_categories'] for l in open(f'{S_}/attributes.jsonl')]
        cats = [cats[i] for i in rows]
        qlabs = [json.loads(l)['label'] for l in open(f'{S_}/qattr_emis.jsonl')]; qlabs = [qlabs[i] for i in qsel]
        inv = {}
        for i, cs in enumerate(cats):
            for c in cs: inv.setdefault(c, []).append(i)
        qoff = [0]; qids = []
        for c in qlabs: ids = np.array(inv.get(c, []), np.int32); qids.append(ids); qoff.append(qoff[-1] + len(ids))
        qoff = np.array(qoff, np.int64); qids = np.concatenate(qids)
        vocab = sorted(inv); vid = {c: j for j, c in enumerate(vocab)}
        ptag_ptr = np.zeros(n + 1, np.int64); ptag_ptr[1:] = np.cumsum([len(cs) for cs in cats])
        ptag_idx = np.array([vid[c] for cs in cats for c in cs], np.int32)
        np.savez(f'{out}/pred.npz', kind='csr', qoff=qoff, qids=qids, ntags=np.ones(len(Q), np.int32), ptag_ptr=ptag_ptr, ptag_idx=ptag_idx,
                 qtag_ptr=np.arange(len(Q) + 1, dtype=np.int64), qtag_idx=np.array([vid.get(c, -1) for c in qlabs], np.int32))
        gt = np.full((len(Q), 10), -1, np.int32)
        for c in sorted(set(qlabs)):
            qi = np.array([i for i, x in enumerate(qlabs) if x == c]); m = np.zeros(n, bool); m[inv.get(c, [])] = True
            if m.any(): g = _rigx().brute(Xc, Q[qi], mask=m, k=10, threads=64)[0]; gt[qi] = g
        sel = np.diff(qoff) / n
        meta = {'kind': 'csr', 'sel_median': float(np.median(sel)), 'sel_quantiles': np.quantile(sel, [0.1, 0.5, 0.9]).tolist(), 'categories': len(vocab)}
    elif sc == 'tags':
        def spmat(p):        # BigANN .spmat: int64 nrow,ncol,nnz | int64 indptr[nrow+1] | int32 indices[nnz] | f32 data
            with open(p, 'rb') as f:
                nr, nc, nnz = (int(v) for v in np.fromfile(f, np.int64, 3))
                indptr = np.fromfile(f, np.int64, nr + 1); indices = np.fromfile(f, np.int32, nnz)
            return indptr, indices, nr, nc
        bp, bi, bn, ntag = spmat('data/yfcc/base.metadata.10M.spmat'); qp, qi_, _, _ = spmat('data/yfcc/query.metadata.public.100K.spmat')
        rows = np.repeat(np.arange(bn, dtype=np.int32), np.diff(bp))            # transpose: tag -> sorted point ids
        order = np.argsort(bi, kind='stable'); tptr = np.zeros(ntag + 1, np.int64); np.cumsum(np.bincount(bi, minlength=ntag), out=tptr[1:])
        tids = rows[order]; del rows, order
        qsel = np.load('data/yfcc/qsel_test.npy' if ds == 'yfcc10m_test' else 'data/yfcc/qsel.npy')
        qoff = [0]; qids = []; ntags = []
        for q in qsel:
            tags = qi_[qp[q]:qp[q + 1]]; ntags.append(len(tags))
            ids = tids[tptr[tags[0]]:tptr[tags[0] + 1]]
            for t in tags[1:]: ids = np.intersect1d(ids, tids[tptr[t]:tptr[t + 1]], assume_unique=True)
            qids.append(np.sort(ids).astype(np.int32)); qoff.append(qoff[-1] + len(ids))
        qoff = np.array(qoff, np.int64); qids = np.concatenate(qids)
        # point tag sets and query tags (AND) are kept for label-native methods (Filtered-DiskANN) and for
        # the predicate-cost measurement (bench/overhead.py)
        qtag = [qi_[qp[q]:qp[q + 1]] for q in qsel]
        qtag_ptr = np.zeros(len(qtag) + 1, np.int64); qtag_ptr[1:] = np.cumsum([len(t) for t in qtag])
        np.savez(f'{out}/pred.npz', kind='csr', qoff=qoff, qids=qids, ntags=np.array(ntags, np.int32),
                 ptag_ptr=bp, ptag_idx=bi, qtag_ptr=qtag_ptr, qtag_idx=np.concatenate(qtag).astype(np.int32))
        gt = np.full((len(Q), 10), -1, np.int32)
        for i in range(len(Q)):
            m = np.zeros(n, bool); m[qids[qoff[i]:qoff[i + 1]]] = True
            if m.any():
                g = _rigx().brute(Xc, Q[i:i + 1], mask=m, k=10, threads=64)[0][0]; c = int(min(10, m.sum())); gt[i, :c] = g[:c]
        sel = np.diff(qoff) / n
        meta = {'kind': 'csr', 'sel_median': float(np.median(sel)), 'sel_quantiles': np.quantile(sel, [0.1, 0.5, 0.9]).tolist(),
                'ntags_hist': np.bincount(ntags).tolist()}
    else:
        # b<beta>_s<s>[_r<rho>]: rho is the query-filter correlation. With probability rho a query asks for the label
        # of its exact (unfiltered) nearest neighbor, otherwise for a uniformly random label; rho=0 is the default.
        # optional _seed<k>: an independent replicate (label direction and noise with seed k, query labels with k + 100);
        # without it the seeds are those of every other schema (labels 11, query labels 7)
        parts = sc.split('_'); beta = float(parts[0][1:]); s = float(parts[1][1:]); L = int(round(1 / s))
        extra = parts[2:]; seed = next((int(x[4:]) for x in extra if x.startswith('seed')), None)
        rho = next((float(x[1:]) for x in extra if x.startswith('r')), 0.0)
        attr = labels(X, beta, L, seed=11 if seed is None else seed)
        qlab = np.random.default_rng(7 if seed is None else seed + 100).integers(0, L, len(Q)).astype(np.int32)
        if rho > 0:
            nn1 = _rigx().brute(Xc, Q, mask=np.ones(n, bool), k=1, threads=64)[0][:, 0]
            near = np.random.default_rng(8).random(len(Q)) < rho; qlab[near] = attr[nn1[near]]
        np.savez(f'{out}/pred.npz', kind='label', attr=attr, qlab=qlab)
        gt = np.full((len(Q), 10), -1, np.int32)
        for b in np.unique(qlab):
            qi = np.where(qlab == b)[0]; gt[qi] = _rigx().brute(Xc, Q[qi], mask=(attr == b), k=10, threads=64)[0]
        meta = {'kind': 'label', 'beta': beta, 's': s, 'labels': L, 'rho': rho, 'seed': seed}
    np.save(f'{out}/gt.npy', gt); json.dump(meta, open(f'{out}/meta.json', 'w'), indent=1)
    log(f'{ds}/{sc}: {meta}')

if __name__ == '__main__':
    ds = sys.argv[1]
    if not os.path.exists(f'{ds_dir(ds)}/meta.json'): base(ds)
    for sc in sys.argv[2:]: schema(ds, sc)

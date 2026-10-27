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
            b0.9_s0.1 schema (prep.labels, beta=0.9, 10 labels, seed 11), so they differ from deep10m's labels.
  arxiv_time: real temporal drift (review experiment D2). arXiv abstracts (4096-d embeddings of Iff et al.'s sample,
            float16 stored, used as float32) ordered by their update_date; the 200K oldest are live at round 0, and
            each of 10 rounds inserts the next 40K papers and deletes the 40K oldest live ones (a sliding time window,
            200% turnover). Label = the paper's first main category, the 9 most frequent in the initial window plus
            'other' (10 labels, real and unequal in size; fixed on the initial window, so no look-ahead). Queries: 500 papers drawn (seed 7) from the 100K papers that
            follow the stream in time, each asking for its own label -- newer papers searching the window.
  amz_time: second real temporal source (follow-up 2, I18). Amazon Reviews 2023, 10 categories, 384-d MiniLM embeddings
            (bench/prep_amazon.py), in review-time order; window and FIFO schedule as arxiv_time (200K live, 10 rounds of
            40K); label = the review's category (fixed list of 10, so no look-ahead); queries: 500 reviews drawn (seed 7)
            from the 100K that follow the stream in time, each asking for its own category.
  yfcc_stream: multi-label AND stream (follow-up 2, I18). 2M random YFCC-10M points (seed 31) with their tag sets; the
            first 1M are live at round 0; each of 10 rounds deletes 10% of the live points uniformly (seed 33) and inserts
            the next 100K. Queries: 500 public YFCC queries (seed 17 order) whose own 1- or 2-tag AND predicate admits at
            least 10 of the 2M points. Stored as CSR (ptag_*, qtag_* in pred.npz); attr.npy is unused (zeros)."""
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
elif name == 'arxiv_time':
    S_ = 'data/arxiv/large_sample'; R, N0, NI = 10, 200_000, 40_000
    att = [json.loads(l) for l in open(f'{S_}/attributes.jsonl')]
    date = np.array([a['update_date'] for a in att]); order = np.argsort(date, kind='stable')
    cat = np.array([a['main_categories'][0] if a['main_categories'] else 'none' for a in att])
    rows = order[:N0 + R * NI]; qpool = order[N0 + R * NI:N0 + R * NI + 100_000]
    # label mapping fixed on the initial window only (no look-ahead): its 9 most frequent first main categories; every
    # other category, including those that only appear later, is 'other'
    names, cnt = np.unique(cat[rows[:N0]], return_counts=True); top = list(names[np.argsort(-cnt)[:9]])
    lab = lambda c: top.index(c) if c in top else 9
    Xf = np.load(f'{S_}/X.f16.npy', mmap_mode='r')
    Xall = np.ascontiguousarray(Xf[np.sort(rows)][np.argsort(np.argsort(rows))], np.float32)    # rows in time order
    attr = np.array([lab(c) for c in cat[rows]], np.int32)
    qrows = np.sort(np.random.default_rng(7).choice(qpool, 500, replace=False))
    Q = np.ascontiguousarray(Xf[qrows], np.float32); qlab = np.array([lab(c) for c in cat[qrows]], np.int32)
    alive = np.zeros(len(Xall), bool); alive[:N0] = True; dels, inss, gts = [], [], []
    def gt_now():
        g = np.full((len(Q), 10), -1, np.int32)
        for b in range(10):
            qi = np.where(qlab == b)[0]
            if len(qi): g[qi] = rigx.brute(Xall, Q[qi], mask=alive & (attr == b), k=10, threads=64)[0]
        return g
    gts.append(gt_now()); oldest = 0
    for r in range(R):
        de = np.arange(oldest, oldest + NI, dtype=np.int32); oldest += NI      # FIFO: the oldest live papers
        ins = np.arange(N0 + r * NI, N0 + (r + 1) * NI, dtype=np.int32)
        alive[de] = False; alive[ins] = True; dels.append(de); inss.append(ins); gts.append(gt_now()); log(name, 'gt round', r + 1)
    gt = np.stack(gts)
    import hashlib
    hs = {hashlib.sha1(np.ascontiguousarray(Xf[i]).tobytes()).hexdigest() for i in rows}
    qdup = int(sum(hashlib.sha1(np.ascontiguousarray(Xf[i]).tobytes()).hexdigest() in hs for i in qrows))
    json.dump({'labels': top + ['other'], 'label_mapping': 'top 9 first main categories of the initial window (rows 0..N0-1)',
               'time_attribute': "update_date of the arXiv metadata (date of the paper's latest version, not first submission)",
               'queries': 'fixed: 500 papers that follow the stream in time (a stress test, not time-matched queries)',
               'query_vectors_identical_to_a_stream_row': qdup, 'date_first': int(date[rows].min()), 'date_last': int(date[rows].max()),
               'query_dates': [int(date[qrows].min()), int(date[qrows].max())],
               'label_share_round0': np.bincount(attr[:N0], minlength=10).tolist(), 'label_share_inserted': np.bincount(attr[N0:], minlength=10).tolist(),
               'query_labels': np.bincount(qlab, minlength=10).tolist()}, open(f'{D}/drift.json', 'w'), indent=1)
elif name == 'amz_time':
    S_ = 'data/amazon'; R, N0, NI = 10, 200_000, 40_000
    att = [json.loads(l) for l in open(f'{S_}/attributes.jsonl')]; ameta = json.load(open(f'{S_}/meta.json'))
    ts = np.array([a['timestamp'] for a in att]); assert (np.diff(ts) >= 0).all(), 'attributes are not in time order'
    cats = ameta['categories']; catid = np.array([cats.index(a['category']) for a in att], np.int32)
    rows = np.arange(N0 + R * NI); qpool = np.arange(N0 + R * NI, min(len(att), N0 + R * NI + 100_000))
    Xf = np.load(f'{S_}/X.f16.npy', mmap_mode='r'); Xall = np.ascontiguousarray(Xf[rows], np.float32); attr = catid[rows]
    qrows = np.sort(np.random.default_rng(7).choice(qpool, 500, replace=False))
    Q = np.ascontiguousarray(Xf[qrows], np.float32); qlab = catid[qrows]
    alive = np.zeros(len(Xall), bool); alive[:N0] = True; dels, inss, gts = [], [], []
    def gt_now():
        g = np.full((len(Q), 10), -1, np.int32)
        for b in range(10):
            qi = np.where(qlab == b)[0]
            if len(qi): g[qi] = rigx.brute(Xall, Q[qi], mask=alive & (attr == b), k=10, threads=64)[0]
        return g
    gts.append(gt_now()); oldest = 0
    for r in range(R):
        de = np.arange(oldest, oldest + NI, dtype=np.int32); oldest += NI
        ins = np.arange(N0 + r * NI, N0 + (r + 1) * NI, dtype=np.int32)
        alive[de] = False; alive[ins] = True; dels.append(de); inss.append(ins); gts.append(gt_now()); log(name, 'gt round', r + 1)
    gt = np.stack(gts)
    json.dump({'labels': cats, 'label_mapping': 'category of the review (fixed list)', 'time_attribute': 'review timestamp',
               'queries': 'fixed: 500 reviews that follow the stream in time', 'ts_first': int(ts[rows].min()), 'ts_last': int(ts[rows].max()),
               'query_ts': [int(ts[qrows].min()), int(ts[qrows].max())],
               'label_share_round0': np.bincount(attr[:N0], minlength=10).tolist(), 'label_share_inserted': np.bincount(attr[N0:], minlength=10).tolist(),
               'query_labels': np.bincount(qlab, minlength=10).tolist()}, open(f'{D}/drift.json', 'w'), indent=1)
elif name == 'yfcc_stream':
    R, N0, NI = 10, 1_000_000, 100_000
    def spmat(p):
        with open(p, 'rb') as f:
            nr, nc, nnz = (int(v) for v in np.fromfile(f, np.int64, 3)); ip = np.fromfile(f, np.int64, nr + 1); ix = np.fromfile(f, np.int32, nnz)
        return ip, ix, nr, nc
    bp, bi, bn, ntag = spmat('data/yfcc/base.metadata.10M.spmat'); qp, qi_, _, _ = spmat('data/yfcc/query.metadata.public.100K.spmat')
    rows = np.random.default_rng(31).choice(bn, N0 + R * NI, replace=False)     # stream order = this random order
    Xall = read_u8bin('data/yfcc/base.10M.u8bin')[rows].astype(np.float32)
    cnt = np.diff(bp)[rows]; ptag_ptr = np.zeros(len(rows) + 1, np.int64); np.cumsum(cnt, out=ptag_ptr[1:])
    ptag_idx = np.concatenate([np.sort(bi[bp[r]:bp[r + 1]]) for r in rows]).astype(np.int32)
    loc = np.repeat(np.arange(len(rows), dtype=np.int32), cnt); order = np.argsort(ptag_idx, kind='stable')
    tptr = np.zeros(ntag + 1, np.int64); np.cumsum(np.bincount(ptag_idx, minlength=ntag), out=tptr[1:]); tids = loc[order]
    def adm(tags):
        ids = tids[tptr[tags[0]]:tptr[tags[0] + 1]]
        for t in tags[1:]: ids = np.intersect1d(ids, tids[tptr[t]:tptr[t + 1]], assume_unique=True)
        return ids
    Qa = read_u8bin('data/yfcc/query.public.100K.u8bin'); qorder = np.random.default_rng(17).permutation(len(Qa)); qrows = []; qtags = []
    for q in qorder:
        tg = np.sort(qi_[qp[q]:qp[q + 1]])
        if 1 <= len(tg) <= 2 and len(adm(tg)) >= 10: qrows.append(q); qtags.append(tg)
        if len(qrows) == 500: break
    Q = Qa[np.array(qrows)].astype(np.float32); qtag_ptr = np.zeros(501, np.int64); qtag_ptr[1:] = np.cumsum([len(t) for t in qtags])
    qtag_idx = np.concatenate(qtags).astype(np.int32); qadm = [adm(t) for t in qtags]
    attr = np.zeros(len(rows), np.int32); qlab = np.zeros(500, np.int32)
    rng = np.random.default_rng(33); alive = np.zeros(len(Xall), bool); alive[:N0] = True; dels, inss, gts = [], [], []
    sq = (Xall * Xall).sum(1)
    def gt_now():
        g = np.full((500, 10), -1, np.int32)
        for i in range(500):
            c = qadm[i][alive[qadm[i]]]
            if not len(c): continue
            dd = sq[c] - 2 * Xall[c] @ Q[i]; kk = min(10, len(c)); p_ = np.argpartition(dd, kk - 1)[:kk]; g[i, :kk] = c[p_[np.argsort(dd[p_], kind='stable')]]
        return g
    gts.append(gt_now())
    for r in range(R):
        cur = np.where(alive)[0]; de = np.sort(rng.choice(cur, int(0.1 * len(cur)), replace=False)).astype(np.int32)
        ins = np.arange(N0 + r * NI, N0 + (r + 1) * NI, dtype=np.int32)
        alive[de] = False; alive[ins] = True; dels.append(de); inss.append(ins); gts.append(gt_now()); log(name, 'gt round', r + 1)
    gt = np.stack(gts)
    sel0 = np.array([np.isin(a_, np.arange(N0)).sum() / N0 for a_ in qadm])
    np.savez(f'{D}/pred.npz', kind='csr', ptag_ptr=ptag_ptr, ptag_idx=ptag_idx, qtag_ptr=qtag_ptr, qtag_idx=qtag_idx, rows=rows.astype(np.int64), qrows=np.array(qrows, np.int64))
    json.dump({'queries': '500 public YFCC queries with 1-2 tags admitting >= 10 of the 2M points', 'ntags_hist': np.bincount([len(t) for t in qtags]).tolist(),
               'sel_round0_quantiles': np.quantile(sel0, [0.1, 0.5, 0.9]).tolist(), 'tags_per_point_mean': float(cnt.mean())}, open(f'{D}/drift.json', 'w'), indent=1)
else:
    X = np.load(f'{ROOT}/deep10m/X.npy', mmap_mode='r'); Q = np.load(f'{ROOT}/deep10m/Q.npy')
    n = len(X); R, N0 = 10, 5_000_000; NI = (n - N0) // R
    # deep5m_seed<k>: an independent replicate of the stream (row sample/order, label direction, query labels, and
    # deletion schedule from seed k); plain deep5m keeps seeds 23 / 11 / 7 / 24
    k = int(name.split('_seed')[1]) if '_seed' in name else None
    perm = np.random.default_rng(23 if k is None else 1000 + k).permutation(n)
    Xall = np.ascontiguousarray(X[perm[:N0 + R * NI]]); attr = labels(Xall, 0.9, 10, seed=11 if k is None else k)
    qlab = np.random.default_rng(7 if k is None else 100 + k).integers(0, 10, len(Q)).astype(np.int32)
    rng = np.random.default_rng(24 if k is None else 2000 + k); alive = np.zeros(len(Xall), bool); alive[:N0] = True; nxt = N0
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
           **({'labels': 'first main category (top 9 + other)', 'order': 'update_date'} if name == 'arxiv_time' else
              {'labels': 'review category (10)', 'order': 'review timestamp'} if name == 'amz_time' else
              {'kind': 'csr', 'predicates': 'AND of 1-2 YFCC tags'} if name == 'yfcc_stream' else {'beta': 0.9, 's': 0.1}),
           'measure_rounds': measure}, open(f'{D}/meta.json', 'w'), indent=1)
log(name, 'stream prepared', len(Xall), 'rows; measure rounds', measure)

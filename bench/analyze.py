#!/usr/bin/env python3
"""Benchmark analysis.  python3 bench/analyze.py  ->  results/bench_summary.json (+ printed tables)

- Passes: records from results/bench (pass 1) and results/bench_pass<k> are merged per
  (workload, method, param); QPS = best pass mean (contention only slows runs down); recall must
  agree across passes (deterministic searches) and a mismatch is reported.
- Frontier: per (workload, method) the recall-QPS Pareto frontier; QPS at a recall target is
  log-linearly interpolated between the two frontier points that bracket it (None if never reached).
- Dispatcher (csr workloads): oracle-threshold hybrid, queries whose admissible set is smaller than T
  are answered by the exact per-query scan, the rest by the index at one parameter; recall and QPS are
  computed from per-query recall/time records; T is swept over powers of two.
- Selectivity buckets (csr): per-bucket mean recall and QPS for each record.
- Streams: per method and measured round, frontier QPS at recall targets, maintenance, bytes."""
import os, sys, json, glob, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import admissible_sizes

TARGETS = (0.9, 0.95, 0.99)

def passes():
    return ['results/bench'] + sorted(glob.glob('results/bench_pass*'))

def load_static():
    W = {}
    for root in passes():
        for f in glob.glob(f'{root}/*/*/*.json'):
            parts = f.split('/'); ds, sc, m = parts[-3], parts[-2], parts[-1][:-5]
            if ds.startswith('stream_'): continue
            for r in json.load(open(f)):
                key = (ds, sc, r['method'], r['param']); old = W.get(key)
                if old is None: W[key] = {**r, 'passes': 1, 'recalls': [r['recall']], 'qpss': [r['qps']['mean']]}; continue
                # keep the whole record of the pass with the best QPS, so that recall, QPS and per-query
                # data always come from the same run (graph builds are multi-threaded, hence not deterministic)
                recalls, qpss, n = old['recalls'] + [r['recall']], old['qpss'] + [r['qps']['mean']], old['passes'] + 1
                if r['qps']['mean'] > old['qps']['mean']: old = {**r}; W[key] = old
                old.update(passes=n, recalls=recalls, qpss=qpss)
                if max(recalls) - min(recalls) > 1e-9: old['recall_mismatch'] = recalls
    return W

def frontier(pts):
    """pts: [(recall, qps)] -> Pareto frontier sorted by recall (each point not dominated)."""
    pts = sorted(pts, key=lambda p: (-p[1], -p[0])); fr = []; best = -1
    for r, q in pts:
        if r > best: fr.append((r, q)); best = r
    return sorted(fr)

def qps_at(fr, target):
    if not fr or fr[-1][0] < target: return None
    for (r0, q0), (r1, q1) in zip([(0.0, fr[0][1])] + fr[:-1], fr):
        if r1 >= target:
            if r0 >= target or r1 == r0: return q1
            w = (target - r0) / (r1 - r0); return math.exp(math.log(q0) * (1 - w) + math.log(q1) * w) if r0 > 0 else q1
    return None

def summarize_static(W):
    out = {}
    for (ds, sc, m, p), r in W.items():
        out.setdefault(f'{ds}/{sc}', {}).setdefault(m, []).append(r)
    S = {}
    for wl, ms in out.items():
        S[wl] = {}
        for m, recs in ms.items():
            fr = frontier([(r['recall'], r['qps']['mean']) for r in recs])
            S[wl][m] = {'frontier': fr, 'max_recall': max(r['recall'] for r in recs),
                        **{f'qps@{t}': qps_at(fr, t) for t in TARGETS},
                        'build_s': recs[0].get('build_s'), 'bytes': recs[0].get('bytes'),
                        'passes': min(r['passes'] for r in recs)}
    return out, S

def dispatcher(wl, ms, sizes):
    """Oracle-threshold hybrid of scan_part (per query) and each index method, from per-query data."""
    if 'scan_part' not in ms: return {}
    scan = [r for r in ms['scan_part'] if 'tq' in r]
    if not scan: return {}
    ts = np.array(scan[0]['tq']); rs = np.array(scan[0]['rq']); H = {}
    for m, recs in ms.items():
        if m in ('scan_part', 'scan_mask'): continue
        pts = []
        for r in recs:
            if 'tq' not in r: continue
            tm = np.array(r['tq']); rm = np.array(r['rq'])
            for T in [2 ** k for k in range(4, 24)]:
                small = sizes < T
                rec = float(np.where(small, rs, rm).mean()); qps = len(sizes) / float(np.where(small, ts, tm).sum())
                pts.append((rec, qps, r['param'], T))
        fr = frontier([(a, b) for a, b, _, _ in pts])
        best = {}
        for t in TARGETS:
            ok = [p for p in pts if p[0] >= t]
            if ok: b_ = max(ok, key=lambda p: p[1]); best[str(t)] = {'recall': b_[0], 'qps': b_[1], 'param': b_[2], 'T': b_[3]}
        # held-out validation: choose (param, T) on even-numbered queries, evaluate on odd-numbered ones
        tr = np.arange(len(sizes)) % 2 == 0; te = ~tr; cv = {}
        cand = []
        for r in recs:
            if 'tq' not in r: continue
            tm = np.array(r['tq']); rm = np.array(r['rq'])
            for T in [2 ** k for k in range(4, 24)]:
                small = sizes < T; rr = np.where(small, rs, rm); tt = np.where(small, ts, tm)
                cand.append((rr[tr].mean(), tr.sum() / tt[tr].sum(), rr[te].mean(), te.sum() / tt[te].sum(),
                             r['param'], T, np.quantile(tt[te], 0.5), np.quantile(tt[te], 0.99)))   # latency of the held-out half
        for t in TARGETS:
            ok = [c for c in cand if c[0] >= t]
            if ok:
                c = max(ok, key=lambda c: c[1])
                cv[str(t)] = {'train_recall': c[0], 'test_recall': c[2], 'test_qps': c[3], 'param': c[4], 'T': c[5],
                              'p50_ms': 1e3 * c[6], 'p99_ms': 1e3 * c[7]}
        H[m] = {'frontier': fr, **{f'qps@{t}': qps_at(fr, t) for t in TARGETS}, 'best': best, 'heldout': cv}
    return H

def crossset(train, train_sizes, test, test_sizes):
    """Dispatcher tuned on one query set and evaluated on a disjoint one: for each method and recall target, the
    (parameter, threshold T) with the highest QPS on the training queries among those meeting the target there,
    then recall, QPS and latency of exactly that configuration on the test queries."""
    def per_q(ms, m):
        return {r['param']: (np.array(r['rq']), np.array(r['tq'])) for r in ms.get(m, []) if 'tq' in r}
    s_tr, s_te = per_q(train, 'scan_part'), per_q(test, 'scan_part')
    if not s_tr or not s_te: return {}
    (rs_tr, ts_tr), (rs_te, ts_te) = next(iter(s_tr.values())), next(iter(s_te.values()))
    out = {}
    for m in train:
        if m in ('scan_part', 'scan_mask'): continue
        tr, te = per_q(train, m), per_q(test, m); res = {}
        for t in TARGETS:
            best = None
            for prm, (rm, tm) in tr.items():
                if prm not in te: continue
                for T in [2 ** k for k in range(4, 24)]:
                    sm = train_sizes < T; rec = float(np.where(sm, rs_tr, rm).mean())
                    if rec < t: continue
                    q = len(sm) / float(np.where(sm, ts_tr, tm).sum())
                    if best is None or q > best[0]: best = (q, prm, T, rec)
            if best is None: continue
            q, prm, T, rec = best; rm2, tm2 = te[prm]; sm2 = test_sizes < T
            rr, tt = np.where(sm2, rs_te, rm2), np.where(sm2, ts_te, tm2)
            res[str(t)] = {'param': prm, 'T': T, 'train_recall': rec, 'train_qps': q, 'test_recall': float(rr.mean()),
                           'test_qps': len(sm2) / float(tt.sum()), 'p50_ms': 1e3 * float(np.quantile(tt, 0.5)),
                           'p99_ms': 1e3 * float(np.quantile(tt, 0.99))}
        out[m] = res
    return out

def pair_dispatch(train, train_sizes, test, test_sizes, small, large):
    """Two-index dispatcher: queries with |A| < T go to `small`, the rest to `large`. For each recall target the
    configuration (params, T) with the best QPS on the training queries is chosen (oracle on train) and then
    applied unchanged to the test queries (if given)."""
    def recs(ms, m): return {r['param']: (np.array(r['rq']), np.array(r['tq'])) for r in ms.get(m, []) if 'tq' in r}
    A, B = recs(train, small), recs(train, large)
    if not A or not B: return {}
    At, Bt = (recs(test, small), recs(test, large)) if test else ({}, {})
    out = {}
    for t in TARGETS:
        best = None
        for pa, (ra, ta) in A.items():
            for pb, (rb, tb) in B.items():
                for T in [2 ** k for k in range(4, 24)]:
                    sm = train_sizes < T; rec = float(np.where(sm, ra, rb).mean())
                    if rec < t: continue
                    q = len(sm) / float(np.where(sm, ta, tb).sum())
                    if best is None or q > best[0]: best = (q, pa, pb, T, rec)
        if best is None: continue
        q, pa, pb, T, rec = best; row = {'qps': q, 'recall': rec, 'small_param': pa, 'large_param': pb, 'T': T}
        if pa in At and pb in Bt:
            sm = test_sizes < T; rr = np.where(sm, At[pa][0], Bt[pb][0]); tt = np.where(sm, At[pa][1], Bt[pb][1])
            row.update(test_recall=float(rr.mean()), test_qps=len(sm) / float(tt.sum()))
        out[str(t)] = row
    return out

def buckets(ms, sizes, edges=(0, 1e3, 1e4, 1e5, 1e6, 1e9)):
    B = {}
    for m, recs in ms.items():
        B[m] = []
        for r in recs:
            rq = np.array(r['rq']); row = {'param': r['param']}
            tq = np.array(r['tq']) if 'tq' in r else None
            for lo, hi in zip(edges[:-1], edges[1:]):
                sel = (sizes >= lo) & (sizes < hi)
                if sel.any():
                    row[f'{int(lo)}-{int(hi)}'] = {'n': int(sel.sum()), 'recall': float(rq[sel].mean()),
                                                   'qps': float(sel.sum() / tq[sel].sum()) if tq is not None else None}
            B[m].append(row)
    return B

def break_even(rebuild_s, ivf_maint_per_round_s, qps_ivf, qps_fd):
    """Queries per round above which one Filtered-DiskANN rebuild per round costs less in total than IVF:
    rebuild - M_ivf = Q * (1/qps_ivf - 1/qps_fd). None if Filtered-DiskANN is not faster per query."""
    if not (qps_ivf and qps_fd) or qps_fd <= qps_ivf: return None
    return (rebuild_s - ivf_maint_per_round_s) / (1 / qps_ivf - 1 / qps_fd)

def merge_stream_passes(files):
    """Merge one method's stream results across passes: per round, the best-QPS record of every
    swept parameter, and the minimum maintenance / rebuild time (contention only slows things down)."""
    runs = [json.load(open(f)) for f in files]; base = runs[0]
    for other in runs[1:]:
        by_round = {o['round']: o for o in other['rounds']}      # match by round, never by position
        for r in base['rounds']:
            o = by_round.get(r['round'])
            if o is None: continue
            for k in ('cum_maint_s', 'rebuild_s'):
                if r.get(k) is not None and o.get(k) is not None: r[k] = min(r[k], o[k])
            if 'sweep' in r and 'sweep' in o:
                by = {s['param']: s for s in r['sweep']}
                for s in o['sweep']:
                    if s['param'] not in by or s['qps']['mean'] > by[s['param']]['qps']['mean']: by[s['param']] = s
                r['sweep'] = list(by.values())
    base['passes'] = len(runs); return base

def summarize_streams():
    S = {}; groups = {}
    for root in passes():
        for f in glob.glob(f'{root}/stream_*/*.json'):
            groups.setdefault((f.split('/')[-2], f.split('/')[-1]), []).append(f)
    for (sdir, fname), files in groups.items():
        name = sdir[len('stream_'):]; d = merge_stream_passes(files); m = d['method']; rows = []
        for r in d['rounds']:
            if 'sweep' not in r: continue
            fr = frontier([(s['recall'], s['qps']['mean']) for s in r['sweep']])
            nbytes = r.get('bytes')
            if m == 'acorn' and nbytes is not None and not r.get('bytes_has_levels'):
                nbytes += 4 * r['resident']   # records from the stream driver before it counted ACORN's int32 level per node
            rows.append({'round': r['round'], 'max_recall': max(s['recall'] for s in r['sweep']),
                         **{f'qps@{t}': qps_at(fr, t) for t in TARGETS},
                         'cum_maint_s': r.get('cum_maint_s'), 'rebuild_s': r.get('rebuild_s'),
                         'bytes': nbytes, 'resident': r.get('resident')})
        S.setdefault(name, {})[m] = {'build_s': d.get('build_s'), 'rounds': rows, 'passes': d['passes'],
                                     'total_maint_s': d['rounds'][-1].get('cum_maint_s') if d['rounds'] else None}
    return S

if __name__ == '__main__':
    W = load_static(); raw, S = summarize_static(W)
    H, B = {}, {}
    for wl, ms in raw.items():
        sizes = admissible_sizes(wl)
        if sizes is not None: H[wl] = dispatcher(wl, raw[wl], sizes); B[wl] = buckets(raw[wl], sizes)
    # measurement-quality statistics: relative 95% CI half-width of QPS within a job, and the spread of
    # QPS across passes (max/min) for configurations measured in more than one pass
    rel = [r['qps']['ci95'] / r['qps']['mean'] for r in W.values() if r['qps']['mean'] > 0]
    spread = [max(r['qpss']) / min(r['qpss']) for r in W.values() if r.get('passes', 1) > 1 and min(r['qpss']) > 0]
    rspread = [max(r['recalls']) - min(r['recalls']) for r in W.values() if r.get('passes', 1) > 1]
    quality = {'ci_rel_median': float(np.median(rel)), 'ci_rel_p90': float(np.quantile(rel, 0.9)), 'n_records': len(rel),
               'pass_spread_median': float(np.median(spread)) if spread else None,
               'pass_spread_p90': float(np.quantile(spread, 0.9)) if spread else None, 'n_multi_pass': len(spread),
               'recall_spread_max': float(max(rspread)) if rspread else None,
               'recall_spread_by_method': {m: {'median': float(np.median(v)), 'p90': float(np.quantile(v, 0.9)), 'max': float(max(v)), 'n': len(v)}
                                           for m in sorted({k[2] for k in W})
                                           for v in [[max(r['recalls']) - min(r['recalls']) for k, r in W.items() if k[2] == m and r.get('passes', 1) > 1]] if v}}
    X = {}      # dispatcher tuned on yfcc10m/tags, evaluated on the disjoint held-out queries yfcc10m_test/tags
    if 'yfcc10m/tags' in raw and 'yfcc10m_test/tags' in raw:
        sz = admissible_sizes
        X['yfcc10m'] = crossset(raw['yfcc10m/tags'], sz('yfcc10m/tags'), raw['yfcc10m_test/tags'], sz('yfcc10m_test/tags'))
        for sm_, lg_ in (('curator', 'acorn'), ('curator', 'ivf')):
            X.setdefault('pairs', {})[f'{sm_}+{lg_}'] = pair_dispatch(raw['yfcc10m/tags'], sz('yfcc10m/tags'), raw['yfcc10m_test/tags'],
                                                                      sz('yfcc10m_test/tags'), sm_, lg_)
    out = {'static': S, 'dispatch': H, 'buckets': B, 'streams': summarize_streams(), 'quality': quality, 'crossset': X,
           'mismatches': [list(k) for k, r in W.items() if 'recall_mismatch' in r]}
    json.dump(out, open('results/bench_summary.json', 'w'), indent=1, default=float)
    fmt = lambda v: '-' if v is None else (f'{v:.0f}' if v >= 10 else f'{v:.1f}')
    for wl in sorted(S):
        print(f'\n== {wl}')
        for m, s in sorted(S[wl].items()):
            print(f"  {m:12s} maxR={s['max_recall']:.4f} " + ' '.join(f"q@{t}={fmt(s[f'qps@{t}'])}" for t in TARGETS)
                  + (f"   [dispatch: " + ' '.join(f"q@{t}={fmt(H[wl][m][f'qps@{t}'])}" for t in TARGETS) + ']' if wl in H and m in H[wl] else ''))
    for name, ms in out['streams'].items():
        print(f'\n== stream {name}')
        for m, d in sorted(ms.items()):
            for r in d['rounds']:
                print(f"  {m:16s} r={r['round']:2d} maxR={r['max_recall']:.4f} " + ' '.join(f"q@{t}={fmt(r[f'qps@{t}'])}" for t in TARGETS)
                      + f"  maint={r['cum_maint_s'] if r['cum_maint_s'] is not None else r['rebuild_s']}")
    if out['mismatches']: print('\nRECALL MISMATCHES ACROSS PASSES:', out['mismatches'][:10])

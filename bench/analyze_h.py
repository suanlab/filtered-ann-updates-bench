#!/usr/bin/env python3
"""Analysis of the follow-up experiments H1-H8 (main env).  python3 bench/analyze_h.py > results/bench_h_analysis.txt
Reads the tagged result directories (results/bench_<tag>[_pass<k>]; best QPS per configuration across passes, as in
bench/analyze.py) and writes results/bench_h_summary.json. Every hypothesis is evaluated against the falsification
criterion stated in the research plan; sections whose data are missing are skipped."""
import sys, os, json, glob, math
sys.path.insert(0, 'bench')
import numpy as np
from analyze import load_static, summarize_static, frontier, qps_at, summarize_streams, merge_stream_passes, passes
from common import ROOT

OUT = {}
def say(*a): print(*a, flush=True)
NS = {'deep300k': 316_228, 'deep1m': 1_000_000, 'deep3m': 3_162_278, 'deep10m': 10_000_000}
INDEX = ('ivf', 'ivf_part', 'acorn', 'postfilter', 'fdiskann', 'curator')
TGT = (0.95, 0.99)

def L_of(sc): return int(round(1 / float(sc.split('_')[1][1:])))

# ------------------------------------------------------------------------------------------------ H1 crossover law
def boot_qps(recs, target, B, rng, scale_to=None):
    """Bootstrap over queries of QPS@target from per-query recall and time; None where it is not reached."""
    recs = [r for r in recs if 'tq' in r and 'rq' in r]
    if not recs: return None
    nq = len(recs[0]['rq']); out = []
    RQ = [np.asarray(r['rq']) for r in recs]; TQ = [np.asarray(r['tq']) for r in recs]
    scale = [r['qps']['mean'] / (nq / t.sum()) if t.sum() > 0 else 1.0 for r, t in zip(recs, TQ)]
    base = qps_at(frontier([(float(rq.mean()), s * nq / float(tq.sum())) for rq, tq, s in zip(RQ, TQ, scale) if tq.sum() > 0]), target)
    for _ in range(B):
        i = rng.integers(0, nq, nq)
        pts = [(float(rq[i].mean()), s * nq / float(tq[i].sum())) for rq, tq, s in zip(RQ, TQ, scale) if tq[i].sum() > 0]
        q = qps_at(frontier(pts), target)
        # a replicate is the point estimate times the resampling ratio, so that the reference mode (e.g. the batched
        # scan, which has no per-query times) and the resampled mode need not be the same
        out.append(None if q is None or base is None or scale_to is None else scale_to * q / base)
    return out

def crossing(ms, rs):
    """m values ascending, ratios (None = target not reached, counted as 0). Returns (m*, censoring)."""
    r = [0.0 if x is None else x for x in rs]
    if r[0] >= 1: return ms[0], 'below'
    for i in range(len(ms) - 1):
        if r[i] < 1 <= r[i + 1]:
            if r[i] <= 0: return ms[i + 1], 'upper'
            a, b = math.log(r[i]), math.log(r[i + 1]); w = (0 - a) / (b - a)
            return math.exp(math.log(ms[i]) + w * (math.log(ms[i + 1]) - math.log(ms[i]))), 'interp'
    return ms[-1], 'above'

def h1(tag='h1', label='H1 (1 thread)', ns=NS):
    W = load_static(tag)
    if not W: return None
    raw, S = summarize_static(W)
    res = {'per_n': {}, 'fit': {}}; rng = np.random.default_rng(0); BOOT = {}
    for t in TGT:
        say(f'\n== {label}: crossover admissible-set size m* at recall {t}, beta=0 ==')
        rows = {}
        for ds, n in ns.items():
            wls = sorted([wl for wl in S if wl.startswith(ds + '/') and wl.split('/')[1].startswith('b0.0_s') and '_r' not in wl], key=lambda wl: -L_of(wl.split('/')[1]))
            if len(wls) < 3 or any('scan_part' not in S[wl] for wl in wls): continue
            m = [n / L_of(wl.split('/')[1]) for wl in wls]; scan = [S[wl]['scan_part'][f'qps@{t}'] for wl in wls]
            row = {'m': m, 'scan_qps': scan}
            for meth in INDEX + ('best',):
                if meth == 'best':
                    rat = [max([S[wl][x][f'qps@{t}'] or 0 for x in INDEX if x in S[wl]] + [0]) / sc for wl, sc in zip(wls, scan)]
                else:
                    if not all(meth in S[wl] for wl in wls): continue
                    rat = [None if S[wl][meth][f'qps@{t}'] is None else S[wl][meth][f'qps@{t}'] / sc for wl, sc in zip(wls, scan)]
                ms_, cen = crossing(m, rat); row[meth] = {'ratio': rat, 'm_star': ms_, 'censor': cen}
                if meth in ('ivf', 'ivf_part', 'curator', 'fdiskann') and cen == 'interp':   # bootstrap where per-query data exist
                    bs = [boot_qps(raw[wl].get(meth, []), t, 200, rng, S[wl][meth][f'qps@{t}']) for wl in wls]
                    bsc = [boot_qps(raw[wl]['scan_part'], 1.0, 200, rng, S[wl]['scan_part'][f'qps@{t}']) for wl in wls]
                    if all(b is not None for b in bs) and all(b is not None for b in bsc):
                        stars = []; boot = []
                        for k in range(200):
                            rr = [None if bs[j][k] is None else bs[j][k] / bsc[j][k] for j in range(len(wls))]
                            s_, c_ = crossing(m, rr)
                            boot.append(s_ if c_ == 'interp' else None)
                            if c_ == 'interp': stars.append(s_)
                        BOOT[(str(t), ds, meth)] = boot
                        if len(stars) > 20: row[meth]['m_star_ci95'] = [float(np.quantile(stars, 0.025)), float(np.quantile(stars, 0.975))]
            rows[ds] = row
            say(f'{ds:9s} n={n:>10,}  ' + '  '.join(f"{k}:{v['m_star']:.3g}({v['censor'][0]})" for k, v in row.items() if isinstance(v, dict)))
        res['per_n'][str(t)] = rows
        fit = {}
        for meth in INDEX + ('best',):
            pts = [(ns[ds], r[meth]['m_star']) for ds, r in rows.items() if meth in r and r[meth]['censor'] == 'interp']
            if len(pts) >= 3:
                x = np.log([p[0] for p in pts]); y = np.log([p[1] for p in pts]); a, c = np.polyfit(x, y, 1)
                # bootstrap the exponent over the per-n m* intervals where available (else point estimates)
                fit[meth] = {'exponent': float(a), 'n_points': len(pts)}
                dss = [ds for ds, r in rows.items() if meth in r and r[meth]['censor'] == 'interp']
                if all((str(t), ds, meth) in BOOT for ds in dss):
                    sl = []
                    for k in range(200):
                        v = [BOOT[(str(t), ds, meth)][k] for ds in dss]
                        if all(x is not None for x in v): sl.append(np.polyfit(np.log([ns[ds] for ds in dss]), np.log(v), 1)[0])
                    if len(sl) > 50:     # basic (pivotal) bootstrap interval: recentred on the point estimate
                        lo, hi = np.quantile(sl, [0.025, 0.975]); fit[meth]['exponent_ci95'] = [float(2 * a - hi), float(2 * a - lo)]
                        fit[meth]['boot_valid'] = len(sl)
                ci_ = fit[meth].get('exponent_ci95'); cis = f' [95% CI {ci_[0]:.2f}, {ci_[1]:.2f}]' if ci_ else ''
                say(f'  m* ~ n^{a:.2f}{cis} for {meth} ({len(pts)} sizes); predicate-agnostic: positive, label-indexed: <= 0')
        res['fit'][str(t)] = fit
    return res

# ------------------------------------------------------------------------------------------------ H2 mechanism
def h2(tags=('h1', 'h4')):
    say(f'\n== H2 {tags}: does (entries checked, admissible distances) explain IVF time? does ndis explain ACORN time? ==')
    rows = []; arows = []
    for tag in tags:
        W = load_static(tag)
        for (ds, sc, m, p), r in W.items():
            ex = r.get('extra') or {}
            if m == 'ivf' and 'entries_q' in ex and r['qps']['mean'] > 0:
                rows.append((1e3 / r['qps']['mean'], float(np.mean(ex['entries_q'])), float(np.mean(ex['admissible_q'])), ds, sc, r['recall']))
            if m == 'acorn' and ex.get('ndis_per_query') and r['qps']['mean'] > 0:
                arows.append((1e3 / r['qps']['mean'], ex['ndis_per_query'], ds, sc))
    res = {}
    if rows:
        t = np.array([r[0] for r in rows]); A = np.c_[np.ones(len(rows)), [r[1] for r in rows], [r[2] for r in rows]]
        coef, *_ = np.linalg.lstsq(A, t, rcond=None); pred = A @ coef; r2 = 1 - ((t - pred) ** 2).sum() / ((t - t.mean()) ** 2).sum()
        rel = np.abs(t - pred) / t
        res['ivf'] = {'n_records': len(rows), 'ms_intercept': coef[0], 'ns_per_entry': 1e6 * coef[1], 'ns_per_admissible': 1e6 * coef[2], 'r2': r2,
                      'rel_err_median': float(np.median(rel)), 'rel_err_p90': float(np.quantile(rel, 0.9))}
        say(f"IVF: t = {coef[0]:.3f} ms + {1e6*coef[1]:.2f} ns/entry + {1e6*coef[2]:.2f} ns/admissible   R^2={r2:.3f}  "
            f"median rel err {np.median(rel):.1%}, p90 {np.quantile(rel, 0.9):.1%}  ({len(rows)} records)   [H2 falsified if residual > 20%]")
        # beta=0: entries needed per collected neighbor vs 1/s at the setting closest to recall 0.95
        best = {}
        for tm, e, a_, ds, sc, rc in rows:
            if not sc.startswith('b0.0') or '_r' in sc: continue
            k_ = (ds, sc)
            if rc >= 0.95 and (k_ not in best or tm < best[k_][0]): best[k_] = (tm, e, a_)
        pts = [(L_of(sc), e / 10, a_ / 10) for (ds, sc), (tm, e, a_) in best.items()]
        if pts:
            x = np.log([p[0] for p in pts]); y = np.log([p[1] for p in pts]); sl = np.polyfit(x, y, 1)[0]
            res['ivf_entries_per_k_vs_inv_s_slope'] = float(sl)
            say(f'  beta=0, cheapest setting with recall >= 0.95: entries/k ~ (1/s)^{sl:.2f} (H2 predicts slope ~1 when admissibility is independent of position)')
    if arows:
        t = np.array([r[0] for r in arows]); A = np.c_[np.ones(len(arows)), [r[1] for r in arows]]
        coef, *_ = np.linalg.lstsq(A, t, rcond=None); pred = A @ coef; r2 = 1 - ((t - pred) ** 2).sum() / ((t - t.mean()) ** 2).sum()
        res['acorn'] = {'n_records': len(arows), 'ms_intercept': coef[0], 'ns_per_distance': 1e6 * coef[1], 'r2': r2}
        say(f'ACORN: t = {coef[0]:.3f} ms + {1e6*coef[1]:.1f} ns/distance  R^2={r2:.3f} ({len(arows)} records; filter checks not counted)')
    # counting build (tag h2c): ACORN time against predicate checks, distance computations and hops
    W = load_static('h2c'); crow = []
    for (ds, sc, m, p), r in W.items():
        ex = r.get('extra') or {}
        if m == 'acorn' and ex.get('checks_per_query') and r['qps']['mean'] > 0:
            crow.append((1e3 / r['qps']['mean'], ex['checks_per_query'], ex['ndis_per_query'], ex['hops_per_query']))
    if len(crow) >= 5:
        t = np.array([c[0] for c in crow]); res['acorn_counts'] = {}
        for name, cols in (('distances', [2]), ('checks', [1]), ('checks+distances', [1, 2]), ('checks+distances+hops', [1, 2, 3])):
            A = np.c_[np.ones(len(crow)), [[c[j] for j in cols] for c in crow]]
            coef, *_ = np.linalg.lstsq(A, t, rcond=None); pred = A @ coef; r2 = 1 - ((t - pred) ** 2).sum() / ((t - t.mean()) ** 2).sum()
            rel = np.abs(t - pred) / t
            res['acorn_counts'][name] = {'coef_ns': [1e6 * c for c in coef[1:]], 'ms_intercept': coef[0], 'r2': r2, 'rel_err_median': float(np.median(rel))}
            say(f'ACORN (counting build, {len(crow)} records) t ~ {name}: R^2={r2:.3f}, median rel err {np.median(rel):.1%}, ns per unit ' + ', '.join(f'{1e6*c:.2f}' for c in coef[1:]))
        ratio = np.array([c[1] / c[2] for c in crow if c[2] > 0])
        res['acorn_counts']['checks_per_distance_median'] = float(np.median(ratio))
        say(f'  predicate checks per distance computation: median {np.median(ratio):.1f}')
    return res

# ------------------------------------------------------------------------------------------------ H4 rho
def h4():
    W = load_static('h4')
    if not W: return None
    _, S = summarize_static(W); res = {}
    say('\n== H4: query-filter correlation rho on SIFT1M, s=0.1 (QPS at recall 0.95; ratio to the per-label scan) ==')
    for beta in ('0.0', '0.9'):
        for rho in ('', '_r0.5', '_r1.0'):
            wl = f'sift1m/b{beta}_s0.1{rho}'
            if wl not in S: continue
            sc = S[wl].get('scan_part', {}).get('qps@0.95')
            row = {m: S[wl][m]['qps@0.95'] for m in S[wl] if m not in ('scan_mask',)}
            res[wl] = {'qps@0.95': row, 'max_recall': {m: S[wl][m]['max_recall'] for m in S[wl]}}
            rank = sorted([(q, m) for m, q in row.items() if q], reverse=True)
            say(f"beta={beta} rho={rho[2:] or '0'}: " + ', '.join(f'{m} {q:,.0f}' for q, m in rank) + f'   (scan {sc:,.0f})' if sc else '')
    res['gain_rho1'] = {}
    for beta in ('0.0', '0.9'):
        a_, b_ = S.get(f'sift1m/b{beta}_s0.1', {}), S.get(f'sift1m/b{beta}_s0.1_r1.0', {})
        res['gain_rho1'][beta] = {m: b_[m]['qps@0.95'] / a_[m]['qps@0.95'] for m in a_ if m in b_ and a_[m]['qps@0.95'] and b_[m]['qps@0.95']}
        say(f'  beta={beta}: QPS@0.95 at rho=1 over rho=0: ' + ', '.join(f'{m} {v:.2f}x' for m, v in res['gain_rho1'][beta].items()))
    return res

# ------------------------------------------------------------------------------------------------ H5 predicate cost
def h5():
    p = 'results/bench_h5/predicate_cost.json'
    if not os.path.exists(p): return None
    s = json.load(open(p))['summary']; summ = json.load(open('results/bench_summary.json'))
    ivf = summ['dispatch']['yfcc10m/tags']['ivf']['best']['0.99']['qps']; t_ivf = 1e3 / ivf
    say('\n== H5: YFCC predicate evaluation per query (mean ms) vs the dispatched IVF query (%.2f ms, paper session) ==' % t_ivf)
    for k in ('numpy', 'roaring_and', 'roaring_ids', 'dense_and'):
        say(f"  {k:12s} mean {s[k]['mean_ms']:.3f} ms  p99 {s[k]['p99_ms']:.3f} ms   = {s[k]['mean_ms'] / t_ivf:.0%} of the IVF query")
    say(f"  AND queries only ({s['n_and_queries']}): " + ', '.join(f'{k} {v:.3f} ms' for k, v in s['and_queries'].items()))
    return {'summary': s, 'ivf_query_ms': t_ivf, 'share': {k: s[k]['mean_ms'] / t_ivf for k in ('numpy', 'roaring_and', 'dense_and')}}

# ------------------------------------------------------------------------------------------------ H3 percolation
def h3():
    rows = []; res = {}
    for root in passes('h3') + passes('h3b'):        # h3b: the same streams with upper-level entry counts
        for f in glob.glob(f'{root}/stream_*/acorn*.json'):
            d = json.load(open(f)); nm = f.split('/')[-2][len('stream_'):]
            for r in d['rounds']:
                if 'perc' not in r or 'sweep' not in r: continue
                for lab in r['perc']:
                    b = lab['label']; rec = max(s['recall_by_label'].get(str(b), s['recall_by_label'].get(b, np.nan)) for s in r['sweep'])
                    rows.append({'stream': nm, 'method': d['method'], 'pass': root, 'round': r['round'], 'label': b, 'recall': rec, **lab})
    if not rows: return None
    say('\n== H3: per-label maximum recall vs the live admissible subgraph (ACORN streams) ==')
    for meth in ('acorn', 'acorn_repair', 'acorn_repair_all'):
        R_ = [x for x in rows if x['method'] == meth]
        if not R_: continue
        deg = np.array([x['deg_mean'] for x in R_]); lcc = np.array([x['lcc_frac'] for x in R_]); rec = np.array([x['recall'] for x in R_])
        bins = [0, 1, 2, 4, 8, 16, 32, 65]; tab = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (deg >= lo) & (deg < hi)
            if m.any(): tab.append((lo, hi, int(m.sum()), float(rec[m].mean()), float(lcc[m].mean())))
        res[meth] = {'by_degree': tab, 'spearman_deg_recall': float(np.corrcoef(np.argsort(np.argsort(deg)), np.argsort(np.argsort(rec)))[0, 1])}
        say(f'{meth}: label-rounds={len(R_)}; recall by mean effective degree: ' + '; '.join(f'[{lo},{hi}) n={c} rec={r_:.3f} lcc={l_:.2f}' for lo, hi, c, r_, l_ in tab))
    S = summarize_streams('h3')
    for nm, ms in summarize_streams('h3b').items():
        if 'acorn_repair_all' in ms: S.setdefault(nm, {})['acorn_repair_all'] = ms['acorn_repair_all']
    for nm, ms in S.items():
        for meth, d in ms.items():
            say(f"  {nm} {meth}: max recall by round " + ' '.join(f"r{r['round']}:{r['max_recall']:.3f}" for r in d['rounds']) + f"  total maintenance {d['total_maint_s']:.0f} s")
    res['streams'] = {nm: {m: {'rounds': [(r['round'], r['max_recall'], r['qps@0.95']) for r in d['rounds']], 'maint': d['total_maint_s']} for m, d in ms.items()} for nm, ms in S.items()}
    T_ = [x for x in rows if x['method'] == 'acorn']; Rp = [x for x in rows if x['method'] == 'acorn_repair']
    def band(R, lo, hi):
        v = [x['recall'] for x in R if lo <= x['deg_mean'] < hi]; return (float(np.mean(v)), len(v)) if v else (None, 0)
    summ = {f'tomb_rec_deg_{lo}_{hi}': band(T_, lo, hi)[0] for lo, hi in ((0, 2), (2, 4), (4, 8), (8, 16), (16, 65))}
    summ.update({f'tomb_n_deg_{lo}_{hi}': band(T_, lo, hi)[1] for lo, hi in ((0, 2), (2, 4), (4, 8), (8, 16), (16, 65))})
    low = [x for x in T_ if x['recall'] < 0.5]
    summ['tomb_lowrec_lcc_mean'] = float(np.mean([x['lcc_frac'] for x in low])) if low else None
    starv = [x for x in Rp if x['recall'] < 0.5]
    summ['repair_lowrec_n'] = len(starv); summ['repair_lowrec_deg_mean'] = float(np.mean([x['deg_mean'] for x in starv])) if starv else None
    summ['repair_lowrec_upper_live'] = [x.get('upper_live') for x in starv]
    adv = res['streams'].get('sift500k_adv', {})
    RA = [x for x in rows if x['method'] == 'acorn_repair_all' and x['recall'] < 0.5]
    summ['repair_all_lowrec_n'] = len(RA)
    for meth in ('acorn', 'acorn_repair', 'acorn_repair_all'):
        if meth in adv:
            for rr, mr, _ in adv[meth]['rounds']: summ[f'adv_{meth}_r{rr}'] = mr
            summ[f'adv_{meth}_maint'] = adv[meth]['maint']
    if 'acorn' in adv and 'acorn_repair' in adv: summ['repair_maint_ratio'] = adv['acorn_repair']['maint'] / adv['acorn']['maint']
    if 'acorn' in adv and 'acorn_repair_all' in adv: summ['repair_all_maint_ratio'] = adv['acorn_repair_all']['maint'] / adv['acorn']['maint']
    res['summary'] = summ
    say('  summary: ' + ', '.join(f'{k}={v:.3f}' if isinstance(v, float) else f'{k}={v}' for k, v in summ.items()))
    return res

# ------------------------------------------------------------------------------------------------ H6 EOQ
def h6():
    res = {}; S = summarize_streams('h6')
    files = {}
    for root in passes('h6'):
        for f in glob.glob(f'{root}/stream_*/fdiskann_delta.json'): files.setdefault(f.split('/')[-2][len('stream_'):], []).append(f)
    if not files: return None
    say('\n== H6: Filtered-DiskANN stale main index + exact delta, total cost of rebuilding every P rounds (recall 0.95) ==')
    for nm, fl in files.items():
        runs = [json.load(open(f)) for f in fl]; B = min(r['rebuild_s'] for r in runs)
        # stream size (rounds R, mean updates per round U) from data/, cached with the results for runs without data/
        cp = f'results/bench_h6/stream_{nm}_sched.json'; meta = None; U = None
        if os.path.exists(f'{ROOT}/stream_{nm}/meta.json') and os.path.exists(f'{ROOT}/stream_{nm}/sched.npz'):
            meta = json.load(open(f'{ROOT}/stream_{nm}/meta.json')); sch = np.load(f'{ROOT}/stream_{nm}/sched.npz')
            U = float(np.mean([len(sch[f'del{r}']) + len(sch[f'ins{r}']) for r in range(meta['R'])]))
            os.makedirs(os.path.dirname(cp), exist_ok=True); json.dump({'R': meta['R'], 'U': U}, open(cp, 'w'), indent=1)
        elif os.path.exists(cp):
            c = json.load(open(cp)); meta = {'R': c['R']}; U = c['U']
        ages = {}
        for r in runs:
            for a in r['ages']:
                fr = frontier([(s['recall'], s['qps']['mean']) for s in a['sweep']]); q = qps_at(fr, 0.95)
                if q and (a['age'] not in ages or q > ages[a['age']]['q']): ages[a['age']] = {'q': q, 'delta': a['delta_per_label_mean']}
        if 0 not in ages: continue
        A = sorted(ages); tq = {a: 1 / ages[a]['q'] for a in A}
        # linear model t(a) = t0 + g*a (g: delta scan growth per round), fitted on the measured ages
        g, t0 = np.polyfit(A, [tq[a] for a in A], 1)
        iv = S.get(nm, {}).get('ivf')
        ivf_m = iv['total_maint_s'] / (len(iv['rounds']) and (meta['R'] if meta else 1)) if iv else None
        ivf_t = 1 / np.mean([r['qps@0.95'] for r in iv['rounds'] if r['qps@0.95']]) if iv else None
        def t_of(a): return float(np.interp(a, A, [tq[x] for x in A])) if a <= A[-1] else t0 + g * a
        rows = []
        for qpu in (0.1, 0.3, 1, 3, 10, 30, 100):
            Qr = qpu * U if U else None
            if Qr is None: continue
            costs = {P: B / P + Qr * np.mean([t_of(a) for a in range(P)]) for P in range(1, max(A) + 2)}
            Pb = min(costs, key=costs.get); Pstar = math.sqrt(2 * B / (Qr * g)) if g > 0 else float('inf')
            c_ivf = ivf_m + Qr * ivf_t if ivf_t else None
            rows.append({'q_per_update': qpu, 'best_P': Pb, 'P_star_eoq': Pstar, 'cost_best': costs[Pb], 'cost_P1': costs[1], 'cost_ivf': c_ivf})
        res[nm] = {'rebuild_s': B, 'ages': {a: ages[a] for a in A}, 't0_ms': 1e3 * t0, 'growth_ms_per_round': 1e3 * g, 'grid': rows}
        win = [r['q_per_update'] for r in rows if r['cost_ivf'] and r['cost_best'] < min(r['cost_P1'], r['cost_ivf'])]
        res[nm]['delta_wins_qpu'] = [min(win), max(win)] if win else None
        agree = [abs(math.log(r['best_P'] / r['P_star_eoq'])) for r in rows if 1 < r['best_P'] < max(A) + 1 and math.isfinite(r['P_star_eoq'])]
        res[nm]['eoq_median_factor'] = math.exp(float(np.median(agree))) if agree else None
        for r in rows: res[nm][f"best_P_qpu{r['q_per_update']:g}"] = r['best_P']; res[nm][f"P_star_qpu{r['q_per_update']:g}"] = r['P_star_eoq']
        say(f'{nm}: B={B:.0f} s; t(a) = {1e3*t0:.3f} + {1e3*g:.3f}*a ms; QPS by age ' + ' '.join(f'a{a}:{ages[a]["q"]:,.0f}' for a in A))
        for r in rows:
            say(f"   Q/U={r['q_per_update']:>5}: best P={r['best_P']:>2} (EOQ P*={r['P_star_eoq']:.1f})  cost/round best {r['cost_best']:.0f} s, P=1 {r['cost_P1']:.0f} s"
                + (f", IVF {r['cost_ivf']:.0f} s" if r['cost_ivf'] else ''))
    return res

# ------------------------------------------------------------------------------------------------ paradigm A
def predictive():
    W = load_static('h1')
    if not W: return None
    _, S = summarize_static(W); res = {}
    say('\n== Paradigm A: fit T = c n^a s^b on n <= 3.16e6, predict QPS and m* at n = 1e7 (recall 0.95) ==')
    for meth in ('scan_part', 'ivf', 'ivf_part', 'curator', 'acorn'):
        tr, te = [], []
        for wl, ms in S.items():
            ds, sc = wl.split('/')
            if ds not in NS or not sc.startswith('b0.0_s') or '_r' in sc or meth not in ms or not ms[meth]['qps@0.95']: continue
            row = (NS[ds], 1 / L_of(sc), 1 / ms[meth]['qps@0.95'], wl)
            (te if ds == 'deep10m' else tr).append(row)
        if len(tr) < 4 or not te: continue
        A = np.array([[1, math.log(n), math.log(s)] for n, s, _, _ in tr]); y = np.log([t for *_, t, _ in tr])
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
        pe = [(wl, 1 / math.exp(c[0] + c[1] * math.log(n) + c[2] * math.log(s)), 1 / t) for n, s, t, wl in te]
        err = [p / q - 1 for _, p, q in pe]
        res[meth] = {'a_n': float(c[1]), 'b_s': float(c[2]), 'pred': pe, 'rel_err_median': float(np.median(np.abs(err)))}
        say(f'{meth:9s} T ~ n^{c[1]:.2f} s^{c[2]:.2f} (train {len(tr)}); n=1e7 QPS predicted/measured: ' + ', '.join(f'{wl.split("/")[1]} {p:,.0f}/{q:,.0f}' for wl, p, q in pe))
    return res

# ------------------------------------------------------------------------------------------------ measurement biases
def biases():
    """(i) Interpolation: for every QPS@target of the paper's results, the measured frontier points that bracket the
    target and the ratio of their QPS, which bounds the error of log-linear interpolation (the interpolated value lies
    between them). (ii) Best-of-two: per method, how much the reported (best-pass) QPS exceeds each single pass."""
    W = load_static(); raw, S = summarize_static(W); res = {'interp': {}, 'best_of_two': {}}
    say('\n== Measurement biases of the paper results ==')
    for t in (0.95, 0.99):
        gaps = {}
        for wl, ms in S.items():
            for m, d in ms.items():
                fr = d['frontier']
                if not d.get(f'qps@{t}') or fr[-1][0] < t: continue
                hi = [p for p in fr if p[0] >= t][0]; lo = [p for p in fr if p[0] < t]
                if not lo or hi[0] == t: gaps.setdefault(m, []).append(1.0); continue
                gaps.setdefault(m, []).append(lo[-1][1] / hi[1])
        res['interp'][str(t)] = {m: {'median_bracket_ratio': float(np.median(v)), 'p90': float(np.quantile(v, 0.9)), 'max': float(max(v)), 'n': len(v)} for m, v in gaps.items()}
        say(f'recall {t}: QPS ratio of the bracketing measured points (interpolation error bound), median / p90 by method: ' +
            ', '.join(f'{m} {np.median(v):.2f}/{np.quantile(v, 0.9):.2f}' for m, v in sorted(gaps.items())))
    flips = []; ncmp = 0
    for wl, ms in S.items():
        if 'scan_part' not in ms: continue
        for m, d in ms.items():
            if m.startswith('scan'): continue
            for t in (0.95, 0.99):
                qi, sc = d.get(f'qps@{t}'), ms['scan_part'].get(f'qps@{t}')
                if not qi or not sc: continue
                fr = d['frontier']; hi = [p for p in fr if p[0] >= t][0]; lo = [p for p in fr if p[0] < t]; lo = lo[-1] if lo else hi
                ncmp += 1; v = [(hi[1] / sc) >= 1, (qi / sc) >= 1, (lo[1] / sc) >= 1]
                if len(set(v)) > 1: flips.append([wl, m, t, hi[1] / sc, qi / sc, lo[1] / sc])
    res['verdicts'] = {'compared': ncmp, 'within_bracket': len(flips), 'cases': flips}
    say(f'index-vs-scan verdicts that change within the bracketing measured points: {len(flips)} of {ncmp}')
    for (ds, sc, m, p), r in W.items():
        if r.get('passes', 1) > 1 and min(r['qpss']) > 0:
            res['best_of_two'].setdefault(m, []).append(max(r['qpss']) / np.mean(r['qpss']))
    res['best_of_two'] = {m: {'median_best_over_mean': float(np.median(v)), 'p90': float(np.quantile(v, 0.9)), 'n': len(v)} for m, v in res['best_of_two'].items()}
    say('best-of-two: reported QPS over the mean of the two passes, median by method: ' +
        ', '.join(f"{m} {v['median_best_over_mean']:.3f}" for m, v in sorted(res['best_of_two'].items())))
    return res

def dense(bias):
    """Re-check the verdicts that change within the bracket of measured points, using the sqrt(2) dense sweeps (tag
    'dense', its own scan reference): a verdict is kept if the dense ratio to the scan is on the same side of 1."""
    W = load_static('dense')
    if not W or not bias: return None
    _, S = summarize_static(W); kept = flipped = 0; cases = []; unresolved = 0
    for wl, m, t, lo_r, mid_r, hi_r in bias['verdicts']['cases']:
        d = S.get(wl, {}); sc = d.get('scan_part', {}).get(f'qps@{t}')
        if m not in d or not sc: continue
        q = d[m].get(f'qps@{t}'); r = (q or 0) / sc
        fr = d[m]['frontier']; hi = [p for p in fr if p[0] >= t]; lo = [p for p in fr if p[0] < t]
        still = bool(hi and lo and ((hi[0][1] / sc >= 1) != (lo[-1][1] / sc >= 1)))
        same = (r >= 1) == (mid_r >= 1); kept += same; flipped += not same; unresolved += still
        cases.append({'workload': wl, 'method': m, 'target': t, 'interpolated_ratio': mid_r, 'dense_ratio': r, 'kept': same, 'still_within_bracket': still})
    fl = [c for c in cases if not c['kept']]; big = max(fl, key=lambda c: abs(c['dense_ratio'] - 1), default=None)
    res = {'kept': kept, 'flipped': flipped, 'checked': kept + flipped, 'still_within_bracket': unresolved, 'cases': cases,
           'not_covered': len(bias['verdicts']['cases']) - kept - flipped,      # methods outside the dense campaign (RIG, ACORN-1)
           'flipped_within_5pct': sum(abs(c['dense_ratio'] - 1) <= 0.05 and abs(c['interpolated_ratio'] - 1) <= 0.05 for c in fl),
           'largest_flip': big}
    say(f'\n== dense sweeps: of {kept + flipped} interpolation-sensitive verdicts, {kept} kept, {flipped} reversed ({unresolved} still within the sqrt(2) bracket) ==')
    for c in cases: say(f"  {c['workload']} {c['method']} @{c['target']}: interpolated {c['interpolated_ratio']:.2f} -> dense {c['dense_ratio']:.2f}" + ('' if c['kept'] else '  REVERSED'))
    return res

def pgv():
    """pgvector inside PostgreSQL (tag 'pg'): QPS of each mode, recall of the optimizer's default plan and the access path
    it chose, and the crossover of the vector-index modes with the DBMS's own exact plan on the DEEP-1M n-sweep schemas."""
    W = load_static('pg')
    if not W: return None
    raw, S = summarize_static(W); res = {'workloads': {}, 'crossover_deep1m': {}}
    say('\n== pgvector (PostgreSQL 16, pgvector 0.8.1) ==')
    for wl in sorted(S):
        ms = S[wl]; a = raw[wl].get('pgvector_auto', [{}])[0]
        res['workloads'][wl] = {'qps@0.95': {m: d['qps@0.95'] for m, d in ms.items()}, 'max_recall': {m: d['max_recall'] for m, d in ms.items()},
                                'auto_recall': a.get('recall'), 'auto_uses_vector_index': 'items_emb_idx' in (a.get('extra') or {}).get('plan_uses', [])}
        say(f"{wl}: auto plan {'HNSW' if res['workloads'][wl]['auto_uses_vector_index'] else 'B-tree exact'} recall {a.get('recall', 0):.3f}; QPS@0.95 " +
            ', '.join(f"{m[9:]} {d['qps@0.95']:,.0f}" for m, d in ms.items() if d['qps@0.95']))
    wls = sorted([wl for wl in S if wl.startswith('deep1m/b0.0_s')], key=lambda wl: -L_of(wl.split('/')[1]))
    if wls:
        m_ = [1e6 / L_of(wl.split('/')[1]) for wl in wls]; ex = [S[wl]['pgvector_exact']['qps@0.95'] for wl in wls]
        for meth in ('pgvector_hnsw', 'pgvector_ivfflat'):
            for t in TGT:
                rat = [None if not S[wl][meth][f'qps@{t}'] else S[wl][meth][f'qps@{t}'] / S[wl]['pgvector_exact'][f'qps@{t}'] for wl in wls]
                ms_, cen = crossing(m_, rat); res['crossover_deep1m'][f'{meth[9:]}_{t}'] = {'m_star': ms_, 'censor': cen}
                say(f'  deep1m crossover with the exact plan, {meth[9:]} @ {t}: m* = {ms_:,.0f} ({cen})')
        bad = [res['workloads'][wl]['auto_recall'] for wl in res['workloads'] if res['workloads'][wl]['auto_uses_vector_index']]
        res['auto_bad_recall_min'] = min(bad) if bad else None; res['auto_bad_recall_max'] = max(bad) if bad else None
        res['auto_vector_plans'] = len(bad); res['auto_plans'] = len(res['workloads'])
        fs = [S[wl]['scan_part']['qps@0.95'] if 'scan_part' in S[wl] else None for wl in wls]
    return res

def mvs():
    """Milvus 2.6 standalone (tag 'mv'): QPS of each mode, the recall of a search with the server's default parameters,
    the crossover of HNSW and IVF_FLAT with Milvus's own exact (FLAT) search on the DEEP-1M n-sweep, the per-query
    floor (a search on a 1-row collection), and whether HNSW at its smallest ef is exact on selective filters (the
    server switching to brute force over the admissible rows)."""
    W = load_static('mv')
    if not W: return None
    raw, S = summarize_static(W); res = {'workloads': {}, 'crossover_deep1m': {}}
    say('\n== Milvus 2.6 standalone ==')
    for wl in sorted(S):
        ms = S[wl]; dflt = (raw[wl].get('milvus_hnsw_default') or [{}])[0]; h = sorted(raw[wl].get('milvus_hnsw', []), key=lambda r: int(r['param']))
        res['workloads'][wl] = {'qps@0.95': {m: d['qps@0.95'] for m, d in ms.items()}, 'qps@0.99': {m: d['qps@0.99'] for m, d in ms.items()},
                                'max_recall': {m: d['max_recall'] for m, d in ms.items()}, 'default_recall': dflt.get('recall'),
                                'hnsw_ef16_recall': h[0]['recall'] if h else None, 'trivial_ms': 1e3 * (dflt.get('extra') or {}).get('trivial_query_s', float('nan'))}
        say(f"{wl}: default-parameter recall {dflt.get('recall', 0):.3f}; HNSW ef=16 recall {res['workloads'][wl]['hnsw_ef16_recall'] or 0:.3f}; QPS@0.95 " +
            ', '.join(f"{m[7:]} {d['qps@0.95']:,.0f}" for m, d in ms.items() if d['qps@0.95']))
    wls = sorted([wl for wl in S if wl.startswith('deep1m/b0.0_s')], key=lambda wl: -L_of(wl.split('/')[1]))
    if wls and all('milvus_flat' in S[wl] for wl in wls):
        m_ = [1e6 / L_of(wl.split('/')[1]) for wl in wls]
        for meth in ('milvus_hnsw', 'milvus_ivf_flat'):
            for t in TGT:
                rat = [None if not S[wl].get(meth, {}).get(f'qps@{t}') else S[wl][meth][f'qps@{t}'] / S[wl]['milvus_flat'][f'qps@{t}'] for wl in wls]
                ms_, cen = crossing(m_, rat); res['crossover_deep1m'][f'{meth[7:]}_{t}'] = {'m_star': ms_, 'censor': cen, 'ratio': rat}
                say(f'  deep1m crossover with FLAT, {meth[7:]} @ {t}: m* = {ms_:,.0f} ({cen}); ratios ' + ' '.join('-' if r is None else f'{r:.2f}' for r in rat))
    gd = [w['default_recall'] for w in res['workloads'].values() if w['default_recall'] is not None and (w['hnsw_ef16_recall'] or 0) < 0.999]
    res['graph_default_recall_min'] = min(gd) if gd else None; res['graph_default_recall_max'] = max(gd) if gd else None
    dr = [w['default_recall'] for w in res['workloads'].values() if w['default_recall'] is not None]
    res['default_recall_min'] = min(dr) if dr else None; res['default_recall_max'] = max(dr) if dr else None
    tr = [w['trivial_ms'] for w in res['workloads'].values() if math.isfinite(w['trivial_ms'])]
    res['trivial_ms_median'] = float(np.median(tr)) if tr else None
    ex = [wl for wl, w in res['workloads'].items() if (w['hnsw_ef16_recall'] or 0) >= 0.999]
    res['hnsw_exact_at_ef16'] = ex; res['hnsw_exact_at_ef16_n'] = len(ex)
    # Knowhere's fixed rule (brute force when >= 93% of a segment is filtered out, i.e. s <= 0.07): on the DEEP-1M
    # n-sweep, compare that brute-force path (HNSW mode at s <= 0.07, exact at every ef) with IVF_FLAT in the same server
    bf = [wl for wl in wls if float(wl.split('_s')[1]) <= 0.07 and 'milvus_hnsw' in S[wl] and 'milvus_ivf_flat' in S[wl]]
    if bf:
        q_bf = [S[wl]['milvus_hnsw']['qps@0.95'] for wl in bf]; q_ivf = [S[wl]['milvus_ivf_flat']['qps@0.95'] for wl in bf]
        g = [wl for wl in wls if float(wl.split('_s')[1]) > 0.07 and 'milvus_hnsw' in S[wl]]
        res['bf_rule'] = {'workloads': bf, 'bf_qps_min': min(q_bf), 'bf_qps_max': max(q_bf), 'ivf_qps_min': min(q_ivf), 'ivf_qps_max': max(q_ivf),
                          'ivf_over_bf': [i / b for i, b in zip(q_ivf, q_bf)], 'ivf_over_bf_max': max(i / b for i, b in zip(q_ivf, q_bf)),
                          'ivf_over_bf_min': min(i / b for i, b in zip(q_ivf, q_bf)),
                          'graph_qps': {wl: S[wl]['milvus_hnsw']['qps@0.95'] for wl in g}}
        say(f"  Knowhere brute-force rule on deep1m (s <= 0.07): HNSW-as-brute-force {min(q_bf):,.0f}-{max(q_bf):,.0f} QPS, IVF_FLAT {min(q_ivf):,.0f}-{max(q_ivf):,.0f} QPS @0.95 "
            f"(IVF_FLAT / brute force {res['bf_rule']['ivf_over_bf_min']:.2f}-{res['bf_rule']['ivf_over_bf_max']:.2f}); graph path above: " +
            ', '.join(f"{wl.split('/')[1]} {q:,.0f}" for wl, q in res['bf_rule']['graph_qps'].items() if q))
    return res

# ------------------------------------------------------------------------------------------------ arXiv (4096-d)
NS_AX = {'arxiv100k': 100_000, 'arxiv300k': 316_228, 'arxiv1m': 999_424}
def arxiv():
    """arXiv abstracts (4096-d embeddings, real categories; tag 'ax'): the H1 crossover law at a high dimension, the
    per-entry vs per-distance cost ratio there (Proposition 1 predicts a lower crossover when c_chk/c_dist is lower),
    and the real category predicates (arxiv1m/cat) as a YFCC-like workload."""
    if not load_static('ax'): return None
    res = {'H1': h1('ax', 'arXiv H1', NS_AX), 'H2': h2(('ax',))}
    if res['H2'] and 'ivf' in res['H2']:
        rc = res['H2']['ivf']['ns_per_entry'] / res['H2']['ivf']['ns_per_admissible']
        res['bound'] = {'chk_over_dist': rc, **{f'm_lower_{ds}': math.sqrt(rc * 10 * n) for ds, n in NS_AX.items()}}
        say(f"arXiv: c_chk/c_dist = {rc:.4f}; Proposition 1 bound " + ', '.join(f'{ds} {math.sqrt(rc * 10 * n):,.0f}' for ds, n in NS_AX.items()))
    raw, S = summarize_static(load_static('ax'))
    if 'arxiv1m/cat' in S:
        ms = S['arxiv1m/cat']; sc = ms.get('scan_part', {}).get('qps@0.95')
        # reference: the faster exact scan (at d=4096 gathering the admissible rows can be slower than a bitmap scan of all rows)
        bsc = max([ms[x]['qps@0.95'] or 0 for x in ('scan_part', 'scan_mask') if x in ms] + [0]) or None
        res['cat'] = {m: {'qps@0.95': d['qps@0.95'], 'qps@0.99': d['qps@0.99'], 'max_recall': d['max_recall'],
                          'over_scan@0.95': (d['qps@0.95'] / sc) if sc and d['qps@0.95'] else None,
                          'over_best_scan@0.95': (d['qps@0.95'] / bsc) if bsc and d['qps@0.95'] else None} for m, d in ms.items()}
        wp, cp = f'{ROOT}/arxiv1m/cat/pred.npz', 'results/bench_ax/cat_workload.json'   # the cached copy ships with the results
        if os.path.exists(wp):
            z = np.load(wp); a_ = np.diff(z['qoff'])
            json.dump({'A_median': float(np.median(a_)), 'A_min': int(a_.min()), 'A_max': int(a_.max()), 'empty_queries': int((a_ == 0).sum()),
                       'categories_in_queries': int(len(np.unique(z['qtag_idx'][z['qtag_idx'] >= 0]))), 'n': int(len(z['ptag_ptr']) - 1)}, open(cp, 'w'), indent=1)
        res['cat_workload'] = json.load(open(cp)); res['cat_workload']['sel_median'] = res['cat_workload']['A_median'] / res['cat_workload']['n']
        say('arXiv-1M categories: ' + ', '.join(f"{m} {d['qps@0.95'] or 0:,.0f}" for m, d in ms.items()) + f"  (QPS@0.95; {res['cat_workload']})")
    return res

if __name__ == '__main__':
    OUT['bias'] = biases(); OUT['dense'] = dense(OUT['bias']); OUT['pg'] = pgv(); OUT['mv'] = mvs()
    OUT['H1'] = h1('h1'); OUT['H7'] = h1('h7mt28', 'H7 (28 threads)'); OUT['H2'] = h2(); OUT['H4'] = h4(); OUT['H5'] = h5()
    OUT['H3'] = h3(); OUT['H6'] = h6(); OUT['A'] = predictive(); OUT['AX'] = arxiv()
    if OUT['H2'] and 'ivf' in OUT['H2']:      # Proposition 1's bound with the measured per-entry and per-distance costs
        rc = OUT['H2']['ivf']['ns_per_entry'] / OUT['H2']['ivf']['ns_per_admissible']
        OUT['bound'] = {'chk_over_dist': rc, **{f'm_lower_n{n:.0e}'.replace('+0', ''): math.sqrt(rc * 10 * n) for n in (3.16e5, 1e6, 3.16e6, 1e7)}}
        say(f"\nProposition 1 bound with c_chk/c_dist = {rc:.3f}: " + ', '.join(f'{k} {v:,.0f}' for k, v in OUT['bound'].items() if k.startswith('m_')))
    for k in ('H8',):
        p = 'results/bench_h8/estimation.json'
        if os.path.exists(p): OUT[k] = json.load(open(p))
    json.dump(OUT, open('results/bench_h_summary.json', 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))

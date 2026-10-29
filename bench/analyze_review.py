#!/usr/bin/env python3
"""Robustness analyses requested by the ICDE review (REVIEW_ICDE.md), from the existing results only. Incomplete
campaigns are an error; REVIEW_PARTIAL=1 skips them with a note (drafting only, never for the released summary).
  python3 bench/analyze_review.py > results/bench_r_analysis.txt      -> results/bench_r_summary.json
  E1  total cost within each pass: maintenance ratios to IVF, IVF's dominance on both cost terms, and the
      Filtered-DiskANN break-even per round, computed from one pass at a time (no mixing of the best QPS of one
      pass with the least maintenance of another, as the merged summary does).
  E2  crossover exponents fitted within each pass (machine-session variation), next to the merged fit.
  B1  crossovers against the per-query scan only (every compared call is one query; no BLAS sharing for the scan).
  B4  28-thread speed-ups of the scan and IVF per admissible-set size at n=1e7.
  D   dispatch policies calibrated on DEEP-1M and applied, unchanged, to other n and to arXiv: a fixed admissible-set
      size, a fixed selectivity, Milvus's 7%, the sqrt(n)-scaled size, and the sqrt(n)-scaled size corrected by the
      measured check/distance cost ratio; cost regret against the per-workload oracle choice of scan or IVF."""
import sys, os, json, glob, math
sys.path.insert(0, 'bench')
import numpy as np
from analyze import summarize_static, frontier, qps_at, break_even
from analyze_h import crossing, L_of, NS, NS_AX
from common import stream_stats

OUT = {}
def say(*a): print(*a, flush=True)

def load_roots(roots, keep=None):
    """load_static for an explicit list of result roots (one pass, or several merged best-QPS as analyze.load_static)."""
    W = {}
    for root in roots:
        for f in glob.glob(f'{root}/*/*/*.json'):
            ds, sc = f.split('/')[-3], f.split('/')[-2]
            if ds.startswith('stream_'): continue
            for r in json.load(open(f)):
                if keep and not keep(r): continue
                key = (ds, sc, r['method'], r['param']); old = W.get(key)
                if old is None or r['qps']['mean'] > old['qps']['mean']: W[key] = {**r, 'passes': 1, 'recalls': [r['recall']], 'qpss': [r['qps']['mean']]}
    return W

def crossovers(S, ns, t, meths=('ivf', 'acorn', 'postfilter', 'curator')):
    rows, fit = {}, {}
    for ds, n in ns.items():
        wls = sorted([wl for wl in S if wl.startswith(ds + '/') and wl.split('/')[1].startswith('b0.0_s') and '_r' not in wl], key=lambda wl: -L_of(wl.split('/')[1]))
        if len(wls) < 3 or any('scan_part' not in S[wl] or not S[wl]['scan_part'][f'qps@{t}'] for wl in wls): continue
        m = [n / L_of(wl.split('/')[1]) for wl in wls]; scan = [S[wl]['scan_part'][f'qps@{t}'] for wl in wls]
        rows[ds] = {}
        for meth in meths:
            if not all(meth in S[wl] for wl in wls): continue
            rat = [None if S[wl][meth][f'qps@{t}'] is None else S[wl][meth][f'qps@{t}'] / sc for wl, sc in zip(wls, scan)]
            ms_, cen = crossing(m, rat); rows[ds][meth] = {'m_star': ms_, 'censor': cen}
    for meth in meths:
        pts = [(ns[ds], r[meth]['m_star']) for ds, r in rows.items() if meth in r and r[meth]['censor'] == 'interp']
        if len(pts) >= 3: fit[meth] = float(np.polyfit(np.log([p[0] for p in pts]), np.log([p[1] for p in pts]), 1)[0])
    return rows, fit

# ------------------------------------------------------------------------------------------------ E2, B1
def exponents():
    res = {}
    for tag, ns in (('h1', NS), ('ax', NS_AX)):
        roots = [f'results/bench_{tag}'] + sorted(glob.glob(f'results/bench_{tag}_pass*'))
        if not os.path.isdir(roots[0]): continue
        r = {}
        for t in ('0.95',):
            r['merged'] = crossovers(summarize_static(load_roots(roots))[1], ns, t)[1]
            for i, root in enumerate(roots):
                r[f'pass{i + 1}'] = crossovers(summarize_static(load_roots([root]))[1], ns, t)[1]
            # B1: the scan as one query per call (records with param 'per_query'), every other method unchanged
            pq = lambda rec: not (rec['method'] == 'scan_part' and rec['param'] != 'per_query')
            rows, fit = crossovers(summarize_static(load_roots(roots, keep=pq))[1], ns, t)
            r['scan_per_query'] = fit; r['scan_per_query_mstar'] = {ds: {m: v['m_star'] for m, v in x.items()} for ds, x in rows.items()}
        for meth in ('ivf', 'acorn', 'postfilter', 'curator'):
            v = [r[k][meth] for k in r if k.startswith('pass') and meth in r[k]]
            if v: r.setdefault('pass_range', {})[meth] = [min(v), max(v)]
        res[tag] = r
        say(f'== {tag}: crossover exponents at recall 0.95 ==')
        for k, v in r.items():
            if isinstance(v, dict) and k != 'scan_per_query_mstar': say(f'  {k:16s} ' + '  '.join(f'{m} {x if isinstance(x, list) else round(x, 2)}' for m, x in v.items()))
    return res

# ------------------------------------------------------------------------------------------------ B4
def threads():
    S1 = summarize_static(load_roots(['results/bench_h1'] + sorted(glob.glob('results/bench_h1_pass*'))))[1]
    S7 = summarize_static(load_roots(['results/bench_h7mt28'] + sorted(glob.glob('results/bench_h7mt28_pass*'))))[1]
    rows = {}
    for wl in sorted(S7):
        if not wl.startswith('deep10m/b0.0_s'): continue
        s = float(wl.split('_s')[1]); row = {'s': s, 'm': 1e7 * s}
        for m in ('scan_part', 'ivf'):
            a, b = S1.get(wl, {}).get(m, {}).get('qps@0.95'), S7[wl].get(m, {}).get('qps@0.95')
            if a and b: row[m] = b / a
        rows[wl] = row
    # h7mt28 and h1 ran in different sessions, so the speed-ups themselves carry the session's speed factor; the ratio of
    # the scan's speed-up to IVF's cancels it (both measured in the same two sessions)
    for r in rows.values():
        if 'scan_part' in r and 'ivf' in r: r['scan_over_ivf'] = r['scan_part'] / r['ivf']
    rel = [r['scan_over_ivf'] for r in rows.values() if 'scan_over_ivf' in r and r['m'] >= 3e4]
    rel_small = [r['scan_over_ivf'] for r in rows.values() if 'scan_over_ivf' in r and r['m'] <= 3.2e3]
    sc = [r['scan_part'] for r in rows.values() if 'scan_part' in r]; iv = [r['ivf'] for r in rows.values() if 'ivf' in r]
    res = {'rows': rows, 'scan_x_min': min(sc), 'scan_x_max': max(sc), 'ivf_x_min': min(iv), 'ivf_x_max': max(iv),
           's_min': min(r['s'] for r in rows.values()), 's_max': max(r['s'] for r in rows.values()),
           'rel_large_min': min(rel), 'rel_large_max': max(rel), 'rel_small_max': max(rel_small)}
    say('== 28-thread speed-up at n=1e7, beta=0 (QPS@0.95, 28 threads / 1 thread) ==')
    for wl, r in rows.items(): say(f"  s={r['s']:<10g} |A|={r['m']:>9,.0f}  scan x{r.get('scan_part', float('nan')):.1f}  IVF x{r.get('ivf', float('nan')):.1f}  scan/IVF {r.get('scan_over_ivf', float('nan')):.2f} (sessions differ; only the ratio is comparable)")
    return res

# ------------------------------------------------------------------------------------------------ D: policies
def policies():
    H = json.load(open('results/bench_h_summary.json'))
    def table(tag, ns, d_name):
        W = load_roots([f'results/bench_{tag}'] + sorted(glob.glob(f'results/bench_{tag}_pass*'))); S = summarize_static(W)[1]; out = {}
        for ds, n in ns.items():
            for wl in [w for w in S if w.startswith(ds + '/b0.0_s') and '_r' not in w]:
                sc, iv = S[wl].get('scan_part', {}).get('qps@0.95'), S[wl].get('ivf', {}).get('qps@0.95')
                if sc: out[wl] = {'n': n, 'm': n / L_of(wl.split('/')[1]), 'c_scan': 1 / sc, 'c_ivf': (1 / iv) if iv else math.inf, 'data': d_name}
        return out
    T = {**table('h1', NS, 'DEEP'), **table('ax', NS_AX, 'arXiv')}
    # unseen replicates (label and IVF-training seeds 12, 13): the rules below are fixed on seed 11's DEEP-1M before
    # these runs, so this group is an evaluation on data the calibration never saw (all four sizes count)
    for sd in ('12', '13'):
        if os.path.isdir(f'results/bench_s{sd}'): T.update({f's{sd}:{k}': v for k, v in table(f's{sd}', NS, 'DEEP-seed').items()})
    m0 = H['H1']['per_n']['0.95']['deep1m']['ivf']['m_star']; rc_d = H['bound']['chk_over_dist']; rc_a = H['AX']['bound']['chk_over_dist']
    pol = {'fixed |A| (DEEP-1M)': lambda w: m0,
           'fixed s (DEEP-1M)': lambda w: m0 / 1e6 * w['n'],
           'fixed s = 7% (Milvus)': lambda w: 0.07 * w['n'],
           'sqrt(n)-scaled (DEEP-1M)': lambda w: m0 * math.sqrt(w['n'] / 1e6),
           'sqrt(n)-scaled, cost-ratio corrected': lambda w: m0 * math.sqrt(w['n'] / 1e6) * (math.sqrt(rc_a / rc_d) if w['data'] == 'arXiv' else 1.0)}
    # the same correction from a workload-independent microbenchmark (bench/microbench_cost.py), if it has run
    if os.path.exists('results/bench_mb/cost.json'):
        mb = json.load(open('results/bench_mb/cost.json')); r_mb = mb['4096']['chk_over_dist'] / mb['96']['chk_over_dist']
        pol['sqrt(n)-scaled, microbenchmark corrected'] = lambda w: m0 * math.sqrt(w['n'] / 1e6) * (math.sqrt(r_mb) if w['data'] == 'arXiv' else 1.0)
    res = {'calibration': {'m_star_deep1m': m0, 'rc_deep': rc_d, 'rc_arxiv': rc_a}, 'policies': {}}
    if os.path.exists('results/bench_mb/cost.json'): res['calibration']['microbenchmark'] = json.load(open('results/bench_mb/cost.json'))
    say('== dispatch policies calibrated on DEEP-1M (scan if |A| < T, else IVF; recall 0.95), applied elsewhere ==')
    for name, f in pol.items():
        per = {}
        for grp in ('DEEP n!=1e6', 'arXiv', 'DEEP unseen seeds'):
            if grp == 'DEEP n!=1e6': ws = [w for w in T.values() if w['data'] == 'DEEP' and w['n'] != 1e6]
            elif grp == 'arXiv': ws = [w for w in T.values() if w['data'] == 'arXiv']
            else: ws = [w for w in T.values() if w['data'] == 'DEEP-seed']
            if not ws: continue
            reg, wrong = [], 0
            for w in ws:
                ch = w['c_scan'] if w['m'] < f(w) else w['c_ivf']; best = min(w['c_scan'], w['c_ivf'])
                reg.append(ch / best - 1); wrong += ch > best * (1 + 1e-9)
            per[grp] = {'workloads': len(ws), 'wrong': wrong, 'regret_mean': float(np.mean(reg)), 'regret_max': float(max(reg)),
                        'inf': int(sum(math.isinf(x) for x in reg))}
        res['policies'][name] = per
        say(f'  {name:40s} ' + '  '.join(f"{g}: wrong {p['wrong']}/{p['workloads']}, mean regret {p['regret_mean']:.0%}, max {p['regret_max']:.0%}" for g, p in per.items()))
    return res

# ------------------------------------------------------------------------------------------------ E1: costs per pass
def stream_pass(root, nm):
    out = {}
    for f in glob.glob(f'{root}/stream_{nm}/*.json'):
        d = json.load(open(f))
        if 'rounds' not in d: continue
        rows = []
        for r in d['rounds']:
            if 'sweep' not in r: continue
            fr = frontier([(s['recall'], s['qps']['mean']) for s in r['sweep']])
            rows.append({'round': r['round'], 'q0.95': qps_at(fr, 0.95), 'q0.99': qps_at(fr, 0.99), 'rebuild_s': r.get('rebuild_s')})
        out[d['method']] = {'rounds': rows, 'maint': d['rounds'][-1].get('cum_maint_s') if d['rounds'] else None,
                            'rebuilds': [r['rebuild_s'] for r in d['rounds'] if r.get('rebuild_s') is not None]}
    return out

def costs():
    res = {}
    for nm in ('sift500k', 'sift500k_adv', 'deep5m'):
        st = stream_stats(nm); R, U = st['R'], st['updates_per_round']; per = {}
        for i, root in enumerate(['results/bench', 'results/bench_pass2']):
            P = stream_pass(root, nm)
            if 'ivf' not in P: continue
            iv = P['ivf']; r = {'x_ivf': {}, 'dominated': {}, 'fd_breakeven': {}}
            for m, d in P.items():
                if m == 'ivf': continue
                if m == 'fdiskann_rebuild' and d['rebuilds']: r['x_ivf'][m] = R * float(np.mean(d['rebuilds'])) / iv['maint']
                elif d['maint']: r['x_ivf'][m] = d['maint'] / iv['maint']
                if m in ('acorn', 'postfilter', 'curator', 'rig'):
                    qi = {x['round']: x['q0.95'] for x in iv['rounds']}
                    faster = all((qi.get(x['round']) or 0) >= (x['q0.95'] or 0) for x in d['rounds'])
                    r['dominated'][m] = bool(faster and d['maint'] and d['maint'] > iv['maint'])
            if 'fdiskann_rebuild' in P and P['fdiskann_rebuild']['rebuilds']:
                reb = float(np.mean(P['fdiskann_rebuild']['rebuilds'])); mi = iv['maint'] / R; fr = {x['round']: x for x in P['fdiskann_rebuild']['rounds']}
                for t in ('0.95', '0.99'):
                    vals, none = [], 0
                    for x in iv['rounds']:
                        y = fr.get(x['round'])
                        if not (y and y[f'q{t}'] and x[f'q{t}']): continue
                        Q = break_even(reb, mi, x[f'q{t}'], y[f'q{t}'])
                        if Q is None: none += 1
                        else: vals.append(Q / U)
                    r['fd_breakeven'][t] = {'min': min(vals) if vals else None, 'max': max(vals) if vals else None, 'nobreak': none, 'n': len(vals) + none}
            per[f'pass{i + 1}'] = r
        res[nm] = per
        say(f'== {nm}: costs within each pass ==')
        for p, r in per.items():
            say(f"  {p}: maintenance / IVF " + ', '.join(f'{m} {v:,.0f}' for m, v in r['x_ivf'].items()) + '; dominated by IVF: ' +
                ', '.join(f'{m}={v}' for m, v in r['dominated'].items()) + '; FD break-even q/update ' +
                ', '.join(f"@{t} {('%.1f-%.1f' % (b['min'], b['max'])) if b['min'] is not None else '-'} (IVF faster in {b['nobreak']}/{b['n']})" for t, b in r['fd_breakeven'].items()))
    # summary over streams and passes
    allx = {}
    for nm, per in res.items():
        for p, r in per.items():
            for m, v in r['x_ivf'].items(): allx.setdefault(m, []).append(v)
    res['x_ivf_range'] = {m: [min(v), max(v)] for m, v in allx.items()}
    res['all_dominated'] = all(v for per in res.values() if isinstance(per, dict) for r in per.values() if isinstance(r, dict) and 'dominated' in r for v in r['dominated'].values())
    say(f"  IVF dominates ACORN, post-filtering, Curator, and RIG on both terms in every stream and pass: {res['all_dominated']}")
    return res

# ------------------------------------------------------------------------------------------------ C3, D2: same-session streams
def same_session(tag, streams, required=()):
    """Per stream and pass of a tagged stream campaign: QPS@0.95/0.99 and maximum recall per measured round,
    total maintenance, and the maintenance ratio to IVF of the same pass (C3: label-partitioned IVF and the repaired
    ACORN next to IVF and tombstone ACORN; D2: the arXiv temporal stream)."""
    res = {}
    for nm in streams:
        per = {}
        for root in [f'results/bench_{tag}'] + sorted(glob.glob(f'results/bench_{tag}_pass*')):
            P = {}
            for f in glob.glob(f'{root}/stream_{nm}/*.json'):
                d = json.load(open(f))
                if 'rounds' not in d: continue
                rows = []
                for r in d['rounds']:
                    if 'sweep' not in r: continue
                    fr = frontier([(x['recall'], x['qps']['mean']) for x in r['sweep']])
                    rows.append({'round': r['round'], 'max_recall': max(x['recall'] for x in r['sweep']), 'q0.95': qps_at(fr, 0.95), 'q0.99': qps_at(fr, 0.99)})
                maint = d['rounds'][-1].get('cum_maint_s') if d['rounds'] else None
                reb = [r['rebuild_s'] for r in d['rounds'] if r.get('rebuild_s') is not None]
                if maint is None and reb: maint = float(sum(reb[1:])) if len(reb) > 1 else float(sum(reb))
                P[d['method']] = {'rounds': rows, 'maint': maint, 'build_s': d.get('build_s')}
            if not P: continue
            miss = [m for m in required if m not in P]
            if miss:
                if os.environ.get('REVIEW_PARTIAL') == '1': say(f'PARTIAL: {tag} {nm} {root} lacks {miss}'); continue
                raise SystemExit(f'{tag} {nm} {root}: methods {miss} missing; the campaign is incomplete')
            iv = (P.get('ivf') or {}).get('maint')
            for m, v in P.items(): v['x_ivf'] = (v['maint'] / iv) if (iv and v['maint'] and m != 'ivf') else None
            per[root.split('/')[-1]] = P
        res[nm] = per
        say(f'== {tag} {nm}: same-session streams ==')
        for pas, P in per.items():
            for m, v in sorted(P.items()):
                say(f"  {pas:16s} {m:18s} maint {v['maint'] if v['maint'] is None else round(v['maint'], 1)} s  x_ivf {v['x_ivf'] and round(v['x_ivf'], 1)}  " +
                    ' '.join(f"r{x['round']}:{x['max_recall']:.3f}/{(x['q0.95'] or 0):,.0f}" for x in v['rounds']))
    return res

# ------------------------------------------------------------------------------------------------ E5: executed rebuild policies
PERIODS = (1, 2, 3, 4, 6, 10, 20)       # fixed before the runs; an incomplete set is an error, not a smaller grid
def executed_policies(tag='r2b'):
    """E5 (tag r2b, the corrected runner). For each target stream and recall target, every executed rebuild period P:
    rounds below the target (violations) and, without violations, the executed total cost = rebuild and delta-
    preparation times after round 0 + Q * sum over rounds 1..R of per-query time. Two search-effort rules:
      oracle  per round, the cheapest L that reaches the target (uses each round's ground truth: an upper bound);
      fixed   one L for the whole stream, the smallest that reaches the target at round 0 of the CALIBRATION stream
              (fresh index, P=1 run), a deployable rule.
    The EOQ period uses the calibration stream's H6 measurements (B, g), rounded to the nearest executed period."""
    H6 = json.load(open('results/bench_h_summary.json'))['H6']; res = {}
    cal = {'sift500k_adv': 'sift500k', 'sift500k': 'sift500k_adv'}
    def load(nm):
        runs = {}
        for f in glob.glob(f'results/bench_{tag}/stream_{nm}/fdiskann_policy_P*.json'):
            d = json.load(open(f)); runs[d['P']] = d
        return runs
    for nm, src in cal.items():
        runs, cruns = load(nm), load(src)
        if not runs: continue
        missing = sorted(set(PERIODS) - set(runs))
        if missing:
            if os.environ.get('REVIEW_PARTIAL') == '1': say(f'PARTIAL: E5 {nm} skipped, periods {missing} missing'); continue
            raise SystemExit(f'E5 {nm}: periods {missing} missing from results/bench_{tag}; the campaign is incomplete')
        expected = set(range(stream_stats(nm)['R'] + 1))     # every period of both streams measured at every round, once
        for lab, group in ((nm, runs), (src, cruns)):
            if set(PERIODS) - set(group): raise SystemExit(f'E5 calibration stream {lab}: periods {sorted(set(PERIODS) - set(group))} missing')
            for P_, d in group.items():
                rr = [r['round'] for r in d['rounds'] if r.get('sweep')]
                if set(rr) != expected or len(rr) != len(d['rounds']) or len(rr) != len(expected):
                    raise SystemExit(f'E5 {lab} P={P_}: incomplete or duplicate rounds')
        U = stream_stats(nm)['updates_per_round']; B, g = H6[src]['rebuild_s'], H6[src]['growth_ms_per_round'] / 1e3
        out = {'calibrated_on': src, 'periods': list(PERIODS), 'targets': {}}
        # sensitivity: B and g re-estimated from the corrected runner on the calibration stream (B: mean rebuild of its
        # P=1 run; g: slope of its never-rebuilt run's per-query time at recall 0.95 against index age)
        cal_new = None
        if 1 in cruns and 20 in cruns:
            Bn = float(np.mean([r['rebuild_s'] for r in cruns[1]['rounds'] if r['rebuilt']]))
            ages, tq = [], []
            for r in cruns[20]['rounds']:
                q_ = qps_at(frontier([(x['recall'], x['qps']['mean']) for x in r['sweep']]), 0.95)
                if q_: ages.append(r['round']); tq.append(1 / q_)
            gn = float(np.polyfit(ages, tq, 1)[0]) if len(ages) > 2 else None
            cal_new = {'B': Bn, 'g': gn}
        out['calibration'] = {'h6': {'B': B, 'g': g}, 'r2b': cal_new}
        for t in (0.95, 0.90):
            Lfix = None
            if 1 in cruns:
                ok = [int(x['param'][2:]) for x in cruns[1]['rounds'][0]['sweep'] if x['recall'] >= t]
                Lfix = min(ok) if ok else None
            per = {}
            for P in PERIODS:
                d = runs[P]; qo, qf, rf = [], [], []; unmeasured = 0
                for r in d['rounds'][1:]:
                    fr = frontier([(x['recall'], x['qps']['mean']) for x in r['sweep']]); qo.append(qps_at(fr, t))
                    x = next((x for x in r['sweep'] if int(x['param'][2:]) == Lfix), None) if Lfix else None
                    if Lfix and x is None: unmeasured += 1; continue     # no fallback to another L
                    if x: qf.append(x['qps']['mean']); rf.append(x['recall'])
                maint = float(sum(r['rebuild_s'] + r.get('t_delta', 0.0) for r in d['rounds'][1:]))   # after the build, as the frontier
                per[P] = {'maint': maint, 'oracle': {'violations': sum(x is None for x in qo), 't_query_sum': float(sum(1 / x for x in qo if x))},
                          'fixed': ({'L': Lfix, 'violations': sum(x < t for x in rf) + unmeasured, 'unmeasured': unmeasured, 'min_recall': min(rf) if rf else None,
                                     't_query_sum': float(sum(1 / x for x in qf))} if Lfix else None)}
            rows = []
            for qpu in (0.1, 0.3, 1.0):
                Qr = qpu * U; Pstar = math.sqrt(2 * B / (Qr * g)); Pcal = min(PERIODS, key=lambda P: abs(math.log(P / Pstar)))
                row = {'q_per_update': qpu, 'P_star_calibrated': Pstar, 'P_chosen': Pcal}
                if cal_new and cal_new['g'] and cal_new['g'] > 0:
                    Pn = math.sqrt(2 * cal_new['B'] / (Qr * cal_new['g'])); row['P_star_recalibrated'] = Pn
                    row['P_chosen_recalibrated'] = min(PERIODS, key=lambda P: abs(math.log(P / Pn)))
                for rule in ('oracle', 'fixed'):
                    if any(v[rule] is None for v in per.values()): row[rule] = None; continue
                    feas = {P: v['maint'] + Qr * v[rule]['t_query_sum'] for P, v in per.items() if v[rule]['violations'] == 0}
                    best = min(feas, key=feas.get) if feas else None
                    row[rule] = {'chosen_violations': per[Pcal][rule]['violations'], 'best_feasible': best, 'feasible': sorted(feas),
                                 'regret': (feas[Pcal] / feas[best] - 1) if (best is not None and Pcal in feas) else None}
                rows.append(row)
            out['targets'][str(t)] = {'L_fixed': Lfix, 'per_P': per, 'grid': rows}
            say(f'== E5 {nm} @ recall {t}: EOQ from {src} (B={B:.0f} s, g={1e3 * g:.3f} ms/round); fixed L={Lfix}; violations oracle/fixed: ' +
                ', '.join(f"P={P}:{v['oracle']['violations']}/{v['fixed']['violations'] if v['fixed'] else '-'}" for P, v in per.items()))
            for r in rows:
                say(f"  Q/U={r['q_per_update']}: P*={r['P_star_calibrated']:.1f} -> P={r['P_chosen']}; " + '; '.join(
                    (f"{rule}: best feasible P={r[rule]['best_feasible']}, chosen violations {r[rule]['chosen_violations']}" +
                     (f", regret {r[rule]['regret']:.1%}" if r[rule]['regret'] is not None else '')) if r[rule] else f'{rule}: n/a (no calibration run)'
                    for rule in ('oracle', 'fixed')))
        res[nm] = out
    return res

# ------------------------------------------------------------------------------------------------ seed replicates
def seeds():
    """Independent replicates of the two decisive comparisons. (A) H1 at recall 0.95: IVF's crossover per n and its
    exponent for label/IVF-training seed 11 (tag h1, pass 1 only, so that each seed is one run), 12 and 13 (tags s12,
    s13). (B) DEEP-5M under uniform deletions: last-round maximum recall and QPS@0.95 of tombstone and all-level-repaired
    ACORN for the base stream (tag r1, each pass) and stream seeds 2 and 3 (tag rs); the paired difference per replicate."""
    res = {'A': {}, 'B': {}}
    for sd, roots in (('11', ['results/bench_h1']), ('12', ['results/bench_s12']), ('13', ['results/bench_s13'])):
        if not os.path.isdir(roots[0]): continue
        rows, fit = crossovers(summarize_static(load_roots(roots))[1], NS, '0.95', meths=('ivf',))
        res['A'][sd] = {'exponent': fit.get('ivf'), 'm_star': {ds: r['ivf']['m_star'] for ds, r in rows.items() if 'ivf' in r}}
    ex = [v['exponent'] for v in res['A'].values() if v['exponent'] is not None]
    if ex: res['A']['summary'] = {'n': len(ex), 'mean': float(np.mean(ex)), 'min': min(ex), 'max': max(ex), 'sd': float(np.std(ex, ddof=1)) if len(ex) > 1 else None}
    say('== seeds A: IVF crossover exponent at recall 0.95 per label/training seed ==')
    for sd, v in res['A'].items():
        if sd != 'summary': say(f"  seed {sd}: n^{v['exponent'] and round(v['exponent'], 3)}  m* " + ', '.join(f'{ds} {m:,.0f}' for ds, m in v['m_star'].items()))
    if 'summary' in res['A']: say(f"  exponent over {res['A']['summary']['n']} seeds: mean {res['A']['summary']['mean']:.3f}, range {res['A']['summary']['min']:.3f}-{res['A']['summary']['max']:.3f}")
    reps = [('base pass1', 'results/bench_r1', 'deep5m'), ('base pass2', 'results/bench_r1_pass2', 'deep5m'),
            ('seed2', 'results/bench_rs', 'deep5m_seed2'), ('seed3', 'results/bench_rs', 'deep5m_seed3')]
    for lab, root, nm in reps:
        P = {}
        for m in ('acorn', 'acorn_repair_all'):
            f = f'{root}/stream_{nm}/{m}.json'
            if not os.path.exists(f): continue
            d = json.load(open(f)); r = [x for x in d['rounds'] if 'sweep' in x][-1]
            fr = frontier([(x['recall'], x['qps']['mean']) for x in r['sweep']])
            P[m] = {'max_recall': max(x['recall'] for x in r['sweep']), 'q0.95': qps_at(fr, 0.95), 'maint': d['rounds'][-1].get('cum_maint_s')}
        if len(P) == 2: P['stream'] = nm; P['recall_diff'] = P['acorn_repair_all']['max_recall'] - P['acorn']['max_recall']; res['B'][lab] = P
    say('== seeds B: DEEP-5M last round, tombstone vs all-level repair ==')
    for lab, P in res['B'].items():
        qs = lambda q: f'{q:,.0f} QPS' if q else 'misses 0.95'
        say(f"  {lab:11s} tombstone {P['acorn']['max_recall']:.3f} ({qs(P['acorn']['q0.95'])})  repaired {P['acorn_repair_all']['max_recall']:.3f} ({qs(P['acorn_repair_all']['q0.95'])})  diff {P['recall_diff']:+.3f}")
    dd = [P['recall_diff'] for P in res['B'].values()]
    if dd: res['B_summary'] = {'n': len(dd), 'min': min(dd), 'max': max(dd), 'all_negative': all(x < 0 for x in dd)}
    return res

# ------------------------------------------------------------------------------------------------ idea 1: feasible cost frontier
def run_costs(d, R, t, include_build=False):
    """One stream run: total maintenance (rebuilds and delta preparation included) and per-query time at recall t for
    rounds 1..R, interpolated linearly between measured rounds; None if t is missed at any measured round.
    Completeness (CHECK.md B06): the run must hold every round 0..R, and its measured rounds must include round 0 and
    round R, so the interpolation never extrapolates; otherwise this raises. The one exception is the snapshot-rebuild
    Filtered-DiskANN run, which rebuilds only at the measured rounds: its fresh-every-round maintenance is estimated as
    the mean rebuild time x R (maintenance_source 'estimated') unless the run rebuilt at every round ('measured')."""
    rounds = [r['round'] for r in d['rounds']]; rows = [r for r in d['rounds'] if 'sweep' in r]; pts = {}
    sparse = d['method'] == 'fdiskann_rebuild' and d.get('maintenance_source', 'estimated') == 'estimated'
    if not sparse and rounds != list(range(R + 1)): raise ValueError(f"{d['method']}: rounds {rounds[:3]}..{rounds[-3:]} are not 0..{R}")
    if not rows or rows[0]['round'] != 0 or rows[-1]['round'] != R:
        raise ValueError(f"{d['method']}: measured rounds {[r['round'] for r in rows]} do not span 0..{R}")
    for r in rows:
        fr = frontier([(x['recall'], x['qps']['mean']) for x in r['sweep']]); q = qps_at(fr, t)
        if q is None: return None
        pts[r['round']] = 1 / q
    A = sorted(pts); tq = [float(np.interp(k, A, [pts[a] for a in A])) for k in range(1, R + 1)]
    last = d['rounds'][-1]; maint = last.get('cum_maint_s'); source = 'measured'
    if maint is None: maint = float(sum((r.get('rebuild_s') or 0) + (r.get('t_delta') or 0) for r in d['rounds'][1:]))
    if d['method'] == 'fdiskann_rebuild':
        if sparse: maint = float(np.mean([r['rebuild_s'] for r in d['rounds'] if r.get('rebuild_s')])) * R; source = 'estimated'
        else: maint = float(sum(r['rebuild_s'] for r in d['rounds'] if r['round'] >= 1))
    if include_build:      # the initial build (for the rebuild-every-round design, its round-0 build)
        b0 = d.get('build_s')
        if b0 is None: b0 = next((r['rebuild_s'] for r in d['rounds'] if r.get('rebuild_s')), 0.0)
        maint += float(b0)
    return {'maint': float(maint), 't_query': tq, 'maintenance_source': source, 'measured_rounds': A}

def cost_frontier(t=0.95, qpus=(0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100), include_build=False):
    """Total cost (maintenance after the initial build, plus Q x per-query time at recall t interpolated between measured
    rounds) of every method relative to IVF measured in the same session and pass. A method enters the frontier only if
    it reaches recall t at every measured round of EVERY run (pass) it has; otherwise it is listed with the number of
    failing runs, so that a failing pass cannot disappear behind a succeeding one. Executed rebuild policies are
    included with the oracle search effort and reported separately (they are not a deployable rule)."""
    sessions = [('main', ['results/bench', 'results/bench_pass2'], None), ('r1', ['results/bench_r1', 'results/bench_r1_pass2'], None),
                ('r4', ['results/bench_r4', 'results/bench_r4_pass2'], None), ('r2b', ['results/bench_r2b'], 'policy'),
                ('r3', ['results/bench_r3', 'results/bench_r3_pass2'], None), ('r6', ['results/bench_r6', 'results/bench_r6_pass2'], None)]
    res = {}
    for nm in ('sift500k', 'sift500k_adv', 'deep5m', 'arxiv_time'):
        st = stream_stats(nm); R, U = st['R'], st['updates_per_round']; rel = {}; runs_n = {}; fails = {}; skipped = []
        for sess, roots, kind in sessions:
            for root in roots:
                runs = {}
                for f in glob.glob(f'{root}/stream_{nm}/*.json'):
                    d = json.load(open(f))
                    if 'rounds' not in d: continue
                    have = {r['round'] for r in d['rounds'] if r.get('sweep')}; need = set(st['measure_rounds']) | {R}
                    if not need <= have: raise SystemExit(f'{f}: rounds {sorted(need - have)} missing (smoke-test or interrupted output?)')
                    if d['method'].startswith('fdiskann_policy') != (kind == 'policy') and d['method'] != 'ivf': continue
                    runs[d['method']] = d
                if 'ivf' not in runs:
                    if runs: skipped.append({'root': root, 'methods': sorted(runs), 'reason': 'no IVF anchor in this pass'}); say(f'SKIPPED {root}/stream_{nm}: no IVF anchor')
                    continue
                base = run_costs(runs['ivf'], R, t, include_build)
                if base is None: raise SystemExit(f'{root}/stream_{nm}: the IVF anchor misses recall {t}')
                for m, d in runs.items():
                    if m == 'ivf' and sess != 'main': continue
                    key = f'{m}@{sess}' if m == 'ivf' else m
                    runs_n[m] = runs_n.get(m, 0) + 1
                    c = run_costs(d, R, t, include_build)
                    if c is None: fails[m] = fails.get(m, 0) + 1; continue
                    for qpu in qpus:
                        Q = qpu * U; cj = c['maint'] + Q * sum(c['t_query']); ci_ = base['maint'] + Q * sum(base['t_query'])
                        rel.setdefault(m, {}).setdefault(qpu, []).append(cj / ci_)
        feasible = [m for m in rel if fails.get(m, 0) == 0]
        table = {m: {str(q): float(np.median(v)) for q, v in rel[m].items()} for m in feasible}
        spread = {m: {str(q): [float(min(v)), float(max(v))] for q, v in rel[m].items()} for m in feasible}
        best = {str(q): min((m for m in table if not m.startswith('fdiskann_policy')), key=lambda m: table[m][str(q)]) for q in qpus}
        res[nm] = {'relative_to_ivf': table, 'range_over_runs': spread, 'best': best, 'runs': runs_n,
                   'infeasible': {m: f'{fails[m]}/{runs_n[m]} runs miss {t}' for m in fails}, 'skipped_sessions': skipped}
        say(f'== cost frontier {nm} @ recall {t}{" incl. initial build" if include_build else ""} (cost / same-session IVF cost, median over runs; feasible in every run only) ==')
        for m in sorted(table, key=lambda m: table[m].get('1', 1e9)):
            say(f'  {m:22s} runs {runs_n[m]}  ' + ' '.join(f"{q}:{table[m][str(q)]:.2f}" for q in qpus if str(q) in table[m]))
        say(f"  best (excluding oracle policies): {best}; infeasible: {res[nm]['infeasible']}")
    return res

if __name__ == '__main__':
    OUT['E2_B1'] = exponents(); OUT['B4'] = threads(); OUT['D'] = policies(); OUT['E1'] = costs()
    if glob.glob('results/bench_r1*'): OUT['C3'] = same_session('r1', ('sift500k', 'sift500k_adv', 'deep5m'), ('ivf', 'ivf_part', 'acorn', 'acorn_repair_all'))
    if glob.glob('results/bench_r4*'): OUT['EMA'] = same_session('r4', ('sift500k', 'sift500k_adv', 'deep5m', 'arxiv_time'), ('ivf', 'ema'))
    if glob.glob('results/bench_r2b*'): OUT['E5'] = executed_policies()
    if glob.glob('results/bench_s1*') or glob.glob('results/bench_rs*'): OUT['SEEDS'] = seeds()
    OUT['FRONTIER'] = {str(t): cost_frontier(t) for t in (0.95, 0.99)}
    OUT['FRONTIER_BUILD'] = {'0.95': cost_frontier(0.95, include_build=True)}
    if glob.glob('results/bench_r3*'): OUT['D2'] = same_session('r3', ('arxiv_time',), ('ivf', 'ivf_rt5', 'ivf_part', 'scan_part', 'acorn', 'acorn_repair_all', 'postfilter', 'fdiskann_rebuild'))
    json.dump(OUT, open('results/bench_r_summary.json', 'w'), indent=1, default=float)

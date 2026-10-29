#!/usr/bin/env python3
"""Follow-up 3 (paper_audit/followup3_prereg.md): the crossover and the threshold rules on a second machine (AMD EPYC
7513), with the decision rules fixed before the runs.
  python3 bench/analyze_followup3.py > results/bench_f3_analysis.txt      -> results/bench_f3_summary.json
Inputs: results/bench_epyc[_pass2] (H1 grid, scan_part and ivf, seed-11 labels) and results/bench_mb_epyc/cost.json,
copied from the second machine; the Xeon's H1 summary and microbenchmark for the transferred rules.
Completeness and feasibility (CHECK.md B07; the rules' text is unchanged, the first version of this code did not enforce
it): every job of results/bench_epyc_log/epyc_jobs.txt must have its output in both passes, P1 must score 17 workloads
and the others 23; a rule's choice of a path that misses recall 0.95 is an infeasible choice with infinite regret, which
fails the "regret at most 10%" test. Infinite values are written as the string "inf" (the JSON has no NaN/Infinity)."""
import sys, os, json, math
sys.path.insert(0, 'bench')
import numpy as np
from analyze import summarize_static
from analyze_h import L_of, NS
from analyze_review import load_roots, crossovers

def say(*a): print(*a, flush=True)
roots = ['results/bench_epyc', 'results/bench_epyc_pass2']
jobs = [l.split() for l in open('results/bench_epyc_log/epyc_jobs.txt') if 'run_faiss.py' in l]
expected = sorted({(j[j.index('bench/run_faiss.py') + 1], j[j.index('bench/run_faiss.py') + 2], m) for j in jobs for m in j[j.index('bench/run_faiss.py') + 3].split(',')})
missing = [f'{r}/{ds}/{wl}/{m}.json' for r in roots for ds, wl, m in expected if not os.path.exists(f'{r}/{ds}/{wl}/{m}.json')]
if len(expected) != 46 or missing: raise SystemExit(f'incomplete: {len(expected)} expected (workload, method) pairs, missing {missing[:5]}')
S = summarize_static(load_roots(roots))[1]
rows, fit = crossovers(S, NS, '0.95', meths=('ivf',))
m_star = {ds: r['ivf']['m_star'] for ds, r in rows.items() if 'ivf' in r}; cens = {ds: r['ivf']['censor'] for ds, r in rows.items() if 'ivf' in r}
H = json.load(open('results/bench_h_summary.json')); m_x = H['H1']['per_n']['0.95']['deep1m']['ivf']['m_star']
mb_x = json.load(open('results/bench_mb/cost.json'))['96']['chk_over_dist']; mb_e = json.load(open('results/bench_mb_epyc/cost.json'))['96']['chk_over_dist']
T = []
for ds, n in NS.items():
    for wl in [w for w in S if w.startswith(ds + '/b0.0_s')]:
        sc, iv = S[wl].get('scan_part', {}).get('qps@0.95'), S[wl].get('ivf', {}).get('qps@0.95')
        if sc: T.append({'wl': wl, 'n': n, 'm': n / L_of(wl.split('/')[1]), 'c_scan': 1 / sc, 'c_ivf': (1 / iv) if iv else math.inf})
m_e = m_star.get('deep1m')
rules = {'P1 EPYC DEEP-1M, sqrt(n)': lambda w: m_e * math.sqrt(w['n'] / 1e6),
         'P2 Xeon DEEP-1M, sqrt(n), microbenchmark corrected': lambda w: m_x * math.sqrt(w['n'] / 1e6) * math.sqrt(mb_e / mb_x),
         'P3 Xeon DEEP-1M, sqrt(n), uncorrected': lambda w: m_x * math.sqrt(w['n'] / 1e6),
         'P4 fixed selectivity 7%': lambda w: 0.07 * w['n'],
         'P5 Xeon fixed size': lambda w: m_x}
res = {'machine': 'AMD EPYC 7513', 'm_star': m_star, 'censor': cens, 'ivf_exponent': fit.get('ivf'), 'xeon_m_star_deep1m': m_x,
       'chk_over_dist': {'xeon': mb_x, 'epyc': mb_e}, 'workloads': len(T), 'rules': {}}
INF = 'inf'
def score_rule(ws, f):
    """wrong choices, infeasible choices (the chosen path misses recall 0.95: infinite cost), and regret of rule f."""
    reg, wrong, infeasible = [], 0, 0
    for w in ws:
        ch = w['c_scan'] if w['m'] < f(w) else w['c_ivf']; best = min(w['c_scan'], w['c_ivf'])
        if not math.isfinite(best): raise SystemExit(f"{w['wl']}: neither path reaches recall 0.95")
        reg.append(ch / best - 1); wrong += ch > best * (1 + 1e-9); infeasible += not math.isfinite(ch)
    fin = [x for x in reg if math.isfinite(x)]
    return {'workloads': len(ws), 'wrong': wrong, 'infeasible_choices': infeasible,
            'regret_max': (INF if infeasible else float(max(fin))) if reg else None, 'regret_max_finite': float(max(fin)) if fin else None,
            'regret_mean': (INF if infeasible else float(np.mean(fin))) if reg else None}
def transfers(r):
    """the pre-registered size-transfer test: at most 1 wrong of the scored workloads, regret at most 10%, none infeasible"""
    return bool(r['workloads'] > 0 and r['infeasible_choices'] == 0 and r['wrong'] <= 1 and r['regret_max'] is not None and r['regret_max'] <= 0.10)
for name, f in rules.items():
    ws = [w for w in T if not (name.startswith('P1') and w['n'] == 1e6)]      # P1 is calibrated on DEEP-1M: score the others
    res['rules'][name] = score_rule(ws, f)
ws_n = {k: v['workloads'] for k, v in res['rules'].items()}
if len(T) != 23 or ws_n['P1 EPYC DEEP-1M, sqrt(n)'] != 17 or any(v != 23 for k, v in ws_n.items() if not k.startswith('P1')):
    raise SystemExit(f'incomplete: {len(T)} workloads with a scan result, rule denominators {ws_n}')
ms = [m_star[d] for d in NS if d in m_star]
P1, P2, P3 = (res['rules'][k] for k in ('P1 EPYC DEEP-1M, sqrt(n)', 'P2 Xeon DEEP-1M, sqrt(n), microbenchmark corrected', 'P3 Xeon DEEP-1M, sqrt(n), uncorrected'))
res['decisions'] = {
    'crossover_grows_on_both': bool(len(ms) == 4 and all(b > a for a, b in zip(ms, ms[1:])) and (fit.get('ivf') or 0) > 0
                                    and all(cens.get(d) == 'interp' for d in NS)),       # a censored crossover has no measured m*
    'sqrtn_transfers_across_sizes': transfers(P1),
    'microbenchmark_correction_transfers': bool(P2['infeasible_choices'] == 0 and P2['wrong'] <= P3['wrong'] and P2['wrong'] <= 2)}
say(f"== follow-up 3 on {res['machine']}: m* {({d: round(v) for d, v in m_star.items()})}, IVF exponent {fit.get('ivf')}")
say(f"   c_chk/c_dist at d=96: Xeon {mb_x:.4f}, EPYC {mb_e:.4f}; Xeon DEEP-1M m* {m_x:.0f}, EPYC {m_e}")
for k, v in res['rules'].items(): say(f'   {k:55s} wrong {v["wrong"]}/{v["workloads"]}, infeasible {v["infeasible_choices"]}, regret max {v["regret_max"]}')
say(f"   decisions: {res['decisions']}")
json.dump(res, open('results/bench_f3_summary.json', 'w'), indent=1, default=float, allow_nan=False)

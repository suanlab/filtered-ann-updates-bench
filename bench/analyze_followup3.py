#!/usr/bin/env python3
"""Follow-up 3 (paper_audit/followup3_prereg.md): the crossover and the threshold rules on a second machine (AMD EPYC
7513), with the decision rules fixed before the runs.
  python3 bench/analyze_followup3.py > results/bench_f3_analysis.txt      -> results/bench_f3_summary.json
Inputs: results/bench_epyc[_pass2] (H1 grid, scan_part and ivf, seed-11 labels) and results/bench_mb_epyc/cost.json,
copied from the second machine; the Xeon's H1 summary and microbenchmark for the transferred rules."""
import sys, os, json, math
sys.path.insert(0, 'bench')
import numpy as np
from analyze import summarize_static
from analyze_h import L_of, NS
from analyze_review import load_roots, crossovers

def say(*a): print(*a, flush=True)
roots = [r for r in ('results/bench_epyc', 'results/bench_epyc_pass2') if os.path.isdir(r)]
if len(roots) < 2: raise SystemExit('both EPYC passes are needed (results/bench_epyc, results/bench_epyc_pass2)')
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
for name, f in rules.items():
    ws = [w for w in T if not (name.startswith('P1') and w['n'] == 1e6)]      # P1 is calibrated on DEEP-1M: score the others
    reg, wrong = [], 0
    for w in ws:
        ch = w['c_scan'] if w['m'] < f(w) else w['c_ivf']; best = min(w['c_scan'], w['c_ivf']); reg.append(ch / best - 1); wrong += ch > best * (1 + 1e-9)
    fin = [x for x in reg if math.isfinite(x)]
    res['rules'][name] = {'workloads': len(ws), 'wrong': wrong, 'regret_max': float(max(fin)) if fin else None, 'regret_mean': float(np.mean(fin)) if fin else None}
ms = [m_star[d] for d in NS if d in m_star]
res['decisions'] = {
    'crossover_grows_on_both': bool(len(ms) == 4 and all(b > a for a, b in zip(ms, ms[1:])) and (fit.get('ivf') or 0) > 0),
    'sqrtn_transfers_across_sizes': bool(res['rules']['P1 EPYC DEEP-1M, sqrt(n)']['wrong'] <= 1 and (res['rules']['P1 EPYC DEEP-1M, sqrt(n)']['regret_max'] or 0) <= 0.10),
    'microbenchmark_correction_transfers': bool(res['rules']['P2 Xeon DEEP-1M, sqrt(n), microbenchmark corrected']['wrong'] <= res['rules']['P3 Xeon DEEP-1M, sqrt(n), uncorrected']['wrong']
                                                and res['rules']['P2 Xeon DEEP-1M, sqrt(n), microbenchmark corrected']['wrong'] <= 2)}
say(f"== follow-up 3 on {res['machine']}: m* {({d: round(v) for d, v in m_star.items()})}, IVF exponent {fit.get('ivf')}")
say(f"   c_chk/c_dist at d=96: Xeon {mb_x:.4f}, EPYC {mb_e:.4f}; Xeon DEEP-1M m* {m_x:.0f}, EPYC {m_e}")
for k, v in res['rules'].items(): say(f'   {k:55s} wrong {v["wrong"]}/{v["workloads"]}, regret max {v["regret_max"]}')
say(f"   decisions: {res['decisions']}")
json.dump(res, open('results/bench_f3_summary.json', 'w'), indent=1, default=float)

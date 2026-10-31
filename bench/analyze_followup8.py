#!/usr/bin/env python3
"""Follow-up 8 (paper_audit/followup8_prereg.md): the deletion lesson on the AMD EPYC 7513.
  python3 bench/analyze_followup8.py > results/bench_f8_analysis.txt      -> results/bench_f8_summary.json
Inputs: results/bench_epyc_str and results/bench_epyc_str_pass2 (copied from the second machine)."""
import sys, os, json
sys.path.insert(0, 'bench')
import numpy as np
from analyze import frontier, qps_at

ROOTS = ('results/bench_epyc_str', 'results/bench_epyc_str_pass2')
def say(*a): print(*a, flush=True)
def run(root, nm, m):
    f = f'{root}/stream_{nm}/{m}.json'
    if not os.path.exists(f): raise SystemExit(f'missing {f}')
    return json.load(open(f))
def maxr(d):
    return {r['round']: max(s['recall'] for s in r['sweep']) for r in d['rounds'] if 'sweep' in r}
def q95(d):
    return {r['round']: qps_at(frontier([(s['recall'], s['qps']['mean']) for s in r['sweep']]), 0.95) for r in d['rounds'] if 'sweep' in r}
def maint(d): return d['rounds'][-1].get('cum_maint_s')

res = {'passes': {}}
for root in ROOTS:
    A = {m: run(root, 'sift500k_adv', m) for m in ('acorn', 'acorn_repair_all', 'acorn_repair_fill', 'ivf', 'ivf_part')}
    D = {m: run(root, 'deep5m', m) for m in ('acorn', 'acorn_repair_all', 'ivf')}
    ma = {m: maxr(d) for m, d in A.items()}; md = {m: maxr(d) for m, d in D.items()}
    for x in list(ma.values()) + list(md.values()):
        if not x: raise SystemExit(f'{root}: a run has no measured round')
    t0, t20, a20, f20 = ma['acorn'][0], ma['acorn'][20], ma['acorn_repair_all'][20], ma['acorn_repair_fill'][20]
    qi, qp = q95(A['ivf']), q95(A['ivf_part'])
    out = {'E1': bool(t20 <= t0 - 0.2), 'E2': bool(a20 >= 0.95 and a20 >= t20 + 0.2), 'E3': bool(f20 >= t20 + 0.5 * (a20 - t20)),
           'E4': bool(md['acorn_repair_all'][10] < md['acorn'][10]),
           'E5': bool(all(qp.get(r) and qi.get(r) and qp[r] > qi[r] for r in sorted(qi))),
           'sift_adv_max_recall': ma, 'deep5m_max_recall': md, 'ivf_q095': qi, 'ivf_part_q095': qp,
           'maintenance_over_ivf': {**{f'sift_adv/{m}': maint(d) / maint(A['ivf']) for m, d in A.items()},
                                    **{f'deep5m/{m}': maint(d) / maint(D['ivf']) for m, d in D.items()}}}
    res['passes'][root] = out
    say(f"== {root}: tombstones {t0:.3f}->{t20:.3f}, all-level {a20:.3f}, slot-fill {f20:.3f}; DEEP-5M r10 tombstones {md['acorn'][10]:.3f}, all-level {md['acorn_repair_all'][10]:.3f}")
    say(f"   E1 {out['E1']} E2 {out['E2']} E3 {out['E3']} E4 {out['E4']} E5 {out['E5']}; label-part/IVF QPS {[round(qp[r] / qi[r], 2) for r in sorted(qi) if qp.get(r) and qi.get(r)]}")
P = list(res['passes'].values())
res['holds'] = {e: all(p[e] for p in P) for e in ('E1', 'E2', 'E3', 'E4', 'E5')}; res['n_hold'] = sum(res['holds'].values())
say(f"== holds in both passes: {res['holds']} ({res['n_hold']} of 5)")
json.dump(res, open('results/bench_f8_summary.json', 'w'), indent=1, default=float, allow_nan=False)

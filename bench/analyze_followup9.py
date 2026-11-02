#!/usr/bin/env python3
"""Follow-up 9 (paper_audit/followup9_prereg.md): the cost lesson on the AMD EPYC 7513.
  python3 bench/analyze_followup9.py > results/bench_f9_analysis.txt      -> results/bench_f9_summary.json
Inputs: results/bench_epyc_cost[_pass2] (methods) and results/bench_epyc_cost_end[_pass2] (second IVF anchor)."""
import sys, os, json, math
sys.path.insert(0, 'bench')
from analyze_review import run_costs
from common import stream_stats

T = 0.95; QPUS = (0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100)
METHODS = ('ivf_part', 'acorn', 'acorn_repair_all', 'fdiskann_rebuild')
GRAPHS = {'acorn', 'acorn_repair_all', 'fdiskann_rebuild'}
def say(*a): print(*a, flush=True)
def load(root, nm, m):
    f = f'{root}/stream_{nm}/{m}.json'
    if not os.path.exists(f): raise SystemExit(f'missing {f}')
    return json.load(open(f))
def total(c, q, U): return c['maint'] + q * U * sum(c['t_query'])

res = {'passes': {}}
for sfx in ('', '_pass2'):
    P = {}
    for nm in ('sift500k', 'arxiv_time'):
        st = stream_stats(nm); R, U = st['R'], st['updates_per_round']
        a1, a2 = run_costs(load(f'results/bench_epyc_cost{sfx}', nm, 'ivf'), R, T), run_costs(load(f'results/bench_epyc_cost_end{sfx}', nm, 'ivf'), R, T)
        if a1 is None or a2 is None: raise SystemExit(f'{nm}{sfx}: an IVF anchor misses {T}')
        anc = {str(q): math.sqrt(total(a1, q, U) * total(a2, q, U)) for q in QPUS}
        rel = {'ivf': {str(q): total(a1, q, U) / anc[str(q)] for q in QPUS}}; infeasible = []
        for m in METHODS:
            c = run_costs(load(f'results/bench_epyc_cost{sfx}', nm, m), R, T)
            if c is None: infeasible.append(m); continue
            rel[m] = {str(q): total(c, q, U) / anc[str(q)] for q in QPUS}
        best = {str(q): min(rel, key=lambda m: rel[m][str(q)]) for q in QPUS}
        P[nm] = {'relative_to_ivf_anchors': rel, 'best': best, 'infeasible': infeasible, 'anchor_drift': {str(q): total(a2, q, U) / total(a1, q, U) for q in QPUS}}
        say(f"== {nm}{sfx or ' (pass 1)'}: best {best}; infeasible {infeasible}; drift at 1: {P[nm]['anchor_drift']['1']:.3f}")
    b_s, b_a = P['sift500k']['best'], P['arxiv_time']['best']
    P['C1'] = all(b_s[str(q)] == 'ivf_part' for q in QPUS)
    P['C2'] = all(b_a[str(q)] in GRAPHS for q in QPUS if q >= 1)
    P['C3'] = all(b_a[str(q)] == 'ivf_part' for q in QPUS if q <= 0.1)
    res['passes'][sfx or 'pass1'] = P
    say(f"   C1 {P['C1']} C2 {P['C2']} C3 {P['C3']}")
res['holds'] = {c: all(p[c] for p in res['passes'].values()) for c in ('C1', 'C2', 'C3')}; res['n_hold'] = sum(res['holds'].values())
say(f"== holds in both passes: {res['holds']} ({res['n_hold']} of 3)")
json.dump(res, open('results/bench_f9_summary.json', 'w'), indent=1, default=float, allow_nan=False)

#!/usr/bin/env python3
"""Follow-up 5 (paper_audit/followup5_prereg.md): executed trace of every round, and memory over the stream.
  python3 bench/analyze_followup5.py > results/bench_f5_analysis.txt      -> results/bench_f5_summary.json
Inputs: results/bench_tr (MEASURE_ALL=1 runs, first IVF anchor) and results/bench_tr_end (second IVF anchor).
Policies for the per-query cost at recall 0.95: (1) oracle at every round; (2) oracle at the paper's measured rounds,
interpolated; (3) one constant effort, the smallest grid value with recall >= 0.95 at every round; (4) the effort
chosen at round 0, kept (infeasible if it misses 0.95 at a later round)."""
import sys, os, json, math, copy
sys.path.insert(0, 'bench')
import numpy as np
from analyze_review import run_costs
from common import stream_stats

T = 0.95; QPUS = (0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100)
PAPER_ROUNDS = {'sift500k_adv': [0, 4, 8, 12, 16, 20], 'arxiv_time': [0, 5, 10]}
METHODS = {'sift500k_adv': ['ivf_part', 'acorn', 'acorn_repair_all', 'ema', 'fdiskann_rebuild'],
           'arxiv_time': ['ivf_part', 'ivf_rt5', 'acorn', 'acorn_repair_all', 'ema', 'fdiskann_rebuild']}
def say(*a): print(*a, flush=True)
def load(root, nm, m):
    f = f'{root}/stream_{nm}/{m}.json'
    if not os.path.exists(f): raise SystemExit(f'missing {f}')
    return json.load(open(f))

def grid(d):
    """the parameter grid in sweep order, and per round {param: (recall, qps)}; every round must sweep the same grid"""
    rows = {r['round']: {s['param']: (s['recall'], s['qps']['mean']) for s in r['sweep']} for r in d['rounds']}
    order = [s['param'] for s in d['rounds'][0]['sweep']]
    for r, g in rows.items():
        if list(g) != order: raise SystemExit(f"{d['method']}: round {r} sweeps {list(g)}, round 0 {order}")
    return order, rows

def policies(d, R):
    """{policy: cost dict for total(), or None if infeasible}, plus the round-0 policy's first failure"""
    out = {'oracle_all': run_costs(d, R, T)}
    sp = copy.deepcopy(d)
    for r in sp['rounds']:
        if r['round'] not in PAPER_ROUNDS[NM]: r.pop('sweep', None)
    out['oracle_paper_rounds'] = run_costs(sp, R, T)
    maint = out['oracle_all']['maint'] if out['oracle_all'] else run_costs(d, R, 0.0)['maint']
    order, rows = grid(d); rr = sorted(rows)
    def fixed(p): return {'maint': maint, 't_query': [1 / rows[r][p][1] for r in rr if r >= 1], 'param': p}
    const = next((p for p in order if all(rows[r][p][0] >= T for r in rr)), None)
    out['constant'] = fixed(const) if const else None
    p0 = next((p for p in order if rows[0][p][0] >= T), None); fail = None
    if p0:
        bad = [r for r in rr if rows[r][p0][0] < T]
        if bad: fail = {'param': p0, 'round': bad[0], 'recall': rows[bad[0]][p0][0]}
    out['round0'] = fixed(p0) if p0 and not fail else None
    return out, fail

def total(c, q, U): return c['maint'] + q * U * sum(c['t_query'])

res = {}
for NM in ('sift500k_adv', 'arxiv_time'):
    st = stream_stats(NM); R, U = st['R'], st['updates_per_round']
    a1, a2 = load('results/bench_tr', NM, 'ivf'), load('results/bench_tr_end', NM, 'ivf')
    P = {}; fails = {}; mem = {}; extra = {}; misses = {}
    for m in METHODS[NM] + ['ivf']:
        d = load('results/bench_tr', NM, m)
        if [r['round'] for r in d['rounds']] != list(range(R + 1)) or not all('sweep' in r for r in d['rounds']):
            raise SystemExit(f'{NM}/{m}: not every round was measured')
        P[m], f = policies(d, R)
        miss = [(r['round'], max(x['recall'] for x in r['sweep'])) for r in d['rounds'] if max(x['recall'] for x in r['sweep']) < T]
        if miss: misses[m] = {'rounds': miss, 'paper_rounds_only': [x for x in miss if x[0] in PAPER_ROUNDS[NM]]}
        if f: fails[m] = f
        r0, rl = d['rounds'][0], d['rounds'][-1]
        if 'rss_anon_mb' in r0 and 'rss_anon_mb' in rl:
            mem[m] = {'rss_anon_mb_r0': r0['rss_anon_mb'], 'rss_anon_mb_last': rl['rss_anon_mb'], 'ratio': rl['rss_anon_mb'] / r0['rss_anon_mb'],
                      'resident_r0': r0.get('resident'), 'resident_last': rl.get('resident')}
        if m == 'ema': extra['ema_configure_share'] = sum(r.get('t_configure', 0) for r in d['rounds']) / d['rounds'][-1]['cum_maint_s']
    Pa2, _ = policies(a2, R)
    out = {'policies': {}, 'drift': {}, 'round0_failures': fails, 'memory': mem, 'max_recall_below_target': misses, **extra}   # every round where even the largest effort misses 0.95
    for pol in ('oracle_all', 'oracle_paper_rounds', 'constant', 'round0'):
        anc = [P['ivf'][pol], Pa2[pol]]
        if any(a is None for a in anc): out['policies'][pol] = {'anchor_infeasible': True}; continue
        rel = {}
        for m in METHODS[NM] + ['ivf']:
            c = P[m][pol]
            if c is None: continue
            rel[m] = {str(q): total(c, q, U) / math.sqrt(total(anc[0], q, U) * total(anc[1], q, U)) for q in QPUS}
        best = {str(q): min(rel, key=lambda m: rel[m][str(q)]) for q in QPUS}
        out['policies'][pol] = {'relative_to_ivf': rel, 'best': best, 'params': {m: P[m][pol].get('param') for m in rel if P[m][pol] is not None}}
        out['drift'][pol] = {str(q): total(anc[1], q, U) / total(anc[0], q, U) for q in QPUS}
    qc = {m: (sum(P[m]['oracle_all']['t_query']) / sum(P[m]['oracle_paper_rounds']['t_query']) - 1)
          for m in P if P[m]['oracle_all'] and P[m]['oracle_paper_rounds']}
    out['interp_query_cost_change'] = qc
    b1, b2 = out['policies']['oracle_all'].get('best', {}), out['policies']['oracle_paper_rounds'].get('best', {})
    out['D1_cells_changed'] = [q for q in b1 if b1[q] != b2.get(q)]
    K = json.load(open('results/bench_r_summary.json'))['FRONTIER']['0.95'][NM]['relative_to_ivf']
    sub = {m: v for m, v in K.items() if m in METHODS[NM] + ['ivf']}
    paper_best = {str(q): min(sub, key=lambda m: sub[m][str(q)]) for q in QPUS if all(str(q) in v for v in sub.values())}
    bc = out['policies']['constant'].get('best', {})
    out['paper_best_same_methods'] = paper_best; out['D2_cells_changed'] = [q for q in paper_best if bc.get(q) != paper_best[q]]
    res[NM] = out
    say(f'== {NM}: winners oracle/every round {b1}; constant effort {bc}; paper (same methods) {paper_best}')
    say(f'   D1 cells changed by interpolation {out["D1_cells_changed"]}; query-cost change {({m: round(100 * v, 1) for m, v in qc.items()})}%')
    say(f'   D2 cells where the constant effort changes the paper winner {out["D2_cells_changed"]}; round-0 effort fails {fails}')
    say(f'   rounds where no effort reaches 0.95: {misses}')
    say(f'   memory {({m: round(v["ratio"], 2) for m, v in mem.items()})}; IVF drift (oracle) {round(out["drift"]["oracle_all"]["1"], 3)}; {extra}')
json.dump(res, open('results/bench_f5_summary.json', 'w'), indent=1, default=float, allow_nan=False)

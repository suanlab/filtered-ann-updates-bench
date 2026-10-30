#!/usr/bin/env python3
"""Follow-up 6 (paper_audit/followup6_prereg.md): the ACORN update paths on three SIFT-500K-adv stream seeds, every
round measured with the mechanism instrumentation.
  python3 bench/analyze_followup6.py > results/bench_f6_analysis.txt      -> results/bench_f6_summary.json
Input: results/bench_rpx/stream_<stream>/<path>.json (bench/stream_acorn.py, MECH=1 MEASURE_ALL=1)."""
import sys, os, json
sys.path.insert(0, 'bench')
import numpy as np
from common import spearman

T = 0.95; R = 20
STREAMS = ('sift500k_adv', 'sift500k_adv_seed2', 'sift500k_adv_seed3')
PATHS = ('acorn', 'acorn_repair', 'acorn_repair_all', 'acorn_repair_upper', 'acorn_repair_fill')
def say(*a): print(*a, flush=True)

def per_round(d):
    rr = [r['round'] for r in d['rounds']]
    if rr != list(range(R + 1)) or not all('sweep' in r and 'mech' in r for r in d['rounds']):
        raise SystemExit(f"{d['method']}: not every round has a sweep and the mechanism metrics")
    maxr = [max(s['recall'] for s in r['sweep']) for r in d['rounds']]
    reach = [float(np.mean([l['reach_frac'] for l in r['mech']['labels']])) for r in d['rounds']]
    return maxr, reach

res = {'streams': {}}
for nm in STREAMS:
    D = {}
    for m in PATHS:
        f = f'results/bench_rpx/stream_{nm}/{m}.json'
        if not os.path.exists(f): raise SystemExit(f'missing {f}')
        D[m] = json.load(open(f))
    P = {m: per_round(d) for m, d in D.items()}
    last = {m: d['rounds'][-1] for m, d in D.items()}
    mx = {m: P[m][0][R] for m in PATHS}; rc = {m: P[m][1][R] for m in PATHS}
    tomb = last['acorn']; labs = sorted(int(l['label']) for l in tomb['mech']['labels'])
    best = max(tomb['sweep'], key=lambda s: s['recall'])['recall_by_label']
    reach_l = {int(l['label']): l['reach_frac'] for l in tomb['mech']['labels']}
    rho, p = spearman([reach_l[l] for l in labs], [best[str(l)] for l in labs])
    allm, allr = P['acorn_repair_all']
    dips = [(r, allm[r]) for r in range(R + 1) if allm[r] < T]
    expl = [{'round': r, 'max_recall': v, 'reach': allr[r], 'reach_median_other_rounds': float(np.median([allr[q] for q in range(R + 1) if q != r]))} for r, v in dips]
    maint = {m: last[m]['cum_maint_s'] / last['acorn']['cum_maint_s'] for m in PATHS}
    fresh = D['acorn'].get('fresh', {}).get('sweep'); fresh_max = max(s['recall'] for s in fresh) if fresh else None
    out = {'max_recall_r20': mx, 'mean_reach_r20': rc, 'max_recall_by_round': {m: P[m][0] for m in PATHS}, 'mean_reach_by_round': {m: P[m][1] for m in PATHS},
           'H1_dips': dips, 'H2': bool(mx['acorn_repair_all'] >= T and mx['acorn_repair_all'] >= mx['acorn'] + 0.2),
           'H3a': bool(rc['acorn_repair_all'] > rc['acorn_repair_upper']), 'H3b_rho': rho, 'H3b_p': p, 'H3b': bool(rho is not None and rho > 0.5),
           'H4': bool(mx['acorn_repair_upper'] <= mx['acorn_repair_all'] - 0.1),
           'H5': bool(mx['acorn_repair_fill'] >= mx['acorn'] + 0.5 * (mx['acorn_repair_all'] - mx['acorn'])),
           'exploratory_dips_vs_reach': expl, 'maintenance_over_tombstones_r20': maint, 'fresh_max_recall_r20': fresh_max}
    res['streams'][nm] = out
    say(f"== {nm}: r20 max recall {({m: round(v, 3) for m, v in mx.items()})}; mean reach {({m: round(v, 3) for m, v in rc.items()})}")
    say(f"   all-level rounds below {T}: {dips}; H2 {out['H2']} H3a {out['H3a']} H3b rho {rho} ({out['H3b']}) H4 {out['H4']} H5 {out['H5']}; fresh {fresh_max}")
    say(f"   maintenance over tombstones {({m: round(v, 2) for m, v in maint.items()})}; dips vs reach {expl}")
S = res['streams']; allmin = [v for x in S.values() for (_, v) in x['H1_dips']]
res['H1'] = {'pairs_below': sum(len(x['H1_dips']) for x in S.values()), 'pairs': 3 * (R + 1), 'streams_with_dip': sum(bool(x['H1_dips']) for x in S.values()),
             'min': min(allmin) if allmin else None, 'holds_every_round': not allmin}
for h in ('H2', 'H3a', 'H3b', 'H4', 'H5'): res[h] = sum(S[nm][h] for nm in STREAMS)
res['all_level_r20_range'] = [min(x['max_recall_r20']['acorn_repair_all'] for x in S.values()), max(x['max_recall_r20']['acorn_repair_all'] for x in S.values())]
say(f"== H1 {res['H1']}; H2 {res['H2']}/3, H3a {res['H3a']}/3, H3b {res['H3b']}/3, H4 {res['H4']}/3, H5 {res['H5']}/3; all-level r20 {res['all_level_r20_range']}")
json.dump(res, open('results/bench_f6_summary.json', 'w'), indent=1, default=float, allow_nan=False)

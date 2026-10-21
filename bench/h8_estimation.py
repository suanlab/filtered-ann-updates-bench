#!/usr/bin/env python3
"""H8: the YFCC dispatcher with estimated instead of exact admissible-set sizes (analysis only, main env).
  python3 bench/h8_estimation.py      -> results/bench_h8/estimation.json
Estimators of |A_p| for a query's tag conjunction: exact; independence (n * prod_t c_t/n, exact for one tag);
uniform point samples of 1% and 0.1% (the count among sampled points, scaled). For each estimator the dispatcher
routes queries whose estimate is below T to the exact scan, with (T, parameter) chosen as before (oracle on the
reported queries) and, separately, with the T and parameter tuned on exact sizes kept fixed. Uses the measured
per-query times and recalls of the final results; nothing is re-measured."""
import sys, os, json
sys.path.insert(0, 'bench')
import numpy as np
from analyze import load_static, summarize_static, dispatcher, TARGETS

W = load_static(); raw, _ = summarize_static(W)
z = np.load('data/bench/yfcc10m/tags/pred.npz'); n = len(z['ptag_ptr']) - 1
cnt = np.bincount(z['ptag_idx'], minlength=int(z['ptag_idx'].max()) + 1)
qoff, qids = z['qoff'], z['qids']; qp, qi = z['qtag_ptr'], z['qtag_idx']; nq = len(qoff) - 1
true = np.diff(qoff).astype(float)
indep = np.array([n * np.prod(cnt[qi[qp[q]:qp[q + 1]]] / n) for q in range(nq)])
def sample_est(frac, seed=0):
    keep = np.random.default_rng(seed).random(n) < frac
    return np.array([keep[qids[qoff[q]:qoff[q + 1]]].sum() / frac for q in range(nq)])
EST = {'exact': true, 'independence': indep, 'sample_1pct': sample_est(0.01), 'sample_0.1pct': sample_est(0.001)}
two = np.diff(qp) > 1
out = {'estimate_over_true_and_queries': {}, 'methods': {}, 'and_size_max': int(true[two].max()), 'and_size_median': float(np.median(true[two])), 'n_and': int(two.sum())}
for k, e in EST.items():
    r = (e[two] + 1) / (true[two] + 1)
    out['estimate_over_true_and_queries'][k] = {'median': float(np.median(r)), 'p10': float(np.quantile(r, 0.1)), 'p90': float(np.quantile(r, 0.9)),
                                                'frac_under_half': float((r < 0.5).mean()), 'frac_over_double': float((r > 2).mean())}
ms = raw['yfcc10m/tags']; ex = dispatcher('yfcc10m/tags', ms, true)
scan = [r for r in ms['scan_part'] if 'tq' in r][0]; ts, rs = np.array(scan['tq']), np.array(scan['rq'])
for m, d in ex.items():
    b = d['best'].get('0.99')
    if not b: continue
    row = {'exact': {'qps': b['qps'], 'recall': b['recall'], 'T': b['T']}}
    rec = next(r for r in ms[m] if r['param'] == b['param']); tm, rm = np.array(rec['tq']), np.array(rec['rq'])
    for k, e in EST.items():
        if k == 'exact': continue
        H = dispatcher('yfcc10m/tags', {'scan_part': ms['scan_part'], m: ms[m]}, e)[m]['best'].get('0.99')
        small = e < b['T']    # exact-tuned threshold and parameter, routed by the estimate
        fixed = {'recall': float(np.where(small, rs, rm).mean()), 'qps': nq / float(np.where(small, ts, tm).sum()),
                 'misrouted': int((small != (true < b['T'])).sum())}
        row[k] = {'retuned': None if H is None else {'qps': H['qps'], 'recall': H['recall'], 'T': H['T'], 'regret': 1 - H['qps'] / b['qps']},
                  'fixed_T': {**fixed, 'qps_change': fixed['qps'] / b['qps'] - 1}}
    out['methods'][m] = row
os.makedirs('results/bench_h8', exist_ok=True); json.dump(out, open('results/bench_h8/estimation.json', 'w'), indent=1)
print(json.dumps(out['estimate_over_true_and_queries'], indent=1))
for m, row in out['methods'].items():
    print(m, 'exact', round(row['exact']['qps']), {k: (None if v['retuned'] is None else (round(v['retuned']['qps']), round(v['retuned']['regret'], 3)),
                                                       round(v['fixed_T']['recall'], 4), round(v['fixed_T']['qps']), v['fixed_T']['misrouted'])
                                                   for k, v in row.items() if k != 'exact'})

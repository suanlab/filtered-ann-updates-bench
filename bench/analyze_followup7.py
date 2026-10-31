#!/usr/bin/env python3
"""Follow-up 7 (paper_audit/followup7_prereg.md): close index-versus-scan verdicts re-measured in four paired blocks.
  python3 bench/analyze_followup7.py > results/bench_f7_analysis.txt      -> results/bench_f7_summary.json
Inputs: results/bench_g6b1..4 (per block) and the stored best-of-two results (results/bench[_pass2], results/bench_h1[_pass2])."""
import sys, math, json
sys.path.insert(0, 'bench')
import numpy as np
from analyze import summarize_static
from analyze_review import load_roots

T = 0.95; T975_DF3 = 3.182446305284263
CANDS = [('sift1m/b0.9_s0.5', 'acorn1', 'ab'), ('sift1m/b0.9_s0.5', 'curator', 'b'), ('sift1m/b0.9_s0.01', 'acorn', 'b'), ('sift1m/b0.9_s0.01', 'acorn_g100', 'b'),
         ('sift1m/b0.0_s0.01', 'acorn1', 'b'), ('deep1m/b0.9_s0.1', 'postfilter', 'b'), ('deep10m/b0.9_s0.01', 'curator', 'a'), ('deep10m/b0.9_s0.01', 'ivf', 'a'),
         ('deep10m/b0.0_s0.01', 'rig', 'a'), ('deep10m/b0.0_s0.01', 'acorn1', 'b'), ('deep3m/b0.9_s0.03125', 'acorn', 'a'), ('deep3m/b0.9_s0.03125', 'curator', 'a'),
         ('deep300k/b0.0_s0.03125', 'ivf', 'a'), ('deep300k/b0.0_s0.1', 'curator', 'a'), ('deep300k/b0.0_s0.1', 'postfilter', 'a'), ('deep1m/b0.0_s0.03125', 'curator', 'a'),
         ('deep1m/b0.0_s0.333333', 'curator', 'b'), ('deep10m/b0.0_s0.03125', 'curator', 'b'), ('deep3m/b0.0_s0.1', 'curator', 'b')]
def say(*a): print(*a, flush=True)
def q(S, wl, m):
    v = S.get(wl, {}).get(m)
    if v is None: raise SystemExit(f'missing {wl} {m}')
    return v.get(f'qps@{T}')

# the stored (paper's) values come from the campaign in which the comparison was made: #1-12 main, #13-19 the n-sweep
STORED = {'main': summarize_static(load_roots(['results/bench', 'results/bench_pass2']))[1],
          'h1': summarize_static(load_roots(['results/bench_h1', 'results/bench_h1_pass2']))[1]}
blocks = [summarize_static(load_roots([f'results/bench_g6b{b}']))[1] for b in (1, 2, 3, 4)]
rows = []
for i, (wl, m, crit) in enumerate(CANDS):
    stored = STORED['main' if i < 12 else 'h1']
    qs0, qm0 = q(stored, wl, 'scan_part'), q(stored, wl, m)
    st = 'infeasible' if qm0 is None else ('index faster' if qm0 > qs0 else 'scan faster')
    L = []
    for S in blocks:
        qs, qm = q(S, wl, 'scan_part'), q(S, wl, m)
        if qs is None: raise SystemExit(f'{wl}: the scan misses {T} in a block')
        L.append(None if qm is None else math.log(qm / qs))
    feas = sum(x is not None for x in L)
    r = {'workload': wl, 'method': m, 'criterion': crit, 'stored_ratio': (qm0 / qs0) if qm0 else None, 'stored_verdict': st,
         'block_log_ratios': L, 'feasible_blocks': feas}
    if feas == 4:
        a = np.array(L); mu = float(a.mean()); h = T975_DF3 * float(a.std(ddof=1)) / 2.0
        r.update(mean_ratio=math.exp(mu), ci95=[math.exp(mu - h), math.exp(mu + h)],
                 verdict='index faster' if mu - h > 0 else 'scan faster' if mu + h < 0 else 'tie')
    else:
        r['verdict'] = 'infeasible' if feas == 0 else 'unstable feasibility'
    if crit.startswith('a') or crit == 'ab':
        r['D1'] = 'confirmed' if r['verdict'] == st else ('tie' if r['verdict'] in ('tie', 'unstable feasibility') else 'reversed')
    if 'b' in crit:
        r['D2_holds'] = (feas == 4 and st != 'infeasible') or (feas == 0 and st == 'infeasible')
    rows.append(r)
    say(f"{wl:26s} {m:11s} [{crit}] stored {st} ({r['stored_ratio'] and round(r['stored_ratio'], 3)}); blocks {[None if x is None else round(math.exp(x), 3) for x in L]} "
        f"-> {r['verdict']} {('mean %.3f CI %.3f-%.3f' % (r['mean_ratio'], *r['ci95'])) if 'mean_ratio' in r else ''} {r.get('D1', '')} {r.get('D2_holds', '')}")
# the threshold policies (analyze_review.policies, rules fixed on DEEP-1M) are not re-scored; report their picks on the
# n-sweep workloads whose IVF-versus-scan verdict is now a tie or reversed (pre-registered)
from analyze_h import L_of, NS
m0 = json.load(open('results/bench_r_summary.json'))['D']['calibration']['m_star_deep1m']
POL = {'fixed |A| (DEEP-1M)': lambda n: m0, 'fixed s (DEEP-1M)': lambda n: m0 / 1e6 * n, 'fixed s = 7% (Milvus)': lambda n: 0.07 * n,
       'sqrt(n)-scaled (DEEP-1M)': lambda n: m0 * math.sqrt(n / 1e6)}
picks = []
for i, r in enumerate(rows):
    if i < 12 or r['method'] != 'ivf' or r.get('D1') not in ('tie', 'reversed'): continue
    ds, sc = r['workload'].split('/'); n = NS[ds]; m = n / L_of(sc)
    for name, f in POL.items():
        ch = 'scan' if m < f(n) else 'ivf'; old = 'ivf' if r['stored_verdict'] == 'index faster' else 'scan'
        picks.append({'workload': r['workload'], 'policy': name, 'pick': ch, 'stored_right': ch == old, 'now': r['verdict']})
    say(f"   policy picks on {r['workload']} (now {r['verdict']}): {[(p['policy'], p['pick'], 'right' if p['stored_right'] else 'wrong') for p in picks if p['workload'] == r['workload']]}")
d1 = [r for r in rows if 'D1' in r]; d2 = [r for r in rows if 'D2_holds' in r]
res = {'rows': rows, 'policy_picks_on_ties': picks,
       'D1': {k: sum(r['D1'] == k for r in d1) for k in ('confirmed', 'tie', 'reversed')} | {'n': len(d1)},
       'D2': {'holds': sum(r['D2_holds'] for r in d2), 'n': len(d2), 'not_holding': [(r['workload'], r['method'], r['feasible_blocks']) for r in d2 if not r['D2_holds']]}}
say(f"== D1 close verdicts: {res['D1']}; D2 edge feasibility: {res['D2']}")
json.dump(res, open('results/bench_f7_summary.json', 'w'), indent=1, default=float, allow_nan=False)

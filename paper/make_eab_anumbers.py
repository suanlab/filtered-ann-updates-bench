#!/usr/bin/env python3
"""Generate paper/eab_anumbers.tex (and results/bench_a_numbers.json): derived numbers that the paper audit of
2026-10-21 added -- per-run ranges of the H3 streams, the same-session maintenance split of the repair runs (h3b),
pgvector-vs-library slowdowns, same-run predicate-cost ratios (H5), the EOQ hit count (H6) and the independence
estimator's underestimate (H8). Reads results/ only. Run from the project root after bench/analyze_h.py."""
import sys, json, glob, os
sys.path.insert(0, 'bench')
import numpy as np
from analyze import load_static, summarize_static
out = {}
# H3: per-run maximum recall at round 20 and maintenance, SIFT-500K-adv (h3 pass 1, h3 pass 2, h3b)
runs = {}
for f in sorted(glob.glob('results/bench_h3*/stream_sift500k_adv/acorn*.json') + glob.glob('results/bench_r1*/stream_sift500k_adv/acorn*.json')):
    d = json.load(open(f)); R = d['rounds']; last = R[-1]
    runs.setdefault(d['method'], []).append({'tag': f.split('/')[1], 'maxr_r20': max(s['recall'] for s in last['sweep']),
        'maint': last['cum_maint_s'], 'repair_s': sum(r.get('t_repair', 0) or 0 for r in R), 'insert_s': sum(r.get('t_insert', 0) for r in R)})
h3_runs = dict(runs)     # `runs` is reused below
for m, rs in runs.items():
    v = [r['maxr_r20'] for r in rs]; out[f'a/h3/{m}/r20_min'] = f'{min(v):.3f}'; out[f'a/h3/{m}/r20_max'] = f'{max(v):.3f}'; out[f'a/h3/{m}/runs'] = str(len(v))
# h3b: same-session triples (tombstones, level-0 repair, all-level repair), one per pass; ratios within a pass, then min-max
def tot(d, k): return sum(r.get(k, 0) or 0 for r in d['rounds'])
trip = {}
for root in ['results/bench_h3b'] + sorted(glob.glob('results/bench_h3b_pass*')):
    h3b = {d['method']: d for d in (json.load(open(f)) for f in glob.glob(f'{root}/stream_sift500k_adv/acorn*.json'))}
    if len(h3b) < 3: continue
    T = h3b['acorn']['rounds'][-1]['cum_maint_s']
    for m in ('acorn_repair', 'acorn_repair_all'):
        trip.setdefault(f'{m}/repair_s', []).append(tot(h3b[m], 't_repair')); trip.setdefault(f'{m}/insert_s', []).append(tot(h3b[m], 't_insert'))
        trip.setdefault(f'{m}/maint_ratio', []).append(h3b[m]['rounds'][-1]['cum_maint_s'] / T)
    trip.setdefault('acorn/insert_s', []).append(tot(h3b['acorn'], 't_insert'))
for k, v in trip.items():
    f_ = '{:.2f}' if k.endswith('ratio') else '{:.0f}'
    out[f'a/h3b/{k}'] = f_.format(v[0]); out[f'a/h3b/{k}_min'] = f_.format(min(v)); out[f'a/h3b/{k}_max'] = f_.format(max(v))
out['a/h3b/runs'] = str(len(trip['acorn/insert_s']))
# pgvector against the library baselines (different sessions: orders of magnitude only)
_, P = summarize_static(load_static('pg')); _, H = summarize_static(load_static('h1'))
wl = 'deep1m/b0.0_s0.1'
out['a/pg/exact_slowdown'] = f"{H[wl]['scan_part']['qps@0.95'] / P[wl]['pgvector_exact']['qps@0.95']:.0f}"
out['a/pg/ivfflat_slowdown'] = f"{H[wl]['ivf']['qps@0.95'] / P[wl]['pgvector_ivfflat']['qps@0.95']:.0f}"
# H5 within the same run
s5 = json.load(open('results/bench_h5/predicate_cost.json'))['summary']
out['a/h5/numpy_ms'] = f"{s5['numpy']['mean_ms']:.2f}"; out['a/h5/roaring_factor'] = f"{s5['numpy']['mean_ms'] / s5['roaring_and']['mean_ms']:.0f}"
out['a/h5/roaring_ids_factor'] = f"{s5['numpy']['mean_ms'] / s5['roaring_ids']['mean_ms']:.0f}"
# H6: interior best periods within 30% of the EOQ prediction
H6 = json.load(open('results/bench_h_summary.json'))['H6']; ok = n = 0
for nm in ('sift500k', 'sift500k_adv', 'deep5m'):
    for r in H6[nm]['grid']:
        if 1 < r['best_P'] < max(int(a) for a in H6[nm]['ages']) + 1:
            n += 1; ok += abs(r['best_P'] / r['P_star_eoq'] - 1) <= 0.3
out['a/h6/eoq_within30'] = str(ok); out['a/h6/eoq_interior'] = str(n)
# Figure 1 / introduction: how wrong a fixed dispatcher threshold is across n (IVF, recall 0.95, beta=0)
HS = json.load(open('results/bench_h_summary.json')); P1 = HS['H1']['per_n']['0.95']
m7, m3 = P1['deep10m']['ivf']['m_star'], P1['deep300k']['ivf']['m_star']
out['a/fig1/ivf_mstar_n3e5'] = f'{m3:,.0f}'; out['a/fig1/ivf_mstar_n1e7'] = f'{m7:,.0f}'
out['a/fig1/A1e4_low_at_1e7'] = f'{m7 / 1e4:.1f}'                  # |A|=1e4 threshold, right near n=3.16e5, too low by this at 1e7
out['a/fig1/s001_high_at_1e7'] = f'{1e-2 * 1e7 / m7:.1f}'          # s=1e-2 threshold at n=1e7 vs the crossover there
ax_ = HS['AX']['H1']['per_n']['0.95']['arxiv1m']['ivf']['m_star']
out['a/fig1/A1e4_high_arxiv1m'] = f'{1e4 / ax_:.1f}'                # the same |A|=1e4 threshold on arXiv-1M
# review robustness analyses (bench/analyze_review.py -> results/bench_r_summary.json)
if os.path.exists('results/bench_r_summary.json'):
    RV = json.load(open('results/bench_r_summary.json')); E = RV['E2_B1']
    for tag in ('h1', 'ax'):
        for meth, (lo, hi) in E[tag].get('pass_range', {}).items(): out[f'a/r/{tag}/{meth}/pass_lo'] = f'{lo:.2f}'; out[f'a/r/{tag}/{meth}/pass_hi'] = f'{hi:.2f}'
        for meth, v in E[tag]['scan_per_query'].items(): out[f'a/r/{tag}/{meth}/pq'] = f'{v:.2f}'
    B4 = RV['B4']
    for k in ('scan_x_min', 'scan_x_max', 'ivf_x_min', 'ivf_x_max', 'rel_large_min', 'rel_large_max', 'rel_small_max'): out[f'a/r/mt/{k}'] = f'{B4[k]:.1f}'
    for name, key in (('fixed |A| (DEEP-1M)', 'fixA'), ('fixed s (DEEP-1M)', 'fixS'), ('fixed s = 7% (Milvus)', 'milvus'),
                      ('sqrt(n)-scaled (DEEP-1M)', 'sqrtn'), ('sqrt(n)-scaled, cost-ratio corrected', 'sqrtnc'),
                      ('sqrt(n)-scaled, microbenchmark corrected', 'sqrtnmb')):
        if name not in RV['D']['policies']: continue
        for grp, g in (('DEEP n!=1e6', 'deep'), ('arXiv', 'ax'), ('DEEP unseen seeds', 'unseen')):
            if grp not in RV['D']['policies'][name]: continue
            P = RV['D']['policies'][name][grp]
            out[f'a/r/pol/{key}/{g}/wrong'] = str(P['wrong']); out[f'a/r/pol/{key}/{g}/n'] = str(P['workloads'])
            out[f'a/r/pol/{key}/{g}/mean'] = f"{100 * P['regret_mean']:.0f}"; out[f'a/r/pol/{key}/{g}/max'] = f"{100 * P['regret_max']:.0f}"
    C = RV['E1']
    out['a/r/cost/all_dominated'] = 'yes' if C['all_dominated'] else 'no'
    xa = [v for m, v in C['x_ivf_range'].items() if m in ('acorn', 'fdiskann_rebuild')]
    out['a/r/cost/graph_x_ivf_lo'] = f"{min(v[0] for v in xa):,.0f}"; out['a/r/cost/graph_x_ivf_hi'] = f"{max(v[1] for v in xa):,.0f}"
    for nm in ('sift500k', 'sift500k_adv', 'deep5m'):
        for t in ('0.95', '0.99'):
            bs = [C[nm][p]['fd_breakeven'].get(t) for p in C[nm]]
            lo = [b['min'] for b in bs if b and b['min'] is not None]; hi = [b['max'] for b in bs if b and b['max'] is not None]
            if lo: out[f'a/r/cost/{nm}/fd{t}'] = f'{min(lo):.1f}--{max(hi):.1f}'
            out[f'a/r/cost/{nm}/nobreak{t}'] = str(sum(b['nobreak'] for b in bs if b)); out[f'a/r/cost/{nm}/n{t}'] = str(sum(b['n'] for b in bs if b))
# C3 (tag r1): label-partitioned IVF and all-level-repaired ACORN next to IVF and tombstone ACORN, same session
if os.path.exists('results/bench_r_summary.json') and 'C3' in json.load(open('results/bench_r_summary.json')):
    C3 = json.load(open('results/bench_r_summary.json'))['C3']; last = {'sift500k': 20, 'sift500k_adv': 20, 'deep5m': 10}
    for nm, per in C3.items():
        for m in ('ivf_part', 'acorn_repair_all', 'acorn', 'ivf'):
            runs = [P[m] for P in per.values() if m in P]
            if not runs: continue
            lr = [next(x for x in r['rounds'] if x['round'] == last[nm]) for r in runs]
            out[f'a/r1/{nm}/{m}/maxr'] = f"{max(x['max_recall'] for x in lr):.3f}"
            q = [x['q0.95'] for x in lr if x['q0.95']]; out[f'a/r1/{nm}/{m}/q0.95'] = f'{max(q):,.0f}' if q else 'n/r'
            out[f'a/r1/{nm}/{m}/maint'] = f"{min(r['maint'] for r in runs):,.1f}" if min(r['maint'] for r in runs) < 100 else f"{min(r['maint'] for r in runs):,.0f}"
            xs = [r['x_ivf'] for r in runs if r.get('x_ivf')]
            if xs: out[f'a/r1/{nm}/{m}/x_ivf_lo'] = f'{min(xs):.1f}' if min(xs) < 10 else f'{min(xs):,.0f}'; out[f'a/r1/{nm}/{m}/x_ivf_hi'] = f'{max(xs):.1f}' if max(xs) < 10 else f'{max(xs):,.0f}'
    # label-partitioned IVF against IVF in the same pass, every measured round: QPS ratio range and minimum recall
    rat, rec, xm = [], [], []
    for nm, per in C3.items():
        for P in per.values():
            if 'ivf_part' in P and 'ivf' in P:
                qi = {x['round']: x['q0.95'] for x in P['ivf']['rounds']}
                rat += [x['q0.95'] / qi[x['round']] for x in P['ivf_part']['rounds'] if x['q0.95'] and qi.get(x['round'])]
                rec += [x['max_recall'] for x in P['ivf_part']['rounds']]; xm.append(P['ivf_part']['x_ivf'])
    out['a/r1/ivfpart/qratio_lo'] = f'{min(rat):.1f}'; out['a/r1/ivfpart/qratio_hi'] = f'{max(rat):.1f}'
    out['a/r1/ivfpart/minrec'] = f'{min(rec):.3f}'; out['a/r1/ivfpart/x_lo'] = f'{min(xm):.1f}'; out['a/r1/ivfpart/x_hi'] = f'{max(xm):.1f}'
# EMA (tag r4) and the arXiv temporal stream (tag r3): last round, best of two passes; ratios to IVF within each pass
if os.path.exists('results/bench_r_summary.json'):
    RV2 = json.load(open('results/bench_r_summary.json'))
    def lastq(runs, key):
        return [r['rounds'][-1][key] for r in runs if r['rounds'] and r['rounds'][-1][key] is not None]
    for grp, tag in (('EMA', 'ema'), ('D2', 'd2')):
        for nm, per in (RV2.get(grp) or {}).items():
            methods = set().union(*[set(P) for P in per.values()])
            for m in methods:
                runs = [P[m] for P in per.values() if m in P]
                q0 = [r['rounds'][0]['q0.95'] for r in runs if r['rounds'][0]['q0.95']]; ql = lastq(runs, 'q0.95')
                out[f'a/{tag}/{nm}/{m}/q_first'] = f'{max(q0):,.0f}' if q0 else 'n/r'
                out[f'a/{tag}/{nm}/{m}/q_last'] = f'{max(ql):,.0f}' if ql else 'n/r'
                out[f'a/{tag}/{nm}/{m}/maxr_min'] = f"{min(min(x['max_recall'] for x in r['rounds']) for r in runs):.3f}"
                out[f'a/{tag}/{nm}/{m}/maxr_last'] = f"{max(r['rounds'][-1]['max_recall'] for r in runs):.3f}"
                mm = min(r['maint'] for r in runs); out[f'a/{tag}/{nm}/{m}/maint'] = f'{mm:,.1f}' if mm < 100 else f'{mm:,.0f}'
                xs = [r['x_ivf'] for r in runs if r.get('x_ivf')]
                if xs: out[f'a/{tag}/{nm}/{m}/x_ivf'] = f'{np.median(xs):,.0f}' if np.median(xs) >= 10 else f'{np.median(xs):.1f}'
                dr = [r['rounds'][0]['q0.95'] / r['rounds'][-1]['q0.95'] for r in runs if r['rounds'][0]['q0.95'] and r['rounds'][-1]['q0.95']]
                if dr: out[f'a/{tag}/{nm}/{m}/drop'] = f'{np.median(dr):.1f}'     # median of per-pass ratios
            # per-pass QPS ratio of a method to IVF in the last round
            for m in methods - {'ivf'}:
                rr = [P[m]['rounds'][-1]['q0.95'] / P['ivf']['rounds'][-1]['q0.95'] for P in per.values()
                      if m in P and 'ivf' in P and P[m]['rounds'][-1]['q0.95'] and P['ivf']['rounds'][-1]['q0.95']]
                if rr: out[f'a/{tag}/{nm}/{m}/q_over_ivf'] = f'{min(rr):.1f}--{max(rr):.1f}' if max(rr) - min(rr) >= 0.1 else f'{np.median(rr):.1f}'
# cost frontier (FRONTIER) and executed rebuild policies (E5)
DISP = {'ivf': 'IVF', 'ivf_part': 'label-partitioned IVF', 'ivf_rt5': 'retrained IVF', 'scan_part': 'exact scan', 'acorn': 'ACORN',
        'acorn_repair_all': 'repaired ACORN', 'postfilter': 'post-filtering', 'curator': 'Curator', 'rig': 'RIG', 'ema': 'EMA',
        'fdiskann_rebuild': 'rebuilt Filtered-DiskANN'}
if os.path.exists('results/bench_r_summary.json'):
    RV3 = json.load(open('results/bench_r_summary.json'))
    for t, FR in (RV3.get('FRONTIER') or {}).items():
        tt = t.replace('.', '')
        for nm, F in FR.items():
            T = F['relative_to_ivf']
            for q, m in F['best'].items():
                out[f'a/fr{tt}/{nm}/best/{q}'] = DISP.get(m, m); out[f'a/fr{tt}/{nm}/bestval/{q}'] = f'{T[m][q]:.2f}'
                ru = sorted((v[q], o) for o, v in T.items() if o != m and q in v and not o.startswith('fdiskann_policy'))
                if ru: out[f'a/fr{tt}/{nm}/runnerup/{q}'] = DISP.get(ru[0][1], ru[0][1]); out[f'a/fr{tt}/{nm}/runnerup_gap/{q}'] = f'{100 * (ru[0][0] / T[m][q] - 1):.0f}'
            for m, v in T.items():
                for q in ('0.1', '1', '10', '100'):
                    if q in v: out[f'a/fr{tt}/{nm}/{m}/q{q}'] = f'{v[q]:.2f}'
            gg = [int(out[f'a/fr{tt}/{nm}/runnerup_gap/{q}']) for q in F['best'] if 1 <= float(q) <= 100 and f'a/fr{tt}/{nm}/runnerup_gap/{q}' in out]
            if gg: out[f'a/fr{tt}/{nm}/runnerup_gap_lo'] = str(min(gg)); out[f'a/fr{tt}/{nm}/runnerup_gap_hi'] = str(max(gg))
            fd = T.get('fdiskann_rebuild')
            if fd:
                cr = [float(q) for q in sorted(fd, key=float) if fd[q] < 1]
                out[f'a/fr{tt}/{nm}/fd_cross'] = (f'{min(cr):g}' if cr else 'none')
            out[f'a/fr{tt}/{nm}/infeasible'] = ', '.join(DISP.get(m, m) for m in sorted(F['infeasible'])) or 'none'
        syn = [v for nm in ('sift500k', 'sift500k_adv', 'deep5m') if nm in FR and 'ivf_part' in FR[nm]['relative_to_ivf'] for v in FR[nm]['relative_to_ivf']['ivf_part'].values()]
        if syn: out[f'a/fr{tt}/syn/ivfpart_lo'] = f'{min(syn):.2f}'; out[f'a/fr{tt}/syn/ivfpart_hi'] = f'{max(syn):.2f}'
    FB = (RV3.get('FRONTIER_BUILD') or {}).get('0.95'); F0 = (RV3.get('FRONTIER') or {}).get('0.95')
    if FB and F0:
        cells = [(nm, q) for nm in F0 for q in F0[nm]['best']]; ch = [(nm, q) for nm, q in cells if FB[nm]['best'].get(q) != F0[nm]['best'][q]]
        out['a/frb/cells'] = str(len(cells)); out['a/frb/changed'] = str(len(ch))
        out['a/frb/changed_list'] = '; '.join(f"{nm} at {q}: {DISP.get(F0[nm]['best'][q])} $\\to$ {DISP.get(FB[nm]['best'][q])}" for nm, q in ch) or 'none'
    for nm, E in (RV3.get('E5') or {}).items():
        for t, T in E['targets'].items():
            tt = t.replace('.', ''); per = T['per_P']
            out[f'a/e5/{nm}/{tt}/L_fixed'] = str(T['L_fixed']) if T['L_fixed'] else 'n/a'
            out[f'a/e5/{nm}/{tt}/oracle_viol'] = str(sum(v['oracle']['violations'] for v in per.values()))
            if all(v['fixed'] for v in per.values()):
                out[f'a/e5/{nm}/{tt}/fixed_viol_periods'] = str(sum(v['fixed']['violations'] > 0 for v in per.values()))
                out[f'a/e5/{nm}/{tt}/fixed_viol_max'] = str(max(v['fixed']['violations'] for v in per.values()))
                out[f'a/e5/{nm}/{tt}/fixed_minrec'] = f"{min(v['fixed']['min_recall'] for v in per.values() if v['fixed']['min_recall'] is not None):.3f}"
            for r in T['grid']:
                q = f"{r['q_per_update']:g}"; out[f'a/e5/{nm}/{tt}/P_chosen/{q}'] = str(r['P_chosen'])
                for rule in ('oracle', 'fixed'):
                    x = r.get(rule)
                    if x and x['regret'] is not None: out[f'a/e5/{nm}/{tt}/{rule}_regret/{q}'] = f"{100 * x['regret']:.1f}"
                    if x: out[f'a/e5/{nm}/{tt}/{rule}_best/{q}'] = str(x['best_feasible'])
                if r.get('P_chosen_recalibrated'): out[f'a/e5/{nm}/{tt}/P_recal/{q}'] = str(r['P_chosen_recalibrated'])
            regs = [r['oracle']['regret'] for r in T['grid'] if r.get('oracle') and r['oracle']['regret'] is not None]
            if regs: out[f'a/e5/{nm}/{tt}/oracle_regret_max'] = f'{100 * max(regs):.1f}'
            fr_ = [r['fixed']['regret'] for r in T['grid'] if r.get('fixed') and r['fixed']['regret'] is not None]
            if fr_ and len(fr_) == len(T['grid']): out[f'a/e5/{nm}/{tt}/fixed_regret_max'] = f'{100 * max(fr_):.1f}'
    E5all = RV3.get('E5') or {}
    if E5all:
        rg = [x['oracle']['regret'] for E in E5all.values() for x in E['targets']['0.95']['grid'] if x.get('oracle') and x['oracle']['regret'] is not None]
        if rg: out['a/e5/oracle_regret_max_both'] = f'{100 * max(rg):.1f}'
        ch = [(x['P_chosen'], x.get('P_chosen_recalibrated')) for E in E5all.values() for x in E['targets']['0.95']['grid']]
        out['a/e5/recal_changed'] = str(sum(1 for a_, b_ in ch if b_ is not None and a_ != b_)); out['a/e5/recal_cases'] = str(len(ch))
# seed replicates (SEEDS) and repair factors (tag r5)
if os.path.exists('results/bench_r_summary.json'):
    RV4 = json.load(open('results/bench_r_summary.json')); SD = RV4.get('SEEDS') or {}
    A_ = (SD.get('A') or {}).get('summary')
    if A_: out['a/seed/ivf_exp_lo'] = f"{A_['min']:.2f}"; out['a/seed/ivf_exp_hi'] = f"{A_['max']:.2f}"; out['a/seed/ivf_exp_mean'] = f"{A_['mean']:.2f}"; out['a/seed/n'] = str(A_['n'])
    B_ = SD.get('B') or {}
    if B_:
        out['a/seed/rep_streams'] = str(len({v['stream'] for v in B_.values()}))
        dd = [v['recall_diff'] for v in B_.values()]; out['a/seed/rep_diff_lo'] = f'{min(dd):+.3f}'; out['a/seed/rep_diff_hi'] = f'{max(dd):+.3f}'; out['a/seed/rep_n'] = str(len(dd))
        out['a/seed/rep_tomb_lo'] = f"{min(v['acorn']['max_recall'] for v in B_.values()):.3f}"; out['a/seed/rep_tomb_hi'] = f"{max(v['acorn']['max_recall'] for v in B_.values()):.3f}"
        out['a/seed/rep_all_lo'] = f"{min(v['acorn_repair_all']['max_recall'] for v in B_.values()):.3f}"; out['a/seed/rep_all_hi'] = f"{max(v['acorn_repair_all']['max_recall'] for v in B_.values()):.3f}"
from analyze import frontier as _fr, qps_at as _qa
R5 = {}
for f in glob.glob('results/bench_r5*/stream_*/*.json'):
    d = json.load(open(f)); last = [r for r in d['rounds'] if 'sweep' in r][-1]; nm = f.split('/')[-2][len('stream_'):]
    q = _qa(_fr([(x['recall'], x['qps']['mean']) for x in last['sweep']]), 0.95)
    R5.setdefault((nm, d['method']), []).append((max(x['recall'] for x in last['sweep']), q))
for (nm, m), v in R5.items():
    rs = [x[0] for x in v]; qs = [x[1] for x in v if x[1]]
    out[f'a/r5/{nm}/{m}/maxr'] = f'{min(rs):.3f}' if max(rs) - min(rs) < 0.0005 else f'{min(rs):.3f}--{max(rs):.3f}'
    out[f'a/r5/{nm}/{m}/q'] = (f'{min(qs):,.0f}--{max(qs):,.0f}' if len(qs) > 1 else f'{qs[0]:,.0f}') if qs else 'n/r'
fb = [max(x['recall'] for x in json.load(open(f))) for f in glob.glob('results/bench_r5*/deep5m_r10/live/acorn.json')]
if fb: out['a/r5/deep5m/fresh_maxr'] = f'{max(fb):.3f}'
H8 = json.load(open('results/bench_h8/estimation.json'))
out['a/h8/indep_under_factor'] = f"{1 / H8['estimate_over_true_and_queries']['independence']['median']:.0f}"
with open('paper/eab_anumbers.tex', 'w') as fo:
    fo.write('% generated by paper/make_eab_anumbers.py from results/ -- do not edit\n')
    for k, v in sorted(out.items()): fo.write('\\expandafter\\def\\csname v@' + k + '\\endcsname{' + v + '}\n')
json.dump({'h3_runs': h3_runs, **out}, open('results/bench_a_numbers.json', 'w'), indent=1)
print(json.dumps(out, indent=1))

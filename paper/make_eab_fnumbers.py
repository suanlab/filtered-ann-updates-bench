#!/usr/bin/env python3
"""Generate paper/eab_fnumbers.tex (macros \\V{f/...}) from results/bench_f2_summary.json (bench/analyze_followup2.py,
follow-up 2 of paper_audit/followup2_prereg.md). F2_PARTIAL=1 reads the partial summary (drafting only)."""
import os, json, math
import numpy as np

src = 'results/bench_f2_summary_partial.json' if os.environ.get('F2_PARTIAL') == '1' else 'results/bench_f2_summary.json'
F = json.load(open(src)); out = {}
pct = lambda x: f'{100 * x:.0f}'; f2 = lambda x: f'{x:.2f}'; f3 = lambda x: f'{x:.3f}'
# I13: seeds, policy on unseen seeds, timing intervals, best-of-two, held-out tuning, repair seeds, E5 replicate
S = (F.get('I13_seeds') or {}).get('summary')
if S:
    out.update({'f/seed/n': str(S['seeds']), 'f/seed/mean': f2(S['mean']), 'f/seed/lo': f2(S['min']), 'f/seed/hi': f2(S['max']),
                'f/seed/tci_lo': f2(S['t_ci95'][0]), 'f/seed/tci_hi': f2(S['t_ci95'][1])})
    if S.get('two_stage_ci95'): out['f/seed/boot_lo'] = f2(S['two_stage_ci95'][0]); out['f/seed/boot_hi'] = f2(S['two_stage_ci95'][1])
P = F.get('I13_policy')
if P:
    out['f/pol/n'] = str(P['workloads']); out['f/pol/seeds'] = str(len(P['seeds']))
    for k, key in (('sqrt(n)', 'sqrt'), ('fixed |A|', 'fixA'), ('fixed s', 'fixs'), ('Milvus 7%', 'milvus')):
        out[f'f/pol/{key}/wrong'] = str(P[k]['wrong'])
        if P[k]['regret_max_finite'] is not None: out[f'f/pol/{key}/max'] = pct(P[k]['regret_max_finite'])
T = F.get('I13_timing')
if T: out.update({'f/tci/median': f"{T['t_median_pct']:.1f}", 'f/tci/p90': f"{T['t_p90_pct']:.0f}", 'f/tci/factor': f"{T['factor_t_over_normal']:.1f}"})
B = F.get('I13_best_of_two')
if B:
    v = B['main']['verdicts'] + B['h1']['verdicts']; out['f/bo2/verdicts'] = str(v)
    out['f/bo2/mean_changed'] = str(B['main']['changed_vs_best']['mean'] + B['h1']['changed_vs_best']['mean'])
    out['f/bo2/worse_changed'] = str(B['main']['changed_vs_best']['worse'] + B['h1']['changed_vs_best']['worse'])
    e = B['h1']['ivf_exponent']; out['f/bo2/exp_best'] = f2(e['best']); out['f/bo2/exp_mean'] = f2(e['mean']); out['f/bo2/exp_worse'] = f2(e['worse'])
H = F.get('I13_heldout')
if H:
    out['f/ho/workloads'] = str(H['workloads']); out['f/ho/lat_changes'] = str(H['latency_changes'])
    out['f/ho/infeasible'] = str(H['feasibility_losses']); out['f/ho/dec_changes'] = str(H['decision_changes'])
    rs = H['recall_shortfall']; out['f/ho/below_pct'] = pct(rs['share_below_target']); out['f/ho/short_max'] = f3(max(0.0, rs['max']))
RS = F.get('I13_repair_seeds')
if RS: out.update({'f/rep/n': str(RS['n']), 'f/rep/streams': str(RS['streams']), 'f/rep/lo': f'{RS["min"]:+.3f}', 'f/rep/hi': f'{RS["max"]:+.3f}'})
E5 = F.get('I13_e5')
if E5:
    rg = [x['oracle']['regret'] for run in E5.values() for E in run.values() for x in E['targets']['0.95']['grid'] if x.get('oracle') and x['oracle']['regret'] is not None]
    rg9 = [x[k]['regret'] for run in E5.values() for E in run.values() for x in E['targets']['0.9']['grid'] for k in ('oracle', 'fixed') if x.get(k) and x[k]['regret'] is not None]
    out['f/e5/regret_max'] = f'{100 * max(rg):.1f}'; out['f/e5/regret09_max'] = f'{100 * max(rg9):.1f}'; out['f/e5/runs'] = str(len(E5))
    vp = [sum(v['fixed']['violations'] > 0 for v in run['sift500k']['targets']['0.95']['per_P'].values()) for run in E5.values()]
    out['f/e5/fixed_viol_lo'] = str(min(vp)); out['f/e5/fixed_viol_hi'] = str(max(vp))
# I07
I7 = F.get('I07')
if I7:
    new = [r for r in I7['runs'] if r['new']]; out['f/blk/n'] = str(len(new)); out['f/blk/runs'] = str(len(I7['runs']))
    out['f/blk/allruns_ema'] = str(len(I7['ruleB_single_graph_winner']['1']['winners']))
    pr = I7['paired_log_ratio']
    for q in ('1', '100'):
        a = pr[f'acorn_repair_all/ema@{q}']; out[f'f/blk/rep_over_ema_lo/{q}'] = pct(math.exp(a['min']) - 1); out[f'f/blk/rep_over_ema_hi/{q}'] = pct(math.exp(a['max']) - 1)
    # EMA's saving relative to repaired ACORN, 1 - EMA/repaired, over the blocks and Q/U in 1..100 (CHECK.md B13: the
    # first text quoted repaired/EMA - 1, the repaired index's extra cost, as EMA's saving)
    sv = [1 - math.exp(-pr[f'acorn_repair_all/ema@{q}'][k]) for q in ('1', '10', '100') for k in ('min', 'max')]
    out['f/blk/ema_saves_lo'] = pct(min(sv)); out['f/blk/ema_saves_hi'] = pct(max(sv))
    out['f/blk/ema_lo'] = f2(min(r['norm']['ema']['1'] for r in new)); out['f/blk/ema_hi'] = f2(max(r['norm']['ema']['1'] for r in new))
    out['f/blk/acorn_lo'] = f2(min(r['norm']['acorn']['1'] for r in new)); out['f/blk/acorn_hi'] = f2(max(r['norm']['acorn']['1'] for r in new))
    out['f/blk/drift_max'] = pct(max(abs(r['anchor_drift']['1'] - 1) for r in new))
    out['f/blk/ruleA'] = 'yes' if all(v['holds'] for v in I7['ruleA_graphs_beat_ivf_part'].values()) else 'no'
    out['f/blk/ruleB'] = 'yes' if all(v['single'] for v in I7['ruleB_single_graph_winner'].values()) else 'no'
# I17
I17 = F.get('I17')
if I17:
    for nm, key in (('sift500k_adv', 'adv'), ('deep5m', 'deep')):
        X = I17.get(nm)
        if not X: continue
        for m, short in (('acorn', 'tomb'), ('acorn_repair', 'l0'), ('acorn_repair_all', 'all'), ('acorn_repair_upper', 'upper'), ('acorn_repair_fill', 'fill'), ('fresh', 'fresh')):
            if m in X['mean_reach']: out[f'f/mech/{key}/{short}/reach'] = f2(X['mean_reach'][m])
            if m in X['mean_nlong_share']: out[f'f/mech/{key}/{short}/long'] = f3(X['mean_nlong_share'][m])
            pm = X['per_mode'].get(m)
            if pm and pm.get('recall_by_label'): out[f'f/mech/{key}/{short}/rec'] = f2(float(np.mean(list(pm['recall_by_label'].values()))))
        if 'entry_account' in X: out['f/mech/entry_rho'] = f2(X['entry_account']['reach_recall_rho_tombstone']); out['f/mech/entry_ok'] = 'yes' if X['entry_account']['supported'] else 'no'
        if 'long_edge_account' in X: out['f/mech/long_rho'] = f2(X['long_edge_account']['b_spearman_rho']); out['f/mech/long_p'] = f2(X['long_edge_account']['b_p']); out['f/mech/long_ok'] = 'yes' if X['long_edge_account']['supported'] else 'no'
        if 'exploratory_gt_reach' in X: out['f/mech/gt_rho'] = f2(X['exploratory_gt_reach']['spearman_rho']); out['f/mech/gt_p'] = f3(X['exploratory_gt_reach']['p']); out['f/mech/gt_min'] = f2(X['exploratory_gt_reach']['rerank_gt_reach_min'])
        pa = X['per_mode'].get('acorn_repair_all')
        if key == 'deep' and pa and pa.get('retention'):
            r_ = [v for v in pa['retention'].values() if v]; l_ = [v for v in pa['long_retention'].values() if v]
            out['f/mech/ret'] = pct(np.mean(r_)); out['f/mech/long_ret'] = pct(np.mean(l_))
# I19
I19 = F.get('I19')
if I19:
    K = json.load(open('results/bench_r_summary.json'))['FRONTIER']['0.95']; cells = sum(len(v['best']) for v in I19['frontier'].values())
    out['f/e2e/cells'] = str(cells); out['f/e2e/changed'] = str(sum(len(v) for v in I19['changed_vs_kernel'].values()))
    sh = [abs(e / K[nm]['relative_to_ivf'][m][q] - 1) for nm, Fr in I19['frontier'].items() for m, v in Fr['relative_to_ivf'].items() for q, e in v.items()
          if nm in K and m in K[nm]['relative_to_ivf'] and q in K[nm]['relative_to_ivf'][m]]
    out['f/e2e/shift_max'] = pct(max(sh))
    lat = [(m, v['p99_ms']) for Fr in I19['frontier'].values() for m, v in Fr['latency_rss'].items() if v.get('p99_ms')]
    for m in ('ivf_part', 'ivf', 'acorn_repair_all'):
        xs = [x for mm, x in lat if mm == m]
        if xs: out[f'f/e2e/p99/{m}_lo'] = f'{min(xs):.1f}'; out[f'f/e2e/p99/{m}_hi'] = f'{max(xs):.1f}'
# I18
C = F.get('I18_cc')      # follow-up 4, cc4: one phase-fair reader-writer lock for every method
if C:
    for k, v in C.items():
        if k.startswith('_'): continue
        nm, m = k.split('/'); key = 'adv' if nm == 'sift500k_adv' else 'sift'
        out[f'f/cc/{key}/{m}/p99'] = f'{v["conc_p99_ms"]:.1f}' if v['conc_p99_ms'] < 100 else f'{v["conc_p99_ms"]:,.0f}'
        out[f'f/cc/{key}/{m}/quiet'] = f'{v["quiet_p99_ms"]:.1f}'; out[f'f/cc/{key}/{m}/x'] = f'{v["conc_p99_ms"] / v["quiet_p99_ms"]:.1f}'
        out[f'f/cc/{key}/{m}/writer'] = f'{v["writer_s_median"]:.1f}'; out[f'f/cc/{key}/{m}/wait'] = f'{v["lock_wait_p99_ms"]:.1f}' if v['lock_wait_p99_ms'] < 100 else f'{v["lock_wait_p99_ms"]:,.0f}'
        out[f'f/cc/{key}/{m}/qps'] = f'{v["conc_qps"]:,.0f}'
    if '_totals' in C:
        out['f/cc/invalid'] = str(sum(C['_totals'].values())); out['f/cc/clean'] = 'yes' if C['_no_invalid_result'] else 'no'
SD = F.get('I18_simd')
if SD:
    for lv, v in SD.items():
        out[f'f/simd/{lv}/ratio'] = f3(v['chk_over_dist']); out[f'f/simd/{lv}/sqrt_wrong'] = str(v['sqrt(n), microbenchmark corrected']['wrong'])
        out[f'f/simd/{lv}/milvus_wrong'] = str(v['Milvus 7%']['wrong']); out[f'f/simd/{lv}/n'] = str(v['Milvus 7%']['workloads'])
M = F.get('I18_multi')
if M:
    if 'ema' in M and 'scan' in M:
        qe = [x for x in M['ema']['qps095'].values() if x]; qs = [x for x in M['scan']['qps095'].values() if x]
        # paired by round (CHECK.md B13: the first text combined extremes of different rounds)
        pr_ = [M['scan']['qps095'][r] / v for r, v in M['ema']['qps095'].items() if v and M['scan']['qps095'].get(r)]
        if pr_: out['f/multi/ema/slower_lo'] = f'{min(pr_):.0f}'; out['f/multi/ema/slower_hi'] = f'{max(pr_):.0f}'; out['f/multi/ema/rounds'] = str(len(pr_))
        out['f/multi/ema/maint_h'] = f"{M['ema']['maint'] / 3600:.1f}"; out['f/multi/ema/invalid'] = str(M['ema']['invalid'])
    for m, v in M.items():
        mr = list(v['maxr'].values()); out[f'f/multi/{m}/maxr_lo'] = f2(min(mr)); out[f'f/multi/{m}/maxr_hi'] = f2(max(mr))
        q = [x for x in v['qps095'].values() if x]
        if q: out[f'f/multi/{m}/q095'] = f'{np.median(q):.0f}'
A = F.get('I18_amz')
if A:
    out['f/amz/part_up_to'] = f"{A['ivf_part_cheapest_up_to']['amz_time']:g}"; out['f/amz/ax_part_up_to'] = f"{A['ivf_part_cheapest_up_to']['arxiv_time']:g}"
    out['f/amz/prediction'] = 'held' if A['prediction_holds'] else 'did not hold'
    b = A['frontier']['amz_time']['best']; out['f/amz/best1'] = b.get('1', ''); out['f/amz/best100'] = b.get('100', '')
    qs = sorted(b, key=float); first = next((q for q in qs if b[q] != 'ivf_part'), None)
    if first: out['f/amz/graph_from'] = f'{float(first):g}'
    rel = A['frontier']['amz_time']['relative_to_ivf']; out['f/amz/part'] = f"{rel['ivf_part']['1']:.2f}"; out['f/amz/best_val100'] = f"{rel[b['100']]['100']:.2f}"
FX = F.get('F4_rebuild')  # follow-up 4, fdx: Filtered-DiskANN rebuilt at every round against the estimate
if FX:
    # the estimate R x mean rebuild against the executed total of the same run; across sessions the rebuild time itself
    # differs (the earlier sessions' estimates are 5-25% higher), which the session-relative costs absorb
    out['f/fdx/err_max'] = f"{100 * max(abs(v['estimate_same_run_err']) for v in FX.values()):.1f}"
    er = [e for v in FX.values() for e in v['estimate_paper_err']]
    if er: out['f/fdx/session_lo'] = f'{100 * min(er):.0f}'; out['f/fdx/session_hi'] = f'{100 * max(er):.0f}'
    out['f/fdx/streams'] = str(len(FX))
R27 = F.get('I27')
if R27:
    out.update({'f/rx/files': str(R27['files']), 'f/rx/rec_ok': str(R27['recall_ok']), 'f/rx/det_ok': str(R27['recall_ok_deterministic'][0]), 'f/rx/det_n': str(R27['recall_ok_deterministic'][1]),
                'f/rx/oth_ok': str(R27['recall_ok_other'][0]), 'f/rx/oth_n': str(R27['recall_ok_other'][1]), 'f/rx/qps_med': f2(R27['qps_ratio_all']['median']),
                'f/rx/qps_p10': f2(R27['qps_ratio_all']['p10_p90'][0]), 'f/rx/qps_p90': f2(R27['qps_ratio_all']['p10_p90'][1]),
                'f/rx/ver_agree': str(R27['verdicts']['agree']), 'f/rx/ver_n': str(R27['verdicts']['pairs']), 'f/rx/jobs': str(R27['jobs'])})
if os.path.exists('results/bench_f3_summary.json'):     # follow-up 3: second machine (paper_audit/followup3_prereg.md)
    H3 = json.load(open('results/bench_f3_summary.json')); rl = H3['rules']
    out.update({'f/hw/exp': f2(H3['ivf_exponent']), 'f/hw/mstar1m': f"{H3['m_star']['deep1m']:,.0f}", 'f/hw/mstar1m_xeon': f"{H3['xeon_m_star_deep1m']:,.0f}",
                'f/hw/ratio': f3(H3['chk_over_dist']['epyc']), 'f/hw/ratio_xeon': f3(H3['chk_over_dist']['xeon']),
                'f/hw/mstar_x': f"{H3['m_star']['deep1m'] / H3['xeon_m_star_deep1m']:.1f}", 'f/hw/n': str(H3['workloads'])})
    for k, key in (('P1 EPYC DEEP-1M, sqrt(n)', 'own'), ('P2 Xeon DEEP-1M, sqrt(n), microbenchmark corrected', 'corr'), ('P3 Xeon DEEP-1M, sqrt(n), uncorrected', 'uncorr'),
                   ('P4 fixed selectivity 7%', 'milvus'), ('P5 Xeon fixed size', 'fixA')):
        out[f'f/hw/{key}/wrong'] = str(rl[k]['wrong']); out[f'f/hw/{key}/n'] = str(rl[k]['workloads'])
        if rl[k]['regret_max'] is not None: out[f'f/hw/{key}/max'] = r'$\infty$' if rl[k]['regret_max'] == 'inf' else pct(rl[k]['regret_max'])
        out[f'f/hw/{key}/infeasible'] = str(rl[k]['infeasible_choices'])
if os.path.exists('results/bench_f5_summary.json'):     # follow-up 5: executed trace of every round, memory (followup5_prereg.md)
    F5 = json.load(open('results/bench_f5_summary.json'))
    qc = [abs(v) for X in F5.values() for m, v in X['interp_query_cost_change'].items() if m != 'ivf_rt5']
    out['f/tr/interp_max'] = pct(max(qc)); out['f/tr/interp_rt5'] = pct(max(abs(X['interp_query_cost_change'].get('ivf_rt5', 0)) for X in F5.values()))
    out['f/tr/d1_cells'] = str(sum(len(X['D1_cells_changed']) for X in F5.values()))
    for nm, key in (('sift500k_adv', 'adv'), ('arxiv_time', 'ax')):
        X = F5[nm]; out[f'f/tr/{key}/d2_cells'] = str(len(X['D2_cells_changed'])); out[f'f/tr/{key}/r0fail'] = str(len(X['round0_failures']))
        out[f'f/tr/{key}/drift'] = pct(abs(X['drift']['oracle_all']['1'] - 1))
        for m, v in X['memory'].items(): out[f'f/mem/{key}/{m}'] = f"{v['ratio']:.1f}"
        mm = X['max_recall_below_target'].get('acorn_repair_all')
        if mm: r_, v_ = min(mm['rounds'], key=lambda x: x[1]); out[f'f/tr/{key}/rep_dip'] = f2(v_); out[f'f/tr/{key}/rep_dip_round'] = str(r_); out[f'f/tr/{key}/rep_dip_unmeasured'] = 'yes' if not mm['paper_rounds_only'] else 'no'
        bc = X['policies']['constant']['best']; out[f'f/tr/{key}/const_best1'] = bc['1']; out[f'f/tr/{key}/const_best100'] = bc['100']
if os.path.exists('results/bench_f6_summary.json'):     # follow-up 6: repairs on three SIFT-500K-adv seeds, every round
    F6 = json.load(open('results/bench_f6_summary.json')); S6 = F6['streams'].values()
    rng = lambda m: [x['max_recall_r20'][m] for x in S6]
    for m, k in (('acorn', 'tomb'), ('acorn_repair', 'l0'), ('acorn_repair_all', 'all'), ('acorn_repair_upper', 'upper'), ('acorn_repair_fill', 'fill')):
        out[f'f/rpx/{k}_lo'] = f2(min(rng(m))); out[f'f/rpx/{k}_hi'] = f2(max(rng(m)))
    rh = [x['H3b_rho'] for x in S6]; out['f/rpx/rho_lo'] = f2(min(rh)); out['f/rpx/rho_hi'] = f2(max(rh))
    out['f/rpx/dips'] = str(F6['H1']['pairs_below']); out['f/rpx/pairs'] = str(F6['H1']['pairs']); out['f/rpx/dip_streams'] = str(F6['H1']['streams_with_dip'])
    out['f/rpx/dip_min'] = f2(F6['H1']['min']) if F6['H1']['min'] is not None else '--'
    dr = [e['reach'] for x in S6 for e in x['exploratory_dips_vs_reach']]
    if dr: out['f/rpx/dip_reach_lo'] = f2(min(dr)); out['f/rpx/dip_reach_hi'] = f2(max(dr))
    out['f/rpx/rules_all'] = 'yes' if all(F6[h] == 3 for h in ('H2', 'H3a', 'H3b', 'H4', 'H5')) else 'no'
    mt = [x['maintenance_over_tombstones_r20']['acorn_repair_all'] for x in S6]; out['f/rpx/maint_lo'] = f2(min(mt)); out['f/rpx/maint_hi'] = f2(max(mt))
    # the all-level repair at round 20 over every run: the main campaigns' (a/h3) and these
    A = json.load(open('results/bench_a_numbers.json'))
    comb = [float(A['a/h3/acorn_repair_all/r20_min']), float(A['a/h3/acorn_repair_all/r20_max'])] + rng('acorn_repair_all')
    out['f/rpx/allruns_lo'] = f3(min(comb)); out['f/rpx/allruns_hi'] = f3(max(comb)); out['f/rpx/allruns_n'] = str(int(A['a/h3/acorn_repair_all/runs']) + len(F6['streams']))
if os.path.exists('results/bench_f7_summary.json'):     # follow-up 7: close verdicts in four paired blocks
    F7 = json.load(open('results/bench_f7_summary.json')); d1, d2 = F7['D1'], F7['D2']
    out.update({'f/g6/close_n': str(d1['n']), 'f/g6/confirmed': str(d1['confirmed']), 'f/g6/ties': str(d1['tie']), 'f/g6/reversed': str(d1['reversed']),
                'f/g6/edge_n': str(d2['n']), 'f/g6/edge_hold': str(d2['holds'])})
    R7 = {(r['workload'], r['method']): r for r in F7['rows']}
    r = R7[('deep10m/b0.9_s0.01', 'ivf')]; out['f/g6/ivf10m'] = f2(r['mean_ratio']); out['f/g6/ivf10m_lo'] = f2(r['ci95'][0]); out['f/g6/ivf10m_hi'] = f2(r['ci95'][1])
    for m, k in (('acorn', 'acorn'), ('acorn_g100', 'g100')): out[f'f/g6/sift_{k}_feas'] = str(R7[('sift1m/b0.9_s0.01', m)]['feasible_blocks'])
DISP = {'ivf_part': 'label-partitioned IVF', 'ema': 'EMA', 'acorn_repair_all': 'repaired ACORN', 'acorn': 'ACORN', 'fdiskann_rebuild': 'rebuilt Filtered-DiskANN', 'ivf': 'IVF'}
for k in ('f/amz/best1', 'f/amz/best100', 'f/tr/adv/const_best1', 'f/tr/adv/const_best100', 'f/tr/ax/const_best1', 'f/tr/ax/const_best100'):
    if k in out: out[k] = DISP.get(out[k], out[k])
with open('paper/eab_fnumbers.tex', 'w') as fo:
    fo.write('% generated by paper/make_eab_fnumbers.py from ' + src + ' -- do not edit\n')
    for k, v in sorted(out.items()): fo.write('\\expandafter\\def\\csname v@' + k + '\\endcsname{' + v + '}\n')
print(len(out), 'macros')

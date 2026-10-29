#!/usr/bin/env python3
"""Analyses of follow-up 2 (paper_audit/followup2_prereg.md), written before the runs finished; decision rules as
pre-registered. Missing campaigns are an error unless F2_PARTIAL=1 (drafting only).
  python3 bench/analyze_followup2.py > results/bench_f2_analysis.txt      -> results/bench_f2_summary.json"""
import sys, os, collections, json, glob, math, subprocess
sys.path.insert(0, 'bench')
import numpy as np
from analyze import summarize_static, frontier, qps_at
from analyze_h import h1, crossing, L_of, NS
from analyze_review import load_roots, crossovers, run_costs, executed_policies
from common import stream_stats, spearman

PARTIAL = os.environ.get('F2_PARTIAL') == '1'
OUT = {}; QPUS = (0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100); GRAPH = ('acorn', 'acorn_repair_all', 'ema', 'fdiskann_rebuild')
def say(*a): print(*a, flush=True)
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}     # Student t, 0.975 quantile
def need(ok, what):
    if ok: return True
    if PARTIAL: say(f'PARTIAL: {what} missing, skipped'); return False
    raise SystemExit(f'{what} missing; the campaign is incomplete')
def runfile(root, nm, m):
    f = f'{root}/stream_{nm}/{m}.json'; return json.load(open(f)) if os.path.exists(f) else None
def total(c, qpu, U): return c['maint'] + qpu * U * sum(c['t_query'])

# ------------------------------------------------------------------------------------------------ I07
def i07(t=0.95):
    nm = 'arxiv_time'; st = stream_stats(nm); R, U = st['R'], st['updates_per_round']; M = ('ivf_part', 'ivf_rt5') + GRAPH
    runs = []
    for tag in ('r7a', 'r7b', 'r7c', 'r7d'):
        pre, post = runfile(f'results/bench_{tag}', nm, 'ivf'), runfile(f'results/bench_{tag}_pass2', nm, 'ivf')
        if not need(pre is not None and post is not None and all(runfile(f'results/bench_{tag}', nm, m) for m in M), f'I07 block {tag}'): continue
        a0, a1 = run_costs(pre, R, t), run_costs(post, R, t)
        row = {'run': tag, 'new': True, 'anchor_drift': {str(q): total(a1, q, U) / total(a0, q, U) for q in QPUS}, 'norm': {}, 'raw': {}, 'feasible': {}}
        for m in M:
            c = run_costs(runfile(f'results/bench_{tag}', nm, m), R, t); row['feasible'][m] = c is not None
            if c is None: continue
            row['raw'][m] = {str(q): total(c, q, U) for q in QPUS}
            row['norm'][m] = {str(q): total(c, q, U) / math.sqrt(total(a0, q, U) * total(a1, q, U)) for q in QPUS}
        runs.append(row)
    for root in ('results/bench_r3', 'results/bench_r3_pass2', 'results/bench_r6', 'results/bench_r6_pass2'):     # earlier runs, same-pass anchor
        iv = runfile(root, nm, 'ivf')
        if iv is None: continue
        a = run_costs(iv, R, t); row = {'run': root.split('bench_')[1], 'new': False, 'norm': {}, 'raw': {}, 'feasible': {}}
        for m in M:
            d = runfile(root, nm, m)
            if d is None: continue
            c = run_costs(d, R, t); row['feasible'][m] = c is not None
            if c is None: continue
            row['raw'][m] = {str(q): total(c, q, U) for q in QPUS}; row['norm'][m] = {str(q): total(c, q, U) / total(a, q, U) for q in QPUS}
        runs.append(row)
    new = [r for r in runs if r['new']]; res = {'runs': runs}
    # rule A: graphs cheaper than the label-partitioned IVF from Q/U = 1, in every new block, normalized and raw
    ruleA = {}
    for q in (str(x) for x in QPUS if x >= 1):
        ok = []
        for r in new:
            g = [m for m in GRAPH if m in r['norm']]
            ok.append(bool(g) and 'ivf_part' in r['norm'] and min(r['norm'][m][q] for m in g) < r['norm']['ivf_part'][q]
                      and min(r['raw'][m][q] for m in g) < r['raw']['ivf_part'][q])
        ruleA[q] = {'blocks_holding': int(sum(ok)), 'blocks': len(new), 'holds': len(new) == 4 and all(ok)}
    res['ruleA_graphs_beat_ivf_part'] = ruleA
    # rule B: a single cheapest graph method only if cheapest in all 4 new blocks and in every earlier run that has all four
    ruleB = {}
    for q in (str(x) for x in QPUS if x >= 1):
        winners = []
        for r in runs:
            if all(m in r['norm'] for m in GRAPH): winners.append(min(GRAPH, key=lambda m: r['norm'][m][q]))
            elif r['new']: winners.append(None)
        uniq = set(winners); ruleB[q] = {'winners': winners, 'single': (len(uniq) == 1 and None not in uniq and len(new) == 4), 'name': winners[0] if len(uniq) == 1 else None}
    res['ruleB_single_graph_winner'] = ruleB
    # paired log ratios between graph methods over every run that has both
    pairs = {}
    for i, a in enumerate(GRAPH):
        for b in GRAPH[i + 1:]:
            for q in ('1', '10', '100'):
                v = [math.log(r['norm'][a][q] / r['norm'][b][q]) for r in runs if a in r['norm'] and b in r['norm']]
                if v: pairs[f'{a}/{b}@{q}'] = {'runs': len(v), 'mean': float(np.mean(v)), 'min': float(min(v)), 'max': float(max(v)), 'excludes_0': bool(min(v) > 0 or max(v) < 0)}
    res['paired_log_ratio'] = pairs
    res['anchor_drift_over_15pct'] = [r['run'] for r in new if abs(r['anchor_drift']['1'] - 1) > 0.15]
    say('== I07 arXiv blocks ==')
    for r in runs: say(f"  {r['run']:10s} " + ' '.join(f"{m}:{r['norm'][m]['1']:.2f}" for m in r['norm']) + (f"  drift {r['anchor_drift']['1']:.2f}" if r['new'] else ''))
    say(f'  rule A (graphs < ivf_part from Q/U=1, every new block, normalized and raw): {ruleA}')
    say(f'  rule B (single cheapest graph): ' + str({q: (v['single'], v['winners']) for q, v in ruleB.items()}))
    return res

# ------------------------------------------------------------------------------------------------ I13
def i13_seeds():
    res = {'per_seed': {}}
    import analyze
    for sd, tag in [('11', 'h1')] + [(str(k), f's{k}') for k in range(12, 18)]:
        if not need(os.path.isdir(f'results/bench_{tag}'), f'I13 seed {sd} (results/bench_{tag})'): continue
        orig = analyze.passes; analyze.passes = lambda tg: [f'results/bench_{tg}']     # one run per seed: seed 11 from pass 1 only
        try: f = h1(tag=tag, label=f'seed {sd}', keep_boot=True)['fit']['0.95'].get('ivf')
        finally: analyze.passes = orig
        if f: res['per_seed'][sd] = {'exponent': f['exponent'], 'ci95_queries': f.get('exponent_ci95'), 'boot': f.get('boot_slopes')}
    ex = np.array([v['exponent'] for v in res['per_seed'].values()])
    if len(ex) >= 2:
        tq = T975[len(ex) - 1]; se = ex.std(ddof=1) / math.sqrt(len(ex))
        res['summary'] = {'seeds': len(ex), 'mean': float(ex.mean()), 'sd': float(ex.std(ddof=1)), 't_ci95': [float(ex.mean() - tq * se), float(ex.mean() + tq * se)],
                          'min': float(ex.min()), 'max': float(ex.max())}
        rng = np.random.default_rng(2027); bs = [v['boot'] for v in res['per_seed'].values() if v['boot']]
        if len(bs) == len(ex):
            two = [np.mean([bs[i][rng.integers(len(bs[i]))] for i in rng.integers(0, len(bs), len(bs))]) for _ in range(4000)]
            res['summary']['two_stage_ci95'] = [float(np.quantile(two, 0.025)), float(np.quantile(two, 0.975))]
        res['summary']['below_half_supported'] = res['summary']['t_ci95'][1] < 0.5
    for v in res['per_seed'].values(): v.pop('boot', None)
    say(f"== I13 seeds: IVF exponent per seed {[(k, round(v['exponent'], 3)) for k, v in res['per_seed'].items()]}; {res.get('summary')}")
    return res

def i13_policy_unseen():
    """thresholds fixed on seed 11 (H1 DEEP-1M), applied to the unseen seeds 12-17, all four sizes"""
    H = json.load(open('results/bench_h_summary.json')); m0 = H['H1']['per_n']['0.95']['deep1m']['ivf']['m_star']
    T = []
    for k in range(12, 18):
        if not os.path.isdir(f'results/bench_s{k}'): continue
        S = summarize_static(load_roots([f'results/bench_s{k}']))[1]
        for ds, n in NS.items():
            for wl in [w for w in S if w.startswith(ds + '/b0.0_s')]:
                sc, iv = S[wl].get('scan_part', {}).get('qps@0.95'), S[wl].get('ivf', {}).get('qps@0.95')
                if sc: T.append({'seed': k, 'n': n, 'm': n / L_of(wl.split('/')[1]), 'c_scan': 1 / sc, 'c_ivf': (1 / iv) if iv else math.inf})
    pol = {'fixed |A|': lambda w: m0, 'fixed s': lambda w: m0 / 1e6 * w['n'], 'Milvus 7%': lambda w: 0.07 * w['n'], 'sqrt(n)': lambda w: m0 * math.sqrt(w['n'] / 1e6)}
    res = {'workloads': len(T), 'seeds': sorted({w['seed'] for w in T})}
    for name, f in pol.items():
        reg = []; wrong = 0
        for w in T:
            ch = w['c_scan'] if w['m'] < f(w) else w['c_ivf']; best = min(w['c_scan'], w['c_ivf']); reg.append(ch / best - 1); wrong += ch > best * (1 + 1e-9)
        fin = [x for x in reg if math.isfinite(x)]
        res[name] = {'wrong': wrong, 'chose_ivf_where_ivf_misses': int(len(reg) - len(fin)), 'regret_max_finite': float(max(fin)) if fin else None,
                     'regret_mean_finite': float(np.mean(fin)) if fin else None}
    say(f'== I13 unseen-seed policy test: {res}')
    return res

def i13_timing_t():
    """within-job intervals with Student t (3 timings, df=2) instead of the normal 1.96"""
    f = T975[2] / 1.96; rel = []
    for p in glob.glob('results/bench/*/*/*.json') + glob.glob('results/bench_pass2/*/*/*.json'):
        for r in json.load(open(p)):
            q = r.get('qps') or {}
            if q.get('reps') == 3 and q.get('mean'): rel.append(q['ci95'] / q['mean'])
    rel = np.array(rel); res = {'configs': len(rel), 'factor_t_over_normal': f, 'normal_median_pct': float(100 * np.median(rel)), 'normal_p90_pct': float(100 * np.quantile(rel, 0.9)),
                                't_median_pct': float(100 * f * np.median(rel)), 't_p90_pct': float(100 * f * np.quantile(rel, 0.9))}
    say(f'== I13 timing intervals: {res}'); return res

def i13_best_of_two():
    """index-vs-scan verdicts and H1 exponents from the best pass (as reported), the mean of passes, and the worse pass"""
    roots = {'best': None, 'mean': None, 'worse': None}; res = {}
    def variant(kind, roots_):
        W = {}
        for root in roots_:
            for f in glob.glob(f'{root}/*/*/*.json'):
                ds, sc = f.split('/')[-3], f.split('/')[-2]
                if ds.startswith('stream_'): continue
                for r in json.load(open(f)): W.setdefault((ds, sc, r['method'], r['param']), []).append(r)
        out = {}
        for k, rs in W.items():
            if len(rs) < 2: continue
            q = [r['qps']['mean'] for r in rs]; v = {'best': max(q), 'mean': float(np.mean(q)), 'worse': min(q)}[kind]
            r0 = dict(rs[int(np.argmax(q))]); r0['qps'] = {**r0['qps'], 'mean': v}; r0['recall'] = rs[int(np.argmax(q))]['recall'] if kind == 'best' else (float(np.mean([r['recall'] for r in rs])) if kind == 'mean' else rs[int(np.argmin(q))]['recall'])
            out[k] = {**r0, 'passes': len(rs), 'recalls': [r['recall'] for r in rs], 'qpss': q}
        return out
    for tag, roots_ in (('main', ['results/bench', 'results/bench_pass2']), ('h1', ['results/bench_h1', 'results/bench_h1_pass2'])):
        V = {kind: summarize_static(variant(kind, roots_))[1] for kind in ('best', 'mean', 'worse')}
        ver = {}
        for kind, S in V.items():
            for wl, ms in S.items():
                sc = ms.get('scan_part', {}).get('qps@0.95')
                if not sc: continue
                for m, v in ms.items():
                    if m.startswith('scan') or v.get('qps@0.95') is None: continue
                    ver.setdefault((wl, m), {})[kind] = v['qps@0.95'] > sc
        changed = {kind: sum(1 for v in ver.values() if 'best' in v and kind in v and v[kind] != v['best']) for kind in ('mean', 'worse')}
        res[tag] = {'verdicts': sum(1 for v in ver.values() if 'best' in v), 'changed_vs_best': changed}
        if tag == 'h1':
            res[tag]['ivf_exponent'] = {kind: crossovers(S, NS, '0.95', meths=('ivf',))[1].get('ivf') for kind, S in V.items()}
    say(f'== I13 best-of-two: {res}'); return res

def i13_heldout(seed=2027, t=0.95):
    """operating point chosen on one half of the queries (per-query recall and time), evaluated on the other half.
    A verdict at the target is three-state: the chosen point misses the target on the held-out half (infeasible), or
    it reaches it and is faster or slower than the scan. The first version compared times only (CHECK.md B03); both
    counts are reported: latency_changes (time ordering only), feasibility_losses, and decision_changes (the index's
    recall-feasible win on all queries differs from its recall-feasible win on the held-out half)."""
    rng = np.random.default_rng(seed); res = {'workloads': 0, 'latency_changes': 0, 'feasibility_losses': 0, 'decision_changes': 0, 'recall_shortfall': [],
                                              'comparisons': [], 'methods': collections.Counter()}
    for root in ('results/bench', 'results/bench_h1'):
        for d in sorted(glob.glob(f'{root}/*/*')):
            if '/stream_' in d: continue
            recs = {}
            for f in glob.glob(f'{d}/*.json'):
                for r in json.load(open(f)):
                    if isinstance(r.get('rq'), list) and isinstance(r.get('tq'), list): recs.setdefault(r['method'], []).append(r)
            if 'scan_part' not in recs or len(recs) < 2: continue
            nq = len(recs['scan_part'][0]['rq']); perm = rng.permutation(nq); A, B = perm[:nq // 2], perm[nq // 2:]
            def pick(rs, idx):
                ok = [(np.sum(np.asarray(r['tq'])[idx]), r) for r in rs if np.mean(np.asarray(r['rq'])[idx]) >= t]
                return min(ok, key=lambda x: x[0])[1] if ok else None
            sA = pick(recs['scan_part'], A); sF = pick(recs['scan_part'], np.arange(nq))
            if sA is None or sF is None: continue
            if not np.mean(np.asarray(sA['rq'])[B]) >= t: raise SystemExit(f'{d}: the scan misses the target on the held-out half')
            for m, rs in sorted(recs.items()):
                if m == 'scan_part': continue
                full = pick(rs, np.arange(nq)); tuned = pick(rs, A)
                if full is None or tuned is None: continue
                rB = float(np.mean(np.asarray(tuned['rq'])[B])); tB = float(np.sum(np.asarray(tuned['tq'])[B])); tsB = float(np.sum(np.asarray(sA['tq'])[B]))
                v_full = float(np.sum(full['tq'])) < float(np.sum(sF['tq']))     # as reported: both tuned and evaluated on all queries
                faster = tB < tsB; feasible = rB >= t                           # both tuned on half A, evaluated on half B
                state = 'infeasible' if not feasible else ('feasible_faster' if faster else 'feasible_slower')
                res['workloads'] += 1; res['recall_shortfall'].append(t - rB); res['methods'][m] += 1
                res['latency_changes'] += bool(v_full) != bool(faster); res['feasibility_losses'] += not feasible
                res['decision_changes'] += bool(v_full) != (state == 'feasible_faster')
                res['comparisons'].append({'workload': d, 'method': m, 'param_full': str(full.get('param')), 'param_half': str(tuned.get('param')),
                                           'recall_half_B': rB, 'full_wins': bool(v_full), 'half_state': state})
    sf = np.array(res['recall_shortfall']); res['recall_shortfall'] = {'median': float(np.median(sf)) if len(sf) else None, 'share_below_target': float(np.mean(sf > 0)) if len(sf) else None,
                                                                         'max': float(sf.max()) if len(sf) else None}
    res['methods'] = dict(res['methods']); res['split'] = {'seed': seed, 'rule': 'numpy default_rng(seed).permutation(nq) per workload in sorted order; A = first half'}
    say(f"== I13 held-out tuning: {res['workloads']} comparisons, latency changes {res['latency_changes']}, below target {res['feasibility_losses']}, "
        f"recall-feasible decision changes {res['decision_changes']}")
    for c in res['comparisons']:
        if c['half_state'] == 'infeasible': say(f"   below target on the held-out half: {c}")
    return res

def i13_e5_replicate():
    if not need(os.path.isdir('results/bench_r2c'), 'I13 E5 replicate (results/bench_r2c)'): return None
    b, c = executed_policies('r2b'), executed_policies('r2c'); say('== I13 E5 replicate done'); return {'r2b': b, 'r2c': c}

def i13_repair_seeds():
    reps = [('base pass1', 'results/bench_r1', 'deep5m'), ('base pass2', 'results/bench_r1_pass2', 'deep5m')] + [(f'seed{k}', 'results/bench_rs', f'deep5m_seed{k}') for k in (2, 3, 4, 5)]
    out = {}
    for lab, root, nm in reps:
        P = {}
        for m in ('acorn', 'acorn_repair_all'):
            d = runfile(root, nm, m)
            if d is None: continue
            r = [x for x in d['rounds'] if 'sweep' in x][-1]; P[m] = max(x['recall'] for x in r['sweep'])
        if len(P) == 2: out[lab] = {'stream': nm, 'tombstone': P['acorn'], 'repaired': P['acorn_repair_all'], 'diff': P['acorn_repair_all'] - P['acorn']}
    need(all(f'seed{k}' in out for k in (4, 5)), 'I13 DEEP-5M stream seeds 4, 5')
    dd = [v['diff'] for v in out.values()]
    res = {'runs': out, 'n': len(dd), 'streams': len({v['stream'] for v in out.values()}), 'all_negative': all(x < 0 for x in dd), 'min': min(dd), 'max': max(dd)}
    say(f'== I13 repair seeds: {res}'); return res

# ------------------------------------------------------------------------------------------------ I17
def i17():
    res = {}
    for nm in ('sift500k_adv', 'deep5m'):
        D = {m: runfile('results/bench_r8', nm, m) for m in ('acorn', 'acorn_repair', 'acorn_repair_all', 'acorn_repair_upper', 'acorn_repair_fill')}
        if not need(all(D.values()), f'I17 r8 {nm}'): continue
        per = {}
        for m, d in D.items():
            last = [r for r in d['rounds'] if 'mech' in r][-1]; L = {l['label']: l for l in last['mech']['labels']}
            rec = max(last['sweep'], key=lambda s: s['recall'])['recall_by_label']
            m3 = {}
            for r in d['rounds']:
                for x in r.get('m3', []):
                    a = m3.setdefault(x['label'], [0, 0, 0, 0]); a[0] += x['live_before']; a[1] += x['kept']; a[2] += x['long_before']; a[3] += x['long_kept']
            per[m] = {'round': last['round'], 'long_share': {k: v['long_share'] for k, v in L.items()}, 'nlong_share': {k: v['nlong_share'] for k, v in L.items()},
                      'reach_frac': {k: v['reach_frac'] for k, v in L.items()}, 'gt_reach': {k: v['gt_reach'] for k, v in L.items()},
                      'recall_by_label': {int(k): v for k, v in rec.items()}, 'm5': last['mech']['m5'],
                      'retention': {k: (v[1] / v[0] if v[0] else None) for k, v in m3.items()}, 'long_retention': {k: (v[3] / v[2] if v[2] else None) for k, v in m3.items()}}
        fr = D['acorn'].get('fresh', {}).get('mech'); per['fresh'] = {'long_share': {l['label']: l['long_share'] for l in fr['labels']}, 'nlong_share': {l['label']: l['nlong_share'] for l in fr['labels']},
                                                                       'reach_frac': {l['label']: l['reach_frac'] for l in fr['labels']}} if fr else None
        avg = lambda dct: float(np.mean([v for v in dct.values() if v is not None]))
        out = {'per_mode': per, 'mean_long_share': {m: avg(v['long_share']) for m, v in per.items() if v}, 'mean_nlong_share': {m: avg(v['nlong_share']) for m, v in per.items() if v},
               'mean_reach': {m: avg(v['reach_frac']) for m, v in per.items() if v}}
        if nm == 'deep5m':    # long-edge account
            ls = out['mean_nlong_share']; a_ = ls['acorn_repair_all'] < ls['acorn_repair_fill'] and per['fresh'] is not None and ls['acorn_repair_all'] < ls['fresh']
            labs = sorted(per['acorn_repair_all']['recall_by_label'])
            dr = [per['acorn_repair_fill']['recall_by_label'][l] - per['acorn_repair_all']['recall_by_label'][l] for l in labs]
            dl = [per['acorn_repair_fill']['nlong_share'][l] - per['acorn_repair_all']['nlong_share'][l] for l in labs]
            rho, p = spearman(dr, dl)
            out['long_edge_account'] = {'a_rerank_below_fill_and_fresh': bool(a_), 'b_spearman_rho': float(rho), 'b_p': float(p), 'labels': len(labs),
                                        'supported': bool(a_ and rho > 0.3 and p < 0.05)}
            # exploratory (not pre-registered): recall loss against the loss of the true neighbours' reachability
            dg = [per['acorn_repair_fill']['gt_reach'][l] - per['acorn_repair_all']['gt_reach'][l] for l in labs]; r2, p2 = spearman(dr, dg)
            out['exploratory_gt_reach'] = {'spearman_rho': float(r2), 'p': float(p2), 'rerank_gt_reach_min': float(min(per['acorn_repair_all']['gt_reach'].values()))}
        else:                 # entry account
            rch = out['mean_reach']; labs = sorted(per['acorn_repair_upper']['recall_by_label'])
            rho, p = spearman([per['acorn']['reach_frac'][l] for l in labs], [per['acorn']['recall_by_label'][l] for l in labs])
            out['entry_account'] = {'upper_below_all_in_reach': bool(rch['acorn_repair_upper'] < rch['acorn_repair_all']), 'reach_recall_rho_tombstone': float(rho), 'p': float(p),
                                    'supported': bool(rch['acorn_repair_upper'] < rch['acorn_repair_all'] and rho > 0.5)}
        res[nm] = out
        say(f"== I17 {nm}: long share {out['mean_nlong_share']}; reach {out['mean_reach']}; " + str(out.get('long_edge_account') or out.get('entry_account')))
    return res

# ------------------------------------------------------------------------------------------------ frontiers (I19, I18a)
E2E_METHODS = {nm: {'ivf', 'ivf_part', 'acorn', 'acorn_repair_all', 'ema', 'fdiskann_rebuild', 'postfilter', 'scan_part'} for nm in ('sift500k', 'sift500k_adv')}
E2E_METHODS['arxiv_time'] = E2E_METHODS['sift500k'] | {'ivf_rt5'}; E2E_METHODS['deep5m'] = {'ivf', 'ivf_part', 'acorn', 'fdiskann_rebuild'}   # followup2_prereg.md, I19
AMZ_METHODS = {'amz_time': E2E_METHODS['arxiv_time']}                                                                                    # the arXiv set (I18a)

def frontier_for(streams, roots, t=0.95, expected=None):
    """Cost frontier relative to IVF of the same root. expected: {stream: set of methods}; every root must hold exactly
    these runs (CHECK.md B06), and run_costs() rejects runs with missing rounds."""
    res = {}
    for nm in streams:
        if not need(all(os.path.exists(f'{r}/stream_{nm}/ivf.json') for r in roots), f'frontier {nm} in {roots}'): continue
        for root in roots:
            got = {json.load(open(f))['method'] for f in glob.glob(f'{root}/stream_{nm}/*.json') if not os.path.basename(f).startswith('cc_')}
            if expected is not None and got != expected[nm]: raise SystemExit(f'{root}/stream_{nm}: methods {sorted(got)} differ from the expected {sorted(expected[nm])}')
        st = stream_stats(nm); R, U = st['R'], st['updates_per_round']; rel = {}; fails = {}; runs_n = {}; lat = {}
        for root in roots:
            base = run_costs(runfile(root, nm, 'ivf'), R, t)
            for f in glob.glob(f'{root}/stream_{nm}/*.json'):
                d = json.load(open(f))
                if 'rounds' not in d or d['method'].startswith('cc_'): continue
                m = d['method']; runs_n[m] = runs_n.get(m, 0) + 1; c = run_costs(d, R, t)
                if c is None: fails[m] = fails.get(m, 0) + 1; continue
                for q in QPUS: rel.setdefault(m, {}).setdefault(str(q), []).append(total(c, q, U) / total(base, q, U))
                last = [r for r in d['rounds'] if 'sweep' in r][-1]; ok = [s for s in last['sweep'] if s['recall'] >= t and s.get('latency')]
                if ok: lat[m] = ok[0]['latency']
                if d.get('peak_rss_mb'): lat.setdefault(m, {})['peak_rss_mb'] = d['peak_rss_mb']
        table = {m: {q: float(np.median(v)) for q, v in d_.items()} for m, d_ in rel.items() if not fails.get(m)}
        best = {str(q): min(table, key=lambda m: table[m][str(q)]) for q in QPUS} if table else {}
        res[nm] = {'relative_to_ivf': table, 'best': best, 'infeasible': fails, 'runs': runs_n, 'latency_rss': lat}
        say(f'== frontier {nm} ({roots}): best {best}; infeasible {fails}')
    return res

E2E4_STREAMS = ('sift500k', 'sift500k_adv', 'arxiv_time')
def i19():
    """End-to-end frontier. EMA's end-to-end cost relative to IVF comes from the e2e4 session (follow-up 4, B2: the e2e
    campaign timed EMA as one batched call, its runner having no E2E mode); the other methods' from e2e."""
    exp = {nm: v - {'ema'} for nm, v in E2E_METHODS.items()}
    e2e = frontier_for(('sift500k', 'sift500k_adv', 'arxiv_time', 'deep5m'), ['results/bench_e2e'], expected=exp)
    ema = frontier_for(E2E4_STREAMS, ['results/bench_e2e4'], expected={nm: {'ivf', 'ema'} for nm in E2E4_STREAMS})
    for nm, F in ema.items():
        if nm not in e2e: continue
        G = e2e[nm]; G['source'] = {'ema': 'results/bench_e2e4', 'others': 'results/bench_e2e'}
        G['runs']['ema'] = F['runs'].get('ema', 0)
        if 'ema' in F['relative_to_ivf']: G['relative_to_ivf']['ema'] = F['relative_to_ivf']['ema']
        if F['infeasible'].get('ema'): G['infeasible']['ema'] = F['infeasible']['ema']
        if 'ema' in F['latency_rss']: G['latency_rss']['ema'] = F['latency_rss']['ema']
        T = G['relative_to_ivf']; G['best'] = {str(q): min(T, key=lambda m: T[m][str(q)]) for q in QPUS}
        say(f"== frontier {nm} with EMA end to end: best {G['best']}")
    kern = json.load(open('results/bench_r_summary.json'))['FRONTIER']['0.95']; changed = {}
    for nm, F in e2e.items():
        if nm in kern: changed[nm] = {q: [kern[nm]['best'].get(q), F['best'].get(q)] for q in F['best'] if kern[nm]['best'].get(q) != F['best'].get(q)}
    say(f'== I19 winners that change from kernel-only to end-to-end: {changed}'); return {'frontier': e2e, 'changed_vs_kernel': changed}

def f4_rebuild():
    """Follow-up 4, B3: Filtered-DiskANN rebuilt at every round (tag fdx) against the estimate mean rebuild x R."""
    res = {}
    for nm, others in (('sift500k', ('results/bench', 'results/bench_pass2')), ('sift500k_adv', ('results/bench', 'results/bench_pass2')),
                       ('arxiv_time', ('results/bench_r3', 'results/bench_r3_pass2'))):
        d = runfile('results/bench_fdx', nm, 'fdiskann_rebuild')
        if not need(d is not None, f'F4 fdx {nm}'): continue
        R = stream_stats(nm)['R']; rr = [r['round'] for r in d['rounds']]
        if d.get('maintenance_source') != 'measured' or rr != list(range(R + 1)): raise SystemExit(f'fdx {nm}: rounds {rr} incomplete')
        executed = float(sum(r['rebuild_s'] for r in d['rounds'] if r['round'] >= 1))
        same = float(np.mean([r['rebuild_s'] for r in d['rounds'] if 'sweep' in r])) * R
        paper = [float(np.mean([r['rebuild_s'] for r in o['rounds'] if r.get('rebuild_s')])) * R for o in (runfile(x, nm, 'fdiskann_rebuild') for x in others) if o]
        res[nm] = {'R': R, 'executed_s': executed, 'estimate_same_run_s': same, 'estimate_same_run_err': same / executed - 1,
                   'estimate_paper_s': paper, 'estimate_paper_err': [x / executed - 1 for x in paper]}
    say(f'== F4 executed rebuilds vs the estimate: {res}'); return res

# ------------------------------------------------------------------------------------------------ I18
def i18_amz():
    F = frontier_for(('amz_time',), ['results/bench_amz'], expected=AMZ_METHODS)
    if not F: return None
    ax = json.load(open('results/bench_r_summary.json'))['FRONTIER']['0.95']['arxiv_time']['best']
    part_up_to = lambda best: max([float(q) for q, m in best.items() if m == 'ivf_part'] or [0])
    res = {'frontier': F, 'ivf_part_cheapest_up_to': {'amz_time': part_up_to(F['amz_time']['best']), 'arxiv_time': part_up_to(ax)}}
    res['prediction_holds'] = res['ivf_part_cheapest_up_to']['amz_time'] > res['ivf_part_cheapest_up_to']['arxiv_time']
    say(f'== I18a Amazon: {res["ivf_part_cheapest_up_to"]}, prediction holds: {res["prediction_holds"]}'); return res

def i18_multi(t=0.95):
    nm = 'yfcc_stream'; res = {}
    for m in ('scan', 'ivf', 'postfilter', 'ema', 'acorn'):
        d = runfile('results/bench_yst', nm, m)
        if not need(d is not None, f'I18b {m}'): continue
        rows = [r for r in d['rounds'] if 'sweep' in r]
        res[m] = {'maxr': {r['round']: max(s['recall'] for s in r['sweep']) for r in rows},
                  'qps095': {r['round']: qps_at(frontier([(s['recall'], s['qps']['mean']) for s in r['sweep']]), t) for r in rows},
                  'qps090': {r['round']: qps_at(frontier([(s['recall'], s['qps']['mean']) for s in r['sweep']]), 0.9) for r in rows},
                  'maint': d['rounds'][-1].get('cum_maint_s'), 'invalid': max((s.get('invalid_queries', 0) for r in rows for s in r['sweep']), default=0)}
    say(f'== I18b multi-label stream: { {m: (v["maxr"], v["qps095"]) for m, v in res.items()} }'); return res

def i18_cc():
    """Follow-up 4, B1 (tag cc4): every method under one phase-fair reader-writer lock; the visibility checks cover the
    whole stream. (The first runs, tag cc, are superseded: results/bench_superseded/cc_unsynchronized.)"""
    res = {}; bad = collections.Counter()
    for nm in ('sift500k', 'sift500k_adv'):
        for m in ('ivf', 'ivf_part', 'postfilter', 'ema'):
            f = f'results/bench_cc4/stream_{nm}/cc_{m}.json'
            if not need(os.path.exists(f), f'F4 cc4 {nm} {m}'): continue
            d = json.load(open(f)); R = stream_stats(nm)['R']
            if d.get('lock') != 'rw' or [r['round'] for r in d['rounds']] != list(range(1, R + 1)): raise SystemExit(f'{f}: lock {d.get("lock")} or rounds incomplete')
            C = [r['concurrent'] for r in d['rounds']]; Qt = [r['quiet'] for r in d['rounds']]
            med = lambda xs, k: float(np.median([x[k] for x in xs if x.get(k) is not None]))
            for k in ('stale_ids', 'unborn_ids', 'out_of_range_ids'): bad[k] += sum(x[k] for x in C + Qt)
            bad['inadmissible_ids'] += int(round(sum(x['inadmissible_share'] * x['returned_ids'] for x in C + Qt)))
            res[f'{nm}/{m}'] = {'param': d.get('param'), 'writer_s_median': float(np.median([r['writer_s'] for r in d['rounds']])), 'conc_p50_ms': med(C, 'p50_ms'), 'conc_p99_ms': med(C, 'p99_ms'),
                                'quiet_p99_ms': med(Qt, 'p99_ms'), 'p99_ratio_median': float(np.median([c['p99_ms'] / q['p99_ms'] for c, q in zip(C, Qt)])),
                                'lock_wait_p99_ms': med(C, 'lock_wait_p99_ms'), 'conc_qps': med(C, 'qps'), 'conc_queries_median': med(C, 'queries'),
                                'stale_ids': sum(x['stale_ids'] for x in C + Qt), 'unborn_ids': sum(x['unborn_ids'] for x in C + Qt),
                                'recall_best_median': med(C, 'recall_best'), 'quiet_recall_median': med(Qt, 'recall_end'), 'rounds': len(d['rounds'])}
    res['_totals'] = dict(bad); res['_no_invalid_result'] = bool(res and not any(bad.values()))
    say(f'== F4 concurrency (cc4): {res}'); return res

def i18_simd():
    """sqrt(n) rule calibrated at the default level (AVX-512), corrected by the microbenchmark ratio of each level"""
    H = json.load(open('results/bench_h_summary.json')); m0 = H['H1']['per_n']['0.95']['deep1m']['ivf']['m_star']
    mb0 = json.load(open('results/bench_mb/cost.json'))['96']['chk_over_dist']; res = {}
    for lv, tag in (('none', 'simdn'), ('avx2', 'simd2')):
        if not need(os.path.isdir(f'results/bench_{tag}') and os.path.exists(f'results/bench_mb/cost_{lv}.json'), f'I18d SIMD {lv}'): continue
        mb = json.load(open(f'results/bench_mb/cost_{lv}.json'))['96']['chk_over_dist']; corr = math.sqrt(mb / mb0)
        S = summarize_static(load_roots([f'results/bench_{tag}']))[1]; rows, fit = crossovers(S, NS, '0.95', meths=('ivf',))
        T = []
        for ds, n in NS.items():
            for wl in [w for w in S if w.startswith(ds + '/b0.0_s')]:
                sc, iv = S[wl].get('scan_part', {}).get('qps@0.95'), S[wl].get('ivf', {}).get('qps@0.95')
                if sc: T.append({'n': n, 'm': n / L_of(wl.split('/')[1]), 'c_scan': 1 / sc, 'c_ivf': (1 / iv) if iv else math.inf})
        out = {'chk_over_dist': mb, 'correction': corr, 'ivf_exponent': fit.get('ivf'), 'm_star': {ds: r['ivf']['m_star'] for ds, r in rows.items() if 'ivf' in r}}
        for name, f in (('sqrt(n), uncorrected', lambda w: m0 * math.sqrt(w['n'] / 1e6)), ('sqrt(n), microbenchmark corrected', lambda w: m0 * math.sqrt(w['n'] / 1e6) * corr),
                        ('Milvus 7%', lambda w: 0.07 * w['n'])):
            wrong = 0; reg = []
            for w in T:
                ch = w['c_scan'] if w['m'] < f(w) else w['c_ivf']; best = min(w['c_scan'], w['c_ivf']); reg.append(ch / best - 1); wrong += ch > best * (1 + 1e-9)
            out[name] = {'wrong': wrong, 'workloads': len(T), 'regret_max': float(max(reg)) if reg else None}
        res[lv] = out
    say(f'== I18d SIMD levels: {res}'); return res

# ------------------------------------------------------------------------------------------------ I27
def i27():
    if not need(glob.glob('results/bench_rep_*/'), 'I27 re-execution outputs'): return None
    subprocess.run([sys.executable, 'bench/reexec_compare.py'], check=True)
    s = json.load(open('results/bench_rep_summary.json')); s.pop('rows', None); return s

if __name__ == '__main__':
    for key, fn in (('I07', i07), ('I13_seeds', i13_seeds), ('I13_policy', i13_policy_unseen), ('I13_timing', i13_timing_t), ('I13_best_of_two', i13_best_of_two),
                    ('I13_heldout', i13_heldout), ('I13_e5', i13_e5_replicate), ('I13_repair_seeds', i13_repair_seeds), ('I17', i17), ('I19', i19),
                    ('I18_amz', i18_amz), ('I18_multi', i18_multi), ('I18_cc', i18_cc), ('I18_simd', i18_simd), ('I27', i27), ('F4_rebuild', f4_rebuild)):
        OUT[key] = fn()
    json.dump(OUT, open('results/bench_f2_summary.json' if not PARTIAL else 'results/bench_f2_summary_partial.json', 'w'), indent=1, default=float)

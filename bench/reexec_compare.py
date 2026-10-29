#!/usr/bin/env python3
"""Compare the re-execution sample with the original results (follow-up 2, I27; paper_audit/followup2_prereg.md).
  python3 bench/reexec_compare.py      -> results/bench_rep_summary.json (and a printed table)
Every JSON file under results/bench_rep_<tag>[_pass<k>] is matched to the original file of the same relative path under
results/bench_<tag>[_pass<k>] (results/bench[_pass<k>] for tag main). Operating points are matched by parameter (static
files: the record list; stream files: round and parameter).
  - recall of deterministic methods (scans, IVF, label-partitioned IVF, RIG): must be identical (|diff| < 1e-9);
  - recall of the others: within max(0.01, that method's largest pass-1 vs pass-2 difference in the main campaign);
  - QPS: rerun / original, reported as a distribution per method family;
  - verdicts: for every pair of methods re-run together on one workload, which is faster at recall 0.95, with the
    paper's operating-point function (Pareto frontier, log-linear QPS interpolation: analyze.frontier/qps_at; the first
    version interpolated QPS linearly over raw points, CHECK.md B12); the share of pairs whose winner agrees.
  - jobs: every rerun line of results/bench_rep_manifest.json must have finished with exit code 0 in the queue's done
    file (results/bench/h.done) before the comparison is made.
Not a hardware replication: same machine, same data, later date."""
import os, re, sys, json, glob, hashlib, collections
import numpy as np
sys.path.insert(0, 'bench')
from analyze import frontier, qps_at

DET = {'scan_part', 'scan_mask', 'scan', 'ivf', 'ivf_part', 'rig', 'scan_part_pq'}
def load(p):
    """None if the file does not exist; a file that exists but does not parse is an error, not a missing result."""
    if not os.path.exists(p): return None
    with open(p) as f: return json.load(f)
def points(d):
    """-> {key: (recall, qps)} for a static record list or a stream result."""
    out = {}
    if isinstance(d, list):
        for r in d:
            if isinstance(r, dict) and 'recall' in r and 'qps' in r:
                q = r['qps']['mean'] if isinstance(r['qps'], dict) else r['qps']; out[('', str(r.get('param')), str(r.get('threads', '')))] = (r['recall'], q)
    elif isinstance(d, dict) and 'rounds' in d:
        for r in d['rounds']:
            for s in r.get('sweep', []):
                q = s['qps']['mean'] if isinstance(s['qps'], dict) else s['qps']; out[(str(r['round']), s['param'], '')] = (s['recall'], q)
    return out
def method_of(path, d):
    if isinstance(d, list) and d and isinstance(d[0], dict): return d[0].get('method', os.path.basename(path)[:-5])
    if isinstance(d, dict): return d.get('method', os.path.basename(path)[:-5])
    return os.path.basename(path)[:-5]
def q_at(pts, t=0.95, rnd=None):
    return qps_at(frontier([(r, q) for (rr, _, _), (r, q) in pts.items() if rnd is None or rr == rnd]), t)

# a sampled job that failed and was re-queued with a changed line, with the reason (paper_audit/followup2_results.md)
RETRIES = {'BENCH_TAG=rep_main BENCH_PASS=1 taskset -c 0-27,56-83 python3 bench/run_acorn.py yfcc10m_test tags gamma=1 M=32 Mb=32':
           ('BENCH_TAG=rep_main BENCH_PASS=1 REGEN=1 taskset -c 0-27,56-83 python3 bench/run_acorn.py yfcc10m_test tags gamma=1 M=32 Mb=32',
            "the float copy of YFCC-10M's vectors had been deleted for disk space; REGEN=1 rebuilds it from the u8bin file")}
def job_status():
    """(jobs in the manifest, jobs finished with exit code 0, the others): a job's done key is the SHA-1 prefix of its
    queue line plus its occurrence number (bench/queue2.sh); a line re-queued after contention counts by its last run."""
    M = json.load(open('results/bench_rep_manifest.json')); lines = [l.rstrip('\n') for l in open('bench/queue_h.txt')]
    done = {}
    for l in open('results/bench/h.done'):
        k, rc = l.split()[:2]; done[k] = int(rc)
    keyed, seen = collections.defaultdict(list), collections.Counter()
    for l in lines:
        if not l or l.startswith('#'): continue
        h = hashlib.sha1(l.encode()).hexdigest()[:16]; seen[h] += 1; keyed[l].append(f'{h}.{seen[h]}')
    ok, bad, retried = 0, [], []
    for j in M['jobs']:
        rcs = [done.get(k) for k in keyed.get(j['rerun'], [])]; rcs = [r for r in rcs if r is not None]
        if rcs and rcs[-1] == 0: ok += 1; continue
        if j['rerun'] in RETRIES:
            line, why = RETRIES[j['rerun']]; r2 = [done.get(k) for k in keyed.get(line, []) if done.get(k) is not None]
            if r2 and r2[-1] == 0: ok += 1; retried.append({'rerun': j['rerun'], 'exit_codes': rcs, 'retry': line, 'retry_exit_codes': r2, 'reason': why}); continue
        bad.append({'rerun': j['rerun'], 'exit_codes': rcs})
    return len(M['jobs']), ok, bad, retried

# pass spread of non-deterministic methods in the main campaign
spread = collections.defaultdict(float)
for p1 in glob.glob('results/bench/*/*/*.json'):
    p2 = p1.replace('results/bench/', 'results/bench_pass2/', 1)
    if not os.path.exists(p2): continue
    a, b = load(p1), load(p2); pa, pb = points(a), points(b); m = method_of(p1, a)
    for k in set(pa) & set(pb): spread[m] = max(spread[m], abs(pa[k][0] - pb[k][0]))
rows = []; verdicts = []; missing = []
for rd in sorted(glob.glob('results/bench_rep_*')):
    if not os.path.isdir(rd): continue
    m_ = re.match(r'results/bench_rep_(.+?)(_pass\d+)?$', rd); tag, ps = m_.group(1), m_.group(2) or ''
    od = f'results/bench{ps}' if tag == 'main' else f'results/bench_{tag}{ps}'
    perwl = collections.defaultdict(dict)
    for f in glob.glob(f'{rd}/**/*.json', recursive=True):
        rel = os.path.relpath(f, rd); of = os.path.join(od, rel); a, b = load(of), load(f)
        if a is None: missing.append(rel); continue
        pa, pb = points(a), points(b); m = method_of(f, b); common = sorted(set(pa) & set(pb))
        if not common and tag == 'main':
            # 28-thread jobs whose queue passed BENCH_TAG through the controller's environment (it is not in the logged
            # command): their outputs are under results/bench_mt28_pool or results/bench_mt28
            for alt in (f'results/bench_mt28_pool{ps}', f'results/bench_mt28{ps}'):
                a2 = load(os.path.join(alt, rel))
                if a2 is not None and set(points(a2)) & set(pb): a, pa = a2, points(a2); common = sorted(set(pa) & set(pb)); of = os.path.join(alt, rel); break
        if not common: missing.append(rel); continue
        dr = [pb[k][0] - pa[k][0] for k in common]; qr = [pb[k][1] / pa[k][1] for k in common if pa[k][1] and pb[k][1]]
        tol = 1e-9 if m in DET else max(0.01, spread.get(m, 0.0))
        rows.append({'campaign': f'{tag}{ps}' if 'bench_mt28' not in of else of.split('results/bench_')[1].split('/')[0], 'superseded': False, 'file': rel, 'method': m, 'points': len(common), 'deterministic': m in DET,
                     'recall_absdiff_max': float(max(abs(x) for x in dr)), 'recall_tol': tol, 'recall_ok': bool(max(abs(x) for x in dr) <= tol),
                     'qps_ratio_median': float(np.median(qr)) if qr else None, 'qps_ratio_min': float(min(qr)) if qr else None, 'qps_ratio_max': float(max(qr)) if qr else None})
        perwl[os.path.dirname(rel)][m] = (pa, pb, isinstance(b, dict) and 'rounds' in b)
    for wl, ms in perwl.items():
        names = sorted(ms)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                (a1, b1, st), (a2, b2, _) = ms[names[i]], ms[names[j]]
                rnds = sorted({k[0] for k in b1} & {k[0] for k in b2}) if st else [None]
                for rr in rnds:
                    oa, ob, ra, rb = q_at(a1, rnd=rr), q_at(a2, rnd=rr), q_at(b1, rnd=rr), q_at(b2, rnd=rr)
                    if None in (oa, ob, ra, rb): continue
                    verdicts.append({'workload': f'{tag}{ps}/{wl}', 'pair': [names[i], names[j]], 'round': rr, 'orig_ratio': oa / ob, 'rerun_ratio': ra / rb, 'agree': (oa > ob) == (ra > rb)})
fam = collections.defaultdict(list)
for r in rows:
    if r['qps_ratio_median']: fam[r['method']].append(r['qps_ratio_median'])
n_jobs, n_ok, bad_jobs, retried_jobs = job_status()
if bad_jobs: raise SystemExit(f'{len(bad_jobs)} re-execution jobs did not finish with exit code 0: {bad_jobs[:3]}')
summ = {'jobs': n_jobs, 'jobs_ok': n_ok, 'jobs_retried': retried_jobs, 'files': len(rows), 'unmatched_files': missing,
        'recall_ok': sum(r['recall_ok'] for r in rows), 'recall_ok_deterministic': [sum(r['recall_ok'] for r in rows if r['deterministic']), sum(r['deterministic'] for r in rows)],
        'recall_ok_other': [sum(r['recall_ok'] for r in rows if not r['deterministic']), sum(not r['deterministic'] for r in rows)],
        'qps_ratio_by_method': {m: {'files': len(v), 'median': float(np.median(v)), 'min': float(min(v)), 'max': float(max(v))} for m, v in fam.items()},
        'qps_ratio_all': {'median': float(np.median([x for v in fam.values() for x in v])) if fam else None,
                          'p10_p90': np.quantile([x for v in fam.values() for x in v], [0.1, 0.9]).tolist() if fam else None},
        'verdicts': {'pairs': len(verdicts), 'agree': sum(v['agree'] for v in verdicts),
                     'disagree': [v for v in verdicts if not v['agree']]},
        'pass_spread_used': dict(spread), 'rows': rows}
json.dump(summ, open('results/bench_rep_summary.json', 'w'), indent=1, default=float)
print(f"jobs {n_jobs} (exit 0: {n_ok}); files {len(rows)}: recall within tolerance {summ['recall_ok']} (deterministic {summ['recall_ok_deterministic']}, other {summ['recall_ok_other']}); "
      f"QPS rerun/original median {summ['qps_ratio_all']['median']}; verdicts agree {summ['verdicts']['agree']}/{summ['verdicts']['pairs']}; unmatched {len(missing)}")
for r in rows:
    if not r['recall_ok']: print('  recall outside tolerance:', r['campaign'], r['file'], round(r['recall_absdiff_max'], 4), 'tol', r['recall_tol'])

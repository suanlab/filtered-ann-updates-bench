#!/usr/bin/env python3
"""Re-execution sample (follow-up 2, I27; paper_audit/followup2_prereg.md).
  python3 bench/reexec_sample.py      -> bench/queue_rep.txt, results/bench_rep_manifest.json
From every queue log, the measurement jobs that exited 0 before the pre-registration (2026-10-24 22:21 KST) are grouped
by campaign (BENCH_TAG, or 'main', and BENCH_PASS); 15% of each campaign, at least one, is drawn with
random.Random(2027) over the jobs in log order. Data preparation and superseded campaigns are excluded. Each drawn job is
re-run with BENCH_TAG=rep_<campaign tag> (same BENCH_PASS), so its outputs land in results/bench_rep_<tag>[_pass<k>]
next to the originals; the pinning is normalized to the current protocol's cores (0-27,56-83) unless the job pinned
physical cores only (0-27, the 28-thread runs). bench/reexec_compare.py compares the outputs."""
import re, json, random, datetime, math, collections
LOGS = ['queue', 'round3', 'fix', 'medium', 'curator_yfcc', 'h']
CUTOFF = datetime.datetime(2026, 10, 24, 22, 21, 42)
SUPERSEDED = {'r2', 'mv0'}          # r2: E5 runner with the candidate-shortage bug (results/bench_superseded/e5_candidate_shortage)
jobs = []
for lg in LOGS:
    start = {}; L = open(f'results/bench/{lg}.log', errors='ignore').read().split('\n')
    for l in L:
        m = re.match(r'=== \[(\d+)[^\]]*\] (\S+ \S+) (.*)$', l)
        if m: start[m.group(1)] = (datetime.datetime.fromisoformat(m.group(2)), m.group(3)); last = m.group(1); continue
        m = re.match(r'--- \[(\d+)\] exit=(\d+)', l)
        if m and m.group(1) in start:
            t, cmd = start.pop(m.group(1))
            if m.group(2) == '0' and t < CUTOFF: jobs.append({'log': lg, 'n': int(m.group(1)), 'start': t.isoformat(), 'cmd': cmd})
def campaign(cmd):
    tag = re.search(r'BENCH_TAG=(\S+)', cmd); ps = re.search(r'BENCH_PASS=(\d+)', cmd)
    return (tag.group(1) if tag else 'main', int(ps.group(1)) if ps else 1)
prep = re.compile(r'bench/(prep|stream_prep|fetch_|prep_snapshot|make_queue|workload_stats|microbench)|^python3 -m|setup_|download')
groups = collections.OrderedDict(); seen = set()
for j in jobs:
    if prep.search(j['cmd']): continue
    c = campaign(j['cmd'])
    if c[0] in SUPERSEDED or c[0].startswith('smoke'): continue
    if j['cmd'] in seen: continue          # a re-run of a contended job: keep one instance
    seen.add(j['cmd']); groups.setdefault(c, []).append(j)
rng = random.Random(2027); out = []; manifest = []
for c, js in groups.items():
    k = max(1, math.ceil(0.15 * len(js))); pick = sorted(rng.sample(range(len(js)), k))
    for i in pick:
        cmd = js[i]['cmd']; tag, ps = c
        new = re.sub(r'BENCH_TAG=\S+ ?', '', cmd).strip()
        new = re.sub(r'taskset -c 0-31\b', 'taskset -c 0-27,56-83', new)
        new = f'BENCH_TAG=rep_{tag} ' + new
        out.append(new); manifest.append({'campaign': f'{tag}/pass{ps}', 'original': cmd, 'rerun': new, 'log': js[i]['log'], 'n': js[i]['n'], 'started': js[i]['start']})
open('bench/queue_rep.txt', 'w').write('# I27 re-execution sample (bench/reexec_sample.py)\n' + '\n'.join(out) + '\n')
json.dump({'cutoff': CUTOFF.isoformat(), 'campaigns': {f'{t}/pass{p}': len(v) for (t, p), v in groups.items()}, 'jobs': manifest},
          open('results/bench_rep_manifest.json', 'w'), indent=1)
print(len(groups), 'campaigns,', sum(len(v) for v in groups.values()), 'jobs,', len(out), 'sampled')

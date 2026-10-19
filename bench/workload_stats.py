#!/usr/bin/env python3
"""Cache the workload statistics the analysis needs in results/workload_stats.json (main env, needs data/bench).
  python3 bench/workload_stats.py
Stores the admissible-set size of every query of each csr workload and each stream's meta.json with its mean
updates per round, so that bench/analyze.py and paper/make_eab_*.py run on the released results alone."""
import sys, os, json, glob
sys.path.insert(0, 'bench')
from common import ROOT, STATS, admissible_sizes, stream_stats, log

out = {'admissible_sizes': {}, 'streams': {}}
for p in sorted(glob.glob(f'{ROOT}/*/*/pred.npz')):
    wl = '/'.join(p.split('/')[-3:-1]); a = admissible_sizes(wl)
    if a is not None: out['admissible_sizes'][wl] = [int(x) for x in a]
for d in sorted(glob.glob(f'{ROOT}/stream_*/meta.json')):
    nm = d.split('/')[-2][len('stream_'):]; out['streams'][nm] = stream_stats(nm)
json.dump(out, open(STATS, 'w'), indent=1)
log(len(out['admissible_sizes']), 'csr workloads and', len(out['streams']), 'streams written to', STATS)

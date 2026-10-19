#!/usr/bin/env python3
"""Static snapshot of an update stream's live set after a round (main env).
  python3 bench/prep_snapshot.py <stream> <round>   ->  data/bench/<stream>_r<round>/ with schema 'live'
The snapshot holds exactly the points live after that round, their labels, the stream's queries and labels, and
the stream's ground truth for that round mapped to snapshot row ids. Indexes built fresh on it answer the same
queries over the same live data as the stream's indexes at that round, which isolates the effect of the update
path (e.g. ACORN's deleted nodes kept as routers) from that of the data and label distribution."""
import sys, os, json
sys.path.insert(0, 'bench')
import numpy as np
from common import ROOT, log

name, r = sys.argv[1], int(sys.argv[2]); S = f'{ROOT}/stream_{name}'; meta = json.load(open(f'{S}/meta.json'))
X = np.load(f'{S}/Xall.npy', mmap_mode='r'); sch = np.load(f'{S}/sched.npz'); attr = np.load(f'{S}/attr.npy')
alive = np.zeros(len(X), bool); alive[:meta['N0']] = True
for k in range(r): alive[sch[f'del{k}']] = False; alive[sch[f'ins{k}']] = True
live = np.where(alive)[0]; new = np.full(len(X), -1, np.int64); new[live] = np.arange(len(live))
gt = np.load(f'{S}/gt.npy')[r]; gt = np.where(gt >= 0, new[np.maximum(gt, 0)], -1).astype(np.int32)
assert (gt[gt >= 0] >= 0).all()
D = f'{ROOT}/{name}_r{r}'; os.makedirs(f'{D}/live', exist_ok=True)
np.save(f'{D}/X.npy', np.ascontiguousarray(X[live], np.float32)); np.save(f'{D}/Q.npy', np.load(f'{S}/Q.npy'))
json.dump({'source': f'live set of stream_{name} after round {r}', 'metric': 'l2', 'n': int(len(live)), 'd': int(X.shape[1]),
           'nq': int(len(gt))}, open(f'{D}/meta.json', 'w'), indent=1)
np.savez(f'{D}/live/pred.npz', kind='label', attr=attr[live].astype(np.int32), qlab=np.load(f'{S}/qlab.npy'))
np.save(f'{D}/live/gt.npy', gt)
log(f'{name}_r{r}: {len(live)} live points; labels per class {np.bincount(attr[live], minlength=10).tolist()}')

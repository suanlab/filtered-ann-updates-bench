#!/usr/bin/env python3
"""ACORN-gamma (AVX2 build: the fork's CMake declares FAISS_OPT_LEVEL with option(), so the documented
build silently compiles generic scalar distance kernels; we build with -DFAISS_OPT_LEVEL=avx2) stream arm (main env; wraps src/acorn_stream, patched ACORN).
  taskset -c <cores> python3 bench/stream_acorn.py <name>
ACORN's own dynamic path: incremental IndexACORN::add for inserts; deletions excluded through the
per-query predicate bitmap (alive AND label), i.e. deleted nodes stay in the graph as routers.
Build/insert 16 threads; search 1 thread, REPS reps, ef sweep at the measured rounds."""
import sys, os, json, subprocess, tempfile
sys.path.insert(0, 'bench')
import numpy as np
from common import ROOT, RES, recall, ci, log, REPS, K, write_fbin

name = sys.argv[1]; D = f'{ROOT}/stream_{name}'; meta = json.load(open(f'{D}/meta.json')); R, N0 = meta['R'], meta['N0']
X = np.load(f'{D}/Xall.npy', mmap_mode='r'); Q = np.load(f'{D}/Q.npy'); gt = np.load(f'{D}/gt.npy'); sch = np.load(f'{D}/sched.npz'); nq = len(Q)
tmp = tempfile.mkdtemp(dir=ROOT); os.makedirs(f'{tmp}/sched'); os.makedirs(f'{tmp}/out')
fbin = write_fbin
fbin(f'{tmp}/X.fbin', X); fbin(f'{tmp}/Q.fbin', Q)
np.load(f'{D}/attr.npy').astype(np.int32).tofile(f'{tmp}/attr.i32'); np.load(f'{D}/qlab.npy').astype(np.int32).tofile(f'{tmp}/qlab.i32')
for r in range(R):
    sch[f'del{r}'].astype(np.int32).tofile(f'{tmp}/sched/del{r}.i32'); sch[f'ins{r}'].astype(np.int32).tofile(f'{tmp}/sched/ins{r}.i32')
efs = [16, 32, 64, 128, 256, 512, 1024, 2048]
cmd = [os.environ.get('ACORN_BIN', './src/acorn_stream_avx2'), f'{tmp}/X.fbin', f'{tmp}/Q.fbin', f'{tmp}/attr.i32', f'{tmp}/qlab.i32', f'{tmp}/sched', str(N0), str(R),
       '32', '12', '64', '16', '1', ','.join(map(str, efs)), f'{tmp}/out']
env = {**os.environ, 'ACORN_REPS': str(REPS), 'MEASURE': ','.join(map(str, meta['measure_rounds']))}
log(name, 'acorn stream start')
p = subprocess.run(cmd, capture_output=True, text=True, env=env)
if p.returncode != 0: log('acorn_stream failed', p.returncode, p.stderr[-1500:]); sys.exit(1)
out = {'method': 'acorn', 'rounds': []}
for line in p.stdout.splitlines():
    if not line.startswith('{'): continue
    ev = json.loads(line)
    if ev['event'] == 'build': out['build_s'] = ev['build_s']; continue
    r = ev['round']; row = {k: ev[k] for k in ('round', 't_delete', 't_insert', 'cum_maint_s') if k in ev}; row['resident'] = ev['ntotal']
    if 'graph_bytes' in ev:
        row['bytes'] = int(ev['ntotal'] * X.shape[1] * 4 + ev['graph_bytes']); row['live'] = ev['live']; row['sweep'] = []
        row['bytes_has_levels'] = bool(ev.get('graph_bytes_has_levels'))
        for ef in efs:
            o = np.fromfile(f'{tmp}/out/acorn_r{r}_ef{ef}.i32', np.int32).reshape(nq, K)
            row['sweep'].append({'param': f'ef={ef}', 'recall': recall(o, gt[r]), 'qps': ci(ev[f'qps_ef{ef}'])})
        log(name, 'acorn round', r, 'cum', row['cum_maint_s'], [(s['param'], round(s['recall'], 4), round(s['qps']['mean'], 1)) for s in row['sweep']])
    out['rounds'].append(row)
os.makedirs(f'{RES}/stream_{name}', exist_ok=True)
json.dump(out, open(f'{RES}/stream_{name}/acorn.json', 'w'), indent=1, default=float)
subprocess.run(['rm', '-rf', tmp]); log('DONE', name, 'acorn')

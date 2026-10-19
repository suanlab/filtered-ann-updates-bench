#!/usr/bin/env python3
"""ACORN-gamma (AVX2 build: the fork's CMake declares FAISS_OPT_LEVEL with option(), so the documented
build silently compiles generic scalar distance kernels; we build with -DFAISS_OPT_LEVEL=avx2) arm (main env, wraps src/acorn_driver; patched ACORN, see patches/).
  taskset -c <cores> python3 bench/run_acorn.py <ds> <schema>[+<schema>...] [gamma=12] [M=32] [Mb=64]
The graph is predicate-agnostic: built once per dataset (32 threads), then searched (1 thread, REPS
timed repetitions after a warm-up) for every schema with the predicate as a per-query bitmap."""
import sys, os, json, subprocess, tempfile
sys.path.insert(0, 'bench')
import numpy as np
from common import load_base, load_schema, recall, per_query, ci, save, log, ds_dir, REPS, K, THREADS, write_fbin

ds = sys.argv[1]; schemas = sys.argv[2].split('+'); kw = dict(a.split('=') for a in sys.argv[3:])
gamma, M, Mb = int(kw.get('gamma', 12)), int(kw.get('M', 32)), int(kw.get('Mb', 64))
method = {12: 'acorn', 1: 'acorn1'}.get(gamma, f'acorn_g{gamma}')
X, Q, meta = load_base(ds); n, d = X.shape; nq = len(Q)
tmp = tempfile.mkdtemp(dir='data/bench'); out = f'{tmp}/out'; os.makedirs(out)
fbin = write_fbin
fbin(f'{tmp}/X.fbin', X); fbin(f'{tmp}/Q.fbin', Q)
jobs = []
for sc in schemas:
    pred, _ = load_schema(ds, sc)
    if str(pred['kind']) == 'label':
        pred['attr'].astype(np.int32).tofile(f'{tmp}/{sc}.attr'); pred['qlab'].astype(np.int32).tofile(f'{tmp}/{sc}.qlab')
        jobs.append(f'{tmp}/{sc}.attr:{tmp}/{sc}.qlab:{sc}')
    else:
        pred['qoff'].astype(np.int64).tofile(f'{tmp}/{sc}.qoff'); pred['qids'].astype(np.int32).tofile(f'{tmp}/{sc}.qids')
        jobs.append(f'csr:{tmp}/{sc}.qoff:{tmp}/{sc}.qids:{sc}')
efs = '16,32,64,128,256,512,1024,2048'
cmd = [os.environ.get('ACORN_BIN', './src/acorn_driver_avx2'), f'{tmp}/X.fbin', f'{tmp}/Q.fbin', str(M), str(gamma), str(Mb), f'32:{THREADS}', efs, out] + jobs
log(ds, method, 'running', ' '.join(cmd[:8]))
p = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, 'ACORN_REPS': str(REPS)})
if p.returncode != 0: log('acorn_driver failed', p.returncode, p.stderr[-2000:]); sys.exit(1)
build = None; recs = {sc: [] for sc in schemas}
for line in p.stdout.splitlines():
    if not line.startswith('{'): continue
    ev = json.loads(line)
    if ev['event'] == 'build': build = ev['build_s']; gbytes = ev['graph_bytes']
    if ev['event'] == 'search':
        sc, ef = ev['tag'], ev['ef']; _, gt = load_schema(ds, sc)
        o = np.fromfile(f'{out}/acorn_{sc}_ef{ef}.i32', np.int32).reshape(nq, K)
        nbytes = int(n * d * 4 + gbytes)
        tpath = f'{out}/acorn_{sc}_ef{ef}.t'; tq = np.fromfile(tpath, np.float64) if os.path.exists(tpath) else None
        recs[sc].append({'method': method, 'param': f'ef={ef}', 'recall': recall(o, gt), 'qps': ci(ev['qps_reps']),
                         'threads': THREADS, 'build_s': build, 'bytes': nbytes, 'extra': {'M': M, 'gamma': gamma, 'M_beta': Mb},
                         **per_query(o, gt, tq)})
        log(ds, sc, method, ef, round(recs[sc][-1]['recall'], 4), round(recs[sc][-1]['qps']['mean'], 1))
for sc in schemas: save(ds, sc, method, recs[sc])
subprocess.run(['rm', '-rf', tmp]); log('DONE', ds, method)

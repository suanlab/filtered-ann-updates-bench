#!/usr/bin/env python3
"""Curator stream arm (.venv-curator).  taskset -c <cores> .venv-curator/bin/python bench/stream_curator.py <name>
Curator's own update path: insertions add the vector to the shared tree and grant its label; deletions revoke the
label (the released code rejects remove_vector, so a deleted vector stays in the tree but no label search can
return it). Maintenance is timed every round (the update calls are single-threaded); search sweeps search_ef
(1 thread, one call per label) at the measured rounds. Parameters as in bench/run_curator.py."""
import sys, os, json, time
sys.path.insert(0, 'bench')
import numpy as np, faiss
if os.environ.get('E2E') == '1': raise SystemExit(f'E2E=1 is not supported by {sys.argv[0]}')   # CHECK.md B05: never ignored silently
from common import ROOT, RES, recall, timed_search, log, K

name = sys.argv[1]; D = f'{ROOT}/stream_{name}'; meta = json.load(open(f'{D}/meta.json')); R, N0 = meta['R'], meta['N0']
X = np.load(f'{D}/Xall.npy', mmap_mode='r'); Q = np.load(f'{D}/Q.npy'); attr = np.load(f'{D}/attr.npy'); qlab = np.load(f'{D}/qlab.npy')
gt = np.load(f'{D}/gt.npy'); sch = np.load(f'{D}/sched.npz'); nq = len(Q); d = X.shape[1]
G = [(b, np.where(qlab == b)[0]) for b in range(10)]
faiss.omp_set_num_threads(16); t0 = time.time()
ix = faiss.MultiTenantIndexIVFHierarchical(d, 32, faiss.METRIC_L2, 1000, 0.01, 256, 20, 256, 3000, 1.6, 0.4, 0, 8)
X0 = np.ascontiguousarray(X[:N0], np.float32); ix.train(X0, 0); ix.add_vector_with_ids(X0, np.arange(N0, dtype=np.int64))
for i in range(N0): ix.grant_access(i, int(attr[i]))
out = {'method': 'curator', 'build_s': time.time() - t0, 'rounds': []}; cum = 0.0; resident = N0
for r in range(R + 1):
    row = {'round': r, 't_delete': 0.0, 't_insert': 0.0}
    if r > 0:
        de, ins = sch[f'del{r-1}'], sch[f'ins{r-1}']
        t = time.time()
        for x in de: ix.revoke_access(int(x), int(attr[x]))
        row['t_delete'] = time.time() - t; t = time.time()
        ix.add_vector_with_ids(np.ascontiguousarray(X[ins], np.float32), ins.astype(np.int64))
        for x in ins: ix.grant_access(int(x), int(attr[x]))
        row['t_insert'] = time.time() - t; resident += len(ins); cum += row['t_delete'] + row['t_insert']
    row['cum_maint_s'] = cum
    if r in meta['measure_rounds']:
        faiss.omp_set_num_threads(1); row['resident'] = resident; row['bytes'] = int(resident * d * 4); sweep = []
        for ef in (32, 64, 128, 256, 512, 1024, 2048, 4096):
            ix.search_ef = ef
            def run():
                o = np.full((nq, K), -1, np.int32); tot = 0.0
                for b, qi in G:
                    t = time.perf_counter(); _, I = ix.search(Q[qi], K, int(b)); tot += time.perf_counter() - t; o[qi] = I
                return o, tot
            o, q = timed_search(run, nq); sweep.append({'param': f'search_ef={ef}', 'recall': recall(o, gt[r]), 'qps': q})
            if sweep[-1]['recall'] > 0.999: break
        row['sweep'] = sweep; faiss.omp_set_num_threads(16)
        log(name, 'curator round', r, 'cum', round(cum, 1), [(s['param'], round(s['recall'], 4), round(s['qps']['mean'], 1)) for s in sweep])
    out['rounds'].append(row)
os.makedirs(f'{RES}/stream_{name}', exist_ok=True)
json.dump(out, open(f'{RES}/stream_{name}/curator.json', 'w'), indent=1, default=float); log('DONE', name, 'curator')

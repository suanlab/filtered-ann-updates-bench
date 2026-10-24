#!/usr/bin/env python3
"""EMA stream arm (.venv-ema: hashannlib built from github.com/lmccccc/EMA at d95fb49973f4fe53f5a2c1ac444dd5fdb879f551,
scripts/setup_ema.sh).   taskset -c <cores> .venv-ema/bin/python bench/stream_ema.py <name>
EMA (Li et al.): HNSW with per-edge Edge Markers that route filtered search, insertion with add_items, and synchronous
deletion with graph and Marker repair (delete_items, IP-DiskANN-style). Parameters as the authors' dynamic pipeline
(exp_benchmark/dynamic/runner.py): M=40, ef_construction=300, ft_bits=128, attr_type=[numeric, categorical] with a
constant numeric value, edge-level Markers, query configuration as their configure_index(). The representative level
plan follows their construction: the nearest initial row to each of 4 sqrt(N0) k-means centroids is placed on level 1;
inserted rows are level 0 (a deleted entry point is replaced by a live one inside EMA). The predicate is the label,
as a one-clause DNF. Build, deletion, and insertion use 16 threads (all counted as maintenance after the build,
including regenerating the attribute indexes before each measurement); search is one call for all 500 queries with
num_threads=1, ef swept, REPS reps."""
import sys, os, json, time
sys.path.insert(0, 'bench')
import numpy as np, faiss, hashannlib
from common import ROOT, RES, recall, timed_search, log, K

name = sys.argv[1]; D = f'{ROOT}/stream_{name}'; meta = json.load(open(f'{D}/meta.json')); R, N0 = meta['R'], meta['N0']
X = np.load(f'{D}/Xall.npy', mmap_mode='r'); Q = np.ascontiguousarray(np.load(f'{D}/Q.npy'), np.float32)
attr = np.load(f'{D}/attr.npy').astype(np.int64); qlab = np.load(f'{D}/qlab.npy').astype(np.int64)
gt = np.load(f'{D}/gt.npy'); sch = np.load(f'{D}/sched.npz'); n, d = X.shape; nq = len(Q); L = int(attr.max()) + 1
MAXR = min(R, int(os.environ.get('MAX_ROUND', R)))      # smoke tests only
rec = lambda ids: [[[0], [int(attr[i])]] for i in ids]

t0 = time.time(); X0 = np.ascontiguousarray(X[:N0], np.float32)
k = int(4 * np.sqrt(N0)); faiss.omp_set_num_threads(16)
km = faiss.Kmeans(d, k, niter=10, seed=1, max_points_per_centroid=50); km.train(X0)
fl = faiss.IndexFlatL2(d); fl.add(X0); _, I = fl.search(km.centroids, 1)
levels = np.zeros(N0, np.int32); levels[np.unique(I[:, 0])] = 1
idx = hashannlib.Index(space='l2', dim=d)
idx.init_index(max_elements=n, top_elements=int(levels.sum()), M=40, ef_construction=300, random_seed=100, ft_bits=128,
               attr_type=[0, 1], max_cate_size=L, allow_replace_deleted=False, edge_level_ft=True)
idx.initAttrMapping([[[0], sorted({int(attr[i]), i % L})] for i in range(N0)])   # every label assigned, as their runner
for s in range(0, N0, 100_000):
    e = min(N0, s + 100_000); idx.add_items(X0[s:e], rec(range(s, e)), np.arange(s, e, dtype=np.uint64), num_threads=16, levels=levels[s:e], replace_deleted=False)
del X0
def configure():
    t = time.time(); idx.generateAttrIndexes(); idx.set_thresholds(0.0001, 0.0001, 0.0001); idx.set_ef_top(1); idx.set_ft_flag(True)
    idx.set_ft_routing_flag(True); idx.set_ft_routing_min_deg(8); idx.set_ft_routing_backfill_tail(False); idx.set_num_threads(1)
    return time.time() - t
c0 = configure()
out = {'method': 'ema', 'build_s': time.time() - t0, 'commit': 'd95fb49973f4fe53f5a2c1ac444dd5fdb879f551', 'rounds': []}; cum = 0.0
alive = np.zeros(n, bool); alive[:N0] = True; preds = [[[[], [int(b)]]] for b in qlab]
for r in range(MAXR + 1):
    row = {'round': r, 't_delete': 0.0, 't_insert': 0.0, 't_configure': 0.0}
    if r > 0:
        de, ins = sch[f'del{r-1}'].astype(np.uint64), sch[f'ins{r-1}'].astype(np.int64); idx.set_num_threads(16)
        t = time.time(); idx.delete_items(de, num_threads=16); row['t_delete'] = time.time() - t
        t = time.time()
        for s in range(0, len(ins), 100_000):
            b = ins[s:s + 100_000]
            idx.add_items(np.ascontiguousarray(X[b], np.float32), rec(b), b.astype(np.uint64), num_threads=16, levels=np.zeros(len(b), np.int32), replace_deleted=False)
        row['t_insert'] = time.time() - t
        alive[de.astype(np.int64)] = False; alive[ins] = True
        if r in meta['measure_rounds']: row['t_configure'] = configure()
        cum += row['t_delete'] + row['t_insert'] + row['t_configure']
    row['cum_maint_s'] = cum
    if r in meta['measure_rounds']:
        sweep = []; row['resident'] = int(alive.sum())
        for ef in (16, 32, 64, 128, 256, 512, 1024, 2048):
            idx.set_ef(ef)
            def run():
                t = time.perf_counter(); lab, _ = idx.hybrid_knn_query_dnf(Q, preds, k=K, num_threads=1); tot = time.perf_counter() - t
                return np.asarray(lab).astype(np.int64), tot
            o, q = timed_search(run, nq)
            assert alive[o[o >= 0]].all() and all((attr[o[i][o[i] >= 0]] == qlab[i]).all() for i in range(nq)), 'EMA returned a dead or inadmissible id'
            sweep.append({'param': f'ef={ef}', 'recall': recall(o, gt[r]), 'qps': q})
            if sweep[-1]['recall'] > 0.999: break
        row['sweep'] = sweep
        log(name, 'ema round', r, 'cum', round(cum, 1), [(s['param'], round(s['recall'], 4), round(s['qps']['mean'], 1)) for s in sweep])
    out['rounds'].append(row)
os.makedirs(f'{RES}/stream_{name}', exist_ok=True)
json.dump(out, open(f'{RES}/stream_{name}/ema.json', 'w'), indent=1, default=float); log('DONE', name, 'ema')

#!/usr/bin/env python3
"""Cost of evaluating YFCC tag predicates with three implementations (main env, 1 thread; H5).
  taskset -c <core> python3 bench/predicate_cost.py      -> results/bench_h5/predicate_cost.json
  (OUT_DIR=<dir>, or a BENCH_TAG, redirects the output: common.output_dir; a tagged re-execution keeps the original)
  numpy    intersect the sorted posting lists of a query's tags, materialize the n-entry mask, pack it
           into a bitmap (the implementation whose cost the paper reports; bench/overhead.py)
  roaring  AND of per-tag Roaring bitmaps (pyroaring), then the admissible ids as an array
  dense    AND of precomputed per-tag dense bitmaps (n/8 bytes each), i.e. a bitmap index on the tags
Per-tag structures are built once, untimed (they are part of the attribute index, like the posting lists);
the timed part is the per-query work. Each query is repeated REPS times and the median kept."""
import sys, os, json, time
sys.path.insert(0, 'bench')
import numpy as np
from pyroaring import BitMap
from common import log, output_dir

REPS = 5
z = np.load('data/bench/yfcc10m/tags/pred.npz'); pp, pi = z['ptag_ptr'], z['ptag_idx']; qp, qi = z['qtag_ptr'], z['qtag_idx']; n = len(pp) - 1
rows = np.repeat(np.arange(n, dtype=np.int32), np.diff(pp)); order = np.argsort(pi, kind='stable'); ntag = int(pi.max()) + 1
tptr = np.zeros(ntag + 1, np.int64); np.cumsum(np.bincount(pi, minlength=ntag), out=tptr[1:]); tids = rows[order]; del rows, order
qtags = [qi[qp[q]:qp[q + 1]] for q in range(len(qp) - 1)]
used = np.unique(np.concatenate(qtags))
post = {int(t): tids[tptr[t]:tptr[t + 1]] for t in used}
roar = {t: BitMap(v) for t, v in post.items()}
def dense_of(ids):
    m = np.zeros(n, bool); m[ids] = True; return np.packbits(m, bitorder='little')
dense = {t: dense_of(v) for t, v in post.items()}
log(len(used), 'queried tags;', round(sum(v.nbytes for v in dense.values()) / 1e9, 2), 'GB of dense tag bitmaps')

def med(f):
    ts = []
    for _ in range(REPS):
        t = time.perf_counter(); r = f(); ts.append(time.perf_counter() - t)
    return float(np.median(ts)), r

out = {'numpy': [], 'roaring_and': [], 'roaring_ids': [], 'dense_and': [], 'ntags': [], 'admissible': []}
for tags in qtags:
    tags = [int(t) for t in tags]
    def f_np():
        ids = post[tags[0]]
        for t in tags[1:]: ids = np.intersect1d(ids, post[t], assume_unique=True)
        m = np.zeros(n, bool); m[ids] = True; return np.packbits(m, bitorder='little'), len(ids)
    def f_roar():
        b = roar[tags[0]]
        for t in tags[1:]: b = b & roar[t]
        return b
    def f_dense():
        b = dense[tags[0]]
        for t in tags[1:]: b = np.bitwise_and(b, dense[t])
        return b
    t_np, (bits_np, na) = med(f_np); t_ra, bm = med(f_roar); t_ri, _ = med(lambda: np.asarray(bm.to_array(), np.int32)); t_dn, bits_dn = med(f_dense)
    assert len(bm) == na and np.array_equal(bits_dn, bits_np)
    for k, v in (('numpy', t_np), ('roaring_and', t_ra), ('roaring_ids', t_ri), ('dense_and', t_dn), ('ntags', len(tags)), ('admissible', na)):
        out[k].append(v)
summ = {k: {'mean_ms': 1e3 * float(np.mean(v)), 'median_ms': 1e3 * float(np.median(v)), 'p99_ms': 1e3 * float(np.quantile(v, 0.99))}
        for k, v in out.items() if k not in ('ntags', 'admissible')}
two = np.array(out['ntags']) > 1
summ['and_queries'] = {k: 1e3 * float(np.mean(np.array(out[k])[two])) for k in ('numpy', 'roaring_and', 'roaring_ids', 'dense_and')}
summ['n_and_queries'] = int(two.sum())
os.makedirs('results/bench_h5', exist_ok=True)
json.dump({'summary': summ, 'per_query': out}, open(f"{output_dir('results/bench_h5')}/predicate_cost.json", 'w'), indent=1)
log(json.dumps(summ))

#!/usr/bin/env python3
"""Curator arm (.venv-curator; hatsu3/curator-v2 at c185042, its Faiss fork built with AVX2 and SWIG 4.1.1).
  taskset -c <cores> .venv-curator/bin/python bench/run_curator.py <ds> <schema>[+<schema>...]
Curator (Jin et al.) indexes all vectors in one clustering tree and embeds a per-label index in it; a point is
indexed under a label by grant_access(point, label). Construction and search parameters are those the authors
use for YFCC (benchmark/overall_results/optimal_baseline_params.json): nlist=32, max_sl_size=max_leaf_size=256,
beam_size=8, variance_boost=0.4, search_ef swept 32..4096. Label predicates use the native per-label search;
YFCC AND predicates (two tags) use the authors' bitmap-filter search over the admissible ids. Build: training
and insertion with 32 threads where the library uses them; search 1 thread, one call per label (labels) or per
query (tags)."""
import sys, time, os
sys.path.insert(0, 'bench')
import numpy as np, faiss
from common import load_base, load_schema, groups, admissible_ids, recall, per_query, timed_search, save, log, K, THREADS

def rss_bytes():
    return int(open('/proc/self/statm').read().split()[1]) * os.sysconf('SC_PAGE_SIZE')

ds = sys.argv[1]; schemas = sys.argv[2].split('+')
X, Q, meta = load_base(ds); X = np.ascontiguousarray(X, np.float32); n, d = X.shape; nq = len(Q)
EFS = (32, 64, 128, 256, 512, 1024, 2048, 4096)

def build(tags_of):
    """tags_of(i) -> iterable of labels for point i. Returns (index, build seconds, bytes)."""
    faiss.omp_set_num_threads(32); r0 = rss_bytes(); t0 = time.time()
    ix = faiss.MultiTenantIndexIVFHierarchical(d, 32, faiss.METRIC_L2, 1000, 0.01, 256, 20, 256, 3000, 1.6, 0.4, 0, 8)
    ix.train(X, 0)
    for s in range(0, n, 1_000_000): ix.add_vector_with_ids(X[s:s + 1_000_000], np.arange(s, min(n, s + 1_000_000), dtype=np.int64))
    for i in range(n):
        for t in tags_of(i): ix.grant_access(i, int(t))
    return ix, time.time() - t0, rss_bytes() - r0

for sc in schemas:
    pred, gt = load_schema(ds, sc); is_label = str(pred['kind']) == 'label'
    if is_label:
        attr = pred['attr']; ix, build_s, nbytes = build(lambda i: (attr[i],))
    else:
        # Curator's label ids are int16 (at most 32,767 labels; its authors subsample YFCC to the 1,000 most
        # frequent tags), while YFCC-10M has ~200K tags. We index exactly the tags that occur in this query set,
        # remapped to dense ids: every query behaves as with all tags indexed, but the build does less work.
        pp, pi = pred['ptag_ptr'], pred['ptag_idx']; qp, qi_ = pred['qtag_ptr'], pred['qtag_idx']
        remap = {int(t): j for j, t in enumerate(np.unique(qi_))}
        ix, build_s, nbytes = build(lambda i: [remap[int(t)] for t in pi[pp[i]:pp[i + 1]] if int(t) in remap])
        qi_ = np.array([remap[int(t)] for t in qi_], np.int64)
    log(ds, sc, 'curator built', round(build_s, 1), 's', round(nbytes / 1e9, 2), 'GB'); faiss.omp_set_num_threads(1)
    G = groups(pred, nq); recs = []
    for ef in EFS:
        ix.search_ef = ef
        def run():
            o = np.full((nq, K), -1, np.int32); tot = 0.0; tq = np.zeros(nq)
            for key, qidx in G:
                if is_label:
                    t = time.perf_counter(); _, I = ix.search(Q[qidx], K, int(key)); dt = time.perf_counter() - t
                    o[qidx] = I; tot += dt; tq[qidx] = dt / len(qidx)
                else:
                    i = int(qidx[0]); tags = qi_[qp[i]:qp[i + 1]]
                    if len(tags) == 1:
                        t = time.perf_counter(); _, I = ix.search(Q[i:i + 1], K, int(tags[0])); dt = time.perf_counter() - t
                    else:
                        adm = admissible_ids(pred, i)
                        t = time.perf_counter(); _, I = ix.search_with_bitmap_filter(Q[i:i + 1], K, adm); dt = time.perf_counter() - t
                    o[i] = I[0]; tot += dt; tq[i] = dt
            return o, tot, tq
        o, q, tq = timed_search(run, nq)
        recs.append({'method': 'curator', 'param': f'search_ef={ef}', 'recall': recall(o, gt), 'qps': q, 'threads': THREADS,
                     'build_s': build_s, 'bytes': nbytes, 'extra': {'nlist': 32, 'max_sl_size': 256, 'beam_size': 8, 'variance_boost': 0.4},
                     **per_query(o, gt, None if is_label else tq)})
        log(ds, sc, 'curator', ef, round(recs[-1]['recall'], 4), round(q['mean'], 1))
        if recs[-1]['recall'] > 0.999: break
    save(ds, sc, 'curator', recs); del ix
log('DONE', ds, 'curator')

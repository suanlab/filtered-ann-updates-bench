"""Shared workload/result format for the filtered-search benchmark (numpy only; imported by the
runners in all three environments: main, .venv-faiss, .venv-diskann).

Workload layout (data/bench/<dataset>/):
  X.npy (float32 n x d), Q.npy (float32 nq x d), meta.json
  <schema>/pred.npz   kind='label': attr int32[n], qlab int32[nq]           (equality on one label)
                      kind='csr'  : qoff int64[nq+1], qids int32[...]        (arbitrary admissible sets)
  <schema>/gt.npy     int32 nq x 10, exact filtered ground truth
Results: results/bench/<dataset>/<schema>/<method>.json, one record per swept parameter:
  {method, param, recall, qps: {mean, ci95, reps}, threads, build_s, bytes, extra}
Protocol: search single-threaded on pinned cores (the caller wraps the runner in taskset), REPS
timed repetitions after one warm-up pass; QPS from the repetition times, results (hence recall) from the
median-time repetition (searches are deterministic, so every repetition returns the same results)."""
import os, json, time
import numpy as np

ROOT = 'data/bench'; REPS = 3; K = 10
# Measurement passes: the whole queue can be repeated (BENCH_PASS=2, 3, ...) at different times; the
# analysis takes, per (workload, method, parameter), the best QPS across passes, since contention on
# the shared machine only ever slows a run down (recall is deterministic and identical across passes).
PASS = int(os.environ.get('BENCH_PASS', '1')); RES = 'results/bench' if PASS == 1 else f'results/bench_pass{PASS}'
# Multi-threaded throughput runs (SEARCH_THREADS > 1) write to results/bench_mt<threads> and are never merged with
# the single-thread passes; runners then use their batched / parallel search path.
THREADS = int(os.environ.get('SEARCH_THREADS', '1'))
# MT_MODE: 'batch' (each method's native batched call: one call per label for IVF/RIG/scans, all queries at once for
# ACORN and hnswlib) or 'pool' (query-parallel: THREADS Python threads each issuing single-query, single-threaded
# calls; the bindings release the GIL). The analysis takes, per method, the better of the two modes.
MT_MODE = os.environ.get('MT_MODE', 'batch')
if THREADS > 1: RES = f'results/bench_mt{THREADS}' + ('_pool' if MT_MODE == 'pool' else '') + ('' if PASS == 1 else f'_pass{PASS}')
# BENCH_TAG: results/bench_<tag>[_pass<k>] -- e.g. same-session single-thread references for the 28-thread runs
# (the machine's single-thread speed varies between sessions, so ratios are always formed within one session)
if os.environ.get('BENCH_TAG'): RES = f"results/bench_{os.environ['BENCH_TAG']}" + ('' if PASS == 1 else f'_pass{PASS}')

# Workload statistics the analysis needs (admissible-set sizes of csr workloads, stream schedules), cached in
# results/workload_stats.json by bench/workload_stats.py so that the analysis, numbers, and figures can be
# regenerated from the released results without the datasets. The data files are used when present.
STATS = 'results/workload_stats.json'

def _stats():
    return json.load(open(STATS)) if os.path.exists(STATS) else {'admissible_sizes': {}, 'streams': {}}

def admissible_sizes(wl):
    """|A_p| per query of a csr workload '<dataset>/<schema>' (None for label workloads)."""
    p = f'{ROOT}/{wl}/pred.npz'
    if os.path.exists(p):
        z = np.load(p)
        return np.diff(z['qoff']) if str(z['kind']) == 'csr' else None
    v = _stats()['admissible_sizes'].get(wl)
    return None if v is None else np.asarray(v, np.int64)

def stream_stats(name):
    """A stream's meta.json plus its mean number of updates (deletions + insertions) per round."""
    D = f'{ROOT}/stream_{name}'
    if os.path.exists(f'{D}/meta.json') and os.path.exists(f'{D}/sched.npz'):
        meta = json.load(open(f'{D}/meta.json')); sch = np.load(f'{D}/sched.npz')
        meta['updates_per_round'] = float(np.mean([len(sch[f'del{r}']) + len(sch[f'ins{r}']) for r in range(meta['R'])]))
        return meta
    return _stats()['streams'][name]

def sweep(grid):
    """A method's search-knob grid. With SWEEP=dense the geometric midpoint of every consecutive pair is added
    (doubling steps become sqrt(2) steps), so that QPS at a recall target can be read from measured points that
    bracket it closely rather than interpolated across a factor of two."""
    g = sorted(set(int(x) for x in grid))
    if os.environ.get('SWEEP') == 'dense':
        g = sorted(set(g + [int(round((a * b) ** 0.5)) for a, b in zip(g, g[1:])]))
    return tuple(g)

def write_fbin(path, A):
    """Write A (n x d) as float32 .fbin (int32 n, d header, then rows), in 1M-row chunks."""
    with open(path, 'wb') as f:
        np.array(A.shape, np.int32).tofile(f)
        for i in range(0, len(A), 1_000_000): np.ascontiguousarray(A[i:i + 1_000_000], np.float32).tofile(f)

def read_u8bin(path, n=None):
    """Read the first n rows (all if None) of a .u8bin file (int32 n, d header, then uint8 rows)."""
    with open(path, 'rb') as f:
        n0, d = np.fromfile(f, np.int32, 2); n = int(n0) if n is None else n
        return np.fromfile(f, np.uint8, n * int(d)).reshape(n, int(d))

def faiss_bitmap(mask, keep):
    """faiss IDSelectorBitmap over a boolean mask. The selector only points at the packed bits, so they are
    appended to `keep`, which the caller holds for as long as the selector is in use."""
    import faiss
    bits = np.packbits(mask.astype(np.uint8), bitorder='little'); keep.append(bits)
    return faiss.IDSelectorBitmap(len(mask), faiss.swig_ptr(bits))

def pool_map(fn, n, threads):
    """Run fn(i) for i in range(n) on a thread pool; returns (results, wall seconds)."""
    import time as _t
    from concurrent.futures import ThreadPoolExecutor
    t0 = _t.perf_counter()
    with ThreadPoolExecutor(threads) as ex: res = list(ex.map(fn, range(n)))
    return res, _t.perf_counter() - t0

def ds_dir(ds): return f'{ROOT}/{ds}'
def load_base(ds):
    """(X, Q, meta). X is memory-mapped; a float16 store (the 4096-dimensional arXiv sets) is upcast to float32 in memory."""
    X = np.load(f'{ds_dir(ds)}/X.npy', mmap_mode='r')
    if X.dtype == np.float16: X = X.astype(np.float32)
    return X, np.load(f'{ds_dir(ds)}/Q.npy'), json.load(open(f'{ds_dir(ds)}/meta.json'))
def load_schema(ds, schema):
    z = np.load(f'{ds_dir(ds)}/{schema}/pred.npz'); gt = np.load(f'{ds_dir(ds)}/{schema}/gt.npy')
    return {k: z[k] for k in z.files}, gt

def groups(pred, nq):
    """Queries that share an admissible set: label schemas group by label; csr gives singletons."""
    if str(pred['kind']) == 'label':
        return [(int(b), np.where(pred['qlab'] == b)[0]) for b in np.unique(pred['qlab'])]
    return [(i, np.array([i])) for i in range(nq)]

def admissible(pred, key, n):
    """Boolean mask of admissible points for a group key."""
    if str(pred['kind']) == 'label': return pred['attr'] == key
    m = np.zeros(n, bool); m[pred['qids'][pred['qoff'][key]:pred['qoff'][key + 1]]] = True; return m

def admissible_ids(pred, key):
    if str(pred['kind']) == 'label': return np.where(pred['attr'] == key)[0].astype(np.int64)
    return pred['qids'][pred['qoff'][key]:pred['qoff'][key + 1]].astype(np.int64)

def recall_vec(out, gt):
    """Per-query recall@10 (queries with fewer than 10 admissible points: recall over those)."""
    r = np.empty(len(gt))
    for i, (o, g) in enumerate(zip(out, gt)):
        g = g[g >= 0]
        r[i] = len(set(o[:K].tolist()) & set(g[:K].tolist())) / min(K, len(g)) if len(g) else 1.0
    return r

def recall(out, gt):
    return float(recall_vec(out, gt).mean())

def per_query(out, gt, tq=None):
    """Compact per-query fields stored with every record: rq (recall), tq (seconds; when the runner
    times queries individually), used for selectivity-bucket and dispatcher analyses."""
    d = {'rq': np.round(recall_vec(out, gt), 3).tolist()}
    if tq is not None: d['tq'] = [float('%.4g' % t) for t in tq]
    return d

def ci(xs):
    xs = np.asarray(xs, float)
    return {'mean': float(xs.mean()), 'ci95': float(1.96 * xs.std(ddof=1) / np.sqrt(len(xs))) if len(xs) > 1 else 0.0, 'reps': len(xs)}

def timed_search(fn, nq, reps=REPS):
    """fn() -> (out, seconds_spent_in_search[, per-query seconds]). Warm-up once, then reps timed
    runs. Returns (out, qps ci) or, if fn reports per-query times, (out, qps ci, per-query times of
    the median-time repetition)."""
    if reps < 1: raise ValueError('timed_search needs at least one timed repetition')
    fn()
    runs = [fn() for _ in range(reps)]
    if any(r[1] <= 0 for r in runs): raise ValueError('a timed repetition reported a non-positive search time')
    qps = ci([nq / r[1] for r in runs]); med = sorted(runs, key=lambda r: r[1])[len(runs) // 2]
    return (med[0], qps, med[2]) if len(med) > 2 else (med[0], qps)

def save(ds, schema, method, records):
    os.makedirs(f'{RES}/{ds}/{schema}', exist_ok=True)
    json.dump(records, open(f'{RES}/{ds}/{schema}/{method}.json', 'w'), indent=1, default=float)

def log(*a):
    print(time.strftime('[%H:%M:%S]'), *a, flush=True)

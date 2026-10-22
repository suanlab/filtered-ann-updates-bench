#!/usr/bin/env python3
"""Milvus arm (.venv-milvus: pymilvus 2.6; server: Milvus 2.6.18 standalone from the official .deb, scripts/setup_milvus.sh).
  taskset -c <cores> .venv-milvus/bin/python bench/run_milvus.py <ds> <schema>[+<schema>...]
The runner re-executes itself inside a private user and network namespace (unshare -rn), starts a private Milvus
there (configuration from bench/milvus_conf.py: embedded etcd, local storage under data/milvus, authentication on),
and stops it at the end; the namespace has only a loopback interface, so none of Milvus's ports is reachable from
outside. The server inherits the job's CPU set and uses its default thread pools; queries are issued one at a time
by one client, so a measurement is the system's latency for a single filtered query, as for PostgreSQL.
A filtered query is  search(q, limit=10, filter="l<j> == label")  on a collection with one INT32 label field per
schema (each with an INVERTED scalar index) and the vectors. Modes, each with its own vector index:
  hnsw      HNSW (M=32, efConstruction=200, as hnswlib), search ef swept 16..1000; plus 'default', a search with no
            search parameters (the server's defaults), recorded as mode hnsw_default.
  ivf_flat  IVF_FLAT (nlist = 4 sqrt n, as Faiss), nprobe swept.
  flat      FLAT: exact search over all rows with the filter's bitset (Milvus's exact plan).
Consistency level 'Eventually' (the data are loaded once and flushed, so every level returns the same rows).
Times are client wall times per query, including the gRPC round trip; a query on a 1-row collection is timed for
reference. Index builds use the server's default build threads. Before any index is built the flushed segments are compacted until
the layout is stable, and every sweep asserts that the loaded segments are that layout at its start and end."""
import sys, os, time, json, subprocess, signal
if os.environ.get('MILVUS_NETNS') != '1':                      # re-exec inside a private user+network namespace
    os.execvp('unshare', ['unshare', '-rn', 'env', 'MILVUS_NETNS=1', sys.executable] + sys.argv)
sys.path.insert(0, 'bench')
import numpy as np
import warnings; warnings.filterwarnings('ignore', category=FutureWarning)   # PyMilvusDeprecationWarning: ORM API
from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection, MilvusClient
from common import load_base, load_schema, recall, per_query, ci, save, log, K, REPS

subprocess.run(['ip', 'link', 'set', 'lo', 'up'], check=True)
subprocess.run([sys.executable, 'bench/milvus_conf.py'], check=True, capture_output=True)
M = os.path.abspath('third_party/milvus/root'); PW = open('data/milvus/root_password').read().strip()
ds = sys.argv[1]; schemas = sys.argv[2].split('+')
X, Q, meta = load_base(ds); n, d = X.shape; nq = len(Q)
preds = {sc: load_schema(ds, sc) for sc in schemas}
assert all(str(p['kind']) == 'label' for p, _ in preds.values()), 'label schemas only'

env = {**os.environ, 'MILVUSCONF': os.path.abspath('data/milvus/configs') + '/', 'DEPLOY_MODE': 'STANDALONE',
       'LD_LIBRARY_PATH': f'{M}/usr/lib/milvus'}
srv = subprocess.Popen([f'{M}/usr/bin/milvus', 'run', 'standalone'], env=env, stdout=open(os.path.abspath('data/milvus/var/stdout.log'), 'a'),
                       stderr=subprocess.STDOUT, start_new_session=True, cwd='data/milvus/var')   # it writes default.etcd/ and index_files/ to its cwd
try:
    for _ in range(180):
        try: connections.connect(uri='http://127.0.0.1:19530', token=f'root:{PW}', timeout=5); break
        except Exception: time.sleep(2)
    else: raise RuntimeError('Milvus did not come up (data/milvus/var/stdout.log)')
    version = utility.get_server_version(); log(ds, 'milvus up', version)
    mc = MilvusClient(uri='http://127.0.0.1:19530', token=f'root:{PW}')
    name = f'b_{ds}'
    if utility.has_collection(name): utility.drop_collection(name)
    fields = [FieldSchema('id', DataType.INT64, is_primary=True), *[FieldSchema(f'l{j}', DataType.INT32) for j in range(len(schemas))],
              FieldSchema('emb', DataType.FLOAT_VECTOR, dim=d)]
    col = Collection(name, CollectionSchema(fields))
    t0 = time.time(); attrs = [preds[sc][0]['attr'] for sc in schemas]
    for s in range(0, n, 50_000):
        e = min(n, s + 50_000)
        col.insert([np.arange(s, e, dtype=np.int64)] + [a[s:e].astype(np.int32) for a in attrs] + [np.ascontiguousarray(X[s:e], np.float32)])
    col.flush(); load_s = time.time() - t0; log(ds, 'milvus loaded', n, 'rows in', round(load_s, 1), 's')
    # Settle the segment layout before any index exists. Flushed segments are merged in the background (up to the default
    # maximum segment size); a merge in the middle of a sweep would change the layout every mode searches and the
    # per-segment filter ratio that decides HNSW's brute-force switch. Compact and wait until two checks agree.
    def layout(): return sorted(int(x.num_rows) for x in mc.list_persistent_segments(name))
    prev = None
    for _ in range(30):
        col.compact(); col.wait_for_compaction_completed(timeout=1800); time.sleep(10); cur = layout()
        if cur == prev and sum(cur) == n: break
        prev = cur
    else: raise RuntimeError(f'segment layout did not settle: {cur}')
    LAYOUT = cur; log(ds, 'segments', LAYOUT)
    for j in range(len(schemas)): col.create_index(f'l{j}', {'index_type': 'INVERTED'}, index_name=f'idx_l{j}')
    # reference: a filtered search on a 1-row collection (client + gRPC + scheduling overhead)
    if utility.has_collection('b_trivial'): utility.drop_collection('b_trivial')
    tr = Collection('b_trivial', CollectionSchema([FieldSchema('id', DataType.INT64, is_primary=True), FieldSchema('l0', DataType.INT32),
                                                   FieldSchema('emb', DataType.FLOAT_VECTOR, dim=d)]))
    tr.insert([np.zeros(1, np.int64), np.zeros(1, np.int32), np.zeros((1, d), np.float32)]); tr.flush()
    tr.create_index('emb', {'index_type': 'FLAT', 'metric_type': 'L2', 'params': {}}); tr.load()
    ts = []
    for i in range(2000):
        t = time.perf_counter(); tr.search([Q[i % nq]], 'emb', {'metric_type': 'L2'}, limit=K, expr='l0 == 0', consistency_level='Eventually'); ts.append(time.perf_counter() - t)
    trivial = float(np.median(ts)); tr.release(); utility.drop_collection('b_trivial'); log(ds, 'trivial query', round(trivial * 1e3, 3), 'ms')

    def segments(): return sorted(int(x.num_rows) for x in mc.list_loaded_segments(name))

    def sweep(mode, sparams, params, build_s, tag=None):
        segs = segments(); out = {sc: [] for sc in schemas}
        assert segs == LAYOUT, f'loaded segments {segs} differ from the settled layout {LAYOUT}'
        for j, sc in enumerate(schemas):
            pred, gt = preds[sc]; qlab = pred['qlab']
            for p in params:
                sp = {'metric_type': 'L2', **({'params': sparams(p)} if sparams(p) is not None else {})}
                def run():
                    o = np.full((nq, K), -1, np.int32); tq = np.zeros(nq)
                    for i in range(nq):
                        t = time.perf_counter()
                        r = col.search([Q[i]], 'emb', sp, limit=K, expr=f'l{j} == {int(qlab[i])}', consistency_level='Eventually')
                        tq[i] = time.perf_counter() - t; ids = r[0].ids; o[i, :len(ids)] = ids
                    return o, tq
                run()                                                   # warm-up
                reps = [run() for _ in range(REPS)]; o, tq = sorted(reps, key=lambda x: x[1].sum())[len(reps) // 2]
                q = ci([nq / x[1].sum() for x in reps]); m = f'milvus_{tag or mode}'
                out[sc].append({'method': m, 'param': f'{p}', 'recall': recall(o, gt), 'qps': q, 'threads': 'server', 'build_s': build_s,
                                'bytes': None, 'extra': {'search_params': sp, 'segments': segs, 'trivial_query_s': trivial, 'server': version,
                                                           'load_s': load_s, 'cpus': len(os.sched_getaffinity(0))}, **per_query(o, gt, tq)})
                log(ds, sc, m, p, round(out[sc][-1]['recall'], 4), round(q['mean'], 1))
                if out[sc][-1]['recall'] > 0.999 and tag is None: break
        assert segments() == segs, f'segment layout changed during the {mode} sweep: {segs} -> {segments()}'
        for sc in schemas: save(ds, sc, f'milvus_{tag or mode}', out[sc])

    def build(params):
        if utility.load_state(name).name != 'NotLoad': col.release()
        for ix in col.indexes:
            if ix.field_name == 'emb': col.drop_index(index_name=ix.index_name)
        t0 = time.time(); col.create_index('emb', {'metric_type': 'L2', **params}, index_name='idx_emb')
        utility.wait_for_index_building_complete(name, index_name='idx_emb'); b = time.time() - t0
        col.load(); return b
    b = build({'index_type': 'HNSW', 'params': {'M': 32, 'efConstruction': 200}})
    sweep('hnsw', lambda _: None, ('default',), b, tag='hnsw_default')
    sweep('hnsw', lambda ef: {'ef': ef}, (16, 32, 64, 128, 256, 512, 1000), b)
    nlist = int(4 * np.sqrt(n)); b = build({'index_type': 'IVF_FLAT', 'params': {'nlist': nlist}})
    sweep('ivf_flat', lambda p: {'nprobe': p}, tuple(p for p in (8, 16, 32, 64, 128, 256, 512, 1024, 2048) if p <= nlist // 2), b)
    b = build({'index_type': 'FLAT', 'params': {}})
    sweep('flat', lambda _: {}, ('exact',), b)
    col.release(); utility.drop_collection(name)
finally:
    try: connections.disconnect('default')
    except Exception: pass
    os.killpg(srv.pid, signal.SIGTERM)
    try: srv.wait(120)
    except subprocess.TimeoutExpired: os.killpg(srv.pid, signal.SIGKILL); srv.wait()
log('DONE', ds, 'milvus')

#!/usr/bin/env python3
"""pgvector arm (main env; PostgreSQL 16 + pgvector 0.8.1, private instance under data/pg, local socket only).
  taskset -c <cores> python3 bench/run_pgvector.py <ds> <schema>[+<schema>...]
A filtered query is the SQL a deployment would issue:  SELECT id FROM items WHERE l<j> = $label
ORDER BY emb <-> $q LIMIT 10. One table per dataset holds the vectors and one integer label column per schema.
Modes, measured in this order (each with its own index set):
  hnsw      HNSW (m=32, ef_construction=200, as hnswlib) with iterative index scans (relaxed order, pgvector 0.8),
            hnsw.ef_search swept 16..1000; sequential scans disabled so that the vector index is used.
  ivfflat   IVFFlat (lists = 4 sqrt n, as Faiss) with iterative scans, ivfflat.probes swept.
  exact     B-tree on the label, no vector index: the planner scans the admissible rows and sorts (pre-filter scan).
  auto      B-tree and HNSW both present, all settings at their defaults (ef_search 40, iterative scans off): the
            plan the optimizer picks by itself, recorded per schema.
The server runs with parallel query off, so each query uses one backend; it inherits the job's CPU set. Times are
client wall times per query (prepared statement, one connection), which include the round trip through the socket;
the cost of a trivial query (SELECT 1) is recorded for reference. Index builds use 16 parallel maintenance workers."""
import sys, os, time, json, subprocess, io
sys.path.insert(0, 'bench')
import numpy as np, psycopg
from common import load_base, load_schema, recall, per_query, ci, save, log, K, REPS

PGBIN = '/usr/lib/postgresql/16/bin'; PGDATA = 'data/pg/cluster'; SOCK = os.path.abspath('data/pg')
ds = sys.argv[1]; schemas = sys.argv[2].split('+')
X, Q, meta = load_base(ds); n, d = X.shape; nq = len(Q)
preds = {sc: load_schema(ds, sc) for sc in schemas}
assert all(str(p['kind']) == 'label' for p, _ in preds.values()), 'label schemas only'

def pg(cmd): subprocess.run([f'{PGBIN}/pg_ctl', '-D', PGDATA, '-l', 'data/pg/server.log', '-w', cmd], check=True, capture_output=True)
pg('start')
try:
    db = f'b_{ds}'
    with psycopg.connect(host=SOCK, dbname='postgres', autocommit=True) as c:
        c.execute(f'DROP DATABASE IF EXISTS {db}'); c.execute(f'CREATE DATABASE {db}')
    con = psycopg.connect(host=SOCK, dbname=db, autocommit=True); cur = con.cursor()
    cur.execute('CREATE EXTENSION vector')
    cols = ', '.join(f'l{j} int' for j in range(len(schemas)))
    cur.execute(f'CREATE UNLOGGED TABLE items (id int, {cols}, emb vector({d}))')
    t0 = time.time()
    with cur.copy(f'COPY items (id, {", ".join(f"l{j}" for j in range(len(schemas)))}, emb) FROM STDIN') as cp:
        attrs = [preds[sc][0]['attr'] for sc in schemas]
        for s in range(0, n, 100_000):
            buf = io.StringIO()
            for i in range(s, min(n, s + 100_000)):
                buf.write(f'{i}\t' + '\t'.join(str(int(a[i])) for a in attrs) + '\t[' + ','.join(f'{v:.7g}' for v in X[i]) + ']\n')
            cp.write(buf.getvalue())
    cur.execute('ALTER TABLE items ADD PRIMARY KEY (id)'); cur.execute('VACUUM ANALYZE items')
    load_s = time.time() - t0; log(ds, 'pgvector loaded', n, 'rows in', round(load_s, 1), 's')
    cur.execute('SET max_parallel_maintenance_workers = 15')

    def ms_trivial():
        ts = []
        for _ in range(2000):
            t = time.perf_counter(); cur.execute('SELECT 1'); cur.fetchone(); ts.append(time.perf_counter() - t)
        return float(np.median(ts))
    trivial = ms_trivial()
    qtext = ['[' + ','.join(f'{v:.7g}' for v in q) + ']' for q in Q]

    def sweep(mode, settings, params, build_s, nbytes):
        out = {sc: [] for sc in schemas}
        for j, sc in enumerate(schemas):
            pred, gt = preds[sc]; qlab = pred['qlab']; sql = f'SELECT id FROM items WHERE l{j} = %s ORDER BY emb <-> %s::vector LIMIT {K}'
            cur.execute('EXPLAIN (FORMAT JSON) ' + sql, (int(qlab[0]), qtext[0])); plan = json.dumps(cur.fetchone()[0])
            for p in params:
                for k_, v in settings(p).items(): cur.execute(f'SET {k_} = {v}')
                cur.execute('EXPLAIN (FORMAT JSON) ' + sql, (int(qlab[0]), qtext[0])); plan = json.dumps(cur.fetchone()[0])
                def run():
                    o = np.full((nq, K), -1, np.int32); tq = np.zeros(nq)
                    for i in range(nq):
                        t = time.perf_counter(); cur.execute(sql, (int(qlab[i]), qtext[i]), prepare=True); r = cur.fetchall(); tq[i] = time.perf_counter() - t
                        o[i, :len(r)] = [x[0] for x in r]
                    return o, tq
                run()                                                   # warm-up (also fills shared buffers)
                reps = [run() for _ in range(REPS)]; o, tq = sorted(reps, key=lambda x: x[1].sum())[len(reps) // 2]
                q = ci([nq / x[1].sum() for x in reps])
                scan = 'Index Scan' if 'Index Scan' in plan else 'Seq Scan' if 'Seq Scan' in plan else 'other'
                idx = 'hnsw' if 'items_emb_idx' in plan and mode in ('hnsw', 'auto') else ('btree' if 'items_l' in plan else '')
                out[sc].append({'method': f'pgvector_{mode}', 'param': f'{p}', 'recall': recall(o, gt), 'qps': q, 'threads': 1, 'build_s': build_s,
                                'bytes': nbytes, 'extra': {'plan_root': json.loads(plan)[0]['Plan']['Node Type'], 'plan_uses': [k for k in ('items_emb_idx', 'items_emb_ivf', 'items_l', 'Seq Scan', 'Sort') if k in plan],
                                                           'trivial_query_s': trivial, 'settings': settings(p)}, **per_query(o, gt, tq)})
                log(ds, sc, f'pgvector_{mode}', p, round(out[sc][-1]['recall'], 4), round(q['mean'], 1), out[sc][-1]['extra']['plan_uses'])
                if out[sc][-1]['recall'] > 0.999 and mode != 'auto': break
            cur.execute('RESET ALL')
        for sc in schemas: save(ds, sc, f'pgvector_{mode}', out[sc])

    def size(rel): cur.execute(f"SELECT pg_total_relation_size('{rel}')"); return int(cur.fetchone()[0])
    tbl = size('items')
    # hnsw
    t0 = time.time(); cur.execute('CREATE INDEX items_emb_idx ON items USING hnsw (emb vector_l2_ops) WITH (m = 32, ef_construction = 200)'); b = time.time() - t0
    sweep('hnsw', lambda ef: {'hnsw.ef_search': ef, 'hnsw.iterative_scan': 'relaxed_order', 'hnsw.max_scan_tuples': 1000000, 'enable_seqscan': 'off'},
          (16, 32, 64, 128, 256, 512, 1000), b, tbl + size('items_emb_idx'))
    # auto: defaults with btree + hnsw
    for j in range(len(schemas)): cur.execute(f'CREATE INDEX items_l{j} ON items (l{j})')
    cur.execute('ANALYZE items')
    sweep('auto', lambda _: {}, ('default',), b, size('items'))
    cur.execute('DROP INDEX items_emb_idx')
    # exact: btree only
    sweep('exact', lambda _: {}, ('btree',), 0.0, size('items'))
    for j in range(len(schemas)): cur.execute(f'DROP INDEX items_l{j}')
    # ivfflat
    lists = int(4 * np.sqrt(n)); t0 = time.time()
    cur.execute(f'CREATE INDEX items_emb_ivf ON items USING ivfflat (emb vector_l2_ops) WITH (lists = {lists})'); b = time.time() - t0
    sweep('ivfflat', lambda p: {'ivfflat.probes': p, 'ivfflat.iterative_scan': 'relaxed_order', 'ivfflat.max_probes': lists, 'enable_seqscan': 'off'},
          tuple(p for p in (8, 16, 32, 64, 128, 256, 512, 1024, 2048) if p <= lists // 2), b, tbl + size('items_emb_ivf'))
    con.close()
    with psycopg.connect(host=SOCK, dbname='postgres', autocommit=True) as c: c.execute(f'DROP DATABASE {db}')
finally:
    pg('stop')
log('DONE', ds, 'pgvector')

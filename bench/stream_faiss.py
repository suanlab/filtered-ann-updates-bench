#!/usr/bin/env python3
"""Stream arms with faiss (.venv-faiss).  taskset -c <cores> .venv-faiss/bin/python bench/stream_faiss.py <name> <ivf|scan_part|ivf_part>
  ivf        IVF-Flat, nlist = 4 sqrt(N0), centroids trained once on the initial rows; every round
             remove_ids(deleted) + add_with_ids(inserted) (no tombstones); predicate as IDSelectorBitmap
  scan_part  exact scan of per-label partitions (IndexIDMap2 over IndexFlatL2 per label), maintained
             with add_with_ids/remove_ids; one query per call
  ivf_part   label-partitioned IVF (as bench/run_faiss.py): one IVF-Flat per label, nlist = 4 sqrt(initial label
             size), centroids trained once on the label's initial rows; every round remove_ids + add_with_ids in the
             deleted and inserted points' labels; a query searches its label's index only (no predicate)
Maintenance 16 threads every round; search sweeps 1 thread at the measured rounds.
E2E=1: per-label bitmaps maintained incrementally (t_pred, maintenance); search parameters built inside the timer;
per-query latency percentiles and peak RSS recorded."""
import sys, os, json, time
sys.path.insert(0, 'bench')
import numpy as np, faiss
from common import ROOT, RES, recall, timed_search, log, K, faiss_bitmap, E2E, latency_pct, peak_rss_mb, bitmap_set

name, method = sys.argv[1], sys.argv[2]; D = f'{ROOT}/stream_{name}'
# ivf_rt<k>: IVF whose centroids are retrained on the live set, and the index rebuilt, every k rounds (cost counted as
# maintenance), to separate centroid drift from the update path
RT = int(method[len('ivf_rt'):]) if method.startswith('ivf_rt') else 0
if RT: method_out = method; method = 'ivf'
else: method_out = method
meta = json.load(open(f'{D}/meta.json')); R, N0 = meta['R'], meta['N0']
X = np.load(f'{D}/Xall.npy'); Q = np.load(f'{D}/Q.npy'); attr = np.load(f'{D}/attr.npy'); qlab = np.load(f'{D}/qlab.npy')
gt = np.load(f'{D}/gt.npy'); sch = np.load(f'{D}/sched.npz'); n, d = X.shape; nq = len(Q)
G = [(b, np.where(qlab == b)[0]) for b in range(10)]; alive = np.zeros(n, bool); alive[:N0] = True
KEEP = []
def bitmap(mask): return faiss_bitmap(mask, KEEP)
faiss.omp_set_num_threads(16); t0 = time.time()
if method == 'ivf':
    nlist = int(4 * np.sqrt(N0)); quant = faiss.IndexFlatL2(d); ivf = faiss.IndexIVFFlat(quant, d, nlist)
    ivf.train(X[np.random.default_rng(0).choice(N0, min(N0, 50 * nlist), replace=False)])
    ivf.add_with_ids(X[:N0], np.arange(N0, dtype=np.int64))
elif method == 'ivf_part':
    parts = {}; nl = {}
    for b in range(10):
        ids = np.where(alive & (attr == b))[0]; nl[b] = max(1, int(4 * np.sqrt(len(ids)))); qb = faiss.IndexFlatL2(d); ix = faiss.IndexIVFFlat(qb, d, nl[b])
        ix.train(X[ids[np.random.default_rng(0).choice(len(ids), min(len(ids), 50 * nl[b]), replace=False)]])
        ix.add_with_ids(X[ids], ids.astype(np.int64)); parts[b] = (ix, qb)
    nlist = sum(nl.values())
else:
    parts = {}
    for b in range(10):
        ids = np.where(alive & (attr == b))[0]; ix = faiss.IndexIDMap2(faiss.IndexFlatL2(d)); ix.add_with_ids(X[ids], ids.astype(np.int64)); parts[b] = ix
if E2E and method == 'ivf':      # one packed bitmap per label, kept current by flipping the updated ids' bits
    PK = {b: np.packbits((alive & (attr == b)).astype(np.uint8), bitorder='little') for b in range(10)}
    SELE = {b: faiss.IDSelectorBitmap(n, faiss.swig_ptr(PK[b])) for b in range(10)}
out = {'method': method_out, 'build_s': time.time() - t0, 'rounds': []}; cum = 0.0
if E2E: out['e2e'] = True
for r in range(R + 1):
    row = {'round': r, 't_delete': 0.0, 't_insert': 0.0}
    if r > 0:
        de, ins = sch[f'del{r-1}'].astype(np.int64), sch[f'ins{r-1}'].astype(np.int64); faiss.omp_set_num_threads(16)
        if method == 'ivf' and RT and r % RT == 0:
            alive_next = alive.copy(); alive_next[de] = False; alive_next[ins] = True; live_ids = np.where(alive_next)[0]
            t = time.time(); quant = faiss.IndexFlatL2(d); ivf = faiss.IndexIVFFlat(quant, d, nlist)
            ivf.train(X[live_ids[np.random.default_rng(r).choice(len(live_ids), min(len(live_ids), 50 * nlist), replace=False)]])
            ivf.add_with_ids(X[live_ids], live_ids.astype(np.int64)); row['t_insert'] = time.time() - t; row['retrained'] = True
        elif method == 'ivf':
            t = time.time(); ivf.remove_ids(de); row['t_delete'] = time.time() - t
            t = time.time(); ivf.add_with_ids(X[ins], ins); row['t_insert'] = time.time() - t
        elif method == 'ivf_part':
            t = time.time()
            for b in range(10): parts[b][0].remove_ids(de[attr[de] == b])
            row['t_delete'] = time.time() - t; t = time.time()
            for b in range(10):
                ib = ins[attr[ins] == b]; parts[b][0].add_with_ids(X[ib], ib)
            row['t_insert'] = time.time() - t
        else:
            t = time.time()
            for b in range(10): parts[b].remove_ids(de[attr[de] == b])
            row['t_delete'] = time.time() - t; t = time.time()
            for b in range(10):
                ib = ins[attr[ins] == b]; parts[b].add_with_ids(X[ib], ib)
            row['t_insert'] = time.time() - t
        alive[de] = False; alive[ins] = True; cum += row['t_delete'] + row['t_insert']
        if E2E and method == 'ivf':
            t = time.time()
            for b in range(10): bitmap_set(PK[b], de[attr[de] == b], False); bitmap_set(PK[b], ins[attr[ins] == b], True)
            row['t_pred'] = time.time() - t; cum += row['t_pred']
    row['cum_maint_s'] = cum
    if r in meta['measure_rounds']:
        faiss.omp_set_num_threads(1); sweep = []; live = int(alive.sum()); row['resident'] = live
        if method == 'ivf':
            sels = SELE if E2E else {b: bitmap(alive & (attr == b)) for b, _ in G}; row['bytes'] = int(live * (d * 4 + 8) + nlist * d * 4)
            for npb in (8, 16, 32, 64, 128, 256, 512, 1024, 2048):
                if npb > nlist // 2: break
                def run():
                    o = np.full((nq, K), -1, np.int32); tot = 0.0; per = np.zeros(nq)
                    for b, qi in G:
                        p = faiss.SearchParametersIVF(sel=sels[b], nprobe=npb)
                        for i in qi:
                            if E2E:
                                t = time.perf_counter(); p = faiss.SearchParametersIVF(sel=sels[int(qlab[i])], nprobe=npb)
                                _, I = ivf.search(Q[i:i + 1], K, params=p); dt = time.perf_counter() - t; tot += dt; per[i] = dt; o[i] = I[0]
                            else:
                                t = time.perf_counter(); _, I = ivf.search(Q[i:i + 1], K, params=p); tot += time.perf_counter() - t; o[i] = I[0]
                    return (o, tot, per) if E2E else (o, tot)
                res = timed_search(run, nq); o, q = res[0], res[1]; sweep.append({'param': f'nprobe={npb}', 'recall': recall(o, gt[r]), 'qps': q})
                if E2E: sweep[-1]['latency'] = latency_pct(res[2])
                if sweep[-1]['recall'] > 0.999: break
        elif method == 'ivf_part':
            row['bytes'] = int(live * (d * 4 + 8) + nlist * d * 4)
            for npb in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
                def run():
                    o = np.full((nq, K), -1, np.int32); tot = 0.0; per = np.zeros(nq)
                    for b, qi in G:
                        ix = parts[b][0]; ix.nprobe = min(npb, nl[b])
                        for i in qi:
                            if E2E:
                                t = time.perf_counter(); _, I = parts[int(qlab[i])][0].search(Q[i:i + 1], K); dt = time.perf_counter() - t; tot += dt; per[i] = dt; o[i] = I[0]
                            else:
                                t = time.perf_counter(); _, I = ix.search(Q[i:i + 1], K); tot += time.perf_counter() - t; o[i] = I[0]
                    return (o, tot, per) if E2E else (o, tot)
                res = timed_search(run, nq); o, q = res[0], res[1]; sweep.append({'param': f'nprobe={npb}', 'recall': recall(o, gt[r]), 'qps': q})
                if E2E: sweep[-1]['latency'] = latency_pct(res[2])
                if sweep[-1]['recall'] > 0.999 or npb >= max(nl.values()): break
        else:
            row['bytes'] = int(live * (d * 4 + 8))
            def run():
                o = np.full((nq, K), -1, np.int32); tot = 0.0; per = np.zeros(nq)
                for b, qi in G:
                    for i in qi:
                        if E2E:
                            t = time.perf_counter(); _, I = parts[int(qlab[i])].search(Q[i:i + 1], K); dt = time.perf_counter() - t; tot += dt; per[i] = dt; o[i] = I[0]
                        else:
                            t = time.perf_counter(); _, I = parts[b].search(Q[i:i + 1], K); tot += time.perf_counter() - t; o[i] = I[0]
                return (o, tot, per) if E2E else (o, tot)
            res = timed_search(run, nq); o, q = res[0], res[1]; sweep.append({'param': 'exact', 'recall': recall(o, gt[r]), 'qps': q})
            if E2E: sweep[-1]['latency'] = latency_pct(res[2])
        row['sweep'] = sweep
        log(name, method_out, 'round', r, 'cum', round(cum, 2), [(s['param'], round(s['recall'], 4), round(s['qps']['mean'], 1)) for s in sweep])
    out['rounds'].append(row)
os.makedirs(f'{RES}/stream_{name}', exist_ok=True)
if E2E: out['peak_rss_mb'] = peak_rss_mb()
json.dump(out, open(f'{RES}/stream_{name}/{method_out}.json', 'w'), indent=1, default=float); log('DONE', name, method_out)

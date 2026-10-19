#!/usr/bin/env python3
"""Write bench/queue_full.txt: every job behind the paper's results, in order (preparation, then two
measurement passes of all static and stream jobs, then the 28-thread runs). Run it with
  QUEUE_FILE=bench/queue_full.txt DONE_FILE=results/bench/full.done LOG_FILE=results/bench/full.log bench/queue2.sh
Preparation jobs run ungated; measurement jobs wait for a quiet machine (see bench/queue2.sh)."""
T = 'taskset -c 0-27,56-83'          # one socket (cores 0-27 and their hyperthreads) on the paper's machine
S1 = ['b0.0_s0.1', 'b0.5_s0.1', 'b0.9_s0.1', 'b1.0_s0.1', 'b0.0_s0.01', 'b0.9_s0.01', 'b0.9_s0.5']
WL = {'sift1m': S1, 'deep1m': ['b0.9_s0.1', 'b0.9_s0.01'], 'deep3m': ['b0.9_s0.1', 'b0.9_s0.03125', 'b0.9_s0.003125'],
      'deep10m': ['b0.0_s0.1', 'b0.9_s0.1', 'b0.0_s0.01', 'b0.9_s0.01', 'b0.9_s0.001'], 'yfcc10m': ['tags']}
STREAMS = ['sift500k', 'sift500k_adv', 'deep5m']
MT_WL = (('deep10m', 'b0.0_s0.1'), ('deep10m', 'b0.9_s0.1'), ('sift1m', 'b0.9_s0.1'))

def extra_jobs():
    """Held-out YFCC queries (dispatcher validation) and live-set snapshots of the SIFT streams after round 20
    (fresh builds on the stream's final live data: control for the update path)."""
    out = ['# held-out YFCC queries and round-20 snapshots (preparation ungated)', 'python3 bench/prep.py yfcc10m_test tags',
           'python3 bench/prep_snapshot.py sift500k_adv 20', 'python3 bench/prep_snapshot.py sift500k 20']
    for p in (1, 2):
        P = f'BENCH_PASS={p} {T}'
        for ds, sc in (('yfcc10m_test', 'tags'), ('sift500k_adv_r20', 'live'), ('sift500k_r20', 'live')):
            out += [f'{P} .venv-faiss/bin/python bench/run_faiss.py {ds} {sc}', f'{P} python3 bench/run_main.py {ds} {sc}',
                    f'{P} python3 bench/run_acorn.py {ds} {sc}', f'{P} .venv-diskann/bin/python bench/run_diskann.py {ds} {sc}']
            out.append(f'{P} .venv-curator/bin/python bench/run_curator.py {ds} {sc}')
            if ds == 'yfcc10m_test': out.append(f'{P} python3 bench/run_acorn.py {ds} {sc} gamma=1 M=32 Mb=32')
    return out

def mt_jobs():
    """28-thread throughput on the 28 physical cores of one socket, two passes, each method in its native batched
    mode and (scans, IVF, RIG) also query-parallel; the analysis keeps the better mode per method."""
    out = ['# 28-thread throughput (two passes; batch and query-parallel modes)']
    for p in (1, 2):
        for mode in ('batch', 'pool'):
            M = f'BENCH_PASS={p} MT_MODE={mode} SEARCH_THREADS=28 OPENBLAS_NUM_THREADS={28 if mode == "batch" else 1} taskset -c 0-27'
            for ds, sc in MT_WL:
                out.append(f'{M} .venv-faiss/bin/python bench/run_faiss.py {ds} {sc} scan_part,ivf')
            out.append(f'{M} python3 bench/run_main.py deep10m b0.0_s0.1+b0.9_s0.1 ' + ('rig,postfilter' if mode == 'batch' else 'rig'))
            out.append(f'{M} python3 bench/run_main.py sift1m b0.9_s0.1 ' + ('rig,postfilter' if mode == 'batch' else 'rig'))
            if mode == 'batch':   # natively batched (ACORN) or thread pool only (Filtered-DiskANN binding)
                out += [f'{M} python3 bench/run_acorn.py deep10m b0.0_s0.1+b0.9_s0.1', f'{M} python3 bench/run_acorn.py sift1m b0.9_s0.1']
                out += [f'{M} .venv-diskann/bin/python bench/run_diskann.py {ds} {sc}' for ds, sc in MT_WL]
    return out
jobs = ['# preparation (ungated)']
for ds in ('sift1m', 'deep10m', 'deep1m', 'deep3m', 'yfcc10m'):        # deep1m/deep3m are subsets of deep10m
    jobs.append(f'python3 bench/prep.py {ds} ' + ' '.join(WL[ds]))
jobs += [f'python3 bench/stream_prep.py {nm}' for nm in STREAMS]
for p in (1, 2):
    P = f'BENCH_PASS={p} {T}'
    jobs.append(f'# measurement pass {p}')
    for ds, scs in WL.items():
        for sc in scs:
            jobs.append(f'{P} .venv-faiss/bin/python bench/run_faiss.py {ds} {sc}')
            jobs.append(f'{P} .venv-diskann/bin/python bench/run_diskann.py {ds} {sc}')
        joined = '+'.join(scs)
        jobs.append(f'{P} python3 bench/run_main.py {ds} {joined}')
        jobs.append(f'{P} python3 bench/run_acorn.py {ds} {joined}')
        jobs.append(f'{P} python3 bench/run_acorn.py {ds} {joined} gamma=1 M=32 Mb=32')
        jobs.append(f'{P} .venv-curator/bin/python bench/run_curator.py {ds} {joined}')
    jobs.append(f'{P} python3 bench/run_acorn.py sift1m b0.0_s0.01+b0.9_s0.01 gamma=100 M=32 Mb=64')
    for nm in STREAMS:
        jobs += [f'{P} .venv-faiss/bin/python bench/stream_faiss.py {nm} ivf', f'{P} .venv-faiss/bin/python bench/stream_faiss.py {nm} scan_part',
                 f'{P} python3 bench/stream_main.py {nm} rig', f'{P} python3 bench/stream_main.py {nm} postfilter',
                 f'{P} python3 bench/stream_acorn.py {nm}', f'{P} .venv-diskann/bin/python bench/stream_diskann.py {nm}',
                 f'{P} .venv-curator/bin/python bench/stream_curator.py {nm}']
jobs += extra_jobs()
jobs += mt_jobs()
jobs += ['# predicate-cost and binding-overhead measurements', 'taskset -c 16 python3 bench/overhead.py',
         'taskset -c 16 .venv-faiss/bin/python bench/overhead.py', 'taskset -c 16 .venv-diskann/bin/python bench/overhead.py']
import sys
if len(sys.argv) > 1 and sys.argv[1] == 'medium':  # re-measurements added after the audit (bench/queue_medium.txt)
    J = mt_jobs() + extra_jobs()[4:]                  # snapshots/held-out set are already prepared
    open('bench/queue_medium.txt', 'w').write('\n'.join(J) + '\n'); print(sum(1 for j in J if not j.startswith('#')), 'jobs written to bench/queue_medium.txt'); sys.exit()
open('bench/queue_full.txt', 'w').write('\n'.join(jobs) + '\n')
print(sum(1 for j in jobs if not j.startswith('#')), 'jobs written to bench/queue_full.txt')

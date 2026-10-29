# Follow-up 4: corrections from the pre-submission audit (pre-registration)

- Written: 2026-10-29, before any of the runs below. The commit that adds this file timestamps it.
- Source: `CHECK.md` (an independent audit of baseline 141ecce; evidence in `paper_audit/check_20261029_000421/`),
  findings B01-B13.
- Every number is reported, including failures. The analysis code was fixed and tested before the runs
  (`tests/test_cc_stats.py`, `tests/test_analysis_gates.py`).

## Part A: re-analysis of stored results (no new runs)

These corrections make the code do what the earlier pre-registrations said. The earlier rules are not changed.

1. **Held-out tuning (B03; follow-up 2, I13-4).**
   - The first code compared times only.
   - A verdict at the target now has three states:
     - the point chosen on half A misses recall 0.95 on half B;
     - it reaches the target and is faster than the scan;
     - it reaches the target and is slower.
   - Reported: latency-order changes, misses of the target, and changes of the recall-feasible decision. The paper
     quotes all three.
2. **Spearman with ties (B11; follow-up 2, I17).**
   - Average ranks are used for ties (`bench/common.py:spearman`), and a constant input gives no value.
   - The entry account's registered threshold (rho > 0.5) and the long-edge account's test (rho > 0.3, p < 0.05)
     are unchanged.
3. **Re-execution (B12; follow-up 2, I27).**
   - Verdicts use the paper's operating-point function (Pareto frontier with log-linear interpolation).
   - The job count is read from the manifest, and every job must have exit code 0 in the queue's done file.
   - One job failed: the float copy of the YFCC vectors had been deleted. It was re-queued with `REGEN=1`. It is
     accounted for as a retry.
4. **Follow-up 3 (B07).**
   - An infeasible choice (a path that misses recall 0.95) has infinite regret and fails "regret at most 10%".
   - Every expected job of the job list must be present, P1 must score 17 workloads, and the other rules 23.
   - A censored crossover cannot support "grows with n".
   - The decisions were recomputed and are unchanged: the stored data has no infeasible choice and no censored
     crossover.
5. **Cost frontiers (B06).**
   - A stream run must hold every round 0..R.
   - Its measured rounds must include 0 and R, so query costs are never extrapolated.
   - The end-to-end and Amazon frontiers must contain exactly the pre-registered methods.
   - The exception is the snapshot-rebuild Filtered-DiskANN. Its fresh-every-round maintenance is labeled
     "estimated" (the mean rebuild time × R).
6. **Wording and denominators (B13).**
   - EMA's saving relative to repaired ACORN is quoted as 1 - EMA/repaired.
   - Multi-label ratios are paired by round.
   - Initial and extended campaigns are named with their own run counts.

## Part B: new runs

### B1. Concurrent maintenance under one visibility contract (tag cc4; replaces cc, B01/B02)

- **Why:** the cc runs of post-filtering (hnswlib) and EMA overlapped mutations with searches. Neither library
  permits this. The stale detector also forgot deletions of earlier rounds.
- **Design:** as cc (`bench/stream_concurrent.py`):
  - streams SIFT-500K and SIFT-500K-adv, 20 rounds;
  - 1 writer with 4 threads, batches of 1000, 8 readers;
  - the same methods: IVF, label-partitioned IVF, post-filtering, EMA.
- **The contract is the same for every method.**
  - Each write batch holds an exclusive lock on the structure it changes, and searches hold it shared.
  - The lock is phase-fair: a reader waits for at most the write in progress plus one more. Readers cannot starve
    the writer, and the writer cannot starve readers by re-acquiring the lock.
  - The C++ and Python implementations follow the same policy.
- **Recorded per query execution:** the request time, the time the lock was granted, the end time, and the returned
  ids.
- **Checked:**
  - stale: the deletion completed, in this or any earlier round, before the request;
  - deleted while the query ran;
  - unborn: the insertion was requested after the query ended;
  - out-of-range ids;
  - inadmissible ids.
- **Reported:**
  - per method and stream: the medians over rounds of the concurrent and quiet p99, the concurrent p99 over the
    quiet p99, the lock-wait p99, the writer seconds, and the query throughput;
  - the totals of stale, unborn, out-of-range, and inadmissible ids.
- **Decision:**
  - The paper may say "no stale result" only if the totals of stale, unborn, out-of-range, and inadmissible ids are
    0 for every method and round. Otherwise it reports the counts.
  - The p99 ratios replace those of cc. The cc results move to `results/bench_superseded/cc_unsynchronized/` with
    the reason.

### B2. EMA end to end (tag e2e4; B05)

- **Why:** the e2e campaign ran EMA without an end-to-end mode, because the runner had none. EMA was timed as one
  batched call over the 500 queries with the predicates built outside the timer.
- **Design:** `bench/stream_ema.py` with `E2E=1`:
  - one call per query;
  - the timer covers building the query's predicate, the call, converting the result, and validating it.
  - IVF is re-run in the same session (tag e2e4, `E2E=1`) as the anchor;
  - streams SIFT-500K, SIFT-500K-adv, arXiv-time.
- **Analysis:**
  - EMA's end-to-end cost relative to IVF comes from e2e4. The other methods' costs come from e2e, relative to the
    IVF of that session.
  - Follow-up 2's I19 decision rule is unchanged: the claims are restated wherever a cell's winner changes.
  - The first EMA files move to `results/bench_superseded/e2e_ema_batched/` with the reason.
- Also reported: the timing stages included for each method (a table in `REPRODUCE.md`).

### B3. Filtered-DiskANN rebuilt at every round (tag fdx; B09)

- **Why:** the cost model estimates its fresh-every-round maintenance as the mean rebuild time × R. The rebuilds were
  measured only at the measured rounds.
- **Design:** `bench/stream_diskann.py` with `REBUILD_ALL=1`:
  - the index is rebuilt at every round, with the same parameters (16 threads);
  - the sweep runs at the measured rounds;
  - streams SIFT-500K, SIFT-500K-adv, arXiv-time.
  - DEEP-5M is not run: 10 rebuilds of about 15 min each.
- **Reported:** the executed maintenance total against the estimate from the same run's measured rounds, and against
  the estimate used in the paper (main campaign), as a relative difference.
- **Decision:** none. The paper keeps the estimate in the frontier, labels it as estimated, and quotes the measured
  difference.

## Queue

- `bench/queue2.sh` with `QUEUE_FILE=bench/queue_f4.txt`, `DONE_FILE=results/bench/f4.done`,
  `LOG_FILE=results/bench/f4.log`, `PROBE_LOG=results/bench/probe_f4.log`.
- Gate 380, the Xeon's gate, as before.
- Same cores as the original jobs: 0-27,56-83, with the probe on core 16.

# Follow-up 4: results (decision rules: followup4_prereg.md)

- Queue: `bench/queue_f4.txt`, 17 jobs, run 2026-10-29 10:13–13:44, all with exit code 0 (`results/bench/f4.done`).
- No job was re-queued for contention.
- Numbers come from `bench/analyze_followup2.py` and `bench/analyze_followup3.py` (strict mode).

## Part A: re-analysis of stored results

| item | first code | corrected |
|---|---|---|
| held-out half (71 comparisons) | 0/71 verdict changes | see below |
| entry account, reach–recall Spearman | 0.915 (p 0.0005) | 0.935 (p 0.0005), average ranks |
| long-edge account | 0.030 (p 0.95) | unchanged |
| exploratory true-neighbor reachability | 0.770 (p 0.012) | unchanged |
| re-execution verdicts | 238/254 | 239/254, paper's operating-point function |
| re-execution jobs | hardcoded 132 | 132 from the manifest, all exit 0 (one retried with `REGEN=1`) |
| follow-up 3 decisions | three hold | three hold; no infeasible choice, no censored crossover, 23 × 2 jobs present |

The corrected held-out half:
- Time orders change in 0/71 comparisons.
- 5 chosen points miss recall 0.95, by at most 0.011.
- The recall-feasible verdict changes in 3/71 comparisons.

Each comparison is listed in `results/bench_f2_summary.json`, `I13_heldout.comparisons`. The five misses:

| workload | method | half-B recall | won on all queries |
|---|---|---|---|
| bench/deep3m/b0.9_s0.003125 | IVF | 0.9496 | no |
| bench/yfcc10m/tags | Curator | 0.946 | yes |
| bench_h1/deep1m/b0.0_s0.00316456 | IVF | 0.9424 | no |
| bench_h1/deep1m/b0.0_s0.1 | IVF | 0.9392 | yes |
| bench_h1/deep1m/b0.0_s0.1 | label-partitioned IVF | 0.9496 | yes |

Other corrections in the paper's text:
- **EMA against repaired ACORN:** the paper now quotes EMA's saving, 1 - EMA/repaired, of 5-14% over Q/U 1-100.
  The first text quoted the repaired index's extra cost, 5-17%.
- **Multi-label scan/EMA ratio:** paired by round, 14-27×. The first text combined extremes of different rounds,
  13-28×.

## B1. Concurrency under one visibility contract (cc4)

Medians over 20 rounds. The ratio is the median concurrent p99 over the median quiet p99. "First run" is the
superseded cc run, without locks for post-filtering and EMA.

| stream / method | concurrent p99 ms | quiet p99 ms | ratio | lock-wait p99 ms | writer s per round | first run ratio |
|---|---|---|---|---|---|---|
| SIFT-500K IVF | 9.1 | 1.0 | 9.0 | 8.3 | 0.5 | 8.9 |
| SIFT-500K label-part. IVF | 0.56 | 0.45 | 1.2 | 0.3 | 0.3 | 1.3 |
| SIFT-500K post-filtering | 426 | 13.1 | 32.6 | 413 | 19.3 | 1.1 |
| SIFT-500K EMA | 1,880 | 21.5 | 87.6 | 1,867 | 100 | 2.7 |
| SIFT-500K-adv IVF | 10.0 | 1.55 | 6.4 | 8.8 | 0.5 | 6.3 |
| SIFT-500K-adv label-part. IVF | 0.82 | 0.60 | 1.4 | 0.4 | 0.3 | 1.4 |
| SIFT-500K-adv post-filtering | 564 | 31.0 | 18.2 | 545 | 25.8 | 1.0 |
| SIFT-500K-adv EMA | 1,890 | 74.6 | 25.3 | 1,861 | 97 | 20.8 |

- **Validity:** over every query execution of all eight runs there are 0 stale, 0 unborn, 0 out-of-range, and 0
  inadmissible ids. The pre-registered condition for "no stale result" holds, and the paper says "0 stale or
  invalid ids".
- **Tail latency:** with the lock that the two libraries' contracts require, the post-filtering and EMA tails are
  almost entirely the wait for a write batch. Their write batches take hundreds of milliseconds (post-filtering) to
  seconds (EMA); IVF's take milliseconds.
- **Earlier runs:** the low ratios of the first post-filtering and EMA runs came from searching during mutations,
  which the libraries do not permit.

## B2. EMA end to end (e2e4)

EMA's cost relative to IVF of the same session:

| stream | Q/U 0.1 | 1 | 100 | first run (batched) at 1 | p99 ms | peak RSS GB |
|---|---|---|---|---|---|---|
| SIFT-500K | 8.81 | 3.30 | 2.68 | 2.93 | 12.6 | 6.2 |
| SIFT-500K-adv | 9.96 | 5.76 | 5.29 | 5.60 | 49.8 | 6.2 |
| arXiv-time | 2.53 | 0.42 | 0.18 | 0.43 | 2.1 | 24.4 |

- With EMA timed per query, the cheapest method changes in 0 of the 36 stream-rate cells against the kernel frontier.
- EMA is still the cheapest on arXiv from Q/U = 1, at 0.42× IVF against repaired ACORN's 0.45×.
- The I19 decision rule needs no restatement.

## B3. Filtered-DiskANN rebuilt at every round (fdx)

| stream | rounds | executed total s | estimate, same run s (error) | paper's estimate s (error) |
|---|---|---|---|---|
| SIFT-500K | 20 | 983 | 980 (-0.3%) | 1065, 1043 (+8.3%, +6.1%) |
| SIFT-500K-adv | 20 | 1034 | 1029 (-0.5%) | 1122, 1084 (+8.5%, +4.9%) |
| arXiv-time | 10 | 1680 | 1663 (-1.0%) | 2102, 2055 (+25%, +22%) |

- The estimate, the mean rebuild time × R, is within 1.0% of the executed total in the same run. The paper says so.
- The rebuild time itself is 5-25% higher in the earlier sessions that the paper's frontier uses. The arXiv
  sessions r3 and r3_pass2 ran when the machine was slower.
- The frontier compares each method with IVF of the same session, which absorbs such session effects; the
  limitations section discusses them.

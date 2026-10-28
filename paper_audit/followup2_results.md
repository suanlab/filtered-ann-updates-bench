# Follow-up 2: results as they finish (decision rules: followup2_prereg.md)

Final numbers come from `bench/analyze_followup2.py` (strict) once every campaign is in; this file records each item's
outcome when its campaign completes.

## I13-6 DEEP-5M repair, stream seeds (done 2026-10-25 04:17)

- 6 runs over 5 streams: all-level repair minus tombstone, last-round maximum recall.
  - base pass 1: −0.037; base pass 2: −0.044
  - seed 2: −0.032; seed 3: −0.035; seed 4: −0.020; seed 5: −0.032
- Rule ("lowers recall" only if every run is negative): holds. The paper keeps the claim and reports −0.044 to −0.020
  over 6 runs.

## I07 arXiv blocks (done 2026-10-25 13:13)

Normalized cost at Q/U = 1 (relative to the geometric mean of the block's two IVF anchors):

| block | ivf_part | ivf_rt5 | acorn | acorn_repair_all | ema | fdiskann_rebuild | anchor drift |
|---|---|---|---|---|---|---|---|
| r7a | 0.51 | 0.59 | 0.66 | 0.41 | 0.37 | 0.57 | 0.77 |
| r7b | 0.59 | 0.66 | 0.45 | 0.45 | 0.42 | 0.61 | 1.02 |
| r7c | 0.57 | 0.67 | 0.47 | 0.45 | 0.41 | 0.60 | 0.99 |
| r7d | 0.58 | 0.68 | 0.47 | 0.46 | 0.43 | 0.63 | 0.99 |

- **Rule A** (graphs cheaper than ivf_part from Q/U = 1, in every new block, normalized and raw): holds at Q/U = 1–100.
- **Rule B** (single cheapest graph): holds. EMA is cheapest in all 4 new blocks and in both earlier runs with all four
  graph methods (r6, r6_pass2), at every Q/U from 1 to 100.
- **Paired log ratios vs EMA, over 6 runs** (excludes 0 in every case):
  - repaired ACORN: +0.05..+0.10 at Q/U = 1; +0.07..+0.15 at 100
  - ACORN: +0.06..+0.57 at 1
  - rebuilt F-DiskANN: EMA cheaper by 0.35..0.41 at 1, 0.06..0.12 at 100
- **Anchor drift above 15%:** r7a (0.77). In raw seconds EMA is still the cheapest there (1,928 vs 2,125).
- **Paper change:** "too close to order" becomes "EMA is the cheapest graph method in every block, by 5–15% over
  repaired ACORN". Session effects are visible (ACORN 0.45–0.66 across blocks).

## I13-5 E5 replicate r2c (done 2026-10-25 18:18)

- Fixed L = 64 (from SIFT-500K-adv) on SIFT-500K at 0.95 misses in 6 of 7 periods in both runs. Rounds missed at P = 1:
  19 in r2b, 17 in r2c.
- Oracle-effort regret of the transferred EOQ period at 0.95, both streams: at most 5.7% (r2b) and 6.8% (r2c). At 0.90:
  at most 8.0%.
- Paper change: report "at most 6.8% over two runs" (macro to be generated from both tags).

## I17 SIFT-500K-adv (r8, done 2026-10-25 18:18)

Reach (m4) is the share of a label's live nodes reachable from the query's level-0 entry, after the exact filtered
descent.

| mode | reach |
|---|---|
| tombstone | 0.57 |
| level-0 repair | 0.88 |
| upper-only | 0.63 |
| all-level | 0.99 |
| slot-fill | 0.99 |
| fresh build | 0.997 |

- Entry account (pre-registered): supported. Upper-only reach is below all-level reach, and reach tracks label recall
  under tombstones (Spearman rho = 0.92, permutation p = 0.0005).
- Both the routes into a label's region and its level-0 lists must be repaired: neither alone recovers reach.

## I17 DEEP-5M long-edge account (r8; modes 0, 1, 2, 4 and fresh done 2026-10-26; mode 3 re-queued after contention)

Normalized long-edge share of live level-0 edges, last round:

| mode | long-edge share |
|---|---|
| tombstone | 0.095 |
| level-0 re-rank | 0.046 |
| all-level re-rank | 0.045 |
| slot-fill | 0.073 |
| fresh build | 0.099 |

- Retention during repair (m3, all-level re-rank): 93% of live neighbors kept, but only 83% of long ones.
- Pre-registered test:
  - (a) re-rank long share below slot-fill and fresh: holds.
  - (b) per-label recall loss (fill − re-rank) vs long-share loss: Spearman rho = 0.03, p = 0.95. Fails.
- **Decision: the long-edge account is not supported.** The paper keeps the policy-level result (re-ranking harms,
  slot-filling does not) and no longer states the long-edge mechanism as the cause.
- **Exploratory (not pre-registered):**
  - After re-ranking, part of each label becomes unreachable: reach 0.86–0.96 against 0.99 with slot-fill. The share
    of the true top-10 reachable from the entry is 0.92–0.99.
  - Per-label recall loss tracks the loss of true-neighbor reachability (rho = 0.77, p = 0.012). It does not track
    whole-label reach (rho = 0.44, p = 0.20).
  - Search effort at ef = 256 is similar (4.4k distance computations vs 5.2k with slot-fill).

## I17 complete (DEEP-5M upper-only rerun done 2026-10-26 02:11)

Mean reach (m4) and mean label recall at the last round:

| mode | SIFT-500K-adv reach / recall | DEEP-5M reach / recall |
|---|---|---|
| tombstone | 0.568 / 0.567 | 0.995 / 0.983 |
| level-0 re-rank | 0.879 / 0.887 | 0.922 / 0.950 |
| all-level re-rank | 0.992 / 0.997 | 0.910 / 0.943 |
| upper-only | 0.633 / 0.646 | 0.996 / 0.984 |
| slot-fill | 0.988 / 0.986 | 0.991 / 0.987 |
| fresh build | 0.997 / – | 0.998 / – |

- Pre-registered outcomes:
  - entry account: supported (SIFT-500K-adv);
  - long-edge account: not supported (DEEP-5M, test (b) fails).
- Across modes, recall follows filtered reachability from the entry in both streams. This is a post-hoc reading of the
  table, not a pre-registered test. It covers both the correlated-deletion collapse (tombstones, upper-only) and the
  uniform-deletion harm of re-ranking.
- Paper change:
  - The mechanism is stated as loss of filtered reachability from the entry, measured.
  - "Re-ranking pushes live long-range neighbors out" is reported as an untested-in-isolation candidate, rejected by
    the per-label test.

## I18(d) Faiss SIMD level (done 2026-10-26 03:04)

| level | c_chk/c_dist (96-d) | ns per distance (96-d / 4096-d) | IVF exponent | sqrt(n) wrong | Milvus 7% wrong |
|---|---|---|---|---|---|
| AVX-512 (default) | 0.051 | 36.8 / 1,550 | 0.48 (seed 11) | calibration | – |
| AVX2 | 0.051 | 35.9 / 1,580 | 0.45 | 1/23 (regret 0.6%) | 5/23 (max 786%) |
| none (generic) | 0.041 | 39.2 / 1,756 | 0.50 | 0/23 | 6/23 (max 806%) |

- The manipulation is weak: generic kernels slow a 96-d distance by 7% (memory-bound), so the cost ratio moves little.
- Within this range the sqrt(n) rule, corrected or not, picks right on 22–23 of 23 workloads; a fixed 7% picks wrong on
  5–6.
- Reported as a SIMD-level check on one machine, not as other hardware. The limitation stays.

## I19 end-to-end cost (done 2026-10-26 09:04)

- Winners that change from kernel-only to end-to-end: none, in all 36 stream × Q/U cells over four streams.
- Relative costs move by at most 18%, mostly down for the methods whose end-to-end overhead is smaller than IVF's
  (per-query SearchParameters construction).
- p99 latency at recall 0.95, last round:
  - label-partitioned IVF: 0.3–0.6 ms
  - IVF: 0.8–3.2 ms
  - rebuilt F-DiskANN: 0.6–0.7 ms
  - repaired ACORN: about 4 ms
- Peak RSS: ACORN about 2.3 GB; the other methods 1.2–1.6 GB on SIFT.
- EMA's batched call has no per-query latency, and its runner records no RSS.
- Decision (pre-registered): the kernel-only cost claims stand; the paper states that the end-to-end frontier has the
  same winners.

## I18(c) concurrent maintenance and search, SIFT streams (done 2026-10-26; SIFT-500K-adv EMA pending)

SIFT-500K, 8 readers and 1 writer (batches of 1000), parameter fixed at round 0 for recall 0.95:

| method | p99 concurrent | p99 quiet | writer s/round | stale |
|---|---|---|---|---|
| IVF, one RW lock | 9.0 ms | 1.0 ms | 0.48 | 0 |
| label-partitioned IVF, per-label locks | 0.56 ms | 0.45 ms | 0.29 | 0 |
| hnswlib post-filter, native concurrency | 12.8 ms | 11.9 ms | 19.3 | 0 |
| EMA, no lock | 54 ms | 20 ms | 98.8 | 0 |

- SIFT-500K-adv, same picture:
  - IVF p99: 9.9 ms concurrent vs 1.6 ms quiet;
  - label-partitioned IVF: 0.80 vs 0.57 ms;
  - post-filter: 30 vs 30 ms.
- No crash and no dead or inadmissible result for EMA without locks over 20 rounds. No stale results for any method.
- Recall during updates (best of the round's two ground-truth endpoints) is 0.02–0.08 below the quiet phase, because
  the live set is mid-change.

## I18(c) SIFT-500K-adv EMA (done 2026-10-26 09:56)

- No lock, 20 rounds: no crash, no stale, dead, or inadmissible result.
- p99 latency 1,517 ms during updates against 73 ms quiet. p50 is 1.3 ms.
- Writer 93 s per round. Recall during updates 0.88 (best endpoint); quiet 0.99.
- Under correlated deletions, EMA's synchronous repair produces long stalls for concurrent queries.

## I18(b) multi-label AND stream, YFCC (4 of 5 methods done 2026-10-26; EMA building)

| method | max recall, rounds 0 / 5 / 10 | QPS at 0.95 | QPS at 0.90 |
|---|---|---|---|
| exact scan of admissible ids | 1.0 / 1.0 / 1.0 | 74 / 73 / 72 | 74 / 73 / 72 |
| IVF + bitmap | 0.937 / 0.939 / 0.939 | not reached | 798 / 795 / 696 |
| post-filter (hnswlib, k' ≤ 6400) | 0.72 / 0.73 / 0.73 | not reached | not reached |
| ACORN | 0.475 / 0.405 / 0.383 | not reached | not reached |

- Only the exact scan holds 0.95. ACORN degrades further as the stream churns.
- The static YFCC finding (no index alone reaches the target) recurs under updates.

## I18(a) second real temporal stream, Amazon Reviews 2023 (done 2026-10-26 12:05)

- Setup: 10 categories, 384-d MiniLM embeddings, FIFO window as arXiv-time (200K live, 10 rounds of 40K).
- Cost relative to IVF:

| method | Q/U 0.1 | 1 | 10 | 100 |
|---|---|---|---|---|
| label-partitioned IVF | 0.53 | 0.53 | 0.53 | 0.53 |
| EMA | 3.90 | 0.68 | 0.35 | 0.32 |
| rebuilt F-DiskANN | 5.50 | 0.96 | 0.50 | 0.46 |
| ACORN | 3.23 | 1.05 | 0.82 | 0.80 |
| repaired ACORN | 3.93 | 1.54 | 1.29 | 1.27 |

- Cheapest: label-partitioned IVF up to Q/U = 1; EMA from 3.
- Prediction (pre-registered: at d = 384 the label-partitioned IVF stays cheapest to a higher Q/U than on arXiv, which
  was up to 0.3): held. The reversal still occurs, at a higher query rate.
- Every method keeps maximum recall at 0.99 or above; no collapse.

## I18(b) multi-label EMA: failed run and rerun

- The first run (job 839) failed at round 0. EMA's batched API raises an error when it cannot return k results for
  every query, and some queries have fewer than 10 live matches.
- The runner now queries one at a time and asks for fewer results when EMA cannot fill k (timed). The run is requeued.

## I18(b) multi-label stream, EMA rerun (done 2026-10-26 20:27)

| method | max recall, rounds 0 / 5 / 10 | QPS at 0.95 | maintenance |
|---|---|---|---|
| exact scan | 1.0 / 1.0 / 1.0 | 72–74 | 0 s |
| EMA (per-query calls) | 1.0 / 0.999 / 0.999 | 5.4 / 2.8 / 2.7 | 11,498 s |
| IVF + bitmap | 0.94 | not reached | 2 s |
| post-filter | 0.73 | not reached | 231 s |
| ACORN | 0.48 → 0.38 | not reached | 317 s |

- 0.95 is held only by the exact scan and by EMA. EMA is 13–28× slower than the scan and needs 3.2 h of maintenance.
- One query returned a dead or inadmissible id at one ef (recorded as `invalid_queries`).

## I27 re-execution (done 2026-10-27 18:34)

- 132 sampled jobs, 347 result files compared. The one unmatched file is `stream_sift500k/fdiskann_delta.json`.
- Recall:
  - deterministic methods: identical in 111/111 files;
  - others: within tolerance in 221/236.
- The 15 files outside tolerance:
  - ACORN stream repairs (h3, h3b, r5, rs). The paper already reports these as ranges over runs; the pre-registered
    tolerance came from static pass spread and is tighter than the run-to-run spread of the streams.
  - pgvector IVFFlat on deep1m: 5 files, 0.012–0.019.
  - F-DiskANN and ACORN-28t: 1 file each, marginal.
- QPS, rerun/original: median 1.01, p10–p90 0.94–1.12.
- Verdicts (which of two methods is faster at 0.95): 238 of 254 pairs agree. Most disagreements are pairs whose ratio
  is near 1. The largest flip is pgvector HNSW vs IVFFlat at s = 0.031 (1.33 → 0.93).
- `predicate_cost.py` writes a fixed path; its rerun overwrote the original. The original was restored and the rerun
  copy kept under `results/bench_rep_main/h5`; means agree within 5%.

## Follow-up 3: second machine (pre-registered in followup3_prereg.md; done 2026-10-29 01:43)

AMD EPYC 7513 (Zen 3, AVX2), shared server. Gate at 85% of the quiet probe median (527 of 620). 7 jobs were re-run after
a post-probe below the gate.

| | Xeon 6258R | EPYC 7513 |
|---|---|---|
| c_chk/c_dist (d = 96, microbenchmark) | 0.051 | 0.103 |
| IVF m\* at DEEP-1M, recall 0.95 | 16,375 | 26,924 |
| IVF exponent (300K–10M) | 0.48 | 0.39 |

Rules on the EPYC's 23 workloads:

| rule | wrong | max regret |
|---|---|---|
| P1 own DEEP-1M m\*, sqrt(n) (17 workloads off DEEP-1M) | 0 | 0% |
| P2 Xeon m\*, sqrt(n), microbenchmark-corrected | 0 | 0% |
| P3 Xeon m\*, sqrt(n), uncorrected | 2 | 49% |
| P4 fixed 7% | 4 | 512% |
| P5 Xeon fixed size | 2 | 147% |

Decisions (pre-registered):
- The crossover grows with n on both machines: holds.
- sqrt(n) scaling transfers across sizes: holds.
- The microbenchmark correction transfers across hardware: holds.

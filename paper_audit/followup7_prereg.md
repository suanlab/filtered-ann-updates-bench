# Follow-up 7: close index-versus-scan verdicts in paired blocks (pre-registration)

- Written: 2026-10-31, before any of the runs below. The commit that adds this file timestamps it.
- Source: `CHECK.md` section G, experiment 6 (paired reliability of results near a boundary).
- Every number is reported, including failures.

## Candidates (fixed now from the stored results)

All index-versus-scan comparisons at recall 0.95 in the main campaign (`results/bench`, `results/bench_pass2`) and in
the n-sweep (`results/bench_h1`, `_pass2`), merged best-of-two as in the paper. A comparison is a candidate if:
- (a) the index's QPS is within a factor of 1.1 of the per-label scan's (a close verdict), or
- (b) the index's maximum recall is in [0.94, 0.96) (a feasibility on the edge).

That gives 19 comparisons: 10 close verdicts (a) and 10 edge cases (b), with #1 in both. Each stored value comes from
the campaign in which the comparison was made: #1-12 from the main campaign, #13-19 from the n-sweep.

| # | workload | index | stored ratio index/scan | stored max recall | criterion |
|---|---|---|---|---|---|
| 1 | sift1m/b0.9_s0.5 | ACORN-1 | 1.043 | 0.952 | a, b |
| 2 | sift1m/b0.9_s0.5 | Curator | – | 0.943 | b |
| 3 | sift1m/b0.9_s0.01 | ACORN-γ | – | 0.946 | b |
| 4 | sift1m/b0.9_s0.01 | ACORN γ=100 | 0.068 | 0.956 | b |
| 5 | sift1m/b0.0_s0.01 | ACORN-1 | – | 0.942 | b |
| 6 | deep1m/b0.9_s0.1 | post-filtering | 0.206 | 0.955 | b |
| 7 | deep10m/b0.9_s0.01 | Curator | 1.034 | 0.966 | a |
| 8 | deep10m/b0.9_s0.01 | IVF | 0.978 | 0.996 | a |
| 9 | deep10m/b0.0_s0.01 | RIG | 0.922 | 0.999 | a |
| 10 | deep10m/b0.0_s0.01 | ACORN-1 | 0.050 | 0.951 | b |
| 11 | deep3m/b0.9_s0.03125 | ACORN-γ | 0.939 | 0.991 | a |
| 12 | deep3m/b0.9_s0.03125 | Curator | 0.970 | 0.969 | a |
| 13 | deep300k/b0.0_s0.03125 | IVF | 1.025 | 1.000 | a |
| 14 | deep300k/b0.0_s0.1 | Curator | 0.972 | 0.996 | a |
| 15 | deep300k/b0.0_s0.1 | post-filtering | 0.997 | 1.000 | a |
| 16 | deep1m/b0.0_s0.03125 | Curator | 0.943 | 0.995 | a |
| 17 | deep1m/b0.0_s0.333333 | Curator | 2.339 | 0.956 | b |
| 18 | deep10m/b0.0_s0.03125 | Curator | – | 0.947 | b |
| 19 | deep3m/b0.0_s0.1 | Curator | 1.915 | 0.951 | b |

"–": the index does not reach 0.95 in the stored result.

## Design (tags g6b1..g6b4)

- Four blocks, run one after another. Each block re-runs, for every candidate workload, the per-label scan and the
  candidate indexes with the original runners and parameters. Each block builds the indexes anew, so ACORN, Curator,
  HNSW, and RIG builds are independent; RIG's build is deterministic.
- IVF's training seed is the default in blocks 1-2 and `FAISS_SEED=12` in blocks 3-4.
- Within a block the jobs run in a random order (`random.Random(2032 + block)`), under the usual gate, with a
  manifest.

## Analysis (`bench/analyze_followup7.py`, fixed now)

Per block, QPS at recall 0.95 comes from that block's results alone, with the paper's operating-point function. A
block where the index misses 0.95 counts as infeasible there.

Per candidate:
- the number of blocks in which the index is feasible;
- if it is feasible in all 4, the mean of the per-block log ratio log(QPS_index/QPS_scan) and a 95% t interval
  (3 degrees of freedom).

**Verdicts:**
- *index faster* if the interval lies above 0, *scan faster* if below, *tie* otherwise;
- *unstable feasibility* if the index is feasible in 1-3 blocks, *infeasible* if in none.

The stored (paper's) verdict is the sign of the stored ratio, or "infeasible" if the index missed 0.95.

**Decisions:**
- **D1:** for the 10 close verdicts (criterion a), the paper reports how many are confirmed (same direction,
  interval excludes 0), how many are ties, and how many are reversed. Every claim of the paper that rests on a tie
  or a reversal is restated as a tie or reversed.
- **D2:** for the 10 edge candidates (criterion b), the paper reports in how many the stored feasibility (reached or
  missed 0.95) holds in all 4 blocks, and names those where it does not.
- The policy evaluations (sqrt(n) and other thresholds) are not re-scored. The paper reports how many of their
  wrong or right picks fall on workloads that turn out to be ties.

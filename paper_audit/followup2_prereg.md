# Follow-up 2: experiments for the open audit issues (pre-registration)

- Written: 2026-10-24, before any of the runs below. The commit that adds this file timestamps it.
- Source: PAPER_AUDIT.md (run_20261024_141023), issues I07, I13, I17, I18, I19, I27; experiment_plan.md.
- Rule: every run is reported, including failures. A result that contradicts a claim in the paper changes the
  claim, not the analysis. Analysis code for each item is written before its runs finish and is not tuned after.
- Protocol: as the main study (one socket, cores 0-27,56-83, one job at a time, gated queue `bench/queue2.sh`,
  calibration probe before and after each job, k=10, 500 queries, recall against exact filtered ground truth).

## I07 — arXiv total cost under session effects (tags r7a..r7d)

- Design: 4 new blocks on `arxiv_time`. In each block an IVF anchor runs first (pass 1) and last (pass 2); the six
  frontier methods (ivf_part, ivf_rt5, acorn, acorn_repair_all, ema, fdiskann_rebuild) run in between in a random
  order (numpy `default_rng(2027 + block).permutation`).
- Metrics: per block and Q/U in {0.01..100}, total cost relative to the geometric mean of the block's two anchors,
  and raw seconds; anchor drift (post/pre); per pair of graph methods the paired log cost ratio over blocks.
- Decision rules:
  - "Graph indexes are cheaper than the label-partitioned IVF from Q/U = 1" stands only if it holds in every block,
    both normalized and raw; otherwise the paper reports the block counts.
  - A single cheapest graph method is named only if it is cheapest in all 4 new blocks and in the 4 earlier runs;
    otherwise the paper keeps "too close to order" and reports the paired ratios' range.
  - A block whose anchor drifts by more than 15% is reported and also analyzed without normalization.

## I13 — uncertainty

1. **Crossover exponent over seeds (tags s14..s17).**
   - Design: label and IVF-training seeds 14-17, same grid as s12/s13, which gives 7 seeds (11-17).
   - Analysis: report each seed's exponent, the mean, the SD, and a Student-t 95% interval over seeds, plus a
     two-stage bootstrap (seeds, then queries).
   - Decision: the paper may say that IVF's exponent is below 0.5 only if the seed-level interval's upper end is
     below 0.5; otherwise it says "compatible with 0.5".
   - Policy transfer: thresholds stay fixed on seed 11; the unseen seeds are 12-17.
2. **Timing intervals.** Recompute the within-job intervals with Student t (3 timings, df=2) and report both.
   Analysis only.
3. **Best-of-two.** Recompute the index-versus-scan verdicts and the H1 exponents from the mean of the two passes and
   from the worse pass, and report how many verdicts change. Analysis only.
4. **Tuning on reported queries.** Where per-query recalls and times are stored, split each workload's queries
   into two halves (seed 2027) and choose the operating point on one half. Report recall and QPS on the other half,
   and whether any index-versus-scan verdict changes.
5. **E5 replicate (tag r2c).** Repeat every period on both SIFT streams once. Report the two runs' regrets and
   violations side by side. The "<= X%" claim is restated over both runs.
6. **DEEP-5M repair, stream seeds 4 and 5 (tag rs).** This gives 6 runs over 5 streams. Report every run.
   - Decision: "lowers recall" stands only if every run is negative; otherwise the counts are reported.

## I17 — mechanism of the repair effects (tag r8)

- Design: on SIFT-500K-adv and DEEP-5M, run modes 0 (tombstone), 1 (L0), 2 (all-level re-rank), 3 (upper only), and
  4 (slot-fill) with graph dumps at the measured rounds. Add a fresh build on the final live set.
- Offline metrics, per label:
  - (m1) live level-0 out-degree;
  - (m2) length of live level-0 edges, normalized by the node's 10th-NN distance, and the share of long edges
    (above the 90th percentile of the round-0 graph);
  - (m3) retention: the share of a node's live neighbors before the repair that remain after it;
  - (m4) directed reachability: the share of a label's live admissible nodes reachable from the level-0 entry
    nodes of that label's queries, under ACORN's filtered expansion rule;
  - (m5) per-query search effort (hops, distance computations) at fixed efSearch.
- Decision rules:
  - The long-edge account ("re-ranking pushes live long-range neighbors out and that harms uniform deletions") is
    supported only if both hold:
    - (a) on DEEP-5M the long-edge share of mode 2 is below both mode 4 and the fresh build;
    - (b) across labels, the recall loss of mode 2 against mode 4 correlates positively with its long-edge loss
      (Spearman rho > 0.3, p < 0.05).
    Otherwise the paper reports it as not supported and keeps only the policy-level result.
  - The entry account ("stale upper levels cut the route into a label's region") is supported only if mode 3 is
    below mode 2 in reachability (m4) on SIFT-500K-adv, and m4 tracks label recall across labels (rho > 0.5).

## I19 — end-to-end cost (tag e2e)

- Design: the stream runners get an `E2E=1` mode.
  - Per-label predicate bitmaps are maintained incrementally and timed as maintenance.
  - Each query's timer covers predicate lookup or materialization, the binding call, result validation, and
    re-search.
  - The policies' survival count is timed as maintenance.
  - Each run records peak RSS and, where queries are single calls, per-query latency p50/p95/p99.
- Streams and methods:
  - SIFT-500K, SIFT-500K-adv, arXiv-time: ivf, ivf_part, acorn, acorn_repair_all, ema, fdiskann_rebuild,
    postfilter, scan_part (arXiv also ivf_rt5).
  - DEEP-5M: ivf, ivf_part, acorn, fdiskann_rebuild.
- Decision: the paper's cost claims are restated on the end-to-end frontier wherever a cell's winner changes; the
  kernel-only frontier stays as a reference.

## I18 — generalization

1. **Second real temporal source (tag amz).**
   - Data: Amazon Reviews 2023 (McAuley Lab), 10 categories. Reviews are ordered by timestamp, the label is the
     category, and the embeddings are all-MiniLM-L6-v2 (384-d, CPU). The window and the FIFO schedule are those of
     arXiv-time: 2x10^5 live, 10 rounds of 4x10^4. Queries are 500 later reviews asking for their own category.
   - Methods: as the arXiv frontier.
   - Prediction (fixed now): at d=384 a distance costs less relative to a check than at 4096, so the
     label-partitioned IVF stays cheapest up to a higher Q/U than on arXiv.
   - Decision: if it does not, the paper's cost-ratio explanation of the arXiv reversal is weakened and it says so.
2. **Multi-label AND stream (tag yst).**
   - Data: 10^6 YFCC-10M vectors with their tag sets live at round 0, then 10 rounds of 10% uniform deletion and
     the same number of unseen insertions.
   - Queries: the YFCC queries with their own 1- or 2-tag AND predicates.
   - Methods with general predicates: scan, IVF (bitmap), ACORN, post-filtering, EMA (DNF).
   - Metrics: as Table IV, plus per-query latency percentiles.
3. **Concurrent maintenance (tag cc).**
   - Updates are applied in batches of 1000 by one writer thread while 8 reader threads query.
   - Methods:
     - IVF with a reader-writer lock;
     - label-partitioned IVF with per-label locks;
     - hnswlib post-filtering (native concurrent add and markDelete);
     - EMA, if its library permits concurrent calls (tested first; otherwise reported as unsupported).
   - Metrics:
     - latency p50/p95/p99;
     - throughput;
     - share of results that were deleted before they were returned;
     - recall against the ground truth at round start and at round end.
4. **Hardware cost ratio.** No second machine is available. Faiss's SIMD level (generic / AVX2 / AVX-512) changes
   the distance cost on the same machine.
   - Design: microbenchmark plus the scan/IVF crossover on DEEP 316K-10M at the generic level.
   - Test: does the sqrt(n) rule corrected by the microbenchmark ratio pick the faster path?
   - This is reported as a SIMD-level check, not as different hardware.

## I27 — re-execution (tag rep_<campaign>)

- Design: a stratified random sample of completed jobs, 15% of each campaign and at least one per campaign
  (`random.Random(2027)` over job lines in queue order), re-run under the same protocol from the prepared data.
- Comparison:
  - recall of deterministic methods (scans, IVF, RIG, label-partitioned IVF): exact;
  - other recall: within max(0.01, that method's observed pass spread);
  - QPS: the rerun/original ratio, reported as a distribution;
  - verdicts (which of two methods is faster at the target): the share that agree.
- Not claimed: replication on other hardware.

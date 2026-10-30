# Follow-up 6: results (decision rules: followup6_prereg.md)

- Queue: `bench/queue_f6.txt`, 2 stream preparations and 15 runs, 2026-10-30 16:11 to 2026-10-31 01:33.
- All jobs exited with code 0, with no re-run for contention.
- Manifest: `results/bench/f6_manifest.jsonl` (commit 8d79bd4, no modified tracked code, post-job probe ≥ 477).
- Numbers come from `bench/analyze_followup6.py`.

## Maximum recall at round 20 (fresh build on the final live set in the last column)

| stream | tombstones | level-0 re-rank | all-level re-rank | upper only | slot-fill | fresh |
|---|---|---|---|---|---|---|
| SIFT-500K-adv | 0.577 | 0.984 | 0.996 | 0.619 | 0.957 | 0.989 |
| seed 2 | 0.601 | 0.974 | 0.992 | 0.600 | 0.883 | 0.998 |
| seed 3 | 0.642 | 0.984 | 0.991 | 0.651 | 0.851 | 0.892 |

## Decisions

- **H1 (all-level repair at every round): does not hold.** 3 of 63 stream-rounds are below 0.95:
  - SIFT-500K-adv: round 10 (0.885) and round 18 (0.942);
  - seed 2: round 15 (0.878).
  - Seed 3 has none.
  - The paper does not say that the repair keeps recall at every round. It reports the 3 of 63 and the minimum.
- **The round-10 dip recurs.** On the plain stream follow-up 5's run dipped at round 10 too (0.888). The two runs
  have independent graph builds, so the dip belongs to that round's deletions, not to one build.
- **H2 (all-level restores the collapse at round 20): 3/3.**
- **H3a (all-level's reachable share exceeds upper-only's): 3/3.**
- **H3b (tombstones' reach–recall Spearman across labels > 0.5): 3/3** (ρ 0.92, 0.98, 0.97).
- **H4 (upper-only at least 0.1 below all-level): 3/3.**
- **H5 (slot-fill recovers at least half of the gap): 3/3.** On the two seeds it stays below 0.95 from round 11–12
  on, so "most" holds by the pre-registered half-gap rule, not as a recovery to the target.

## Exploratory: the dips against reachability

| stream, round | all-level max recall | its mean reachable share | median over its other rounds |
|---|---|---|---|
| SIFT-500K-adv, 10 | 0.885 | 0.891 | 0.992 |
| SIFT-500K-adv, 18 | 0.942 | 0.943 | 0.992 |
| seed 2, 15 | 0.878 | 0.892 | 0.992 |

- Each dip coincides with a drop of the reachable share by about the same amount. This is consistent with the entry
  account and not a test of it.
- The level-0 repair also dips at rounds the main design does not measure: 0.888 at round 10 and 0.882 at round 16
  (plain stream), and 0.938 at round 17 (seed 2).

## Maintenance at round 20, relative to tombstones

| stream | level-0 | all-level | upper only | slot-fill |
|---|---|---|---|---|
| SIFT-500K-adv | 1.28 | 1.17 | 0.99 | 1.19 |
| seed 2 | 1.30 | 1.26 | 0.99 | 1.29 |
| seed 3 | 1.30 | 1.27 | 0.99 | 1.31 |

## Paper

- The abstract and introduction ranges for the all-level repair now cover 8 runs on three stream seeds (0.991–0.998
  at the last round).
- Section VI reports the 3 of 63 dips and the reachable share at each.
- Table IV has a new row, and the follow-up 5 dip clause is folded into it.
- The slot-fill statement adds the seeds' 0.85–0.96.
- The limitations sentence names the failed every-round test.

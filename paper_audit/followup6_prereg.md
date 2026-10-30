# Follow-up 6: the repair interventions on independent stream seeds, every round measured (pre-registration)

- Written: 2026-10-30, before any of the runs below. The commit that adds this file timestamps it.
- Source: `CHECK.md` section G, experiment 7 (repair mechanism, matched intervention).
- Trigger: follow-up 5's finding that all-level repaired ACORN reached at most 0.888 at round 10 of SIFT-500K-adv, a
  round the main design does not measure.
- Every number is reported, including failures.

## Design (tag rpx)

**Streams.** SIFT-500K-adv and two independent replicates, `sift500k_adv_seed2` and `sift500k_adv_seed3`
(`bench/stream_prep.py`). Seed k sets:
- the order of the 1.5M BIGANN rows (which are live at round 0 and when the others arrive);
- the label direction and the label noise;
- the deletion noise;
- the query labels.

The plain stream keeps its seeds.

**Update paths.** All with `MECH=1 MEASURE_ALL=1` (the instrumented `src/acorn_stream_mech_avx2`). The mechanism
metrics (exact filtered descent replay, reachability, live retention, effective degree) and the full ef sweep are
recorded at every one of the 21 rounds.
- tombstones (`REPAIR=0`, with `FRESH=1`: a fresh build on the final live set);
- level-0 re-rank (`REPAIR=1`);
- all-level re-rank (`REPAIR=2`);
- upper levels only (`REPAIR=3`);
- slot-fill (`REPAIR=4`).

The 15 runs are in a random order (`random.Random(2031)`), after the two stream preparations, under the usual gate
(`bench/queue2.sh`, gate 380, with a manifest).

## Analysis (`bench/analyze_followup6.py`, fixed now)

For each stream and path:
- per round: the maximum recall over the ef sweep, the mean over labels of the reachable share, and the cumulative
  maintenance;
- at round 20: the recall by label at the highest-recall setting.

**Rules, each scored on each of the 3 streams.** The paper reports "k of 3" for each.
- **H1 (does all-level repair hold the target at every round?):** the number of (stream, round) pairs, out of 63,
  where all-level repair's maximum recall is below 0.95, and their minimum.
  - The paper may say that the all-level repair keeps recall at or above 0.95 at every round only if this count
    is 0.
  - Otherwise it reports the count, the streams, and the minimum.
- **H2 (all-level repair restores the correlated collapse):** at round 20, all-level's maximum recall is at least
  0.95 and at least 0.2 above tombstones'.
- **H3 (entry account):** at round 20,
  - (a) the mean reachable share of all-level exceeds that of upper-only, and
  - (b) across labels, the Spearman correlation of tombstones' reachable share with their recall exceeds 0.5,
    with average ranks.
- **H4 (the level-0 lists must be repaired too):** at round 20, upper-only's maximum recall is at least 0.1 below
  all-level's.
- **H5 (slot-fill recovers most of the collapse):** at round 20, slot-fill's maximum recall is at least tombstones'
  plus half of the gap between tombstones and all-level.
- **Exploratory, not a decision:** at every round where all-level is below 0.95, its mean reachable share there
  against the median over its other rounds.

**Also reported:**
- each path's maintenance over tombstones' at round 20;
- the fresh build's maximum recall at round 20.

**Use in the paper.** Section VI's statements on the repairs are restated with the counts across the 3 streams. The
abstract's range for the all-level repair is widened to include these runs.

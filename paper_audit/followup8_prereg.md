# Follow-up 8: the deletion lesson on a second machine (pre-registration)

- Written: 2026-10-31, before any of the runs below. The commit that adds this file timestamps it.
- **Why:** follow-up 3 repeated the crossover lesson on a second machine. The deletion lesson ran only on the Xeon:
  - tombstoned ACORN collapses under correlated deletions;
  - the all-level repair restores it;
  - slot-fill recovers most of it;
  - the repair lowers recall under uniform deletions.

  This follow-up repeats it on the same AMD EPYC 7513.
- Every number is reported, including failures.

## Machine and protocol (as follow-up 3)

- **Machine:** AMD EPYC 7513 (Zen 3, AVX2), shared with other users. Code and data live under `/mnt/nas`.
- **Placement:** each job is pinned to 16 physical cores of one socket (40-55), and the calibration probe to core 39.
- **Gate:** the probe's quiet value is the median of 10 probes before the first job. A job starts after two
  consecutive probes at or above 85% of it, and a job whose post-probe falls below is re-run (`bench/queue2.sh` with a
  manifest and run directories).
- **Software:**
  - ACORN at c259f11 with the same patches, built with `FAISS_OPT_LEVEL=avx2` against MKL 2026.1.0 from pip, as on
    the Xeon (`scripts/build_acorn.sh` adapted only in paths);
  - Faiss 1.15.1 and numpy 2.2.6 for IVF;
  - the stream data copied byte-identically (SHA-256 checked).

## Runs (tag epyc_str; two passes, random order within a pass, `random.Random(2033 + pass)`)

- SIFT-500K-adv: ACORN with tombstones, all-level repair (`REPAIR=2`), slot-fill (`REPAIR=4`), IVF, and
  label-partitioned IVF.
- DEEP-5M: ACORN with tombstones, all-level repair, and IVF.
- Measured rounds as in the paper: 0, 4, ..., 20 for SIFT and 0, 5, 10 for DEEP.

## Decisions (`bench/analyze_followup8.py`, fixed now)

Each rule is scored in each pass; "holds" means it holds in both passes. "Maximum recall" is the maximum over the ef
sweep at the round.
- **E1 (collapse):** tombstoned ACORN's maximum recall on SIFT-500K-adv at round 20 is at least 0.2 below its round-0
  value.
- **E2 (restore):** the all-level repair's maximum recall at round 20 is at least 0.95 and at least 0.2 above
  tombstones'.
- **E3 (slot-fill):** slot-fill's maximum recall at round 20 is at least tombstones' plus half the gap between
  tombstones and the all-level repair.
- **E4 (uniform harm):** on DEEP-5M, the all-level repair's maximum recall at round 10 is below tombstones'.
- **E5 (IVF baselines):** on SIFT-500K-adv, the label-partitioned IVF's QPS at recall 0.95 exceeds IVF's at every
  measured round.

**Also reported:**
- the per-round values;
- each path's maintenance relative to IVF's;
- the ratios to the Xeon's values for the same runs.

**Use in the paper:** Table V's second-machine row reports E1-E5 as "k of 5 hold". The abstract and introduction may
say both lessons were repeated on a second machine only if all five hold. Otherwise they name what did not hold.

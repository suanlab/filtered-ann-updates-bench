# Follow-up 9: the cost lesson on a second machine (pre-registration)

- Written: 2026-11-03, before any of the runs below. The commit that adds this file timestamps it.
- Why: follow-ups 3 and 8 repeated the crossover and the deletion lessons on a second machine. The second half of
  lesson A ran only on the Xeon: which method is cheapest over a stream depends on the query-to-update ratio and the
  data. This follow-up repeats it on the AMD EPYC 7513.
- Every number is reported, including failures.

## Machine, software, protocol (as follow-up 8)

- AMD EPYC 7513. Cores 40-55; probe on core 39. The gate is 85% of a fresh quiet median of 10 probes. The queue
  keeps a manifest and run directories.
- Software:
  - ACORN c259f11 with the same patches, AVX2, MKL 2026.1.0 from pip;
  - Faiss 1.15.1 and numpy 2.2.6;
  - diskannpy 0.7.0 with numpy 1.25.0 under Python 3.10, as on the Xeon.
- The stream data is copied and checked by SHA-256.

## Runs (tag epyc_cost; two passes, random order within a pass, `random.Random(2034 + pass)`)

In each stream and pass, the methods run between two IVF runs, the anchors.
- SIFT-500K (uniform deletions): label-partitioned IVF, ACORN, all-level repaired ACORN, and Filtered-DiskANN
  rebuilt at the measured rounds.
- arXiv-time (real drift): the same four methods. EMA is not run: it has no build on this machine. The arXiv
  claim tested here is therefore about graph indexes, not EMA.

## Analysis (`bench/analyze_followup9.py`, fixed now)

- **Cost:** maintenance plus Q × the per-query time at recall 0.95, relative to the geometric mean of the two IVF
  anchors. This is the paper's Figure 4 model (`analyze_review.run_costs`). Filtered-DiskANN's fresh-every-round
  maintenance is estimated as in the paper.
- **Cells:** the nine query-to-update ratios of Figure 4, with the IVF anchors included as a method. A method that
  misses 0.95 at a measured round is left out of that run's frontier, as in the paper.

**Decisions** (a rule holds if it holds in both passes):
- **C1 (synthetic):** on SIFT-500K the label-partitioned IVF is the cheapest method in all nine cells.
- **C2 (arXiv, high query rates):** on arXiv-time a graph index is the cheapest in every cell from 1 to 100 queries
  per update. The graph indexes are ACORN, repaired ACORN, and Filtered-DiskANN rebuilt.
- **C3 (arXiv, low query rates):** on arXiv-time the label-partitioned IVF is the cheapest in every cell up to 0.1
  queries per update.

**Also reported:**
- every relative cost;
- the cell where the winner changes;
- the anchor drift.

**Use in the paper:**
- Table V's second-machine row reports C1-C3 as "k of 3 hold".
- The limitations and abstract may extend "both lessons were repeated on a second machine" to the cost ranking only
  if all three hold. Otherwise they name the rule that failed.

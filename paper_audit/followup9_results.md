# Follow-up 9: results (decision rules: followup9_prereg.md)

- **Machine:** AMD EPYC 7513, cores 40–55, probe on core 39.
- **Queue:** `bench/queue_f9.txt`, 24 jobs, 2026-11-03 01:43–08:32. Every job finished with exit code 0.
- **Re-runs:** 4 ACORN jobs (2 on SIFT-500K, 2 on arXiv-time) ended below the gate and were re-run, as the protocol
  requires. The first attempts' outputs are kept in their run directories (`results/bench_epyc_cost_log/runs/`,
  status "contended").
- **Gate (deviation from the pre-registration):** the "fresh quiet median" measured just before the first job was
  taken under a load average of 100–120 on the 128 threads. It came out at about 595, below follow-up 8's quiet median
  of 653 on the same core. We kept follow-up 8's gate of 555, the stricter one, instead of 85% of the loaded median
  (about 506).
- **Inputs:** the stream data was copied and checked by SHA-256 (15 files).
- **Software:** ACORN as in follow-up 8; diskannpy 0.7.0 with numpy 1.25.0 under Python 3.10.22, installed with uv.
- **Analysis:** `bench/analyze_followup9.py`.

## Cost relative to the IVF anchors, at 1 query per update (pass 1 / pass 2)

| | IVF | label-part. IVF | ACORN | repaired ACORN | Filtered-DiskANN rebuilt |
|---|---|---|---|---|---|
| SIFT-500K | 1.01 / 1.11 | 0.25 / 0.24 | 3.01 / 3.08 | 3.78 / 3.55 | 2.71 / 2.94 |
| arXiv-time | 0.98 / 1.00 | 0.57 / 0.54 | 0.71 / 0.84 | 0.62 / 0.66 | 1.08 / 1.26 |

On arXiv at 3 queries per update the repaired ACORN costs 0.33–0.37 and the label-partitioned IVF 0.54–0.56. The
Xeon's values at 1 query per update are 0.46 and 0.59.

## Decisions

- **C1 holds:** on SIFT-500K the label-partitioned IVF is cheapest in all nine cells, in both passes.
- **C2 fails:** on arXiv-time a graph index is cheapest from 3 queries per update, not from 1. At 1, the
  label-partitioned IVF is still cheapest in both passes (0.54–0.57 against repaired ACORN's 0.62–0.66).
- **C3 holds:** on arXiv-time the label-partitioned IVF is cheapest up to 0.1, and in fact up to 1.

2 of 3 hold. The direction of the cost lesson repeats: the label-partitioned IVF wins at low query rates and on
synthetic streams, graph indexes win at high query rates on drifting data. The crossover moves from about 1 (Xeon) to
about 3 (EPYC) queries per update, because the graph indexes cost relatively more on this machine.

## Paper

- The abstract and lesson (4) now say "(3 on a second machine)" after "from about one query per update".
- Table V's second-machine row reports "costs: 2/3 rules hold (arXiv graphs from 3, not 1)".
- "Both lessons repeated on a second machine" stays in the abstract, qualified by the shifted crossover, as the
  pre-registration requires.

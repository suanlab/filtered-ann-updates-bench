# Follow-up 8: results (decision rules: followup8_prereg.md)

- **Machine:** AMD EPYC 7513, cores 40–55, probe on core 39, gate 555 (85% of a quiet median of 653).
- **Queue:** `bench/queue_f8.txt`, 16 jobs, 2026-10-31 19:44 to 2026-11-01 01:21.
  - Every job had status ok (exit 0, post-probe 566–720). None was contended.
- **Kept in `results/bench_epyc_str_log/`:** manifest, run directories, log, probes, `lscpu`, `pip freeze`.
- **Inputs:** the stream data was copied and checked by SHA-256 (14 files).
- **Build:** ACORN c259f11 with the same patches, AVX2, MKL 2026.1.0 from pip, plus a locally built zlib 1.3.1
  (REPRODUCE.md section 13).
- **Analysis:** `bench/analyze_followup8.py`.

## Results

| | pass 1 | pass 2 | Xeon (paper) |
|---|---|---|---|
| SIFT-500K-adv tombstoned ACORN, max recall r0 → r20 | 1.000 → 0.596 | 1.000 → 0.590 | 1.000 → 0.546–0.615 |
| all-level repair, r20 | 0.997 | 0.997 | 0.991–0.998 |
| slot-fill, r20 | 0.972 | 0.961 | 0.983–0.988 (seeds: 0.85–0.96) |
| DEEP-5M r10: tombstones / all-level repair | 0.982 / 0.943 | 0.982 / 0.941 | 0.983 / 0.943 |
| label-partitioned IVF / IVF, QPS at 0.95 (all six rounds) | 3.5–4.5× | 3.6–4.7× | faster at every round |
| maintenance relative to IVF, SIFT-500K-adv: tombstones / all-level / slot-fill | 143 / 146 / 149× | 145 / 143 / 149× | |
| maintenance relative to IVF, DEEP-5M: tombstones / all-level | 67 / 80× | 70 / 80× | |

## Decisions

E1 (collapse), E2 (restore), E3 (slot-fill), E4 (uniform harm), and E5 (IVF baselines) hold in both passes: 5 of 5.
As the pre-registration allows, the abstract and the limitations now say that both lessons were repeated on a
second machine. Table V's second-machine row gives these numbers.

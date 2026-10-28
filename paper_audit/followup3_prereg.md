# Follow-up 3: a second machine (pre-registration)

- Written: 2026-10-29, before any run on the second machine. The commit that adds this file timestamps it.
- Motivation: every measurement so far ran on one Intel Xeon Gold 6258R (Cascade Lake, AVX-512). The paper's
  threshold guideline says to set the scan-or-index threshold for the hardware at hand and to scale it with sqrt(n).

## Machine and protocol

- **Machine:** AMD EPYC 7513 (Zen 3, 2x32 cores, AVX2, no AVX-512), 1.1 TB RAM. It is a shared server (load about 15
  of 128 CPUs, other users' GPU jobs).
- **Placement:** search on one physical core (single thread), and the calibration probe on another core of the same
  socket, as on the Xeon.
- **Gate:** the probe's quiet value is measured before the first job (median of 10 probes). A job starts only after two
  consecutive probes reach 85% of that value, and a job whose post-probe falls below it is re-run.
- **Software:** the same harness code and Python packages as the Xeon (Faiss 1.15.1, numpy 2.2.6), with Faiss's
  default SIMD dispatch (AVX2 here).

## Workloads (identical inputs)

- `bench/microbench_cost.py`: the dataset-free check/distance ratio at d = 96 and 4096.
- The H1 grid at recall 0.95: DEEP 3.16e5, 1e6, 3.16e6, 1e7 with the seed-11 beta = 0 labels (the same files,
  copied), methods `scan_part` and `ivf`, in two passes. In total 23 workloads × 2 passes.

## Analysis and decision rules (fixed now)

1. **Crossover m\* per n, and IVF's exponent on the EPYC.**
   - Rule: the paper may say the crossover grows with n on both machines only if the EPYC exponent is positive at all
     four sizes, i.e. m\* increases with n.
2. **Threshold rules on the EPYC's 23 workloads.** Each rule chooses scan if |A| < T, else IVF; it is scored against
   the faster of the two at recall 0.95 (wrong choices, regret).
   - (P1) the EPYC's own DEEP-1M m\*, scaled with sqrt(n);
   - (P2) the Xeon's DEEP-1M m\*, scaled with sqrt(n) and corrected by the square root of the microbenchmark ratio
     (EPYC / Xeon) at d = 96;
   - (P3) the Xeon's DEEP-1M m\*, scaled with sqrt(n), uncorrected;
   - (P4) a fixed selectivity of 7% (Milvus);
   - (P5) the Xeon's fixed admissible-set size.
3. **Decisions:**
   - "sqrt(n)-scaled transfers across sizes" holds on the EPYC if P1 picks wrong on at most 1 of the 17 non-DEEP-1M
     workloads, with regret at most 10%.
   - The microbenchmark correction "transfers across hardware" only if P2 picks wrong on at most as many workloads as
     P3, and on at most 2. Otherwise the paper says the threshold must be re-measured on new hardware.
   - P4 and P5 are reported as measured.
4. **Every number is reported, including failures.** If the shared machine prevents a quiet measurement (the gate is
   never reached for 6 hours), that is reported instead of a result.

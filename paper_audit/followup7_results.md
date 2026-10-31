# Follow-up 7: results (decision rules: followup7_prereg.md)

- Queue: `bench/queue_f7.txt`, 104 jobs in four blocks, 2026-10-31 07:47–13:36.
- Every job exited with code 0. One job was contended (post probe 370 < 380); it was re-queued and re-run, so the
  manifest has 105 lines.
- The manifest (`results/bench/f7_manifest.jsonl`, commit 05adc09) flags 1 modified tracked file from 08:11 on. That
  file is `bench/queue_f7.txt` itself: the queue appends the re-queued job to it. No code changed during the runs.
- Numbers come from `bench/analyze_followup7.py`.

## Per candidate (ratio index/scan in QPS at recall 0.95; "miss": the index did not reach 0.95)

| # | workload | index | crit. | stored | blocks 1–4 | mean (95% t interval) | verdict | decision |
|---|---|---|---|---|---|---|---|---|
| 1 | sift1m/b0.9_s0.5 | acorn1 | ab | 1.043 | 1.253, miss, miss, 1.161 | – | unstable feasibility | tie / does not hold |
| 2 | sift1m/b0.9_s0.5 | curator | b | miss | miss, miss, miss, miss | – | infeasible | holds |
| 3 | sift1m/b0.9_s0.01 | acorn | b | miss | miss, miss, miss, 0.043 | – | unstable feasibility | does not hold |
| 4 | sift1m/b0.9_s0.01 | acorn_g100 | b | 0.068 | 0.044, 0.107, miss, 0.049 | – | unstable feasibility | does not hold |
| 5 | sift1m/b0.0_s0.01 | acorn1 | b | miss | miss, miss, miss, miss | – | infeasible | holds |
| 6 | deep1m/b0.9_s0.1 | postfilter | b | 0.206 | 0.212, 0.217, 0.217, 0.227 | 0.218 (0.209–0.229) | scan faster | holds |
| 7 | deep10m/b0.9_s0.01 | curator | a | 1.034 | 1.129, 1.167, 1.159, 1.143 | 1.150 (1.123–1.177) | index faster | confirmed |
| 8 | deep10m/b0.9_s0.01 | ivf | a | 0.978 | 1.070, 1.113, 1.215, 1.168 | 1.140 (1.044–1.245) | index faster | reversed |
| 9 | deep10m/b0.0_s0.01 | rig | a | 0.922 | 1.069, 1.062, 1.018, 1.042 | 1.048 (1.012–1.085) | index faster | reversed |
| 10 | deep10m/b0.0_s0.01 | acorn1 | b | 0.050 | 0.059, 0.082, 0.091, 0.073 | 0.075 (0.056–0.101) | scan faster | holds |
| 11 | deep3m/b0.9_s0.03125 | acorn | a | 0.939 | 1.020, 1.041, 1.082, 1.002 | 1.036 (0.983–1.091) | tie | tie |
| 12 | deep3m/b0.9_s0.03125 | curator | a | 0.970 | 0.933, 0.964, 0.967, 1.002 | 0.966 (0.922–1.012) | tie | tie |
| 13 | deep300k/b0.0_s0.03125 | ivf | a | 1.025 | 1.004, 1.017, 1.001, 1.016 | 1.010 (0.997–1.022) | tie | tie |
| 14 | deep300k/b0.0_s0.1 | curator | a | 0.972 | 0.944, 0.959, 0.946, 0.950 | 0.950 (0.939–0.960) | scan faster | confirmed |
| 15 | deep300k/b0.0_s0.1 | postfilter | a | 0.997 | 0.917, 0.967, 0.934, 0.948 | 0.941 (0.908–0.975) | scan faster | confirmed |
| 16 | deep1m/b0.0_s0.03125 | curator | a | 0.943 | 0.946, 0.920, 0.943, 0.938 | 0.937 (0.918–0.956) | scan faster | confirmed |
| 17 | deep1m/b0.0_s0.333333 | curator | b | 2.339 | 2.807, 2.786, 2.843, 2.878 | 2.828 (2.765–2.893) | index faster | holds |
| 18 | deep10m/b0.0_s0.03125 | curator | b | miss | miss, miss, miss, miss | – | infeasible | holds |
| 19 | deep3m/b0.0_s0.1 | curator | b | 1.915 | 2.319, 2.441, 2.329, 2.382 | 2.368 (2.280–2.458) | index faster | holds |

## Decisions

- **D1 (10 close verdicts):** 4 confirmed, 4 ties, 2 reversed.
  - Ties: SIFT1M β=0.9 s=0.5 ACORN-1 (also unstable feasibility), DEEP-3M β=0.9 s=0.031 ACORN and Curator, and DEEP-300K
    β=0 s=0.031 IVF.
  - Reversed: both at n = 10^7.
    - DEEP-10M β=0.9 s=0.01 IVF: stored 0.98×, now 1.14× (1.04–1.25).
    - DEEP-10M β=0 s=0.01 RIG: stored 0.92×, now 1.05× (1.01–1.09).
  - The pass-to-pass spread and the session effects documented in Section IX are larger than these margins.
- **D2 (10 edge cases):** 7 of 10 keep their stored feasibility in all four blocks. The three that do not
  are all ACORN on SIFT1M:
  - ACORN-1 at β=0.9 s=0.5 reaches 0.95 in 2 of 4 blocks;
  - ACORN-γ at β=0.9 s=0.01 in 1 of 4;
  - ACORN γ=100 at β=0.9 s=0.01 in 3 of 4.
  
  ACORN's multi-threaded build is not deterministic.
- **Policy picks on the changed n-sweep workload** (only DEEP-300K β=0 s=0.031 IVF, now a tie):
  - the fixed-size and 7% rules picked the scan there, counted as wrong;
  - the fixed-selectivity and √n rules picked IVF, counted as right.
  
  The policy conclusions do not change. The √n rule has no wrong pick on the other sizes in any case, and the
  fixed-size and 7% rules keep wrong picks on workloads that are not ties.

## Paper

- Section V-A: "IVF merely ties the scan at n = 10^7" is restated. The stored 1.0× is given with the re-measured 1.14×
  (interval 1.04–1.25), and the lead is said to shrink with n rather than the winner to depend on n.
- The ACORN γ=100 sentence adds the counts: 1 and 3 of 4 rebuilds reach 0.95.
- Table IV has a new row.
- RIG's verdict at DEEP-10M β=0 s=0.01 is not quoted in the text; only Table IV's count covers it.

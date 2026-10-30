# Follow-up 5: results (decision rules: followup5_prereg.md)

- Queue: `bench/queue_f5.txt`, 15 jobs, 2026-10-30 01:49–12:17, all with exit code 0.
- Each job has a line in `results/bench/f5_manifest.jsonl`: commit 30ae136, gate probes 433–528 after the jobs.
- The manifest's `dirty_code_files` counted 4 untracked binaries (src/acorn_*_avx2, src/cc_ivf), not modified code.
  The queue now counts tracked files only.
- Numbers come from `bench/analyze_followup5.py`. IVF anchor drift, second run over first: 0.99 (SIFT-adv), 0.98
  (arXiv).

## Decisions

- **D1 (interpolation):** no cheapest-method cell changes on either stream.
  - Query-cost change from the paper's rounds to every round: −1.4 to +2.8% (SIFT-adv); −4.8 to +1.3% (arXiv),
    except the retrained IVF at +14.1%. Its retraining every 5 rounds is not captured by rounds 0, 5, 10.
- **D2 (one effort per stream):**
  - SIFT-adv: the label-partitioned IVF stays cheapest in all 9 cells.
  - arXiv: the winner changes in 5 of 9 cells. At Q/U = 1 ACORN (0.35× IVF) beats EMA (0.38×); at Q/U 3–100 the
    rebuilt Filtered-DiskANN (L = 16) does.
  - At Q/U = 3 the margins are within 1% (Filtered-DiskANN 0.243, ACORN 0.245, repaired ACORN 0.247), well within the
    anchor drift. That is not a win.
  - The effort grid is the cause: EMA's smallest effort that holds 0.95 at every round (ef = 32) gives recall
    0.98–0.99, while the oracle interpolates to exactly 0.95.
  - The paper now says that EMA's lead on arXiv needs the per-round effort.
- **D3 (round-0 effort):** on SIFT-adv the effort chosen at round 0 later misses 0.95 for 4 of 6 methods:
  - IVF: nprobe = 128, round 4, 0.945;
  - label-partitioned IVF: nprobe = 64, round 18, 0.949;
  - ACORN: ef = 64, round 3, 0.936;
  - repaired ACORN: ef = 64, round 2, 0.940.
  
  On arXiv none misses.

## Rounds that no effort holds (not a pre-registered decision; reported as the rule requires every number)

- Repaired ACORN on SIFT-adv reaches at most 0.888 at round 10, while every round the main design measures is
  ≥ 0.97. Rounds 13, 15, and 20 reach 0.988, 0.976, and 0.972.
  - With only the paper's rounds, this run would enter the frontier. With every round, it does not.
  - Section VI and Table IV now say so.
  - This is one run; whether the dip recurs at round 10 is not known.
- Tombstoned ACORN misses from round 6 on, as in the paper.

## Memory (RssAnon of the index process, last round over round 0, MB)

| method | SIFT-500K-adv | arXiv-time |
|---|---|---|
| IVF | 1120 → 1344 (1.20×) | 13582 → 15887 (1.17×) |
| label-partitioned IVF | 1131 → 1371 (1.21×) | 13208 → 15845 (1.20×) |
| retrained IVF | – | 13585 → 13645 (1.00×) |
| ACORN, repaired ACORN | 1257 → 2268 (1.80×), resident nodes 3× | 12621 → 19082 (1.51×), resident nodes 3× |
| EMA | 2139 → 5509 (2.58×) | 6996 → 14406 (2.06×) |
| Filtered-DiskANN, rebuilt | 474 → 563 (1.19×) | 3239 → 3277 (1.01×) |

- The live set is constant in both streams.
- IVF grows although the live set is constant. The retrained IVF, which rebuilds its lists every 5 rounds, does not
  grow. This is consistent with list capacity left behind by removals; we did not measure that directly.
- EMA keeps deleted elements, without slot reuse.
- EMA's per-round regeneration of its attribute indexes is 1.4% (SIFT-adv) and 0.2% (arXiv) of its maintenance.

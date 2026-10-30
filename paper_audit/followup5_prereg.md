# Follow-up 5: executed trace of every round, and memory over the stream (pre-registration)

- Written: 2026-10-30, before any of the runs below. The commit that adds this file timestamps it.
- Source: `CHECK.md` section G.
  - Experiment 8: the cost model against an executed trace. It addresses the oracle tuning and the unobserved
    rounds.
  - Experiment 9: memory over the stream's lifetime.
  - Finding B14: run lineage.
- Every number is reported, including failures.

## Why

The paper's cost frontier (Figure 4) has two simplifications:
- Each method's per-query cost at recall 0.95 is measured only at rounds 0, 4, ..., 20 (SIFT) or 0, 5, 10 (arXiv),
  and interpolated in between.
- The search effort is chosen per measured round from the ground truth (an oracle). A deployment cannot do that.

Memory is not costed, and the `bytes` column means different things for different methods.

## Design (tag tr)

**Mode.** `MEASURE_ALL=1` makes the stream runners sweep at every round, over their full grid (no early stop at
recall 0.999).
- EMA's attribute indexes are then regenerated every round, which counts as maintenance.
- Filtered-DiskANN is then rebuilt every round.

**Streams and methods.** In each stream the methods run in a random order (`random.Random(2030)`), between two IVF
runs, as in follow-up 2's arXiv blocks.
- SIFT-500K-adv: label-partitioned IVF, ACORN (tombstones), all-level repaired ACORN, EMA, Filtered-DiskANN rebuilt
  every round.
- arXiv-time: label-partitioned IVF, IVF retrained every 5 rounds, ACORN, repaired ACORN, EMA, Filtered-DiskANN
  rebuilt every round.

**Memory.** At every round the runner records the index process's anonymous resident memory and total resident
memory (`RssAnon` and `VmRSS` from `/proc`).
- For ACORN, the C++ process reads them before the sweep's filter maps are allocated.
- File-backed pages, such as the memory-mapped input vectors, are not in `RssAnon`.

**Lineage.** `bench/queue2.sh` with `MANIFEST=` writes one JSON line per job: key, line, command, exit code,
times, commit, dirty code files, and probes.

## Analysis (`bench/analyze_followup5.py`, fixed now)

The cost of a method is $M + Q\sum_r t_r$, at recall 0.95, relative to the geometric mean of the two IVF runs'
costs under the same policy. The query-to-update grid is that of Figure 4.

Four query-cost policies:
1. **Oracle, every round:** the best effort per round, from the ground truth. No interpolation.
2. **Oracle, the paper's rounds:** the same run, using only the paper's measured rounds, interpolated as in Figure 4.
   The maintenance is the run's own.
3. **Constant effort:** the smallest grid value that reaches 0.95 at every round. It is chosen in hindsight, but it
   is one setting for the whole stream.
4. **Round-0 effort:** the smallest grid value that reaches 0.95 at round 0, kept. It is infeasible if it misses
   0.95 at any round.

A method enters a policy's frontier only if that policy is feasible for it.

**Reported:**
- per method, the change of the total query cost from policy 2 to policy 1;
- the winner per cell under each policy;
- the methods for which policy 4 is infeasible, with their first failing round and its recall;
- per method, RssAnon at round 0 and at the last round, with its ratio;
- EMA's attribute-index regeneration as a share of its maintenance;
- the IVF anchor drift (second run over first).

**Decisions:**
- **D1:** if the winner of any cell differs between policies 1 and 2, the paper reports the number of cells and
  says interpolation changed them. Otherwise it reports the largest cost change and says interpolation changed no
  winner.
- **D2:** compare the winner under policy 3 with the paper's Figure 4 winner among these methods (results of
  `bench/analyze_review.py`, kernel frontier, recall 0.95). If any cell differs, the paper names the cells and says
  that the oracle frontier is an envelope that a single effort does not reach there. Otherwise it says the winners
  held with one effort per stream.
- **D3:** the paper lists the methods for which the round-0 effort misses the target.
- Memory is reported as measured, with no decision.
- This is one run per method and stream. Differences between methods within the IVF drift are not called wins.

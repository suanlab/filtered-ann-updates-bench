# Reproducing "Filtered Vector Search Under Updates: When Does an Index Beat a Scan?"

This document reproduces every table, figure, and number of `paper/paper_eab.tex` (ICDE 2027,
Experiment, Analysis, and Benchmark). Every number in the paper that is derived from the final results is a
LaTeX macro generated from the analysis output (`paper/eab_numbers.tex`), so re-running the pipeline below
regenerates the paper's numbers.

## 1. Requirements

- Linux x86-64 with AVX2 (the paper's machine: 2x Xeon Gold 6258R, 56 cores, 502 GB RAM, Ubuntu 22.04,
  kernel 6.8, GCC 11.4). Peak memory is about 100 GB (the RIG index on YFCC-10M alone is 73 GB); disk about 60 GB (26 GB of inputs and workloads plus temporary index files).
- Python 3.10, CMake >= 3.23, Intel MKL (`libmkl_rt.so`) for ACORN's Faiss fork.

```bash
scripts/setup_envs.sh      # main env (requirements.txt), .venv-faiss (faiss-cpu 1.15.1), .venv-diskann (diskannpy 0.7.0)
scripts/build_kernels.sh   # src/librig.so, src/libcarve.so (the RIG design point)
scripts/build_acorn.sh     # ACORN c259f11 + patches, AVX2 build: src/acorn_driver_avx2, acorn_stream_avx2, acorn_stream_mech_avx2,
                           # src/cc_ivf, src/acorn_driver_count (BUILD_ONLY_DRIVERS=1 relinks the drivers only)
scripts/build_curator.sh   # Curator (hatsu3/curator-v2 c185042), its Faiss fork with Python bindings, in .venv-curator
```

`patches/acorn_c259f11_cmake_and_add_link.patch` fixes (a) the CMake configuration with a system nlohmann_json and
(b) an out-of-bounds write in ACORN's incremental insertion (`add_link` on full upper-level lists). The fork's
documented build compiles scalar distance kernels (`FAISS_OPT_LEVEL` is declared with `option()`); the script
passes `-DFAISS_OPT_LEVEL=avx2`.

## 2. Data

```bash
scripts/download_data.sh          # SIFT1M and DEEP (ann-benchmarks), YFCC-10M (Big-ANN'23 filtered track), BIGANN prefix
python3 bench/fetch_arxiv.py      # arXiv 4096-d sample (Hugging Face SPCL/arxiv-for-fanns-large, pinned revision 79a4cff)
python3 -m venv .venv-embed && .venv-embed/bin/pip install -r bench/requirements-embed.txt   # CPU torch: see the file
.venv-embed/bin/python bench/prep_amazon.py download && python3 bench/prep_amazon.py select &&
  .venv-embed/bin/python bench/prep_amazon.py embed   # Amazon Reviews 2023, 10 categories (follow-up 2, tag amz)
# all-MiniLM-L6-v2 at revision 1110a243 (pinned in prep_amazon.py); the run's X.f16.npy has sha256 c9aa62fc...8d385
sha256sum -c data_sha256.txt      # every downloaded or sampled input file
```

Terms of use and citations of every dataset: `release/README.md` (Data sources and terms). The arXiv sample's
`meta.json` written on 2026-10-20 says "64-row blocks"; the code read 1024-row blocks (fixed in the code since).

`data/bigann/base.1.5M.u8bin` is the first 192,000,008 bytes of `base.1B.u8bin` (header plus 1.5M vectors);
its header still says 10^9 rows, and the stream generator reads only the first 1.5M.

## 3. Workloads and measurements

```bash
python3 bench/make_queue.py       # writes bench/queue_full.txt
QUEUE_FILE=bench/queue_full.txt DONE_FILE=results/bench/full.done LOG_FILE=results/bench/full.log bench/queue2.sh
```

- Preparation jobs (`bench/prep.py`, `bench/stream_prep.py`) build the workloads under `data/bench/` with
  fixed seeds: label schemas `b<beta>_s<s>`, the YFCC tag predicates, exact filtered ground truth, and the three
  update streams with per-round ground truth. `python3 bench/check_yfcc.py` verifies the YFCC ground truth against
  the official one and that regeneration reproduces the stored workload bit for bit.
- Measurement jobs run strictly one at a time, pinned with `taskset` to one socket of the paper's machine
  (`0-27,56-83`; adjust in `bench/make_queue.py`). `bench/queue2.sh` gates every job on a calibration kernel
  (`bench/probe.py`): a job starts only when the kernel reaches `GATE` queries per second (380 on the paper's
  machine, about two thirds of its quiet value; recalibrate on other hardware) and is re-queued if it ends below.
- Two passes (`BENCH_PASS=1,2`) write to `results/bench/` and `results/bench_pass2/`; the 28-thread runs write to
  `results/bench_mt28/`. On the paper's machine the full queue takes about two days.
- Builds and graph construction are multi-threaded; ACORN, HNSW, and Filtered-DiskANN builds are therefore not
  bit-deterministic, and their recall varies slightly between passes (reported in the paper).

## 4. Analysis, numbers, figures, paper

The analysis needs, besides the raw results, the admissible-set size of every YFCC query and each stream's
schedule summary. `python3 bench/workload_stats.py` caches them in `results/workload_stats.json` (run it after the
preparation jobs); the scripts below read `data/bench/` when it exists and this cache otherwise, so with the released
results alone (no datasets) the commands below regenerate every number, table, figure, and the PDF.

```bash
python3 bench/analyze.py > results/bench_analysis.txt   # merges passes (best QPS per configuration), frontiers,
                                                       # dispatcher (oracle and held-out), buckets, streams
python3 paper/make_eab_numbers.py                      # paper/eab_numbers.tex (all macros)
python3 paper/make_eab_figures.py                      # paper/figures/eab_*.pdf
cd paper && pdflatex paper_eab.tex && pdflatex paper_eab.tex
```

## 5. What is in the repository

| path | content |
|---|---|
| `bench/` | harness: workload generators, one runner per method family and environment, stream runners, gated queue, analysis |
| `src/acorn_driver.cpp`, `src/acorn_stream.cpp` | ACORN drivers (static, and incremental insertion with deletions through the predicate bitmap) |
| `src/rig.cpp`, `src/carve.cpp`, `experiments/rigx.py` | the regenerable pool index (RIG), a design point of the author's own |
| `patches/` | ACORN patch |
| `results/bench*/` | raw per-configuration results, including per-query recall and time |
| `results/bench_superseded/` | results discarded and why (scalar ACORN build; SIFT stream with re-used vectors) |
| `paper/` | paper source, number and figure generators |

Superseded runs are kept under `results/bench_superseded/` (one directory per reason); every defect found during
the study is described in the paper's lessons section (Section VII).

## 6. Follow-up campaign (cost model, Sections IV-VIII)

The follow-up experiments were run from one gated queue (`bench/make_queue.py hplan`, then entries appended for the
streams, counting build, pgvector, and dense sweeps; the queue file actually run is `bench/queue_h.txt`). Results go
to tagged directories `results/bench_<tag>[_pass2]`, which `bench/analyze.py` never merges with the paper's main
results.

| tag | experiment | paper | runner |
|---|---|---|---|
| `h1` | beta=0 n-sweep (DEEP subsets 3.16e5-1e7), H1/H2 | V-A | `run_faiss.py` (`scan_part,ivf,ivf_part`), `run_main.py postfilter`, `run_acorn.py`, `run_curator.py`, `run_diskann.py` |
| `h2c` | ACORN counting build (hops, predicate checks) | V-A | `ACORN_BIN=./src/acorn_driver_count run_acorn.py` |
| `h3`, `h3b` | ACORN streams with `PERC=1`, `REPAIR=1` (level 0), `REPAIR=2` (all levels) | VI | `stream_acorn.py` |
| `h4` | query-filter correlation rho on SIFT1M | V-B | all static runners |
| `h5` | predicate cost (numpy, Roaring, dense bitmaps) | IX | `predicate_cost.py` |
| `h6` | Filtered-DiskANN stale index + exact delta by age; IVF reference | VII | `stream_diskann_delta.py`, `stream_faiss.py` |
| `h7mt28` | 28 threads, n=1e7 slice of h1 | V-F | static runners with `SEARCH_THREADS=28` |
| `h8` | dispatcher with estimated cardinalities (analysis only) | V-C | `h8_estimation.py` |
| `pg` | PostgreSQL 16 + pgvector 0.8.1 | V-E | `run_pgvector.py` (needs `postgresql-16`, `postgresql-16-pgvector`, `psycopg`) |
| `mv` | Milvus 2.6.18 standalone (private, own network namespace) | V-E | `run_milvus.py` (`scripts/setup_milvus.sh`: official .deb extracted without root, `.venv-milvus`) |
| `r1` | streams, same session: IVF, label-partitioned IVF, tombstone ACORN, all-level repair | VI, VII | `stream_faiss.py ivf_part`, `stream_acorn.py` (`REPAIR=2`) |
| `r2b` | executed rebuild policies (Filtered-DiskANN + exact delta every P rounds), IVF anchor | VII | `stream_diskann_policy.py <stream> <P>` |
| `r3` | arXiv temporal stream (`stream_prep.py arxiv_time`), seven methods incl. retrained IVF (`ivf_rt5`) | VI, VII | stream runners |
| `r4` | EMA (`scripts/setup_ema.sh`, commit d95fb49) with an IVF anchor, four streams | VI, VII | `stream_ema.py` |
| `r5` | repair factors: upper levels only (`REPAIR=3`), slot filling (`REPAIR=4`); fresh build on DEEP-5M after round 10 (`prep_snapshot.py deep5m 10`) | VI | `stream_acorn.py`, `run_acorn.py deep5m_r10 live` |
| `r6` | arXiv frontier methods in one session block, pass 2 in reverse order | VII | stream runners |
| `s12`, `s13` | H1 replicates: label/query-label seed and IVF training seed 12, 13 (`prep.py ... b0.0_s<s>_seed<k>`, `FAISS_SEED`) | V-A | `run_faiss.py` |
| `rs` | DEEP-5M stream replicates (`stream_prep.py deep5m_seed<k>`), tombstone vs all-level repair | VI | `stream_acorn.py` |
| `mb` | workload-independent check/distance cost ratio; cost of the policies' untimed survival count | V-A, VII | `microbench_cost.py`, `survival_cost.py` |
| `s14`–`s17` | label/IVF-training seeds 14–17 (follow-up 2, I13) | V-A, Table IV | `run_faiss.py` |
| `rs` (seeds 4, 5) | further DEEP-5M stream replicates (I13) | Table IV | `stream_acorn.py` |
| `r2c` | second run of every executed rebuild period (I13) | VII | `stream_diskann_policy.py` |
| `r7a`–`r7d` | arXiv blocks: IVF anchor, the frontier methods in random order, IVF anchor (pass 2) (I07) | VII, Table IV | stream runners |
| `r8` | repair mechanism: `MECH=1` edge lengths, retention, exact filtered descent and reachability, `FRESH=1` rebuild (I17) | VI | `stream_acorn.py` with `src/acorn_stream_mech_avx2` |
| `e2e` | end-to-end timing, `E2E=1` (I19) | VII, Table IV | stream runners |
| `cc` | concurrent maintenance and search (I18); superseded by `cc4` (`results/bench_superseded/cc_unsynchronized/`) | - | `stream_concurrent.py`, `src/cc_ivf` |
| `cc4` | concurrency under one phase-fair reader-writer lock for every method (follow-up 4, B1) | Table IV | `stream_concurrent.py`, `src/cc_ivf` |
| `e2e4` | EMA end to end (`E2E=1`, one call per query) with an IVF anchor (follow-up 4, B2) | VII, Table IV | `stream_ema.py`, `stream_faiss.py ivf` |
| `fdx` | Filtered-DiskANN rebuilt at every round (`REBUILD_ALL=1`) against the estimate (follow-up 4, B3) | VI, Table IV | `stream_diskann.py` |
| `g6b1`–`g6b4` | 19 close index-versus-scan comparisons re-measured in four blocks (follow-up 7) | V, Table IV | static runners |
| `rpx` | ACORN update paths on SIFT-500K-adv and `sift500k_adv_seed2`, `_seed3` (`stream_prep.py`), `MECH=1 MEASURE_ALL=1` (follow-up 6) | VI, Table IV | `stream_acorn.py` |
| `tr`, `tr_end` | every round measured (`MEASURE_ALL=1`), random method order between two IVF runs; memory per round (follow-up 5) | VI, VII, Table IV | stream runners |
| `yst` | multi-label AND stream on YFCC tags (`stream_prep.py yfcc_stream`) (I18) | Table IV | `stream_multi.py` |
| `amz` | Amazon Reviews 2023 temporal stream (`prep_amazon.py`, `stream_prep.py amz_time`) (I18) | Table IV | stream runners |
| `simdn`, `simd2` | Faiss SIMD level generic / AVX2 (`FAISS_SIMD`) (I18) | Table IV | `run_faiss.py`, `microbench_cost.py` |
| `rep_<tag>` | re-execution of a random 15% of all measurement jobs (`reexec_sample.py`, `reexec_compare.py`) (I27) | Table IV | as the original jobs |
| `dense` | sqrt(2) knob steps on the key workloads | VIII | static runners with `SWEEP=dense` |
| `ax` | arXiv abstracts, 4096-d embeddings, real categories (`prep.py arxiv*`) | V-A | static runners |

```bash
python3 bench/analyze_h.py > results/bench_h_analysis.txt   # results/bench_h_summary.json
python3 paper/make_eab_hnumbers.py                           # paper/eab_hnumbers.tex (macros \V{h/...})
python3 paper/make_eab_hfigures.py                           # paper/figures/eab_teaser.pdf (Fig. 1), eab_law.pdf, eab_percolation.pdf
python3 bench/analyze_review.py > results/bench_r_analysis.txt   # results/bench_r_summary.json (strict: fails on any incomplete campaign, missing round, or missing IVF anchor): per-pass costs, per-query-scan
                                                             # crossovers, 28-thread speed-ups, transfer of dispatch thresholds
python3 paper/make_eab_anumbers.py                           # paper/eab_anumbers.tex (\V{a/...}): per-run ranges, same-run ratios
python3 paper/make_eab_rfigures.py                           # paper/figures/eab_frontier.pdf (total-cost frontier)
python3 bench/analyze_followup2.py > results/bench_f2_analysis.txt   # results/bench_f2_summary.json (follow-up 2, decision rules of
                                                              # paper_audit/followup2_prereg.md; runs bench/reexec_compare.py)
python3 paper/make_eab_fnumbers.py                           # paper/eab_fnumbers.tex (\V{f/...}, Table IV)
```

`h3b` passes 2 and 3 replicate the same-session tombstone / level-0 repair / all-level repair triple (audit item I31).
All follow-up jobs of `bench/queue_h.txt` completed on 2026-10-22 (queue log `results/bench/h.log`, done file
`results/bench/h.done`). `bench/analyze_h.py` reads `data/` only for two workload statistics (the arXiv category
predicate sizes and the stream schedules of H6) and caches them in `results/bench_ax/cat_workload.json` and
`results/bench_h6/stream_*_sched.json`, so the follow-up analysis and its macros regenerate byte-identically without
`data/` (checked by `scripts/make_release.sh`). One arXiv query asks for a category that no sampled paper carries; it
has an empty admissible set, counts as recall 1 for every method, and `run_curator.py` answers it without calling
Curator, which rejects unknown labels. In `stream_diskann_policy.py` the per-label survival used to size the stale index's request is computed outside the timers: `bench/survival_cost.py` replays both SIFT schedules and measures it at a median of about 20 ms per round (at most 30 ms; `results/bench_mb/survival.json`), under 0.06% of one rebuild (52 s).
Milvus (`mv`): the brute-force switch quoted in Section V-E is Knowhere's `HnswSearchThresholds::kHnswSearchKnnBFFilterThreshold
= 0.93` (`src/index/hnsw/impl/IndexConditionalWrapper.h`, applied per segment in `IndexConditionalWrapper.cc`, Knowhere
v2.6.15 as bundled with Milvus 2.6.18). `run_milvus.py` compacts the flushed segments until their layout is stable before
building indexes (all reported runs: one segment of 10^6 rows) and asserts that layout around every sweep; a first pass
without this step is kept in `results/bench_superseded/milvus_unsettled_segments/` with its reason.


## 7. Follow-up 2 (pre-registered; Table IV)

`paper_audit/followup2_prereg.md` fixed the designs and decision rules before the first run (commit 0ee6515);
`paper_audit/followup2_results.md` records each outcome as its campaign finished, including the failed and rerun jobs.
Notes:
- `stream_acorn.py` with `MECH=1`, `FRESH=1`, `E2E=1`, or `ATTR_CSR` needs `src/acorn_stream_mech_avx2`, and
  `stream_concurrent.py` needs `src/cc_ivf`; `scripts/build_acorn.sh` builds both (the first from the same
  `src/acorn_stream.cpp` as `acorn_stream_avx2`, with the instrumented modes off unless their environment switch is set).
- The multi-label EMA run calls EMA once per query and asks for fewer than k results when EMA cannot fill k. Its
  batched API raises an error otherwise; the first run failed on that and was rerun.
- `bench/predicate_cost.py` writes a fixed path. Its re-execution overwrote `results/bench_h5/predicate_cost.json`, so
  that file was restored from git and the rerun copy kept as `results/bench_rep_main/h5/predicate_cost.json`.
- In the re-execution, 28-thread jobs whose queue passed BENCH_TAG through the controller's environment are compared
  with `results/bench_mt28` and `results/bench_mt28_pool`.

## 8. Follow-up 3: a second machine (pre-registered; Table IV, Section V-A)

`paper_audit/followup3_prereg.md` (commit a77ad28) fixed the design and the decision rules before the runs. The H1 grid
(scan and IVF, seed-11 labels, two passes) and `bench/microbench_cost.py` ran on an AMD EPYC 7513 with the same harness
and packages (`results/bench_epyc_log/` holds the queue log, the job list, `lscpu`, and `pip freeze`). The jobs were
pinned to cores 48-55, with the probe on core 47 and a gate at 85% of the quiet probe median.
```bash
python3 bench/analyze_followup3.py > results/bench_f3_analysis.txt   # results/bench_f3_summary.json
python3 paper/make_eab_ffigures.py                                   # paper/figures/eab_reach.pdf (follow-up 2, tag r8)
```

## 9. Follow-up 4: corrections from the pre-submission audit (pre-registered; Table IV)

`CHECK.md` is an independent audit of commit 141ecce. Its evidence is in `paper_audit/check_20261029_000421/`.
`paper_audit/followup4_prereg.md` (commit eff1255) fixed the corrections and three new runs before the runs. The
analysis corrections make the code do what the earlier pre-registrations said:
- held-out verdicts with recall feasibility;
- tie-aware Spearman;
- the paper's operating-point function in the re-execution comparison, with job accounting;
- infeasible choices and completeness in follow-up 3;
- completeness gates in every cost frontier.

Tests: `tests/test_cc_stats.py`, `tests/test_analysis_gates.py`.
```bash
QUEUE_FILE=bench/queue_f4.txt DONE_FILE=results/bench/f4.done LOG_FILE=results/bench/f4.log \
  PROBE_LOG=results/bench/probe_f4.log bench/queue2.sh          # cc4, e2e4, fdx
python3 bench/analyze_followup2.py > results/bench_f2_analysis.txt  # now also reads cc4, e2e4, fdx
python3 bench/analyze_followup3.py > results/bench_f3_analysis.txt
python3 paper/make_eab_fnumbers.py
```

**Concurrency contract (cc4).**
- Every method uses one phase-fair reader-writer lock per structure: `bench/cc_stats.py:FairRW` for the Python
  readers and `src/cc_ivf.cpp:FairRW` for IVF. The structure is one index, or one label's index for the
  label-partitioned IVF.
- A write batch of 1000 points is atomic for searches.
- A reader that arrives while a writer holds or waits for the lock enters after that writer's release.
- Latency runs from the request to the end, so it includes the lock wait; the wait is also reported on its own.
- `bench/cc_stats.py:Tracker` checks every returned id against the whole stream's history:
  - deleted before the request (stale);
  - deleted while the query ran;
  - inserted after the query ended (unborn);
  - out of range;
  - inadmissible.

**Timing stages of the end-to-end mode (`E2E=1`).** Predicate state that the updates keep current is timed as
maintenance (`t_pred`).

| method (runner) | calls | predicate state | inside the query timer |
|---|---|---|---|
| IVF, retrained IVF (`stream_faiss.py`) | one per query | per-label bitmap, flipped by the updates | building the search parameters and selector, the call |
| label-partitioned IVF, per-label scan (`stream_faiss.py`) | one per query | the partition | looking up the label's index, the call |
| post-filtering (`stream_main.py`) | one per label | per-label mask, kept current | the call, filtering by the mask, truncation to k |
| ACORN, repaired ACORN (`acorn_stream.cpp`) | one per query | per-label map, kept current | the call |
| Filtered-DiskANN (`stream_diskann.py`) | one per query | label string | the call, mapping ids, dropping padding, checking the label |
| EMA (`stream_ema.py`, e2e4) | one per query | the DNF predicate, built per query | building the predicate, the call, converting and validating the result |

- ACORN's and IVF's results are live and admissible by construction (the map or bitmap excludes deleted and
  inadmissible ids), so they are not re-checked.
- The other runners refuse `E2E=1`: Curator's stream, the delta and multi-label streams, and the concurrency runner.
- The first e2e campaign's EMA files were timed as one batched call. They are kept in
  `results/bench_superseded/e2e_ema_batched/`.

**Memory columns.** `bytes` is computed differently by each runner, so the values are not the same physical quantity:
- Faiss and ACORN: vectors + ids (+ centroids, + graph lists);
- Filtered-DiskANN: index files;
- Curator: process RSS growth during the build;
- RIG: its own `nbytes()`.

The end-to-end runs also record peak RSS (`peak_rss_mb`).

**Curator on YFCC.** Curator's label ids are 16-bit, so `run_curator.py` indexes the tags that occur in the query set,
remapped to dense ids. This is a workload-informed build. Every query behaves as with all tags indexed.

**Queue lineage.**
- A done file records one line per job: content key, exit code, start, end, line number. Only exit code 0 is a
  success, and 99 marks a contended run that was re-queued.
- `bench/reexec_compare.py` checks the re-execution sample against `results/bench/h.done`. One job was retried
  with a changed line, `REGEN=1` (listed in the script).
- `bench/queue_h.txt` is the executed queue, including lines appended during the campaign. `make_queue.py hplan`
  generates only its first part.
- `bench/predicate_cost.py`, `bench/microbench_cost.py`, and `bench/survival_cost.py` first ignored `BENCH_TAG` and
  wrote fixed paths. The re-execution copy of the first is kept under `results/bench_rep_main/` (section 7).
  They now write to `OUT_DIR`, else to the `BENCH_TAG` directory, else to the historical path (`common.output_dir`).
- Immutable run directories (`RUNS_DIR=results/runs`, `bench/queue2.sh`): every executed job gets a read-only
  directory `<start>_<key>/` with these files:
  - `cmd.txt`, `git_head.txt`, and `code.diff` (uncommitted changes to tracked code);
  - `job.log`;
  - `status.json`: ok, failed (non-zero exit), or contended (re-queued), with the probes;
  - a copy of every file the job wrote under `results/` (`outputs/`, with `outputs.sha256`).

  A later re-run that overwrites a result file keeps the earlier attempt's output there. The manifest line records
  the status and the run directory. Only `status == ok` counts as a completed job (`reexec_compare.py` checks exit
  code 0 in the done file).
- `results/workload_stats.json` lets the analysis run without `data/`. Whenever `data/` is present, every cached
  entry is checked against the data and the analysis stops if they differ (`common._check_cache`).
  `bench/workload_stats.py` regenerates the cache and writes the inputs' SHA-256 to
  `results/workload_stats.inputs.json`.

## 10. Follow-up 5: executed trace of every round, and memory (pre-registered; Table IV)

`paper_audit/followup5_prereg.md` (commit 30ae136) fixed the design, the four query-cost policies, and the decisions
before the runs. `paper_audit/followup5_results.md` records the outcome.
- `MEASURE_ALL=1` makes the stream runners sweep their full grid at every round, without the early stop at 0.999.
  EMA then regenerates its attribute indexes, and Filtered-DiskANN rebuilds, at every round.
- Each measured round records `rss_anon_mb` and `vm_rss_mb` from `/proc`. ACORN reads them inside
  `src/acorn_stream.cpp`, before the sweep allocates its filter maps.
- `bench/queue2.sh` with `MANIFEST=<file>` appends one JSON line per job: key, line, command, exit code, times,
  commit, modified tracked code files, and gate probes.
```bash
QUEUE_FILE=bench/queue_f5.txt DONE_FILE=results/bench/f5.done LOG_FILE=results/bench/f5.log \
  PROBE_LOG=results/bench/probe_f5.log MANIFEST=results/bench/f5_manifest.jsonl bench/queue2.sh
python3 bench/analyze_followup5.py > results/bench_f5_analysis.txt   # results/bench_f5_summary.json
```

## 11. Follow-up 6: the repairs on three stream seeds (pre-registered; Section VI, Table IV)

`paper_audit/followup6_prereg.md` (commit 8d79bd4) fixed the streams, the five update paths, and the rules H1–H5
before the runs. `paper_audit/followup6_results.md` records the outcome. `sift500k_adv_seed<k>` draws the row order,
the label direction and noise, the deletion noise, and the query labels from seed k; the plain stream is unchanged.
```bash
QUEUE_FILE=bench/queue_f6.txt DONE_FILE=results/bench/f6.done LOG_FILE=results/bench/f6.log \
  PROBE_LOG=results/bench/probe_f6.log MANIFEST=results/bench/f6_manifest.jsonl bench/queue2.sh
python3 bench/analyze_followup6.py > results/bench_f6_analysis.txt   # results/bench_f6_summary.json
```

## 12. Follow-up 7: close verdicts in paired blocks (pre-registered; Section V, Table IV)

`paper_audit/followup7_prereg.md` (commit 05adc09) fixed the 19 candidates, the blocks, and decisions D1 and D2
before the runs. `paper_audit/followup7_results.md` records the outcome.
```bash
QUEUE_FILE=bench/queue_f7.txt DONE_FILE=results/bench/f7.done LOG_FILE=results/bench/f7.log \
  PROBE_LOG=results/bench/probe_f7.log MANIFEST=results/bench/f7_manifest.jsonl bench/queue2.sh
python3 bench/analyze_followup7.py > results/bench_f7_analysis.txt   # results/bench_f7_summary.json
```

## 13. Follow-up 8: the deletion lesson on the second machine (pre-registered; Table V)

`paper_audit/followup8_prereg.md` (commit 5f32308) fixed the runs and decisions E1–E5 before the runs.

**Setup on the AMD EPYC 7513.** The machine is shared; there is no root and no system BLAS.
- Code: `git archive` of the commit, plus the patched `third_party/ACORN` without its build directories.
- Stream data: copied and SHA-256 checked.
- Python environment: `python3 -m venv venv && venv/bin/pip install numpy==2.2.6 faiss-cpu==1.15.1 mkl==2026.1.0`,
  then link `venv/lib/libmkl_rt.so` to `libmkl_rt.so.3`.
- zlib 1.3.1 built into a local prefix, because the machine has no zlib headers (tarball SHA-256
  9a93b2b7dfdac77ceba5a558a580e74667dd6fede4585b91eefb60f03b72df23).
- ACORN configured as in `scripts/build_acorn.sh`, adding `-DZLIB_ROOT=<prefix>` and building with
  `taskset -c 40-55 make -j16`.
- `src/acorn_stream.cpp` linked as `src/acorn_stream_avx2`.

**Placement and gate.**
- Jobs run on cores 40–55 and the probe on core 39.
- The quiet probe median was 653, so the gate is 555 (85%).
```bash
export PATH=$PWD/venv/bin:$PATH
GATE=555 PROBE_CPU=39 QUEUE_FILE=bench/queue_f8.txt DONE_FILE=results/bench/f8.done LOG_FILE=results/bench/f8.log \
  PROBE_LOG=results/bench/probe_f8.log MANIFEST=results/bench/f8_manifest.jsonl RUNS_DIR=results/runs bench/queue2.sh
# copied back to results/bench_epyc_str[_pass2], results/bench_epyc_str_log/
python3 bench/analyze_followup8.py > results/bench_f8_analysis.txt   # results/bench_f8_summary.json
```

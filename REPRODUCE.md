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
scripts/build_acorn.sh     # ACORN c259f11 + patches/acorn_c259f11_cmake_and_add_link.patch, AVX2 build, src/acorn_*_avx2
scripts/build_curator.sh   # Curator (hatsu3/curator-v2 c185042), its Faiss fork with Python bindings, in .venv-curator
```

`patches/acorn_c259f11_cmake_and_add_link.patch` fixes (a) the CMake configuration with a system nlohmann_json and
(b) an out-of-bounds write in ACORN's incremental insertion (`add_link` on full upper-level lists). The fork's
documented build compiles scalar distance kernels (`FAISS_OPT_LEVEL` is declared with `option()`); the script
passes `-DFAISS_OPT_LEVEL=avx2`.

## 2. Data

```bash
scripts/download_data.sh          # SIFT1M and DEEP (ann-benchmarks), YFCC-10M (Big-ANN'23 filtered track), BIGANN prefix
sha256sum -c data_sha256.txt      # the YFCC and BIGANN entries; the hdf5 entries cover the ann-benchmarks files
```

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

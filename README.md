# Filtered Vector Search Under Updates — benchmark artifact

Artifact of *"[Experiment, Analysis, and Benchmark] Filtered Vector Search Under Updates: When Does an Index Beat a
Scan?"* (ICDE 2027 submission; `paper/paper_eab.pdf`).

It contains the benchmark harness (workload generators, one runner per method and environment, update-stream
runners, gated measurement queue, analysis), build scripts and patches for the baselines (Faiss IVF, hnswlib,
ACORN, Filtered-DiskANN via diskannpy, Curator), the regenerable pool index RIG (the author's own design point),
all raw per-configuration and per-query results, and the scripts that turn them into every number, table, and
figure of the paper.

## Regenerate the paper from the released results (minutes, no datasets)

```bash
pip install -r requirements.txt            # numpy, matplotlib, pytest, ...
python3 -m pytest -q tests/test_bench.py tests/test_cc_stats.py tests/test_analysis_gates.py   # harness tests, analysis gates
# tests/test_units.py needs the RIG kernels (scripts/build_kernels.sh)
python3 bench/analyze.py > results/bench_analysis.txt
python3 paper/make_eab_numbers.py          # paper/eab_numbers.tex: every number in the text is a macro from here
python3 paper/make_eab_figures.py          # paper/figures/eab_*.pdf
cd paper && pdflatex paper_eab.tex && pdflatex paper_eab.tex
```

The follow-up campaign (cost model, crossover law, percolation and repair, rebuild law, PostgreSQL, Milvus, arXiv 4096-d;
paper Sections IV-VIII) has its own analysis and macro files:

```bash
python3 bench/analyze_h.py > results/bench_h_analysis.txt   # results/bench_h_summary.json
python3 bench/analyze_review.py > results/bench_r_analysis.txt   # results/bench_r_summary.json (review campaigns, cost frontier)
python3 paper/make_eab_hnumbers.py                           # paper/eab_hnumbers.tex  (\V{h/...})
python3 paper/make_eab_anumbers.py                           # paper/eab_anumbers.tex  (\V{a/...}; needs both summaries)
python3 paper/make_eab_hfigures.py                           # paper/figures/eab_teaser.pdf (Fig. 1), eab_law.pdf, eab_percolation.pdf
python3 paper/make_eab_rfigures.py                           # paper/figures/eab_frontier.pdf (total-cost frontier)
python3 bench/analyze_followup2.py > results/bench_f2_analysis.txt   # results/bench_f2_summary.json (pre-registered follow-ups 2 and 4)
python3 bench/analyze_followup3.py > results/bench_f3_analysis.txt   # results/bench_f3_summary.json (second machine)
python3 bench/analyze_followup5.py > results/bench_f5_analysis.txt   # results/bench_f5_summary.json (every round, memory)
python3 paper/make_eab_fnumbers.py                           # paper/eab_fnumbers.tex  (\V{f/...}, Table IV)
python3 paper/make_eab_ffigures.py                           # paper/figures/eab_reach.pdf (reachability figure)
```

`analyze_review.py` and `analyze_followup2.py` stop on an incomplete campaign instead of reporting a partial result:
a missing period, round, measured end round, IVF anchor, expected method, or re-execution job. `analyze_followup3.py` stops
if any job of the second machine's list is missing. `results/bench_summary.json`, `results/bench_h_summary.json`, `results/bench_r_summary.json`, `results/bench_f2_summary.json`, and the four macro files as shipped are the outputs
of these commands; `scripts/make_release.sh` checked that they regenerate byte-identically from this bundle alone.

## Re-run the measurements

See **[REPRODUCE.md](REPRODUCE.md)**: environments and pinned versions, baseline builds (including the ACORN
patch in `patches/`), data download and checksums, the gated queue, and the analysis pipeline. The full queue takes
about two days on a 2-socket, 56-core machine.

## Layout

| path | content |
|---|---|
| `bench/` | harness, runners, stream runners, queue (`queue2.sh`, `make_queue.py`), analysis (`analyze.py`, `analyze_h.py`) |
| `src/`, `experiments/` | ACORN drivers (`acorn_*.cpp`); RIG kernels (`rig.cpp`, `carve.cpp`, `rigx.py`) and the modules its unit tests use |
| `patches/`, `scripts/` | ACORN patch; environment setup, builds, data download |
| `results/bench*/` | raw results: two single-thread passes, 28-thread runs, same-session 1-thread references |
| `results/bench_<tag>[_pass<k>]/` | follow-up campaign by tag (`h1`...`h8`, `pg`, `mv`, `dense`, `ax`; table in REPRODUCE.md section 6); queue file `bench/queue_h.txt`, log `results/bench/h.log` |
| `results/bench_superseded/` | results discarded during the study, one directory per reason |
| `results/workload_stats.json` | admissible-set sizes and stream summaries the analysis needs without the datasets |
| `paper/` | paper source, generated macros, number and figure generators |

## Data sources and terms

This artifact contains no dataset content: no vectors, tags, attributes, ground-truth ids, or text. The
`results/` files hold only our measurements (per-query recall and time, admissible-set sizes). The inputs are
fetched from the hosts below by `scripts/download_data.sh` and `bench/fetch_arxiv.py` and verified with
`data_sha256.txt`. They remain under their providers' terms; if you use them, cite their sources.

| Data | Fetched from | Terms (provider) | Please cite |
|---|---|---|---|
| SIFT1M (ANN_SIFT1M, TEXMEX/IRISA) | ann-benchmarks `sift-128-euclidean.hdf5` | CC0 1.0 (http://corpus-texmex.irisa.fr/) | Jégou, Douze, Schmid, IEEE TPAMI 2011; Aumüller et al., Inf. Syst. 2020 (ann-benchmarks) |
| BIGANN / ANN_SIFT1B (first 1.5M vectors, 10K queries) | `dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/` | CC0 (https://big-ann-benchmarks.com/neurips21.html) | Jégou, Tavenard, Douze, Amsaleg, ICASSP 2011 |
| DEEP (Deep1B, Yandex; 10M subset) | ann-benchmarks `deep-image-96-angular.hdf5` | CC BY 4.0 (https://research.yandex.com/datasets/biganns) | Babenko, Lempitsky, CVPR 2016 |
| YFCC-10M (Big-ANN NeurIPS'23 filtered track; CLIP embeddings and tags derived from YFCC100M) | `dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/yfcc100M/` | CC BY 4.0 (https://big-ann-benchmarks.com/neurips23.html); the underlying YFCC100M is subject to Yahoo's Webscope terms and to each photo's Creative Commons license | Simhadri et al., NeurIPS 2023 competition report; Thomee et al., CACM 2016 |
| arXiv abstracts, 4096-d stella_en_400M_v5 embeddings with categories (arxiv-for-fanns-large) | `huggingface.co/datasets/SPCL/arxiv-for-fanns-large` (revision 79a4cff) | No license stated by the provider. Built from arXiv metadata (CC0 1.0, https://info.arxiv.org/help/license/index.html); individual papers keep their own licenses; embedding model MIT | Iff et al., SIGIR 2026 (arXiv:2507.21989) |
| Amazon Reviews 2023 (McAuley Lab), 10 categories, embedded by us with all-MiniLM-L6-v2 (follow-up 2) | `huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023` (raw review files) | No data license published by the provider; we release no review text, vectors, or ids, only measurements. Embedding model: Apache-2.0 | Hou et al., "Bridging language and items for retrieval and recommendation," arXiv:2403.03952, 2024 |

Changes we make (all by script, with fixed seeds): query subsets, L2 normalization of DEEP, random subsets of
DEEP and of the arXiv set, synthetic labels, predicates, exact filtered ground truth, and update streams. The
results and figures are CC BY 4.0 (`LICENSE-results`); that license does not extend to any of the datasets.

## License

Code: MIT (`LICENSE`). Results and figures: CC BY 4.0 (`LICENSE-results`). The manuscript is included for
reference only. Datasets are not redistributed.

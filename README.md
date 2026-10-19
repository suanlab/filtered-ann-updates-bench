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
python3 -m pytest -q tests/test_bench.py   # harness unit tests
python3 bench/analyze.py > results/bench_analysis.txt
python3 paper/make_eab_numbers.py          # paper/eab_numbers.tex: every number in the text is a macro from here
python3 paper/make_eab_figures.py          # paper/figures/eab_*.pdf
cd paper && pdflatex paper_eab.tex && pdflatex paper_eab.tex
```

`results/bench_summary.json` and `paper/eab_numbers.tex` as shipped are the outputs of these commands.

## Re-run the measurements

See **[REPRODUCE.md](REPRODUCE.md)**: environments and pinned versions, baseline builds (including the ACORN
patch in `patches/`), data download and checksums, the gated queue, and the analysis pipeline. The full queue takes
about two days on a 2-socket, 56-core machine.

## Layout

| path | content |
|---|---|
| `bench/` | harness, runners, stream runners, queue (`queue2.sh`, `make_queue.py`), analysis (`analyze.py`) |
| `src/`, `experiments/` | ACORN drivers (`acorn_*.cpp`); RIG kernels (`rig.cpp`, `carve.cpp`, `rigx.py`) and the modules its unit tests use |
| `patches/`, `scripts/` | ACORN patch; environment setup, builds, data download |
| `results/bench*/` | raw results: two single-thread passes, 28-thread runs, same-session 1-thread references |
| `results/bench_superseded/` | results discarded during the study, one directory per reason |
| `results/workload_stats.json` | admissible-set sizes and stream summaries the analysis needs without the datasets |
| `paper/` | paper source, generated macros, number and figure generators |

## License

Code: MIT (`LICENSE`). Results and figures: CC BY 4.0 (`LICENSE-results`). The manuscript is included for
reference only. Datasets are not redistributed.

#!/usr/bin/env bash
# Python environments: main (hnswlib, RIG, analysis), .venv-faiss (Faiss: scans, IVF), .venv-diskann (Filtered-DiskANN).
# diskannpy 0.7.0 requires numpy<2, hence its own environment.
set -euo pipefail; cd "$(dirname "$0")/.."
python3 -m pip install -r requirements.txt
python3 -m venv .venv-faiss   && .venv-faiss/bin/pip install -r requirements-faiss.txt
python3 -m venv .venv-diskann && .venv-diskann/bin/pip install -r requirements-diskann.txt

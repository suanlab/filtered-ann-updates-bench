#!/usr/bin/env bash
# EMA arm (bench/stream_ema.py): the authors' public code at a fixed commit, Python bindings built into .venv-ema.
set -euo pipefail; cd "$(dirname "$0")/.."
C=d95fb49973f4fe53f5a2c1ac444dd5fdb879f551
[[ -d third_party/EMA ]] || git clone -q https://github.com/lmccccc/EMA.git third_party/EMA
git -C third_party/EMA checkout -q "$C"
python3 -m venv .venv-ema && .venv-ema/bin/pip -q install --upgrade pip "numpy==2.2.6" "pybind11==3.1.0" setuptools wheel "faiss-cpu==1.15.1"
.venv-ema/bin/pip -q install -e third_party/EMA/hnswlib
.venv-ema/bin/python -c "import hashannlib; print('EMA (hashannlib) ready at', '$C')"

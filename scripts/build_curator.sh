#!/usr/bin/env bash
# Curator (Jin et al., hatsu3/curator-v2 at c185042): its Faiss fork with Python bindings in .venv-curator.
# The fork needs CMake >= 3.23.1 and a SWIG older than 4.2 (4.2 changed long long typemaps), both from pip.
set -euo pipefail; cd "$(dirname "$0")/.."
python3 -m venv .venv-curator && .venv-curator/bin/pip install numpy==1.25.0 swig==4.1.1 cmake==3.26.4 setuptools wheel
C=third_party/curator-v2
[ -d $C ] || { git clone https://github.com/hatsu3/curator-v2 $C; git -C $C checkout c185042; }
export PATH=$PWD/.venv-curator/bin:$PATH
cmake -S $C/3rd_party/faiss -B $C/3rd_party/faiss/build -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=ON -DCMAKE_BUILD_TYPE=Release \
      -DFAISS_OPT_LEVEL=avx2 -DBUILD_TESTING=OFF -DBLA_VENDOR=Intel10_64_dyn -DMKL_LIBRARIES=${MKL_LIB:-$HOME/.local/lib/libmkl_rt.so} \
      -DPython_EXECUTABLE=$PWD/.venv-curator/bin/python -DSWIG_EXECUTABLE=$PWD/.venv-curator/bin/swig
make -C $C/3rd_party/faiss/build -j"$(nproc)" faiss_avx2 swigfaiss_avx2
(cd $C/3rd_party/faiss/build/faiss/python && $OLDPWD/.venv-curator/bin/python setup.py install)

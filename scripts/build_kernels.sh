#!/usr/bin/env bash
# RIG kernels (search kernel and C++ carver), loaded through ctypes by experiments/rigx.py. Requires g++ >= 11, OpenMP.
set -euo pipefail; cd "$(dirname "$0")/../src"
g++ -O3 -march=native -fopenmp -shared -fPIC -o librig.so rig.cpp
g++ -O3 -march=native -fopenmp -shared -fPIC -o libcarve.so carve.cpp

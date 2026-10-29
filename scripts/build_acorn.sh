#!/usr/bin/env bash
# ACORN (stanford-futuredata/ACORN) at commit c259f11 with our patch, built with AVX2, plus the two drivers.
# The patch (patches/acorn_c259f11_cmake_and_add_link.patch) does two things:
#  - CMakeLists.txt: exposes the nlohmann_json include directory only at build time (the stock file fails to
#    configure with a system nlohmann_json);
#  - faiss/impl/ACORN.cpp: truncates the merged candidate set of a full upper-level neighbor list to its capacity
#    (upstream writes one entry past the node's range, which corrupts the graph and crashes incremental add).
# The fork declares FAISS_OPT_LEVEL with option(), so the documented build silently compiles scalar distance
# kernels; we pass -DFAISS_OPT_LEVEL=avx2 and link libfaiss_avx2. BLAS: Intel MKL (set MKL_LIB below).
set -euo pipefail; cd "$(dirname "$0")/.."
MKL_LIB=${MKL_LIB:-$HOME/.local/lib/libmkl_rt.so}
A=third_party/ACORN
if [ ! -d $A ]; then git clone https://github.com/stanford-futuredata/ACORN.git $A; git -C $A checkout c259f11; git -C $A apply ../../patches/acorn_c259f11_cmake_and_add_link.patch; git -C $A apply ../../patches/acorn_c259f11_count_checks.patch; fi
[ "${BUILD_ONLY_DRIVERS:-0}" = 1 ] || cmake -S $A -B $A/build_avx2 -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=OFF -DBUILD_SHARED_LIBS=ON \
      -DCMAKE_BUILD_TYPE=Release -DFAISS_OPT_LEVEL=avx2 -DBLA_VENDOR=Intel10_64_dyn -DMKL_LIBRARIES=$MKL_LIB
[ "${BUILD_ONLY_DRIVERS:-0}" = 1 ] || make -C $A/build_avx2 -j"$(nproc)" faiss_avx2
link() { g++ -O3 -march=native -fopenmp -std=c++17 -I$A "src/$1.cpp" -o "src/$2" -L$A/build_avx2/faiss -lfaiss_avx2 \
             -L"$(dirname $MKL_LIB)" -lmkl_rt -Wl,-rpath,$PWD/$A/build_avx2/faiss -Wl,-rpath,"$(dirname $MKL_LIB)"; }
link acorn_driver acorn_driver_avx2
link acorn_stream acorn_stream_avx2
# Follow-up 2 and 4: the same acorn_stream.cpp under a second name (the instrumented modes MECH, FRESH, E2E, ATTR_CSR
# are environment switches, off by default; the first campaigns ran a binary built before they were added, so the
# follow-up runs use a separate file), and the concurrency driver.
link acorn_stream acorn_stream_mech_avx2
link cc_ivf cc_ivf
[ "${BUILD_ONLY_DRIVERS:-0}" = 1 ] && exit 0
# Counting build (cost accounting only, H2): the same source compiled with -DACORN_COUNT_CHECKS, which enables
# per-thread counters of level-0 hops and predicate checks (patches/acorn_c259f11_count_checks.patch; without the
# define the patch compiles to nothing, so the measured build above is unaffected).
cmake -S $A -B $A/build_count -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=OFF -DBUILD_SHARED_LIBS=ON \
      -DCMAKE_BUILD_TYPE=Release -DFAISS_OPT_LEVEL=avx2 -DBLA_VENDOR=Intel10_64_dyn -DMKL_LIBRARIES=$MKL_LIB -DCMAKE_CXX_FLAGS=-DACORN_COUNT_CHECKS
make -C $A/build_count -j"$(nproc)" faiss_avx2
g++ -O3 -march=native -fopenmp -std=c++17 -DACORN_COUNT_CHECKS -I$A src/acorn_driver.cpp -o src/acorn_driver_count -L$A/build_count/faiss -lfaiss_avx2 \
    -L"$(dirname $MKL_LIB)" -lmkl_rt -Wl,-rpath,$PWD/$A/build_count/faiss -Wl,-rpath,"$(dirname $MKL_LIB)"

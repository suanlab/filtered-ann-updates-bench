#!/usr/bin/env bash
# Inputs (about 9 GB): ann-benchmarks SIFT1M and DEEP (hdf5), the Big-ANN'23 filtered track (YFCC-10M), and the
# first 1.5M vectors of BIGANN (SIFT1B) for the SIFT update streams. Verify with: sha256sum -c data_sha256.txt
set -euo pipefail; cd "$(dirname "$0")/.."; mkdir -p data/yfcc data/bigann
AB=http://ann-benchmarks.com; FB=https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks
curl -fL -o data/sift-128-euclidean.hdf5 $AB/sift-128-euclidean.hdf5
curl -fL -o data/deep-image-96-angular.hdf5 $AB/deep-image-96-angular.hdf5
for f in base.10M.u8bin base.metadata.10M.spmat query.public.100K.u8bin query.metadata.public.100K.spmat GT.public.ibin; do
  curl -fL -o data/yfcc/$f $FB/yfcc100M/$f; done
curl -fL -r 0-$((8 + 1500000 * 128 - 1)) -o data/bigann/base.1.5M.u8bin $FB/bigann/base.1B.u8bin   # header says 1e9 rows; we read 1.5M
curl -fL -o data/bigann/query.public.10K.u8bin $FB/bigann/query.public.10K.u8bin

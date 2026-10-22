#!/usr/bin/env bash
# Milvus arm (bench/run_milvus.py): Milvus 2.6.18 standalone from the official .deb, extracted without root, and a
# client venv with pymilvus 2.6. No system packages, no Docker; the runner starts the server in a private
# user+network namespace (unshare -rn), so unprivileged user namespaces must be enabled.
set -euo pipefail; cd "$(dirname "$0")/.."
V=2.6.18; DEB=milvus_${V}-1_amd64.deb; SHA=5305a66f83fcde501f1716c878026a64b16f015ab30dc471a7239b59d3ea6bf9
mkdir -p third_party/milvus && cd third_party/milvus
[[ -f $DEB ]] || curl -sSL -o "$DEB" "https://github.com/milvus-io/milvus/releases/download/v${V}/${DEB}"
echo "$SHA  $DEB" | sha256sum -c
rm -rf root && dpkg-deb -x "$DEB" root
cd ../..
python3 -m venv .venv-milvus && .venv-milvus/bin/pip -q install "pymilvus==2.6.17" "numpy==2.2.6"
python3 bench/milvus_conf.py
unshare -rn true && echo "milvus ready: taskset -c <cores> .venv-milvus/bin/python bench/run_milvus.py <ds> <schemas>"

Milvus pass 1 (2026-10-22 14:32-14:59), superseded. The runner built indexes right after the flush, while Milvus was
still merging the flushed segments in the background: the HNSW sweeps ran on four segments (about 2.9e5, 2.9e5,
2.95e5, and 1.25e5 rows) and the IVF_FLAT and FLAT sweeps on one merged segment of 1e6 rows, and the merge happened
during the HNSW ef sweep (the same brute-force path ran at 148 QPS in the default-parameter sweep and at 89 QPS in
the ef sweep on DEEP-1M, s=0.01). bench/run_milvus.py now compacts until the layout is stable before building any
index and asserts the layout at the start and end of every sweep. The SIFT1M pass-1 job was stopped (exit 143).

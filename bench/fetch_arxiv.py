#!/usr/bin/env python3
"""Fetch a random sample of the arxiv-for-fanns-large dataset (Iff et al., SIGIR'26; HuggingFace SPCL/arxiv-for-fanns-large):
4096-dimensional stella_en_400M_v5 embeddings of arXiv abstracts with real attributes.
  python3 bench/fetch_arxiv.py [n_sample=1000000]   -> data/arxiv/large_sample/{X.f16.npy, rows.npy, attributes.jsonl, Q.npy, qattr_emis.jsonl}
The full vector file is 44.8 GB; we read N_SAMPLE rows as random blocks of 1024 consecutive rows (seed 19) through HTTP
range requests and store them as float16 (the runners upcast to float32). Attributes are kept for the sampled rows; the
query vectors and their category labels (emis_query_attributes) are kept whole."""
import sys, os, json, urllib.request, concurrent.futures as cf
import numpy as np

BASE = 'https://huggingface.co/datasets/SPCL/arxiv-for-fanns-large/resolve/main/'
D = 4096; ROW = 4 + 4 * D; NROWS = 44825506432 // ROW; BLOCK = 1024
OUT = 'data/arxiv/large_sample'; os.makedirs(OUT, exist_ok=True)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000

import time, urllib.error
def get(url, lo, hi):
    for attempt in range(12):
        try:
            req = urllib.request.Request(url, headers={'Range': f'bytes={lo}-{hi}'})
            with urllib.request.urlopen(req, timeout=300) as r: return r.read()
        except Exception as e:       # 429 (rate limit) and transient errors: exponential backoff
            if attempt == 11: raise
            time.sleep(min(600, 15 * 2 ** attempt) if isinstance(e, urllib.error.HTTPError) and e.code == 429 else 5 * (attempt + 1))
rng = np.random.default_rng(19)
nb = NROWS // BLOCK; blocks = np.sort(rng.choice(nb, N // BLOCK, replace=False))
rows = (blocks[:, None] * BLOCK + np.arange(BLOCK)[None, :]).ravel(); np.save(f'{OUT}/rows.npy', rows)
fx = f'{OUT}/X.f16.npy'; fdone = f'{OUT}/blocks_done.txt'
X = np.lib.format.open_memmap(fx, mode='r+' if os.path.exists(fx) else 'w+', dtype=np.float16, shape=(len(rows), D))
done = set(int(l) for l in open(fdone)) if os.path.exists(fdone) else set(); flog = open(fdone, 'a')
def fetch(k):
    if k in done: return k
    b = int(blocks[k]); raw = get(BASE + 'database_vectors.fvecs', b * BLOCK * ROW, (b + 1) * BLOCK * ROW - 1)
    a = np.frombuffer(raw, np.int32).reshape(BLOCK, ROW // 4)
    assert (a[:, 0] == D).all(), 'unexpected fvecs header'
    X[k * BLOCK:(k + 1) * BLOCK] = a[:, 1:].view(np.float32).astype(np.float16); flog.write(f'{k}\n'); flog.flush(); return k
with cf.ThreadPoolExecutor(4) as ex:
    for i, _ in enumerate(ex.map(fetch, range(len(blocks)))):
        if i % 50 == 0: print(f'{i}/{len(blocks)} blocks', flush=True)
X.flush(); del X
# attributes of the sampled rows (stream the 1.1 GB jsonl once)
want = set(rows.tolist()); keep = {}
with urllib.request.urlopen(BASE + 'database_attributes.jsonl', timeout=600) as r:
    for i, line in enumerate(r):
        if i in want: keep[i] = line.decode()
with open(f'{OUT}/attributes.jsonl', 'w') as f:
    for i in rows: f.write(keep[int(i)])
q = get(BASE + 'query_vectors.fvecs', 0, 163880000 - 1); qa = np.frombuffer(q, np.int32).reshape(-1, ROW // 4)
np.save(f'{OUT}/Q.npy', qa[:, 1:].view(np.float32).copy())
with urllib.request.urlopen(BASE + 'emis_query_attributes.jsonl', timeout=120) as r: open(f'{OUT}/qattr_emis.jsonl', 'wb').write(r.read())
json.dump({'source': 'SPCL/arxiv-for-fanns-large (HuggingFace), random sample of 64-row blocks, seed 19', 'n': int(len(rows)), 'd': D,
           'rows_total': int(NROWS)}, open(f'{OUT}/meta.json', 'w'), indent=1)
print('done', len(rows), 'rows')

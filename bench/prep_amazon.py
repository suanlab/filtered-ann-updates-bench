#!/usr/bin/env python3
"""Second real temporal source (follow-up 2, I18): Amazon Reviews 2023 (McAuley Lab), 10 categories.
  .venv-embed/bin/python bench/prep_amazon.py download     -> data/amazon/raw/<category>.jsonl (Hugging Face)
  .venv-embed/bin/python bench/prep_amazon.py select       -> data/amazon/selected.npz (time, category, row offsets)
  taskset -c <cores> .venv-embed/bin/python bench/prep_amazon.py embed   -> data/amazon/X.f16.npy, attributes.jsonl
Reviews from 2016-01-01 to 2022-12-31 of the 10 categories below, text = title + ". " + text (empty, very short, and
exactly repeated texts dropped), a uniform sample of 700,000 kept in time order (seed 2027). Embeddings:
sentence-transformers/all-MiniLM-L6-v2 (384-d, mean pooling, L2-normalized, max 256 tokens), CPU.
No review text is written to disk beyond the downloaded raw files (the dataset publishes no redistribution license):
attributes.jsonl holds the timestamp, the category, and a SHA-1 of (user_id, parent_asin, timestamp)."""
import sys, os, json, hashlib, subprocess
import numpy as np

CATS = ['All_Beauty', 'Appliances', 'Digital_Music', 'Gift_Cards', 'Handmade_Products', 'Health_and_Personal_Care',
        'Magazine_Subscriptions', 'Software', 'Amazon_Fashion', 'Musical_Instruments']
URL = 'https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories/{}.jsonl'
D = 'data/amazon'; T0, T1 = 1451606400000, 1672531200000        # 2016-01-01, 2023-01-01 (ms, UTC)
N_KEEP = 700_000
step = sys.argv[1]
if step == 'download':
    os.makedirs(f'{D}/raw', exist_ok=True)
    for c in CATS:
        f = f'{D}/raw/{c}.jsonl'
        if not os.path.exists(f): subprocess.run(['curl', '-sSLf', '-o', f + '.part', URL.format(c)], check=True); os.rename(f + '.part', f)
        print(c, os.path.getsize(f) // 10**6, 'MB', flush=True)
elif step == 'select':
    ts, cat, off, seen = [], [], [], set()
    for ci, c in enumerate(CATS):
        n0 = len(ts)
        with open(f'{D}/raw/{c}.jsonl', 'rb') as fh:
            pos = 0
            for line in fh:
                o = json.loads(line); t = int(o['timestamp']); here = pos; pos += len(line)
                if not (T0 <= t < T1): continue
                txt = (o.get('title') or '').strip() + '. ' + (o.get('text') or '').strip()
                if len(txt) < 20: continue
                h = hashlib.sha1(txt.encode()).digest()
                if h in seen: continue
                seen.add(h); ts.append(t); cat.append(ci); off.append(here)
        print(c, len(ts) - n0, flush=True)
    ts, cat, off = np.array(ts, np.int64), np.array(cat, np.int16), np.array(off, np.int64)
    keep = np.sort(np.random.default_rng(2027).choice(len(ts), N_KEEP, replace=False))
    order = keep[np.argsort(ts[keep], kind='stable')]
    np.savez(f'{D}/selected.npz', ts=ts[order], cat=cat[order], off=off[order], candidates=len(ts),
             candidates_per_cat=np.bincount(cat, minlength=len(CATS)))
    print('selected', N_KEEP, 'of', len(ts))
elif step == 'embed':
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(int(os.environ.get('EMBED_THREADS', '16')))
    S = dict(np.load(f'{D}/selected.npz')); n = len(S['ts']); texts = [None] * n; fh = {c: open(f'{D}/raw/{c}.jsonl', 'rb') for c in CATS}   # dict: load each array once
    ids = []
    for i in range(n):
        f = fh[CATS[S['cat'][i]]]; f.seek(int(S['off'][i])); o = json.loads(f.readline())
        texts[i] = (o.get('title') or '').strip() + '. ' + (o.get('text') or '').strip()
        ids.append(hashlib.sha1(f"{o.get('user_id')}|{o.get('parent_asin')}|{o.get('timestamp')}".encode()).hexdigest())
    m = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu'); m.max_seq_length = 256
    X = np.zeros((n, 384), np.float16); B = 20_000
    for s in range(0, n, B):
        X[s:s + B] = m.encode(texts[s:s + B], batch_size=256, normalize_embeddings=True, show_progress_bar=False).astype(np.float16)
        print('embedded', min(n, s + B), flush=True)
    np.save(f'{D}/X.f16.npy', X)
    with open(f'{D}/attributes.jsonl', 'w') as fo:
        for i in range(n): fo.write(json.dumps({'id': ids[i], 'timestamp': int(S['ts'][i]), 'category': CATS[S['cat'][i]]}) + '\n')
    json.dump({'model': 'sentence-transformers/all-MiniLM-L6-v2', 'dim': 384, 'normalized': True, 'max_seq_length': 256, 'n': n,
               'window_ms': [T0, T1], 'categories': CATS, 'candidates': int(S['candidates']),
               'candidates_per_cat': S['candidates_per_cat'].tolist(), 'source': URL.format('<category>')}, open(f'{D}/meta.json', 'w'), indent=1)

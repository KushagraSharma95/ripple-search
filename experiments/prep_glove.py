"""Parse GloVe (word2vec text format, gzipped) into a normalized float32
matrix and build the shared kNN-graph substrate used by exp10.

Usage: python -I prep_glove.py <glove.gz> <out_dir>
The graph (33 approximate NNs per node via hnswlib, self stripped) is the
same substrate for fixed-ef, heuristic early-exit and the learned predictor,
so comparisons isolate the termination policy.
"""

import gzip
import sys
import time
from pathlib import Path

import numpy as np

src, out_dir = sys.argv[1], Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)

t0 = time.time()
with gzip.open(src, "rt", encoding="utf-8", errors="replace") as f:
    header = f.readline().split()
    n, d = int(header[0]), int(header[1])
    X = np.empty((n, d), dtype=np.float32)
    for i, line in enumerate(f):
        parts = line.rstrip().split(" ")
        X[i] = np.asarray(parts[-d:], dtype=np.float32)  # token may contain spaces
print(f"parsed {X.shape} in {time.time()-t0:.0f}s")

norms = np.linalg.norm(X, axis=1, keepdims=True)
keep = norms[:, 0] > 1e-6
X = (X[keep] / norms[keep]).astype(np.float32)
print(f"kept {len(X)} non-degenerate vectors")
np.save(out_dir / "glove100.npy", X)

import hnswlib

t0 = time.time()
idx = hnswlib.Index(space="cosine", dim=X.shape[1])
idx.init_index(max_elements=len(X), M=32, ef_construction=400, random_seed=0)
idx.add_items(X, np.arange(len(X)))
idx.set_ef(200)
labels, _ = idx.knn_query(X, k=33)
nbr = np.empty((len(X), 32), dtype=np.int32)
for i in range(len(X)):
    row = labels[i]
    row = row[row != i][:32]
    nbr[i, : len(row)] = row
    if len(row) < 32:  # degenerate duplicates: pad with first neighbor
        nbr[i, len(row):] = row[0] if len(row) else (i + 1) % len(X)
np.save(out_dir / "glove100_knn32.npy", nbr)
print(f"graph built in {time.time()-t0:.0f}s -> {out_dir}")

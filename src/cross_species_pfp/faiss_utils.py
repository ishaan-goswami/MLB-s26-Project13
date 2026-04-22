from __future__ import annotations

import faiss
import numpy as np


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return x / norms


def build_index(vectors: np.ndarray) -> faiss.Index:
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors.astype(np.float32))
    return index


def search_index(index: faiss.Index, queries: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    scores, indices = index.search(queries.astype(np.float32), top_k)
    return scores, indices


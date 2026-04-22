#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# macOS can load multiple OpenMP runtimes through numpy/faiss stacks.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import _bootstrap
from cross_species_pfp.config import ensure_dirs, load_config
from cross_species_pfp.faiss_utils import build_index, l2_normalize, search_index
from cross_species_pfp.projector import ProjectionMLP


def load_embeddings(path: Path) -> tuple[list[str], np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return data["ids"].tolist(), data["embeddings"]


def maybe_project(embeddings: np.ndarray, checkpoint_path: Path, device: str) -> np.ndarray:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = ProjectionMLP(
        input_dim=checkpoint["input_dim"],
        hidden_dims=checkpoint["hidden_dims"],
        projection_dim=checkpoint["projection_dim"],
        dropout=checkpoint["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.no_grad():
        tensor = torch.tensor(embeddings, dtype=torch.float32, device=device)
        projected = model(tensor).cpu().numpy()
    return projected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True, choices=["esm2_650m", "prott5_xl", "esmc_600m"])
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--use-projector", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)

    source_species = cfg["retrieval"]["source_species"]
    target_species = cfg["retrieval"]["target_species"]
    top_k = args.top_k or cfg["retrieval"]["top_k"]

    embeddings_dir = Path(cfg["paths"]["embeddings_dir"])
    search_dir = Path(cfg["paths"]["search_dir"])
    models_dir = Path(cfg["paths"]["models_dir"])
    search_dir.mkdir(parents=True, exist_ok=True)

    source_ids, source_embeddings = load_embeddings(embeddings_dir / f"{source_species}_{args.model}.npz")
    target_ids, target_embeddings = load_embeddings(embeddings_dir / f"{target_species}_{args.model}.npz")

    suffix = "raw"
    if args.use_projector:
        checkpoint_path = models_dir / f"{args.model}_projector.pt"
        source_embeddings = maybe_project(source_embeddings, checkpoint_path, args.device)
        target_embeddings = maybe_project(target_embeddings, checkpoint_path, args.device)
        suffix = "projected"

    source_embeddings = l2_normalize(source_embeddings)
    target_embeddings = l2_normalize(target_embeddings)
    index = build_index(target_embeddings)
    scores, indices = search_index(index, source_embeddings, top_k)

    rows = []
    for i, query_id in enumerate(source_ids):
        for rank in range(top_k):
            neighbor_index = int(indices[i, rank])
            rows.append(
                {
                    "query_id": query_id,
                    "rank": rank + 1,
                    "neighbor_id": target_ids[neighbor_index],
                    "score": float(scores[i, rank]),
                }
            )

    output_path = search_dir / f"{source_species}_to_{target_species}_{args.model}_{suffix}_top{top_k}.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Wrote search results to {output_path}")


if __name__ == "__main__":
    main()

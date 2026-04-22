#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

import _bootstrap
from cross_species_pfp.config import ensure_dirs, load_config
from cross_species_pfp.evaluation import jaccard
from cross_species_pfp.io_utils import read_json
from cross_species_pfp.projector import ProjectionMLP


class PairDataset(Dataset):
    def __init__(self, left: np.ndarray, right: np.ndarray, targets: np.ndarray):
        self.left = torch.tensor(left, dtype=torch.float32)
        self.right = torch.tensor(right, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int):
        return self.left[idx], self.right[idx], self.targets[idx]


def load_embeddings(path: Path) -> tuple[list[str], np.ndarray]:
    data = np.load(path, allow_pickle=True)
    ids = data["ids"].tolist()
    embeddings = data["embeddings"]
    return ids, embeddings


def load_annotations(path: Path) -> dict[str, dict[str, set[str]]]:
    raw = read_json(path)
    return {
        protein_id: {aspect: set(terms) for aspect, terms in aspect_map.items()}
        for protein_id, aspect_map in raw.items()
    }


def any_go_overlap(left: dict[str, set[str]], right: dict[str, set[str]]) -> float:
    all_left = set().union(*left.values()) if left else set()
    all_right = set().union(*right.values()) if right else set()
    return jaccard(all_left, all_right)


def build_training_pairs(
    source_ids: list[str],
    source_embeddings: np.ndarray,
    target_ids: list[str],
    target_embeddings: np.ndarray,
    source_annotations: dict[str, dict[str, set[str]]],
    target_annotations: dict[str, dict[str, set[str]]],
    positives_per_query: int,
    negatives_per_query: int,
    min_jaccard_for_positive: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    target_index = {protein_id: i for i, protein_id in enumerate(target_ids)}
    term_to_targets: dict[str, set[str]] = defaultdict(set)
    for protein_id, aspect_map in target_annotations.items():
        for terms in aspect_map.values():
            for term in terms:
                term_to_targets[term].add(protein_id)

    left_rows = []
    right_rows = []
    targets = []

    for i, source_id in enumerate(source_ids):
        src_ann = source_annotations.get(source_id, {})
        candidate_targets = set()
        for terms in src_ann.values():
            for term in terms:
                candidate_targets.update(term_to_targets.get(term, set()))

        positive_candidates = []
        for candidate in candidate_targets:
            score = any_go_overlap(src_ann, target_annotations.get(candidate, {}))
            if score >= min_jaccard_for_positive:
                positive_candidates.append((candidate, score))

        rng.shuffle(positive_candidates)
        selected_positives = positive_candidates[:positives_per_query]
        for target_id, score in selected_positives:
            left_rows.append(source_embeddings[i])
            right_rows.append(target_embeddings[target_index[target_id]])
            targets.append(score)

        excluded = {protein_id for protein_id, _ in selected_positives}
        negative_pool = [protein_id for protein_id in target_ids if protein_id not in excluded]
        rng.shuffle(negative_pool)
        for target_id in negative_pool[:negatives_per_query]:
            left_rows.append(source_embeddings[i])
            right_rows.append(target_embeddings[target_index[target_id]])
            targets.append(0.0)

    return np.asarray(left_rows), np.asarray(right_rows), np.asarray(targets, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True, choices=["esm2_650m", "prott5_xl", "esmc_600m"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)

    embeddings_dir = Path(cfg["paths"]["embeddings_dir"])
    models_dir = Path(cfg["paths"]["models_dir"])
    processed_dir = Path(cfg["paths"]["processed_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    source_species = cfg["retrieval"]["source_species"]
    target_species = cfg["retrieval"]["target_species"]
    training_cfg = cfg["training"]

    source_ids, source_embeddings = load_embeddings(embeddings_dir / f"{source_species}_{args.model}.npz")
    target_ids, target_embeddings = load_embeddings(embeddings_dir / f"{target_species}_{args.model}.npz")
    source_annotations = load_annotations(processed_dir / f"{source_species}_go_annotations.json")
    target_annotations = load_annotations(processed_dir / f"{target_species}_go_annotations.json")

    left, right, targets = build_training_pairs(
        source_ids,
        source_embeddings,
        target_ids,
        target_embeddings,
        source_annotations,
        target_annotations,
        training_cfg["positives_per_query"],
        training_cfg["negatives_per_query"],
        training_cfg["min_jaccard_for_positive"],
        cfg["project"]["seed"],
    )

    dataset = PairDataset(left, right, targets)
    loader = DataLoader(dataset, batch_size=training_cfg["batch_size"], shuffle=True)

    model = ProjectionMLP(
        input_dim=source_embeddings.shape[1],
        hidden_dims=training_cfg["projector_hidden_dims"],
        projection_dim=training_cfg["projection_dim"],
        dropout=training_cfg["projector_dropout"],
    ).to(args.device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_cfg["lr"],
        weight_decay=training_cfg["weight_decay"],
    )
    loss_fn = torch.nn.MSELoss()

    for epoch in range(training_cfg["epochs"]):
        model.train()
        running_loss = 0.0
        for left_batch, right_batch, target_batch in loader:
            left_batch = left_batch.to(args.device)
            right_batch = right_batch.to(args.device)
            target_batch = target_batch.to(args.device)

            optimizer.zero_grad()
            left_proj = model(left_batch)
            right_proj = model(right_batch)
            pred = torch.sum(left_proj * right_proj, dim=-1)
            loss = loss_fn(pred, target_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(target_batch)

        epoch_loss = running_loss / max(len(dataset), 1)
        print(f"epoch={epoch + 1} loss={epoch_loss:.6f}")

    checkpoint_path = models_dir / f"{args.model}_projector.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": source_embeddings.shape[1],
            "hidden_dims": training_cfg["projector_hidden_dims"],
            "projection_dim": training_cfg["projection_dim"],
            "dropout": training_cfg["projector_dropout"],
            "model_key": args.model,
        },
        checkpoint_path,
    )
    print(f"Saved projector to {checkpoint_path}")


if __name__ == "__main__":
    main()

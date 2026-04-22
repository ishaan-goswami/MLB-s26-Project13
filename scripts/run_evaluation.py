#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import _bootstrap
from cross_species_pfp.config import ensure_dirs, load_config
from cross_species_pfp.evaluation import evaluate_retrieval
from cross_species_pfp.io_utils import read_json, write_json


def normalize_annotation_json(raw: dict) -> dict[str, dict[str, set[str]]]:
    output = {}
    for protein_id, aspect_map in raw.items():
        output[protein_id] = {aspect: set(terms) for aspect, terms in aspect_map.items()}
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True, choices=["esm2_650m", "prott5_xl", "esmc_600m"])
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--use-projector", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)

    source_species = cfg["retrieval"]["source_species"]
    target_species = cfg["retrieval"]["target_species"]
    top_k = args.top_k or cfg["retrieval"]["top_k"]
    suffix = "projected" if args.use_projector else "raw"

    search_path = Path(cfg["paths"]["search_dir"]) / f"{source_species}_to_{target_species}_{args.model}_{suffix}_top{top_k}.csv"
    eval_dir = Path(cfg["paths"]["eval_dir"])
    processed_dir = Path(cfg["paths"]["processed_dir"])
    eval_dir.mkdir(parents=True, exist_ok=True)

    search_results = pd.read_csv(search_path)
    source_annotations = normalize_annotation_json(read_json(processed_dir / f"{source_species}_go_annotations.json"))
    target_annotations = normalize_annotation_json(read_json(processed_dir / f"{target_species}_go_annotations.json"))

    detailed, summary = evaluate_retrieval(search_results, source_annotations, target_annotations)
    detailed_path = eval_dir / f"{args.model}_{suffix}_detailed.csv"
    summary_path = eval_dir / f"{args.model}_{suffix}_summary.csv"
    metrics_path = eval_dir / f"{args.model}_{suffix}_metrics.json"

    detailed.to_csv(detailed_path, index=False)
    summary.to_csv(summary_path, index=False)
    write_json(
        {
            row["aspect"]: {
                "mean_go_jaccard": row["mean_go_jaccard"],
                "precision_at_k_proxy": row["precision_at_k_proxy"],
            }
            for row in summary.to_dict(orient="records")
        },
        metrics_path,
    )
    print(f"Wrote {detailed_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()

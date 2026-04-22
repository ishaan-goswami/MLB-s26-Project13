#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import _bootstrap
from cross_species_pfp.config import ensure_dirs, load_config
from cross_species_pfp.embedding_models import (
    embed_with_esmc,
    embed_with_transformers_esm,
    embed_with_transformers_t5,
)
from cross_species_pfp.fasta_utils import parse_fasta_to_frame


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True, choices=["esm2_650m", "prott5_xl", "esmc_600m", "prot_t5_xl", "prott5"])
    parser.add_argument("--species", nargs="*", default=["human", "yeast"])
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--limit", type=int, default=None, help="Only embed the first N proteins per species for a quick test run.")
    args = parser.parse_args()

    model_aliases = {
        "prot_t5_xl": "prott5_xl",
        "prott5": "prott5_xl",
    }
    model_key = model_aliases.get(args.model, args.model)

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    raw_dir = Path(cfg["paths"]["raw_dir"])
    embeddings_dir = Path(cfg["paths"]["embeddings_dir"])
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = cfg["models"][model_key]

    for species_key in args.species:
        taxid = cfg["species"][species_key]["taxid"]
        fasta_path = raw_dir / f"{taxid}.protein.sequences.fa.gz"
        frame = parse_fasta_to_frame(fasta_path, truncate_to=model_cfg["max_length"])
        if args.limit is not None:
            frame = frame.head(args.limit).copy()
        protein_ids = frame["protein_id"].tolist()
        sequences = frame["sequence"].tolist()

        if model_cfg["family"] == "transformers_esm":
            out_ids, embeddings = embed_with_transformers_esm(
                protein_ids,
                sequences,
                model_cfg["model_name"],
                model_cfg["batch_size"],
                args.device,
            )
        elif model_cfg["family"] == "transformers_t5":
            out_ids, embeddings = embed_with_transformers_t5(
                protein_ids,
                sequences,
                model_cfg["model_name"],
                model_cfg["batch_size"],
                args.device,
            )
        elif model_cfg["family"] == "esmc":
            out_ids, embeddings = embed_with_esmc(
                protein_ids,
                sequences,
                model_cfg["model_name"],
                model_cfg["batch_size"],
                args.device,
            )
        else:
            raise ValueError(f"Unsupported model family: {model_cfg['family']}")

        output_path = embeddings_dir / f"{species_key}_{model_key}.npz"
        np.savez_compressed(output_path, ids=np.array(out_ids), embeddings=embeddings.astype(np.float32))
        pd.DataFrame(
            {
                "protein_id": out_ids,
                "embedding_index": range(len(out_ids)),
            }
        ).to_csv(embeddings_dir / f"{species_key}_{model_key}_index.csv", index=False)
        print(f"Wrote {len(out_ids)} embeddings for {species_key} to {output_path}")


if __name__ == "__main__":
    main()

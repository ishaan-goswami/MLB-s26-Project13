#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
from pathlib import Path
import subprocess

import _bootstrap
from cross_species_pfp.config import ensure_dirs, load_config


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def decompress_if_needed(source: Path, destination: Path) -> Path:
    if source.suffix != ".gz":
        return source
    destination.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(source, "rb") as src, open(destination, "wb") as dst:
        dst.write(src.read())
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)

    raw_dir = Path(cfg["paths"]["raw_dir"])
    search_dir = Path(cfg["paths"]["search_dir"])
    tmp_dir = search_dir / "mmseqs_tmp"
    search_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    source_taxid = cfg["species"][cfg["retrieval"]["source_species"]]["taxid"]
    target_taxid = cfg["species"][cfg["retrieval"]["target_species"]]["taxid"]
    source_fasta = decompress_if_needed(
        raw_dir / f"{source_taxid}.protein.sequences.fa.gz",
        search_dir / f"{source_taxid}.protein.sequences.fa",
    )
    target_fasta = decompress_if_needed(
        raw_dir / f"{target_taxid}.protein.sequences.fa.gz",
        search_dir / f"{target_taxid}.protein.sequences.fa",
    )
    output_path = search_dir / "mmseqs_baseline.tsv"

    run(
        [
            "mmseqs",
            "easy-search",
            str(source_fasta),
            str(target_fasta),
            str(output_path),
            str(tmp_dir),
            "--format-output",
            "query,target,pident,bits,evalue",
        ]
    )
    print(f"Wrote MMseqs2 baseline to {output_path}")


if __name__ == "__main__":
    main()

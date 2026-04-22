#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap
from cross_species_pfp.config import ensure_dirs, load_config
from cross_species_pfp.io_utils import download_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    raw_dir = Path(cfg["paths"]["raw_dir"])

    for species_key, species_cfg in cfg["species"].items():
        taxid = species_cfg["taxid"]
        download_file(species_cfg["fasta_url"], raw_dir / f"{taxid}.protein.sequences.fa.gz")
        download_file(species_cfg["aliases_url"], raw_dir / f"{taxid}.protein.aliases.txt.gz")
        download_file(species_cfg["goa_url"], raw_dir / f"{species_key}.goa.gaf.gz")

    download_file(cfg["ontology"]["go_basic_url"], raw_dir / "go-basic.obo")
    print("Finished downloading configured raw files.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap
from cross_species_pfp.config import ensure_dirs, load_config
from cross_species_pfp.go_utils import (
    invert_alias_map,
    load_go_dag,
    load_string_aliases,
    parse_gaf_to_frame,
    write_annotations_json,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    raw_dir = Path(cfg["paths"]["raw_dir"])
    processed_dir = Path(cfg["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    go_dag = load_go_dag(raw_dir / "go-basic.obo")
    evidence_codes = set(cfg["ontology"]["experimental_evidence_codes"])
    aspect_map = cfg["ontology"]["aspects"]

    for species_key, species_cfg in cfg["species"].items():
        taxid = species_cfg["taxid"]
        alias_map = load_string_aliases(raw_dir / f"{taxid}.protein.aliases.txt.gz")
        inverse_aliases = invert_alias_map(alias_map)
        frame = parse_gaf_to_frame(
            raw_dir / f"{species_key}.goa.gaf.gz",
            evidence_codes,
            go_dag,
            inverse_aliases,
            aspect_map,
        )
        frame.to_csv(processed_dir / f"{species_key}_go_annotations.csv", index=False)
        write_annotations_json(frame, processed_dir / f"{species_key}_go_annotations.json")
        print(f"{species_key}: wrote {len(frame)} protein-aspect annotation rows.")


if __name__ == "__main__":
    main()

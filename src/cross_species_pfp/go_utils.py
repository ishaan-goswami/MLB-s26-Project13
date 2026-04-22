from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re

import pandas as pd
from goatools.obo_parser import GODag

from .io_utils import open_maybe_gzip, write_json


UNIPROT_ACCESSION_PATTERN = re.compile(r"^[A-NR-Z][0-9][A-Z0-9]{3}[0-9](-\d+)?$|^[OPQ][0-9][A-Z0-9]{3}[0-9](-\d+)?$")


def load_string_aliases(path: str | Path) -> dict[str, set[str]]:
    mapping: dict[str, set[str]] = defaultdict(set)
    with open_maybe_gzip(path, "rt") as handle:
        next(handle)
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            string_id, alias, source = parts[:3]
            if "UniProt" in source or UNIPROT_ACCESSION_PATTERN.match(alias):
                mapping[string_id].add(alias.split("-")[0])
    return dict(mapping)


def invert_alias_map(alias_map: dict[str, set[str]]) -> dict[str, set[str]]:
    inverse: dict[str, set[str]] = defaultdict(set)
    for string_id, aliases in alias_map.items():
        for alias in aliases:
            inverse[alias].add(string_id)
    return dict(inverse)


def load_go_dag(path: str | Path) -> GODag:
    return GODag(str(path))


def propagate_terms(terms: set[str], go_dag: GODag) -> set[str]:
    expanded = set()
    for term in terms:
        expanded.add(term)
        if term in go_dag:
            expanded.update(go_dag[term].get_all_parents())
    return expanded


def parse_gaf_to_frame(
    path: str | Path,
    allowed_evidence_codes: set[str],
    go_dag: GODag,
    accession_to_string_ids: dict[str, set[str]],
    aspect_map: dict[str, str],
) -> pd.DataFrame:
    raw_records = []
    with open_maybe_gzip(path, "rt") as handle:
        for line in handle:
            if not line or line.startswith("!"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 17:
                continue
            accession = parts[1].split("-")[0]
            go_id = parts[4]
            evidence = parts[6]
            aspect = parts[8]
            if evidence not in allowed_evidence_codes:
                continue
            if accession not in accession_to_string_ids:
                continue
            for string_id in accession_to_string_ids[accession]:
                raw_records.append(
                    {
                        "protein_id": string_id,
                        "uniprot_accession": accession,
                        "go_id": go_id,
                        "evidence": evidence,
                        "aspect": aspect_map.get(aspect, aspect),
                    }
                )

    frame = pd.DataFrame(raw_records)
    if frame.empty:
        return frame

    aggregated = []
    for protein_id, group in frame.groupby("protein_id"):
        for aspect, aspect_group in group.groupby("aspect"):
            terms = set(aspect_group["go_id"].tolist())
            propagated = sorted(propagate_terms(terms, go_dag))
            aggregated.append(
                {
                    "protein_id": protein_id,
                    "aspect": aspect,
                    "go_terms": propagated,
                    "n_terms": len(propagated),
                }
            )
    return pd.DataFrame(aggregated)


def write_annotations_json(frame: pd.DataFrame, path: str | Path) -> None:
    by_protein: dict[str, dict[str, list[str]]] = defaultdict(dict)
    for row in frame.itertuples(index=False):
        by_protein[row.protein_id][row.aspect] = row.go_terms
    write_json(by_protein, path)


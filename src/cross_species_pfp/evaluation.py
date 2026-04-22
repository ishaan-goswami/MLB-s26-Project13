from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from .io_utils import read_json


def load_annotation_frame(path: str) -> pd.DataFrame:
    raw = read_json(path)
    rows = []
    for protein_id, aspect_map in raw.items():
        row = {"protein_id": protein_id}
        row.update(aspect_map)
        rows.append(row)
    return pd.DataFrame(rows)


def annotation_dict_from_json(path: str) -> dict[str, dict[str, set[str]]]:
    raw = read_json(path)
    result: dict[str, dict[str, set[str]]] = {}
    for protein_id, aspect_map in raw.items():
        normalized = {}
        for aspect, terms in aspect_map.items():
            normalized[aspect] = set(terms) if isinstance(terms, list) else set()
        result[protein_id] = normalized
    return result


def jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def shared_term_indicator(a: set[str], b: set[str]) -> float:
    return float(len(a & b) > 0)


def evaluate_retrieval(
    search_results: pd.DataFrame,
    source_annotations: dict[str, dict[str, set[str]]],
    target_annotations: dict[str, dict[str, set[str]]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    detailed_rows = []
    summary_rows = []
    for aspect in ["MF", "BP", "CC"]:
        aspect_scores = []
        aspect_precisions = []
        for row in search_results.itertuples(index=False):
            source_terms = source_annotations.get(row.query_id, {}).get(aspect, set())
            target_terms = target_annotations.get(row.neighbor_id, {}).get(aspect, set())
            score = jaccard(source_terms, target_terms)
            precision_hit = shared_term_indicator(source_terms, target_terms)
            aspect_scores.append(score)
            aspect_precisions.append(precision_hit)
            detailed_rows.append(
                {
                    "query_id": row.query_id,
                    "neighbor_rank": row.rank,
                    "neighbor_id": row.neighbor_id,
                    "aspect": aspect,
                    "cosine_score": row.score,
                    "go_jaccard": score,
                    "shared_term_hit": precision_hit,
                }
            )

        summary_rows.append(
            {
                "aspect": aspect,
                "mean_go_jaccard": float(np.mean(aspect_scores)) if aspect_scores else 0.0,
                "precision_at_k_proxy": float(np.mean(aspect_precisions)) if aspect_precisions else 0.0,
            }
        )
    return pd.DataFrame(detailed_rows), pd.DataFrame(summary_rows)


def summarize_top1(summary_frame: pd.DataFrame) -> dict[str, float]:
    return dict(zip(summary_frame["aspect"], summary_frame["mean_go_jaccard"]))

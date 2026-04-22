from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from tqdm import tqdm


@dataclass
class EmbeddingBatch:
    protein_ids: list[str]
    sequences: list[str]


def batched_records(protein_ids: list[str], sequences: list[str], batch_size: int) -> Iterable[EmbeddingBatch]:
    for i in range(0, len(protein_ids), batch_size):
        yield EmbeddingBatch(
            protein_ids=protein_ids[i : i + batch_size],
            sequences=sequences[i : i + batch_size],
        )


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    masked = last_hidden_state * mask
    summed = masked.sum(dim=1)
    lengths = mask.sum(dim=1).clamp(min=1)
    return summed / lengths


def embed_with_transformers_esm(
    protein_ids: list[str],
    sequences: list[str],
    model_name: str,
    batch_size: int,
    device: str,
) -> tuple[list[str], np.ndarray]:
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    outputs = []
    kept_ids = []
    with torch.no_grad():
        for batch in tqdm(batched_records(protein_ids, sequences, batch_size), total=(len(protein_ids) + batch_size - 1) // batch_size):
            encoded = tokenizer(
                batch.sequences,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            hidden = model(**encoded).last_hidden_state
            pooled = _mean_pool(hidden, encoded["attention_mask"])
            outputs.append(pooled.cpu().numpy())
            kept_ids.extend(batch.protein_ids)
    return kept_ids, np.vstack(outputs)


def embed_with_transformers_t5(
    protein_ids: list[str],
    sequences: list[str],
    model_name: str,
    batch_size: int,
    device: str,
) -> tuple[list[str], np.ndarray]:
    from transformers import T5EncoderModel, T5Tokenizer

    tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(model_name).to(device)
    model.eval()

    outputs = []
    kept_ids = []
    with torch.no_grad():
        for batch in tqdm(batched_records(protein_ids, sequences, batch_size), total=(len(protein_ids) + batch_size - 1) // batch_size):
            spaced_sequences = [" ".join(list(seq)) for seq in batch.sequences]
            encoded = tokenizer(
                spaced_sequences,
                add_special_tokens=True,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            hidden = model(**encoded).last_hidden_state
            pooled = _mean_pool(hidden, encoded["attention_mask"])
            outputs.append(pooled.cpu().numpy())
            kept_ids.extend(batch.protein_ids)
    return kept_ids, np.vstack(outputs)


def embed_with_esmc(
    protein_ids: list[str],
    sequences: list[str],
    model_name: str,
    batch_size: int,
    device: str,
) -> tuple[list[str], np.ndarray]:
    try:
        from esm.models.esmc import ESMC
        from esm.sdk.api import ESMProtein, LogitsConfig
    except ImportError as exc:
        raise RuntimeError(
            "ESM-C embedding requires the `esm` package from EvolutionaryScale. "
            "Use Python 3.12 and `pip install esm`."
        ) from exc

    if batch_size != 1:
        raise ValueError("Current ESM-C wrapper expects batch_size=1.")

    client = ESMC.from_pretrained(model_name).to(device)
    client.eval()

    outputs = []
    kept_ids = []
    for protein_id, sequence in tqdm(list(zip(protein_ids, sequences)), total=len(protein_ids)):
        protein = ESMProtein(sequence=sequence)
        encoded = client.encode(protein)
        logits = client.logits(encoded, LogitsConfig(sequence=True, return_embeddings=True))
        residue_embeddings = logits.embeddings
        pooled = residue_embeddings.mean(axis=0)
        outputs.append(np.asarray(pooled))
        kept_ids.append(protein_id)
    return kept_ids, np.vstack(outputs)


def load_projector_checkpoint(path: str, device: str):
    checkpoint = torch.load(path, map_location=device)
    return checkpoint


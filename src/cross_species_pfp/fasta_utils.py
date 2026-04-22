from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from Bio import SeqIO

from .io_utils import open_maybe_gzip


NON_STANDARD_AA = re.compile(r"[^ACDEFGHIKLMNPQRSTVWYUXOBZ]")


def clean_sequence(sequence: str) -> str:
    sequence = sequence.upper().replace("*", "")
    return NON_STANDARD_AA.sub("X", sequence)


def parse_fasta_to_frame(path: str | Path, truncate_to: int | None = None) -> pd.DataFrame:
    records = []
    with open_maybe_gzip(path, "rt") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            seq = clean_sequence(str(record.seq))
            original_len = len(seq)
            if truncate_to is not None:
                seq = seq[:truncate_to]
            records.append(
                {
                    "protein_id": record.id,
                    "description": record.description,
                    "sequence": seq,
                    "sequence_length": original_len,
                    "truncated_length": len(seq),
                    "was_truncated": truncate_to is not None and original_len > truncate_to,
                }
            )
    return pd.DataFrame(records)


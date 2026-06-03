"""CertFE Redundancy module — dual-channel (value + structure) redundancy detection."""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import rankdata

logger = logging.getLogger(__name__)


@dataclass
class RedundancyResult:
    redundancy_max: float
    redundancy_argmax: str  # feature_id of closest match
    value_sim: float
    struct_sim: float


def compute_redundancy(
    candidate_col: pd.Series,
    candidate_hash: str,
    accepted_features: list[tuple[str, pd.Series, str]],
    sample_size: int = 5000,
    seed: int = 42,
) -> RedundancyResult:
    """Compute max redundancy of candidate vs accepted set.

    Args:
        candidate_col: feature values (train split)
        candidate_hash: canonical program hash
        accepted_features: list of (feature_id, col_values, program_hash)
        sample_size: max rows for value channel
        seed: deterministic sample seed

    Returns:
        RedundancyResult with max redundancy and argmax
    """
    if not accepted_features:
        return RedundancyResult(redundancy_max=0.0, redundancy_argmax="", value_sim=0.0, struct_sim=0.0)

    best_sim = 0.0
    best_fid = ""
    best_val = 0.0
    best_struct = 0.0

    for fid, accepted_col, accepted_hash in accepted_features:
        s_struct = 1.0 if candidate_hash == accepted_hash else 0.0

        s_val = _spearman_rank_cosine(candidate_col, accepted_col, sample_size, seed)

        combined = max(s_val, s_struct)
        if combined > best_sim:
            best_sim = combined
            best_fid = fid
            best_val = s_val
            best_struct = s_struct

    return RedundancyResult(
        redundancy_max=best_sim,
        redundancy_argmax=best_fid,
        value_sim=best_val,
        struct_sim=best_struct,
    )


def _spearman_rank_cosine(
    a: pd.Series, b: pd.Series, sample_size: int, seed: int
) -> float:
    """Spearman rank cosine similarity between two series."""
    aligned = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(aligned) < 30:
        return 0.0

    if len(aligned) > sample_size:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(aligned), size=sample_size, replace=False)
        aligned = aligned.iloc[idx]

    if not pd.api.types.is_numeric_dtype(aligned["a"]):
        return 0.0
    if not pd.api.types.is_numeric_dtype(aligned["b"]):
        return 0.0

    a_vals = aligned["a"].values.astype(float)
    b_vals = aligned["b"].values.astype(float)

    if np.std(a_vals) < 1e-12 or np.std(b_vals) < 1e-12:
        return 0.0

    ra = rankdata(a_vals).astype(float)
    rb = rankdata(b_vals).astype(float)

    ra -= ra.mean()
    rb -= rb.mean()

    dot = np.dot(ra, rb)
    norm_a = np.linalg.norm(ra)
    norm_b = np.linalg.norm(rb)

    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0

    return float(dot / (norm_a * norm_b))


def canonical_hash(program_json: dict) -> str:
    """Compute canonical hash of a program for structural dedup."""
    import json
    canonical = json.dumps(program_json, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]

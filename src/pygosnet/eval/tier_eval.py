"""L0-L4 density-tier evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import summarize_counts, to_numpy

TIER_NAMES = ("L0", "L1", "L2", "L3", "L4")
# right-open bins: <1, 1..100, 101..500, 501..1000, >1000
TIER_BINS = (1, 101, 501, 1001)
TIER_RULES = {
    "L0": "count = 0",
    "L1": "1 <= count <= 100",
    "L2": "101 <= count <= 500",
    "L3": "501 <= count <= 1000",
    "L4": "count > 1000",
}


def assign_tier(counts):
    array = np.asarray(to_numpy(counts), dtype=np.float64)
    tier_index = np.digitize(array, TIER_BINS)
    names = np.asarray(TIER_NAMES, dtype=object)[tier_index]
    if array.ndim == 0:
        return str(names)
    return names


def tier_distribution(counts) -> pd.DataFrame:
    counts = np.asarray(to_numpy(counts), dtype=np.float64).ravel()
    tiers = assign_tier(counts)
    rows = []
    for tier in TIER_NAMES:
        mask = tiers == tier
        rows.append(
            {
                "tier": tier,
                "range": TIER_RULES[tier],
                "n_images": int(mask.sum()),
                "image_share": float(mask.mean()) if counts.size else 0.0,
                "gt_total": float(counts[mask].sum()),
            }
        )
    frame = pd.DataFrame(rows)
    total = float(counts.sum())
    frame["penguin_share"] = frame["gt_total"] / total if total > 0 else 0.0
    return frame


def evaluate_by_tier(pred_counts, gt_counts) -> pd.DataFrame:
    pred_counts = np.asarray(to_numpy(pred_counts), dtype=np.float64).ravel()
    gt_counts = np.asarray(to_numpy(gt_counts), dtype=np.float64).ravel()
    if pred_counts.size != gt_counts.size:
        raise ValueError(f"pred and gt lengths differ: {pred_counts.size} vs {gt_counts.size}")

    tiers = assign_tier(gt_counts)
    rows = []
    for tier in TIER_NAMES:
        mask = tiers == tier
        if not mask.any():
            continue
        row = {"tier": tier, "range": TIER_RULES[tier]}
        row.update(summarize_counts(pred_counts[mask], gt_counts[mask]))
        rows.append(row)
    return pd.DataFrame(rows)

"""Error decomposition: background false positives, missed targets, dense excess."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import EPS, to_numpy
from .tier_eval import TIER_NAMES, assign_tier


def _density_map(values, name: str = "density") -> np.ndarray:
    array = np.squeeze(np.asarray(to_numpy(values), dtype=np.float64))
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D single-channel density map, got shape={array.shape}")
    return array


def _max_filter3(array: np.ndarray) -> np.ndarray:
    # 3x3 max filter
    height, width = array.shape
    padded = np.pad(array, 1, mode="constant")
    out = np.zeros_like(array)
    for dy in range(3):
        for dx in range(3):
            out = np.maximum(out, padded[dy : dy + height, dx : dx + width])
    return out


def build_gt_support(gt_density, tau: float = 1e-3, dilate: int = 1) -> np.ndarray:
    support = (_density_map(gt_density) > tau).astype(np.float64)
    for _ in range(int(dilate)):
        support = _max_filter3(support)
    return support


def support_from_points(
    points_yx,
    shape_hw: tuple[int, int],
    stride: int = 8,
    radius_cells: int = 1,
) -> np.ndarray:
    # points -> stride-8 cells -> dilate
    height, width = int(shape_hw[0]), int(shape_hw[1])
    support = np.zeros((height, width), dtype=np.float64)
    points = np.asarray(to_numpy(points_yx), dtype=np.float64).reshape(-1, 2)
    if points.size == 0:
        return support

    rows = np.clip((points[:, 0] / stride).astype(int), 0, height - 1)
    cols = np.clip((points[:, 1] / stride).astype(int), 0, width - 1)
    support[rows, cols] = 1.0
    for _ in range(int(radius_cells)):
        support = _max_filter3(support)
    return support


def decompose_with_support(pred_density, support, gt_total: float) -> dict:
    pred_map = _density_map(pred_density, "pred_density")
    support_map = _density_map(support, "support")
    if pred_map.shape != support_map.shape:
        raise ValueError(f"support and prediction shapes differ: {support_map.shape} vs {pred_map.shape}")

    foreground = support_map > 0
    true_positive = float(pred_map[foreground].sum())
    background_fp = float(pred_map[~foreground].sum())
    pred_total = true_positive + background_fp
    gt_total = float(gt_total)
    missed = max(gt_total - true_positive, 0.0)

    return {
        "gt_total": gt_total,
        "pred_total": pred_total,
        "tp_mass": true_positive,
        "bg_fp_mass": background_fp,
        "missed_mass": missed,
        "bg_fp_rate": background_fp / pred_total if pred_total > EPS else 0.0,
        "miss_rate": missed / gt_total if gt_total > EPS else 0.0,
        "net_error": pred_total - gt_total,
    }


def counting_breakdown(
    pred_density,
    gt_points_yx,
    gt_count: float,
    stride: int = 8,
    radius_cells: int = 1,
) -> dict:
    # per-cell clipping: matched = sum(min(pred_cell, gt_cell))
    pred_map = _density_map(pred_density, "pred_density")
    height, width = pred_map.shape
    points = np.asarray(to_numpy(gt_points_yx), dtype=np.float64).reshape(-1, 2)

    gt_cells = np.zeros_like(pred_map)
    if points.size:
        rows = np.clip((points[:, 0] / stride).astype(int), 0, height - 1)
        cols = np.clip((points[:, 1] / stride).astype(int), 0, width - 1)
        np.add.at(gt_cells, (rows, cols), 1.0)

    support = support_from_points(points, (height, width), stride=stride, radius_cells=radius_cells)
    foreground = support > 0

    matched = float(np.minimum(pred_map, gt_cells)[foreground].sum())
    background_fp = float(pred_map[~foreground].sum())
    excess = float(np.maximum(pred_map - gt_cells, 0.0)[foreground].sum())
    pred_total = float(pred_map.sum())
    gt_total = float(gt_count)
    missed = max(gt_total - matched, 0.0)

    return {
        "gt_total": gt_total,
        "pred_total": pred_total,
        "tp_mass": matched,
        "bg_fp_mass": background_fp,
        "missed_mass": missed,
        "excess_mass": excess,
        "bg_fp_rate": background_fp / pred_total if pred_total > EPS else 0.0,
        "miss_rate": missed / gt_total if gt_total > EPS else 0.0,
        "excess_rate": excess / pred_total if pred_total > EPS else 0.0,
        "net_error": pred_total - gt_total,
    }


def density_mass_decomposition(
    gt_density,
    pred_density,
    support=None,
    tau: float = 1e-3,
    dilate: int = 1,
) -> dict:
    gt_map = _density_map(gt_density, "gt_density")
    pred_map = _density_map(pred_density, "pred_density")
    if gt_map.shape != pred_map.shape:
        raise ValueError(f"GT and prediction shapes differ: {gt_map.shape} vs {pred_map.shape}")

    if support is None:
        support = build_gt_support(gt_map, tau=tau, dilate=dilate)
    return decompose_with_support(pred_map, support, float(gt_map.sum()))


def image_error_record(
    gt_density,
    pred_density,
    gt_count: float,
    tier: str | None = None,
    support=None,
    tau: float = 1e-3,
    dilate: int = 1,
) -> dict:
    record = {
        "tier": tier if tier is not None else str(assign_tier(gt_count)),
        "gt_count": float(gt_count),
    }
    record.update(
        density_mass_decomposition(gt_density, pred_density, support=support, tau=tau, dilate=dilate)
    )
    return record


def _aggregate(subset: pd.DataFrame, tier: str) -> dict:
    gt_total = float(subset["gt_total"].sum())
    pred_total = float(subset["pred_total"].sum())
    tp_mass = float(subset["tp_mass"].sum())
    bg_fp_mass = float(subset["bg_fp_mass"].sum())
    missed_mass = float(subset["missed_mass"].sum())
    row = {
        "tier": tier,
        "n": int(len(subset)),
        "gt_total": gt_total,
        "pred_total": pred_total,
        "tp_mass": tp_mass,
        "bg_fp_mass": bg_fp_mass,
        "missed_mass": missed_mass,
        # mass-weighted rates
        "bg_fp_rate": bg_fp_mass / pred_total if pred_total > EPS else 0.0,
        "miss_rate": missed_mass / gt_total if gt_total > EPS else 0.0,
        "net_error": pred_total - gt_total,
    }
    if "excess_mass" in subset.columns:
        excess_mass = float(subset["excess_mass"].sum())
        row["excess_mass"] = excess_mass
        row["excess_rate"] = excess_mass / pred_total if pred_total > EPS else 0.0
    return row


def masking_effect_record(
    raw_density,
    masked_density,
    gt_points_yx,
    gt_count: float,
    tier: str | None = None,
    stride: int = 8,
    radius_cells: int = 1,
) -> dict:
    # raw vs masked output; separates background-FP removal from true-target suppression
    raw = counting_breakdown(raw_density, gt_points_yx, gt_count, stride=stride, radius_cells=radius_cells)
    masked = counting_breakdown(masked_density, gt_points_yx, gt_count, stride=stride, radius_cells=radius_cells)

    return {
        "tier": tier if tier is not None else str(assign_tier(gt_count)),
        "gt_count": float(gt_count),
        "raw_count": raw["pred_total"],
        "masked_count": masked["pred_total"],
        "raw_mae": abs(raw["pred_total"] - float(gt_count)),
        "masked_mae": abs(masked["pred_total"] - float(gt_count)),
        "raw_bg_fp_mass": raw["bg_fp_mass"],
        "masked_bg_fp_mass": masked["bg_fp_mass"],
        "raw_tp_mass": raw["tp_mass"],
        "masked_tp_mass": masked["tp_mass"],
        "fp_removed_mass": raw["bg_fp_mass"] - masked["bg_fp_mass"],
        "true_suppressed_mass": raw["tp_mass"] - masked["tp_mass"],
        "excess_suppressed_mass": raw["excess_mass"] - masked["excess_mass"],
    }


def summarize_masking_effect(records) -> pd.DataFrame:
    # per-tier gain/loss of D_out = D * 1[S >= 0.5]
    frame = pd.DataFrame(list(records))
    required = {"tier", "gt_count", "raw_mae", "masked_mae", "raw_bg_fp_mass", "masked_bg_fp_mass",
                "raw_tp_mass", "masked_tp_mass"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"Masking records missing fields: {sorted(missing)}")

    rows = []
    for tier in list(TIER_NAMES) + ["ALL"]:
        subset = frame if tier == "ALL" else frame[frame["tier"] == tier]
        if subset.empty:
            continue
        gt_total = float(subset["gt_count"].sum())
        raw_total = float(subset["raw_count"].sum())
        masked_total = float(subset["masked_count"].sum())
        fp_removed = float(subset["fp_removed_mass"].sum())
        true_suppressed = float(subset["true_suppressed_mass"].sum())
        row = {
            "tier": tier,
            "n": len(subset),
            "gt_total": gt_total,
            "raw_total": raw_total,
            "masked_total": masked_total,
            "raw_mae": float(subset["raw_mae"].mean()),
            "masked_mae": float(subset["masked_mae"].mean()),
            "mae_delta": float(subset["masked_mae"].mean() - subset["raw_mae"].mean()),
            "mass_suppressed": raw_total - masked_total,
            "mass_suppressed_share": (raw_total - masked_total) / raw_total if raw_total > EPS else 0.0,
            "fp_removed_mass": fp_removed,
            "true_suppressed_mass": true_suppressed,
            # share of surviving prediction mass coming from background
            "bg_fp_rate_raw": float(subset["raw_bg_fp_mass"].sum()) / raw_total if raw_total > EPS else 0.0,
            "bg_fp_rate_masked": float(subset["masked_bg_fp_mass"].sum()) / masked_total if masked_total > EPS else 0.0,
            # true mass lost to the mask relative to GT; the L3/L4 over-suppression check
            "over_suppression_rate": true_suppressed / gt_total if gt_total > EPS else 0.0,
            "miss_rate": max(gt_total - float(subset["masked_tp_mass"].sum()), 0.0) / gt_total
            if gt_total > EPS else 0.0,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def select_representative_cases(frame, metrics=("bg_fp_rate", "miss_rate", "excess_rate")) -> pd.DataFrame:
    # highest-error image per tier for each error source; supports qualitative inspection
    frame = pd.DataFrame(frame)
    rows = []
    for tier in TIER_NAMES:
        subset = frame[frame["tier"] == tier]
        if subset.empty:
            continue
        chosen = set()
        for metric in metrics:
            if metric not in subset.columns:
                continue
            index = subset[metric].idxmax()
            if index in chosen:
                continue
            chosen.add(index)
            row = subset.loc[index].to_dict()
            row["case_type"] = metric
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_error_sources(records) -> pd.DataFrame:
    frame = pd.DataFrame(list(records))
    required = {"tier", "gt_total", "pred_total", "tp_mass", "bg_fp_mass", "missed_mass"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"Error records missing fields: {sorted(missing)}")

    rows = []
    for tier in list(TIER_NAMES) + ["ALL"]:
        subset = frame if tier == "ALL" else frame[frame["tier"] == tier]
        if subset.empty:
            continue
        rows.append(_aggregate(subset, tier))
    return pd.DataFrame(rows)

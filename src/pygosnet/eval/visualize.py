"""Evaluation figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from .metrics import summarize_counts, to_numpy
from .tier_eval import assign_tier


def _prepare(path):
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)


def to_display_image(image) -> np.ndarray:
    # [3,H,W] / [H,W] / float images -> imshow-ready
    array = np.asarray(to_numpy(image))
    if array.ndim == 3 and array.shape[0] in (1, 3) and array.shape[0] != array.shape[-1]:
        array = np.transpose(array, (1, 2, 0))
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    if array.ndim == 3 and array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Cannot convert image shape: {array.shape}")

    if array.dtype == np.uint8:
        return array
    array = array.astype(np.float64)
    if array.max() > 1.5:
        array = array / 255.0
    return np.clip(array, 0.0, 1.0)


def _density(values) -> np.ndarray:
    return np.clip(np.squeeze(np.asarray(to_numpy(values), dtype=np.float64)), 0.0, None)


def plot_prediction_panels(
    image,
    gt_density=None,
    pred_density=None,
    segmentation=None,
    masked_pred=None,
    gt_count: float | None = None,
    points_yx=None,
    include_error: bool = False,
    title: str | None = None,
    save_path=None,
    dpi: int = 200,
    cmap: str = "magma",
):
    # shared color scale across GT / prediction / masked prediction
    panels: list[tuple[str, np.ndarray, str]] = []
    if gt_density is not None:
        panels.append(("GT density", _density(gt_density), "density"))
    if pred_density is not None:
        panels.append(("Prediction (raw)", _density(pred_density), "density"))
    if segmentation is not None:
        panels.append(("Segmentation support", np.squeeze(np.asarray(to_numpy(segmentation), np.float64)), "prob"))
    if masked_pred is not None:
        panels.append(("Masked prediction", _density(masked_pred), "density"))
    if include_error:
        if gt_density is None or pred_density is None:
            raise ValueError("include_error=True requires both gt_density and pred_density")
        panels.append(("Signed error (pred - GT)", _density(pred_density) - _density(gt_density), "error"))

    density_values = [values for _, values, kind in panels if kind == "density"]
    density_max = max([float(values.max()) for values in density_values], default=0.0)
    density_max = density_max if density_max > 0 else 1.0
    error_max = max([float(np.abs(v).max()) for _, v, k in panels if k == "error"], default=0.0)

    ncols = 1 + len(panels)
    fig, axes = plt.subplots(1, ncols, figsize=(3.4 * ncols, 3.9))
    axes = np.atleast_1d(axes)

    axes[0].imshow(to_display_image(image))
    if points_yx is not None:
        points = np.asarray(to_numpy(points_yx), dtype=np.float64).reshape(-1, 2)
        if points.size:
            axes[0].scatter(points[:, 1], points[:, 0], s=3, c="#ff2d55", linewidths=0, alpha=0.9)
    axes[0].set_title("Input" + ("" if gt_count is None else f"  GT N={int(gt_count)}"), fontsize=9)

    for axis, (label, values, kind) in zip(axes[1:], panels):
        if kind == "density":
            image_artist = axis.imshow(
                values, cmap=cmap, norm=Normalize(vmin=0.0, vmax=density_max), interpolation="nearest"
            )
            axis.set_title(f"{label}  N={values.sum():.0f}", fontsize=9)
        elif kind == "prob":
            image_artist = axis.imshow(values, cmap="viridis", vmin=0.0, vmax=1.0, interpolation="nearest")
            axis.set_title(label, fontsize=9)
        else:
            image_artist = axis.imshow(
                values, cmap="coolwarm", norm=Normalize(vmin=-error_max, vmax=error_max or 1.0),
                interpolation="nearest",
            )
            axis.set_title(label, fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])
        fig.colorbar(image_artist, ax=axis, fraction=0.046, pad=0.03)

    axes[0].set_xticks([])
    axes[0].set_yticks([])
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    _prepare(save_path)
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


def plot_count_scatter(gt_counts, pred_counts, tiers=None, save_path=None, dpi: int = 200):
    gt = np.asarray(to_numpy(gt_counts), dtype=np.float64).ravel()
    pred = np.asarray(to_numpy(pred_counts), dtype=np.float64).ravel()
    if tiers is None:
        tiers = assign_tier(gt)

    fig, axis = plt.subplots(figsize=(4.8, 4.6))
    colors = plt.get_cmap("tab10")
    for index, tier in enumerate(("L0", "L1", "L2", "L3", "L4")):
        mask = np.asarray(tiers) == tier
        if mask.any():
            axis.scatter(gt[mask], pred[mask], s=14, alpha=0.75, color=colors(index), label=f"{tier} (n={int(mask.sum())})")

    limit = max(float(gt.max()), float(pred.max()), 10.0) * 1.06
    axis.plot([0, limit], [0, limit], "k--", linewidth=1.0, label="y = x")
    axis.set_xlim(0, limit)
    axis.set_ylim(0, limit)
    axis.set_aspect("equal")
    axis.set_xlabel("Ground-truth count")
    axis.set_ylabel("Predicted count")
    axis.set_title("Per-image counting accuracy", fontsize=10)

    stats = summarize_counts(pred, gt)
    axis.text(
        0.04, 0.96,
        f"MAE={stats['mae']:.2f}\nRMSE={stats['rmse']:.2f}\nRelErr={stats['relative_count_error'] * 100:.2f}%",
        transform=axis.transAxes, va="top", fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    axis.legend(fontsize=7, loc="lower right", frameon=False)
    fig.tight_layout()
    _prepare(save_path)
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


def plot_tier_metrics(tier_frame, save_path=None, dpi: int = 200):
    tiers = list(tier_frame["tier"])
    positions = np.arange(len(tiers))
    width = 0.38
    fig, axis = plt.subplots(figsize=(7.0, 3.9))
    axis.bar(positions - width / 2, tier_frame["mae"], width, label="MAE", color="#4c72b0")
    axis.bar(positions + width / 2, tier_frame["rmse"], width, label="RMSE", color="#dd8452")

    top = max(float(tier_frame["rmse"].max()), 1e-6)
    for position, mae_value, rmse_value, n in zip(positions, tier_frame["mae"], tier_frame["rmse"], tier_frame["n"]):
        axis.text(position, max(mae_value, rmse_value) + top * 0.03, f"n={int(n)}", ha="center", fontsize=7)

    axis.set_xticks(positions)
    axis.set_xticklabels(tiers)
    axis.set_ylabel("Error (penguins)")
    axis.set_title("Per-tier counting error", fontsize=10)
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _prepare(save_path)
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


def plot_masking_effect(effect_frame, save_path=None, dpi: int = 200):
    # left: what the mask removes; right: MAE before/after, to expose over-suppression
    tiers = list(effect_frame["tier"])
    positions = np.arange(len(tiers))
    width = 0.38

    fig, (left, right) = plt.subplots(1, 2, figsize=(10.4, 3.9))
    left.bar(positions - width / 2, effect_frame["fp_removed_mass"], width,
             label="background FP removed", color="#4c72b0")
    left.bar(positions + width / 2, effect_frame["true_suppressed_mass"], width,
             label="true-target mass suppressed", color="#c44e52")
    left.set_xticks(positions)
    left.set_xticklabels(tiers)
    left.set_ylabel("Density mass")
    left.set_title("What the segmentation mask removes", fontsize=10)
    left.legend(frameon=False, fontsize=8)

    right.bar(positions - width / 2, effect_frame["raw_mae"], width, label="raw D", color="#8c8c8c")
    right.bar(positions + width / 2, effect_frame["masked_mae"], width, label="masked D_out", color="#55a868")
    right.set_xticks(positions)
    right.set_xticklabels(tiers)
    right.set_ylabel("MAE (penguins)")
    right.set_title("Counting error before/after masking", fontsize=10)
    right.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    _prepare(save_path)
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


def plot_count_histogram(counts, bins: int = 40, save_path=None, dpi: int = 200):
    values = np.asarray(to_numpy(counts), dtype=np.float64).ravel()
    fig, axis = plt.subplots(figsize=(6.2, 3.6))
    axis.hist(values, bins=bins, color="#55a868", edgecolor="white", linewidth=0.4)
    axis.set_yscale("log")
    axis.set_xlabel("Penguins per 448x448 tile")
    axis.set_ylabel("Number of tiles (log)")

    for boundary, tier in ((100.5, "L2"), (500.5, "L3"), (1000.5, "L4")):
        if values.size and values.max() >= boundary:
            axis.axvline(boundary, color="0.4", linestyle=":", linewidth=0.9)
            axis.text(boundary, axis.get_ylim()[1] * 0.35, f" {tier}", fontsize=7, color="0.3")

    axis.set_title(f"Ground-truth count distribution (n={values.size})", fontsize=10)
    fig.tight_layout()
    _prepare(save_path)
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig

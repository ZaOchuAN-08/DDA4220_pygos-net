"""Counting and density metrics for evaluation."""

from __future__ import annotations

import warnings

import numpy as np

EPS = 1e-12


def to_numpy(value):
    # numpy / torch / list -> numpy; None passes through
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _like(reference, array):
    if hasattr(reference, "detach"):
        import torch

        return torch.as_tensor(array, dtype=reference.dtype, device=reference.device)
    return array


def _as_1d(values, name: str) -> np.ndarray:
    array = np.asarray(to_numpy(values), dtype=np.float64).ravel()
    if array.size == 0:
        raise ValueError(f"{name} is empty, cannot compute metrics")
    return array


def _paired(pred, gt) -> tuple[np.ndarray, np.ndarray]:
    pred_array = _as_1d(pred, "pred")
    gt_array = _as_1d(gt, "gt")
    if pred_array.size != gt_array.size:
        raise ValueError(f"pred and gt lengths differ: {pred_array.size} vs {gt_array.size}")
    return pred_array, gt_array


# counts
def count_from_density(density) -> float:
    return float(np.asarray(to_numpy(density), dtype=np.float64).sum())


def counts_from_batch(densities) -> np.ndarray:
    # [B,1,H,W] or [B,H,W]
    array = np.asarray(to_numpy(densities), dtype=np.float64)
    if array.ndim == 4:
        return array.sum(axis=(1, 2, 3))
    if array.ndim == 3:
        return array.sum(axis=(1, 2))
    raise ValueError(f"Expected [B,C,H,W] or [B,H,W], got shape={array.shape}")


def density_counts(density):
    # float for one map, array for a batch
    array = np.asarray(to_numpy(density), dtype=np.float64)
    if array.ndim == 2:
        return float(array.sum())
    if array.ndim == 3:
        if array.shape[0] == 1:
            return float(array.sum())
        return array.sum(axis=(1, 2))
    if array.ndim == 4:
        return array.sum(axis=(1, 2, 3))
    raise ValueError(f"Unrecognized density shape: {array.shape}")


def check_count_preservation(density, count, rel_tol: float = 1e-3) -> dict:
    # verifies density.sum() == count
    density_sum = count_from_density(density)
    count = float(count)
    abs_error = abs(density_sum - count)
    rel_error = abs_error / max(count, EPS)
    return {
        "density_sum": density_sum,
        "count": count,
        "abs_error": abs_error,
        "rel_error": rel_error,
        "ok": bool(rel_error <= rel_tol or abs_error <= 1e-6),
    }


# output masking
def _segmentation_prob(segmentation, from_logits: bool) -> np.ndarray:
    prob = np.asarray(to_numpy(segmentation), dtype=np.float64)
    if prob.ndim == 2:
        prob = prob[None, None]
    elif prob.ndim == 3:
        prob = prob[None] if prob.shape[0] <= 2 else prob[:, None]
    elif prob.ndim != 4:
        raise ValueError(f"Unrecognized segmentation shape: {prob.shape}")

    channels = prob.shape[1]
    if channels == 2:
        if from_logits:
            shifted = prob - prob.max(axis=1, keepdims=True)
            exp = np.exp(shifted)
            prob = exp[:, 1:2] / exp.sum(axis=1, keepdims=True)
        else:
            prob = prob[:, 1:2]
    elif channels == 1:
        if from_logits:
            prob = 1.0 / (1.0 + np.exp(-prob))
    else:
        raise ValueError(f"segmentation must have 1 or 2 channels, got {channels}")
    return prob


def apply_output_mask(density, segmentation=None, threshold: float | None = 0.5, from_logits: bool = False):
    # D_out = D * 1[S >= 0.5]; threshold=None gives the soft mask
    if segmentation is None:
        return density

    density_array = np.asarray(to_numpy(density), dtype=np.float64)
    if density_array.ndim == 2:
        density_array = density_array[None, None]
    elif density_array.ndim == 3:
        density_array = density_array[None] if density_array.shape[0] == 1 else density_array[:, None]
    elif density_array.ndim != 4:
        raise ValueError(f"Unrecognized density shape: {density_array.shape}")

    prob = _segmentation_prob(segmentation, from_logits)
    if prob.shape[-2:] != density_array.shape[-2:]:
        raise ValueError(
            f"segmentation and density resolutions differ: {prob.shape[-2:]} vs {density_array.shape[-2:]}"
        )

    mask = prob if threshold is None else (prob >= threshold).astype(np.float64)
    return _like(density, density_array * mask)


# metrics
def mae(pred, gt) -> float:
    pred_array, gt_array = _paired(pred, gt)
    return float(np.mean(np.abs(pred_array - gt_array)))


def rmse(pred, gt) -> float:
    pred_array, gt_array = _paired(pred, gt)
    return float(np.sqrt(np.mean((pred_array - gt_array) ** 2)))


def bias(pred, gt) -> float:
    # mean(pred - gt); negative means undercounting
    pred_array, gt_array = _paired(pred, gt)
    return float(np.mean(pred_array - gt_array))


def image_relative_errors(pred, gt) -> np.ndarray:
    # |pred-gt|/gt; GT=0 gives 0 if pred=0, else NaN
    pred_array, gt_array = _paired(pred, gt)
    errors = np.zeros_like(pred_array)
    positive = gt_array > 0
    errors[positive] = np.abs(pred_array[positive] - gt_array[positive]) / gt_array[positive]
    empty = ~positive
    errors[empty] = np.where(np.isclose(pred_array[empty], 0.0), 0.0, np.nan)
    return errors


def relative_count_error(pred, gt) -> float:
    # dataset-level |sum(pred)-sum(gt)| / sum(gt)
    pred_array, gt_array = _paired(pred, gt)
    gt_total = float(gt_array.sum())
    if gt_total <= EPS:
        return 0.0 if abs(float(pred_array.sum())) <= EPS else float("nan")
    return float(abs(float(pred_array.sum()) - gt_total) / gt_total)


def summarize_counts(pred_counts, gt_counts) -> dict:
    pred_array, gt_array = _paired(pred_counts, gt_counts)
    diffs = pred_array - gt_array

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_image_relative_error = float(np.nanmean(image_relative_errors(pred_array, gt_array)))

    return {
        "n": int(pred_array.size),
        "gt_total": float(gt_array.sum()),
        "pred_total": float(pred_array.sum()),
        "mae": float(np.mean(np.abs(diffs))),
        "rmse": float(np.sqrt(np.mean(diffs**2))),
        "bias": float(np.mean(diffs)),
        "relative_count_error": relative_count_error(pred_array, gt_array),
        "mean_image_relative_error": mean_image_relative_error,
        "images_over": int((diffs > 0).sum()),
        "images_under": int((diffs < 0).sum()),
        "images_exact": int((diffs == 0).sum()),
    }

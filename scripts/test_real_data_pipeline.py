#!/usr/bin/env python3
"""
Run PygosNet's end-to-end evaluation pipeline on the held-out BAS penguin test set.

Expected project layout:
  <project-root>/data/splits/test.jsonl
  <project-root>/data/raw/images/...
  <project-root>/pygosnet/data/dataset.py
  <project-root>/pygosnet/.../pygosnet.py

Example:
  python scripts/test_real_data_pipeline.py \
      --project-root . \
      --checkpoint checkpoints/best.pt \
      --manifest data/splits/test.jsonl \
      --output outputs/test_predictions.csv
"""
from __future__ import annotations

import argparse
import csv
import importlib
import math
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


def load_model_class():
    """Try common package layouts; keep the model's relative imports intact."""
    candidates = (
        "pygosnet.models.pygosnet",
        "pygosnet.model.pygosnet",
        "pygosnet.pygosnet",
    )
    errors = []
    for module_name in candidates:
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, "PygosNet"):
                print(f"Using model class from {module_name}.PygosNet")
                return module.PygosNet
        except Exception as exc:
            errors.append(f"{module_name}: {type(exc).__name__}: {exc}")
    raise ImportError(
        "Could not import PygosNet. Make sure the project root is on PYTHONPATH "
        "and the model file is inside its original package (so relative imports work). "
        "Tried:\n  " + "\n  ".join(errors)
    )


def unwrap_state_dict(checkpoint: Any) -> dict:
    """Support raw state_dicts and common training-checkpoint dictionaries."""
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a state_dict or a checkpoint dictionary.")
    for key in ("model_state_dict", "state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, dict) and value:
            checkpoint = value
            break
    if not checkpoint or not all(isinstance(k, str) for k in checkpoint):
        raise ValueError("No valid model state_dict found in checkpoint.")
    # Handle checkpoints saved by DistributedDataParallel / DataParallel.
    cleaned = {}
    for key, value in checkpoint.items():
        cleaned[key[7:] if key.startswith("module.") else key] = value
    return cleaned


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", default="data/splits/test.jsonl")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/test_predictions.csv"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--sigma", type=float, default=4.0)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--support-threshold", type=float, default=1e-3)
    parser.add_argument("--use-dcb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dcb-branch-channels", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    root = args.project_root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"Project root does not exist: {root}")
    sys.path.insert(0, str(root))

    manifest = Path(args.manifest)
    if not manifest.is_absolute():
        manifest = root / manifest
    checkpoint_path = args.checkpoint
    if not checkpoint_path.is_absolute():
        checkpoint_path = root / checkpoint_path
    output_path = args.output
    if not output_path.is_absolute():
        output_path = root / output_path

    if not manifest.is_file():
        raise FileNotFoundError(f"Test manifest not found: {manifest}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # The dataset class is the same manifest-backed class used by training.
    from pygosnet.data.dataset import PenguinDataset

    dataset = PenguinDataset(
        manifest_path=manifest,
        project_root=root,
        split="test",
        crop_size=256,  # ignored by EvalTransform; kept for API compatibility
        sigma=args.sigma,
        stride=args.stride,
        support_threshold=args.support_threshold,
    )
    if len(dataset) == 0:
        raise ValueError(f"Test manifest contains no samples: {manifest}")

    # Fail early on missing images instead of silently evaluating a partial set.
    missing = []
    for row in dataset.rows:
        image_path = root / row["image_path"]
        if not image_path.is_file():
            missing.append(str(image_path))
            if len(missing) >= 5:
                break
    if missing:
        raise FileNotFoundError(
            "One or more image files referenced by the manifest are missing. "
            "First missing paths:\n  " + "\n  ".join(missing)
        )

    # Basic manifest integrity checks.
    ids = [row["image_id"] for row in dataset.rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate image_id values found in test manifest.")
    for i, row in enumerate(dataset.rows):
        points = row.get("points")
        if not isinstance(points, list):
            raise ValueError(f"Row {i} ({row.get('image_id')}) has no valid points list.")
        if row.get("count") is not None and int(row["count"]) != len(points):
            raise ValueError(
                f"Count/annotation mismatch for {row.get('image_id')}: "
                f"count={row['count']} but len(points)={len(points)}"
            )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,  # preserve manifest order for reproducible output
        num_workers=args.num_workers,
        pin_memory=str(args.device).startswith("cuda"),
    )

    device = torch.device(args.device)
    PygosNet = load_model_class()
    # pretrained=False avoids downloading ImageNet weights during evaluation;
    # the checkpoint supplies the trained parameters.
    model = PygosNet(
        use_dcb=args.use_dcb,
        pretrained=False,
        dcb_branch_channels=args.dcb_branch_channels,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = unwrap_state_dict(checkpoint)
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint architecture/state_dict mismatch.\n"
            f"Missing keys ({len(incompatible.missing_keys)}): {incompatible.missing_keys[:10]}\n"
            f"Unexpected keys ({len(incompatible.unexpected_keys)}): "
            f"{incompatible.unexpected_keys[:10]}\n"
            "Check --use-dcb and --dcb-branch-channels against training settings."
        )

    model.eval()
    records = []
    abs_errors, squared_errors = [], []
    total_gt, total_pred = 0.0, 0.0
    intersection = 0
    union = 0
    raw_density_sums = []

    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            gt_count = batch["count"].to(device).reshape(-1)
            outputs = model(images)

            if "density" not in outputs or "segmentation" not in outputs:
                raise KeyError("Model forward() must return 'density' and 'segmentation'.")
            density = outputs["density"]
            seg_prob = outputs["segmentation"]
            if density.ndim != 4 or density.shape[1] != 1:
                raise ValueError(f"Expected density [B,1,H,W], got {tuple(density.shape)}")
            if seg_prob.shape != density.shape:
                raise ValueError(
                    f"Segmentation/density shape mismatch: {tuple(seg_prob.shape)} vs "
                    f"{tuple(density.shape)}"
                )
            if not torch.isfinite(density).all() or not torch.isfinite(seg_prob).all():
                raise FloatingPointError(f"NaN/Inf model output in batch {batch_index}.")

            # Match PygosNet.predict(): threshold segmentation then sum density.
            masked_density = density * (seg_prob >= 0.5).to(density.dtype)
            pred_count = masked_density.sum(dim=(1, 2, 3))
            raw_sum = density.sum(dim=(1, 2, 3))
            gt_mask = batch["seg_mask"].to(device) >= 0.5
            pred_mask = seg_prob >= 0.5
            intersection += int((pred_mask & gt_mask).sum().item())
            union += int((pred_mask | gt_mask).sum().item())

            for j in range(images.shape[0]):
                gt = float(gt_count[j].item())
                pred = float(pred_count[j].item())
                error = pred - gt
                abs_errors.append(abs(error))
                squared_errors.append(error * error)
                total_gt += gt
                total_pred += pred
                raw_density_sums.append(float(raw_sum[j].item()))
                records.append({
                    "image_id": batch["image_id"][j],
                    "gt_count": gt,
                    "pred_count": pred,
                    "absolute_error": abs(error),
                    "signed_error": error,
                    "raw_density_sum": float(raw_sum[j].item()),
                    "segmentation_iou_per_image": "",  # computed globally below
                })

            print(f"Evaluated batch {batch_index}/{len(loader)}")

    n = len(records)
    mae = sum(abs_errors) / n
    rmse = math.sqrt(sum(squared_errors) / n)
    mean_bias = sum(r["signed_error"] for r in records) / n
    global_iou = intersection / union if union else 1.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    summary_path = output_path.with_name(output_path.stem + "_summary.json")
    summary = {
        "manifest": str(manifest),
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "num_images": n,
        "mae_count": mae,
        "rmse_count": rmse,
        "mean_signed_error": mean_bias,
        "total_ground_truth_count": total_gt,
        "total_predicted_count": total_pred,
        "global_count_ratio_pred_over_gt": total_pred / total_gt if total_gt else None,
        "segmentation_global_iou": global_iou,
        "mean_raw_density_sum": sum(raw_density_sums) / n,
        "note": (
            "Count prediction is the sum of density values where segmentation probability >= 0.5, "
            "matching PygosNet.predict(). Global IoU is calculated over all test pixels."
        ),
    }
    import json
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== Real-data test results ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"\nPer-image predictions: {output_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()

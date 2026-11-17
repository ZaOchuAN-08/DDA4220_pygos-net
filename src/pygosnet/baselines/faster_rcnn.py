"""Faster R-CNN reference baseline and bbox geometry check."""

from __future__ import annotations

import json

import numpy as np

from ..eval.labelbox import load_index

# measured: 137k boxes, median 5x5 px, p95 max side <= 8 px, max 20 px,
# so 6 px targets are sub-cell at stride 8 and detectors stay reference-only.


def analyze_bboxes(index: dict[str, dict]) -> dict:
    sizes = []
    images_with_boxes = 0
    for sample in index.values():
        boxes = np.asarray(sample.get("boxes_xyxy", np.zeros((0, 4))))
        if boxes.size == 0:
            continue
        images_with_boxes += 1
        sizes.append(np.stack([boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]], axis=1))

    if not sizes:
        return {"num_images_with_boxes": 0, "num_boxes": 0}

    size_array = np.concatenate(sizes, axis=0)
    max_dim = size_array.max(axis=1)
    unique_pairs = {(int(round(w)), int(round(h))) for w, h in size_array}
    return {
        "num_images_with_boxes": images_with_boxes,
        "num_boxes": int(size_array.shape[0]),
        "median_w": float(np.median(size_array[:, 0])),
        "median_h": float(np.median(size_array[:, 1])),
        "median_max_dim_px": float(np.median(max_dim)),
        "p95_max_dim_px": float(np.percentile(max_dim, 95)),
        "max_dim_px": float(max_dim.max()),
        "tiny_fraction_le4": float(np.mean(max_dim <= 4)),
        "tiny_fraction_le8": float(np.mean(max_dim <= 8)),
        "unique_size_pairs": int(len(unique_pairs)),
    }


def check_feasibility(
    stats: dict,
    min_median_px: float = 8.0,
    max_tiny_fraction: float = 0.30,
    stride: int = 8,
) -> dict:
    if not stats.get("num_boxes"):
        return {
            "suitable_as_primary_method": False,
            "usable_as_reference_baseline": False,
            "reason": "no bbox annotations",
            "stride": stride,
        }

    median_dim = stats["median_max_dim_px"]
    tiny_fraction = stats["tiny_fraction_le4"]
    problems = []
    if median_dim < min_median_px:
        problems.append(f"median target size {median_dim:.1f}px below {min_median_px:.1f}px (~one stride-{stride} cell)")
    if tiny_fraction > max_tiny_fraction:
        problems.append(f"<=4px point-like fraction {tiny_fraction:.1%} exceeds {max_tiny_fraction:.0%}")

    if problems:
        return {
            "suitable_as_primary_method": False,
            "usable_as_reference_baseline": True,
            "reason": "; ".join(problems),
            "stride": stride,
            "recommendation": "Reference-only detector; needs high-resolution FPN and small anchors",
        }
    return {
        "suitable_as_primary_method": True,
        "usable_as_reference_baseline": True,
        "reason": "bbox sizes support regular detector supervision",
        "stride": stride,
        "recommendation": "Train the reference baseline with build_faster_rcnn()",
    }


def build_faster_rcnn(
    num_classes: int = 2,
    anchor_sizes=(4, 8, 16, 32, 64),
    min_size: int = 448,
    max_size: int = 448,
    trainable_backbone_layers: int = 3,
):
    try:
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        from torchvision.models.detection.anchor_utils import AnchorGenerator
    except ImportError as exc:  # pragma: no cover
        raise ImportError("torch/torchvision is required to build the Faster R-CNN baseline") from exc

    model = fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
        num_classes=num_classes,
        min_size=min_size,
        max_size=max_size,
        trainable_backbone_layers=trainable_backbone_layers,
    )
    model.rpn.anchor_generator = AnchorGenerator(
        sizes=tuple((size,) for size in anchor_sizes),
        aspect_ratios=tuple((0.5, 1.0, 2.0) for _ in anchor_sizes),
    )
    return model


def boxes_to_targets(boxes_xyxy, labels=None) -> dict:
    # xyxy pixels, label 1 = penguin
    import torch

    boxes = torch.as_tensor(np.asarray(boxes_xyxy, dtype=np.float32)).reshape(-1, 4)
    if labels is None:
        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)
    else:
        labels = torch.as_tensor(np.asarray(labels), dtype=torch.int64)
    return {"boxes": boxes, "labels": labels}


def main() -> None:
    index = load_index()
    stats = analyze_bboxes(index)
    report = {"bbox_stats": stats, "feasibility": check_feasibility(stats)}
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

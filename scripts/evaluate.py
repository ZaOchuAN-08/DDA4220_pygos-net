"""Evaluation entry point: rebuild GT, compute overall and L0-L4 metrics, write figures."""

# prediction layout: <pred-dir>/<Site>/<tile>.npy (or .npz with "density"/"segmentation")
# fallback names: <Site>__<tile>.npy, <tile>.npy; masks: <tile>_seg.npy


from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pygosnet.evaluation import (  # noqa: E402
    apply_output_mask,
    assign_tier,
    evaluate_by_tier,
    load_index,
    load_split,
    split_counts,
    summarize_counts,
)
from pygosnet.evaluation.error_analysis import (  # noqa: E402
    counting_breakdown,
    summarize_error_sources,
)


def _candidate_paths(pred_dir: Path, site: str, name: str, suffix: str, allow_flat: bool = True):
    stem = Path(name).stem
    candidates = []
    for base, prefix in ((pred_dir / site, ""), (pred_dir, f"{site}__"), (pred_dir, "")):
        if not allow_flat and base == pred_dir:
            continue
        candidates.append(base / f"{prefix}{stem}{suffix}.npy")
        candidates.append(base / f"{prefix}{stem}{suffix}.npz")
    return candidates


def _load_output(path: Path) -> dict:
    # .npy -> density; .npz -> density and optional segmentation
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        output = {"density": loaded["density"]}
        if "segmentation" in loaded.files:
            output["segmentation"] = loaded["segmentation"]
        loaded.close()
        return output
    return {"density": loaded}


def find_prediction(pred_dir: Path, site: str, name: str, suffix: str = "", allow_flat: bool = True):
    for path in _candidate_paths(pred_dir, site, name, suffix, allow_flat=allow_flat):
        if path.exists():
            return path
    return None


def load_predictions(pred_dir, keys, duplicate_names, args):
    # density maps are [56,56] or [1,56,56]
    predictions: dict[str, np.ndarray] = {}
    missing: list[str] = []
    for key in keys:
        site, name = key.split("/", 1)
        allow_flat = Path(name).name not in duplicate_names
        path = find_prediction(pred_dir, site, name, allow_flat=allow_flat)
        if path is None:
            missing.append(key)
            continue
        output = _load_output(path)
        density = output["density"]
        if args.use_seg:
            seg = output.get("segmentation")
            if seg is None:
                seg_path = find_prediction(pred_dir, site, name, suffix=args.seg_suffix, allow_flat=allow_flat)
                if seg_path is None:
                    raise FileNotFoundError(f"Missing segmentation mask: {key} (expected {args.seg_suffix}.npy)")
                seg = np.load(seg_path)
            density = apply_output_mask(
                density, seg, threshold=args.mask_threshold, from_logits=args.seg_logits
            )
        predictions[key] = np.asarray(density, dtype=np.float64)
    return predictions, missing


def build_tables(index, keys, predictions, args):
    rows = []
    for key in keys:
        site, name = key.split("/", 1)
        sample = index[key]
        density = np.squeeze(predictions[key])
        if density.ndim != 2:
            raise ValueError(f"Prediction for {key} must be a 2D density map, got {predictions[key].shape}")

        decomposition = counting_breakdown(
            density,
            sample["points_yx"],
            sample["count"],
            stride=args.input_stride,
            radius_cells=args.support_radius,
        )
        rows.append(
            {
                "key": key,
                "site": site,
                "name": name,
                "tier": str(assign_tier(sample["count"])),
                "gt_count": float(sample["count"]),
                "pred_count": decomposition["pred_total"],
                "signed_error": decomposition["pred_total"] - sample["count"],
                **decomposition,
            }
        )
    return pd.DataFrame(rows)


def _json_safe(value):
    # NaN / Inf -> null
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _optional_float(text):
    # "none" selects the soft mask
    if str(text).strip().lower() in {"none", "null", "off"}:
        return None
    return float(text)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Pygos-Net evaluation")
    parser.add_argument("--split", default="data/splits/test.json", help="official split file")
    parser.add_argument("--pred-dir", required=True, help="directory of predicted density maps")
    parser.add_argument("--ann-glob", default="*_export-*.json", help="Labelbox export glob")
    parser.add_argument("--ann-root", default=".", help="directory containing the exports")
    parser.add_argument("--out-dir", default=None, help="output directory; defaults to outputs/eval/<split>")
    parser.add_argument("--seg-suffix", default="_seg", help="segmentation mask filename suffix")
    parser.add_argument("--use-seg", action="store_true", help="apply output masking")
    parser.add_argument("--seg-logits", action="store_true", help="segmentation file stores logits")
    parser.add_argument("--mask-threshold", type=_optional_float, default=0.5, help="mask threshold, default 0.5")
    parser.add_argument("--input-stride", type=int, default=8, help="density downsampling factor")
    parser.add_argument("--support-radius", type=int, default=1, help="GT support dilation in cells")
    parser.add_argument("--dpi", type=int, default=200, help="figure dpi")
    parser.add_argument("--no-figures", action="store_true", help="write tables only")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    split_path = Path(args.split)
    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.out_dir) if args.out_dir else Path("outputs/eval") / split_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    index = load_index(args.ann_glob, root=args.ann_root)
    keys = load_split(split_path)
    gt_counts = split_counts(index, keys)

    duplicate_names = {Path(key.split("/", 1)[1]).name for key in keys}
    duplicated = {name for name in duplicate_names if sum(key.endswith("/" + name) for key in keys) > 1}

    predictions, missing = load_predictions(pred_dir, keys, duplicated, args)
    if missing:
        example = missing[0].split("/", 1)
        expected = _candidate_paths(pred_dir, example[0], example[1], "")[0]
        raise SystemExit(
            f"Missing {len(missing)}/{len(keys)} prediction files, e.g. {missing[0]}\n"
            f"Expected path: {expected}\n"
            f"Check --pred-dir, site subdirectories, and flat Site__tile.npy naming"
        )

    table = build_tables(index, keys, predictions, args)
    pred_counts = table["pred_count"].to_numpy()

    overall = summarize_counts(pred_counts, gt_counts)
    tier_frame = evaluate_by_tier(pred_counts, gt_counts)
    sources = summarize_error_sources(table.to_dict(orient="records"))

    table.to_csv(out_dir / "per_image.csv", index=False, encoding="utf-8-sig")
    tier_frame.to_csv(out_dir / "tier_metrics.csv", index=False, encoding="utf-8-sig")
    sources.to_csv(out_dir / "error_sources.csv", index=False, encoding="utf-8-sig")

    report = {
        "split": str(split_path),
        "split_size": len(keys),
        "pred_dir": str(pred_dir),
        "annotation_glob": args.ann_glob,
        "use_seg_mask": bool(args.use_seg),
        "seg_logits": bool(args.seg_logits),
        "mask_threshold": args.mask_threshold,
        "input_stride": args.input_stride,
        "support_radius": args.support_radius,
        "overall": overall,
        "tiers": tier_frame.to_dict(orient="records"),
        "error_sources": sources.to_dict(orient="records"),
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(_json_safe(report), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if not args.no_figures:
        from pygosnet.evaluation.visualize import (
            plot_count_histogram,
            plot_count_scatter,
            plot_tier_metrics,
        )

        plot_count_scatter(gt_counts, pred_counts, save_path=out_dir / "count_scatter.png", dpi=args.dpi)
        plot_tier_metrics(tier_frame, save_path=out_dir / "tier_metrics.png", dpi=args.dpi)
        plot_count_histogram(gt_counts, save_path=out_dir / "count_histogram.png", dpi=args.dpi)

    print(f"\nsplit={split_path}  n={overall['n']}  gt_total={overall['gt_total']:.0f}  pred_total={overall['pred_total']:.0f}")
    print(
        f"MAE={overall['mae']:.3f}  RMSE={overall['rmse']:.3f}  "
        f"RelErr={overall['relative_count_error'] * 100:.2f}%  bias={overall['bias']:+.1f}"
    )
    columns = ["tier", "n", "gt_total", "pred_total", "mae", "rmse", "relative_count_error", "bias"]
    print(tier_frame[[c for c in columns if c in tier_frame.columns]].to_string(index=False))
    print("\nError sources:")
    source_columns = ["tier", "n", "gt_total", "pred_total", "bg_fp_rate", "miss_rate", "net_error"]
    if "excess_rate" in sources.columns:
        source_columns.insert(-1, "excess_rate")
    print(sources[source_columns].to_string(index=False))
    print(f"\nResults written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


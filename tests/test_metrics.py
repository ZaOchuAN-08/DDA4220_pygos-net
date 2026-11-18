"""Tests for evaluation metrics and utilities."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pygosnet.baselines.faster_rcnn import analyze_bboxes, check_feasibility  # noqa: E402
from pygosnet.eval.error_analysis import (  # noqa: E402
    counting_breakdown,
    density_mass_decomposition,
    image_error_record,
    masking_effect_record,
    select_representative_cases,
    summarize_error_sources,
    summarize_masking_effect,
    support_from_points,
)
from pygosnet.eval.labelbox import parse_record, split_counts  # noqa: E402
from pygosnet.eval.metrics import (  # noqa: E402
    apply_output_mask,
    check_count_preservation,
    count_from_density,
    counts_from_batch,
    density_counts,
    image_relative_errors,
    mae,
    relative_count_error,
    rmse,
    summarize_counts,
)
from pygosnet.eval.tier_eval import assign_tier, evaluate_by_tier, tier_distribution  # noqa: E402


# counts and metrics
def test_density_counts_shapes():
    single = np.ones((4, 4))
    assert count_from_density(single) == 16.0
    assert density_counts(single) == 16.0
    assert count_from_density(single[None, None]) == 16.0

    batch = np.stack([np.ones((4, 4)), np.full((4, 4), 2.0)])
    assert np.allclose(counts_from_batch(batch), [16.0, 32.0])
    assert np.allclose(density_counts(batch), [16.0, 32.0])
    assert np.allclose(counts_from_batch(batch[:, None]), [16.0, 32.0])


def test_count_preservation():
    density = np.zeros((8, 8))
    density[2, 3] = 7.0
    assert check_count_preservation(density, 7)["ok"] is True
    assert check_count_preservation(density, 8)["ok"] is False


def test_mae_rmse_bias():
    gt = np.array([0.0, 10.0, 100.0])
    pred = np.array([0.0, 12.0, 90.0])
    assert abs(mae(pred, gt) - (0 + 2 + 10) / 3) < 1e-9
    assert abs(rmse(pred, gt) - np.sqrt((0 + 4 + 100) / 3)) < 1e-9
    assert abs(summarize_counts(pred, gt)["bias"] - (0 + 2 - 10) / 3) < 1e-9


def test_relative_count_error():
    gt = np.array([10.0, 20.0])
    assert abs(relative_count_error(np.array([11.0, 22.0]), gt) - 0.1) < 1e-9
    # totals, not mean of per-image ratios
    assert abs(relative_count_error(np.array([0.0, 60.0]), gt) - 1.0) < 1e-9
    assert relative_count_error(np.array([0.0]), np.array([0.0])) == 0.0
    assert np.isnan(relative_count_error(np.array([3.0]), np.array([0.0])))


def test_image_relative_errors_empty_images():
    gt = np.array([0.0, 50.0])
    pred = np.array([0.0, 40.0])
    errors = image_relative_errors(pred, gt)
    assert errors[0] == 0.0
    assert abs(errors[1] - 0.2) < 1e-9
    # Predictions on an empty tile give NaN.
    assert np.isnan(image_relative_errors(np.array([5.0]), np.array([0.0]))[0])


# output masking
def test_apply_output_mask_soft():
    density = np.ones((1, 1, 2, 2)) * 4.0
    segmentation = np.array([[0.5, 0.0], [1.0, 0.25]])
    masked = apply_output_mask(density, segmentation, threshold=None)
    assert masked.shape == density.shape
    assert np.allclose(masked.squeeze(), [[2.0, 0.0], [4.0, 1.0]])


def test_apply_output_mask_default_is_hard():
    # hard mask by default
    density = np.ones((2, 2)) * 4.0
    segmentation = np.array([[0.5, 0.49], [1.0, 0.0]])
    assert np.allclose(apply_output_mask(density, segmentation), [[4.0, 0.0], [4.0, 0.0]])


def test_apply_output_mask_hard_threshold():
    density = np.ones((2, 2)) * 4.0
    segmentation = np.array([[0.6, 0.4], [0.9, 0.1]])
    masked = apply_output_mask(density, segmentation, threshold=0.5)
    assert np.allclose(masked, [[4.0, 0.0], [4.0, 0.0]])


def test_apply_output_mask_logits_matches_probabilities():
    logits = np.array([[[-2.0]]])
    expected = 1.0 / (1.0 + np.exp(2.0))  # sigmoid(-2)
    masked = apply_output_mask(np.ones_like(logits), logits, threshold=None, from_logits=True)
    assert abs(float(masked.sum()) - expected) < 1e-6

    # default hard mask zeros this
    assert float(apply_output_mask(np.ones_like(logits), logits, from_logits=True).sum()) == 0.0


def test_apply_output_mask_no_segmentation():
    density = np.ones((1, 1, 3, 3))
    assert apply_output_mask(density) is density


def test_apply_output_mask_shape_mismatch():
    try:
        apply_output_mask(np.ones((1, 1, 4, 4)), np.ones((1, 1, 3, 3)))
    except ValueError:
        return
    raise AssertionError("mismatched resolutions should raise")


# tiers
def test_assign_tier_boundaries():
    cases = {0: "L0", 1: "L1", 100: "L1", 101: "L2", 500: "L2", 501: "L3", 1000: "L3", 1001: "L4", 2682: "L4"}
    for count, expected in cases.items():
        assert assign_tier(count) == expected, (count, assign_tier(count))
    assert list(assign_tier(np.array([0, 150, 2000]))) == ["L0", "L2", "L4"]


def test_tier_distribution():
    counts = np.array([0.0, 0.0, 50.0, 200.0, 600.0, 1200.0])
    frame = tier_distribution(counts).set_index("tier")
    assert frame.loc["L0", "n_images"] == 2
    assert frame.loc["L4", "gt_total"] == 1200.0
    assert abs(frame["image_share"].sum() - 1.0) < 1e-9
    assert abs(frame["penguin_share"].sum() - 1.0) < 1e-9


def test_evaluate_by_tier():
    gt = np.array([0.0, 50.0, 300.0, 800.0, 1500.0])
    pred = np.array([1.0, 45.0, 330.0, 700.0, 1000.0])
    frame = evaluate_by_tier(pred, gt).set_index("tier")

    assert list(frame.index) == ["L0", "L1", "L2", "L3", "L4"]
    assert frame.loc["L0", "mae"] == 1.0
    assert frame.loc["L2", "mae"] == 30.0
    assert frame.loc["L3", "rmse"] == 100.0
    assert frame.loc["L4", "bias"] == -500.0


# error decomposition
def _synthetic_pair():
    gt = np.zeros((14, 14))
    gt[3, 3] = 10.0
    gt[10, 10] = 6.0
    pred = gt * 0.8
    pred[0, 13] = 4.0  # background false positive
    return gt, pred


def test_mass_decomposition_identity():
    gt, pred = _synthetic_pair()
    result = density_mass_decomposition(gt, pred, tau=1e-6, dilate=0)
    assert abs(result["pred_total"] - pred.sum()) < 1e-9
    assert abs(result["tp_mass"] + result["bg_fp_mass"] - result["pred_total"]) < 1e-9
    assert abs(result["missed_mass"] - max(gt.sum() - result["tp_mass"], 0.0)) < 1e-9
    assert result["bg_fp_mass"] > 0 and result["missed_mass"] > 0


def test_support_from_points_dilation():
    support = support_from_points([[10.0, 20.0]], (28, 28), stride=8, radius_cells=1)
    assert support.sum() == 9
    assert support[1, 2] == 1.0


def test_counting_breakdown_detects_missed_targets():
    # 4 GT, 3 predicted, 2 background
    points = np.array([[8.0, 8.0], [8.0, 24.0], [24.0, 8.0], [24.0, 24.0]])
    density = np.zeros((7, 7))
    density[1, 1] = 1.0
    density[1, 3] = 1.0
    density[3, 1] = 1.0
    density[6, 6] = 2.0

    result = counting_breakdown(density, points, gt_count=4, stride=8, radius_cells=0)
    assert abs(result["tp_mass"] - 3.0) < 1e-9
    assert abs(result["missed_mass"] - 1.0) < 1e-9
    assert abs(result["bg_fp_mass"] - 2.0) < 1e-9
    assert abs(result["excess_mass"] - 0.0) < 1e-9
    identity = result["tp_mass"] + result["excess_mass"] + result["bg_fp_mass"]
    assert abs(identity - result["pred_total"]) < 1e-9
    assert abs(result["miss_rate"] - 0.25) < 1e-9


def test_counting_breakdown_reports_dense_excess():
    # 2 GT, 5 predicted -> 3 excess
    points = np.array([[8.0, 8.0], [9.0, 9.0]])
    density = np.zeros((7, 7))
    density[1, 1] = 5.0
    result = counting_breakdown(density, points, gt_count=2, stride=8, radius_cells=0)
    assert abs(result["tp_mass"] - 2.0) < 1e-9
    assert abs(result["excess_mass"] - 3.0) < 1e-9
    assert abs(result["missed_mass"] - 0.0) < 1e-9


def test_error_summary_aggregation():
    gt, pred = _synthetic_pair()
    records = [
        image_error_record(gt, pred, gt_count=16.0),
        image_error_record(np.zeros((14, 14)), np.full((14, 14), 0.5), gt_count=0.0),
    ]
    frame = summarize_error_sources(records).set_index("tier")
    assert frame.loc["L0", "gt_total"] == 0.0
    assert frame.loc["L0", "bg_fp_mass"] > 0
    assert abs(frame.loc["L1", "pred_total"] + frame.loc["L0", "pred_total"] - frame.loc["ALL", "pred_total"]) < 1e-9


# output masking effect
def _masking_pair():
    # 2 GT on support, 2 units of background FP, mask suppresses one GT cell and all FP
    points = np.array([[8.0, 8.0], [24.0, 8.0]])
    raw = np.zeros((7, 7))
    raw[1, 1] = 1.0
    raw[3, 1] = 1.0
    raw[6, 6] = 2.0
    mask = np.zeros((7, 7))
    mask[1, 1] = 1.0
    return points, raw, raw * mask


def test_masking_effect_separates_fp_from_true_suppression():
    points, raw, masked = _masking_pair()
    record = masking_effect_record(raw, masked, points, gt_count=2.0, stride=8, radius_cells=0)
    assert record["tier"] == "L1"
    assert abs(record["raw_count"] - 4.0) < 1e-9
    assert abs(record["masked_count"] - 1.0) < 1e-9
    assert abs(record["fp_removed_mass"] - 2.0) < 1e-9
    assert abs(record["true_suppressed_mass"] - 1.0) < 1e-9
    assert abs(record["raw_mae"] - 2.0) < 1e-9
    assert abs(record["masked_mae"] - 1.0) < 1e-9


def test_summarize_masking_effect_per_tier():
    points, raw, masked = _masking_pair()
    record = masking_effect_record(raw, masked, points, gt_count=2.0, stride=8, radius_cells=0)
    empty = masking_effect_record(np.zeros((7, 7)), np.zeros((7, 7)), np.zeros((0, 2)), gt_count=0.0,
                                  stride=8, radius_cells=0)
    frame = summarize_masking_effect([record, empty]).set_index("tier")

    assert abs(frame.loc["L1", "mass_suppressed_share"] - 0.75) < 1e-9
    assert abs(frame.loc["L1", "over_suppression_rate"] - 0.5) < 1e-9
    assert abs(frame.loc["L1", "bg_fp_rate_raw"] - 0.5) < 1e-9
    assert abs(frame.loc["L1", "bg_fp_rate_masked"] - 0.0) < 1e-9
    assert abs(frame.loc["L1", "mae_delta"] - (-1.0)) < 1e-9
    assert abs(frame.loc["ALL", "masked_total"] - 1.0) < 1e-9


def test_select_representative_cases_picks_worst_per_tier_and_source():
    frame = pd.DataFrame([
        {"tier": "L1", "key": "a", "bg_fp_rate": 0.1, "miss_rate": 0.2, "excess_rate": 0.0},
        {"tier": "L1", "key": "b", "bg_fp_rate": 0.7, "miss_rate": 0.1, "excess_rate": 0.0},
        {"tier": "L4", "key": "c", "bg_fp_rate": 0.0, "miss_rate": 0.6, "excess_rate": 0.4},
    ])
    cases = select_representative_cases(frame)
    l1 = cases[cases["tier"] == "L1"]
    # bg false positives and missed targets come from different images
    assert set(l1["key"]) == {"a", "b"}
    assert set(l1["case_type"]) == {"bg_fp_rate", "miss_rate"}
    assert cases[cases["tier"] == "L4"].iloc[0]["key"] == "c"


# parsing and feasibility
def _record(objects, name="tile.png", skipped=False):
    return {"External ID": name, "Skipped": skipped, "Label": {"objects": objects} if objects else {}}


def _box(left, top, width, height, value="penguin"):
    return {"value": value, "bbox": {"left": left, "top": top, "width": width, "height": height}}


def test_parse_record_filters_non_penguin():
    record = _record([_box(10, 20, 4, 4), _box(30, 40, 6, 2), _box(1, 1, 8, 8, value="other"),
                      _box(2, 2, 8, 8, value="elephant_seal")])
    sample = parse_record(record, site="Jack")
    assert sample["count"] == 2
    assert sample["key"] == "Jack/tile.png"
    assert np.allclose(sample["points_yx"][0], [22.0, 12.0])
    assert np.allclose(sample["boxes_xyxy"][1], [30, 40, 36, 42])


def test_parse_record_empty_and_skipped():
    sample = parse_record(_record([], skipped=True), site="Luke")
    assert sample["count"] == 0
    assert sample["skipped"] is True
    assert sample["boxes_xyxy"].shape == (0, 4)


def test_split_counts_fails_fast():
    index = {"Jack/a.png": {"count": 3}}
    assert np.allclose(split_counts(index, ["Jack/a.png"]), [3.0])
    try:
        split_counts(index, ["Jack/missing.png"])
    except KeyError:
        return
    raise AssertionError("missing samples should raise instead of being skipped")


def _index_with_box_size(size: float, n_images: int = 2) -> dict:
    boxes = np.array([[0.0, 0.0, size, size]])
    return {f"Jack/{i}.png": {"boxes_xyxy": boxes} for i in range(n_images)}


def test_feasibility_rejects_point_like_boxes():
    stats = analyze_bboxes(_index_with_box_size(5.0))
    verdict = check_feasibility(stats)
    assert stats["num_boxes"] == 2
    assert stats["median_max_dim_px"] == 5.0
    # tiny boxes: reference baseline only
    assert verdict["suitable_as_primary_method"] is False
    assert verdict["usable_as_reference_baseline"] is True


def test_feasibility_accepts_real_boxes():
    stats = analyze_bboxes(_index_with_box_size(40.0))
    verdict = check_feasibility(stats)
    assert verdict["suitable_as_primary_method"] is True
    assert verdict["usable_as_reference_baseline"] is True


def main() -> int:
    tests = [(name, func) for name, func in sorted(globals().items()) if name.startswith("test_") and inspect.isfunction(func)]
    failures = 0
    for name, func in tests:
        try:
            func()
            print(f"PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

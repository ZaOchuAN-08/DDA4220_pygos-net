# Evaluation Protocol, Error Analysis and Reproducibility (Workstream D)

This document specifies how every model in the project is evaluated: metric definitions, output-mask effects, error decomposition, visualization, the Faster R-CNN reference baseline, and reproduction commands. Everything is implemented in `src/pygosnet/eval/`, `src/pygosnet/baselines/` and `scripts/evaluate.py`, so all four ablation variants are scored by the same code.

## 1. Ground truth and splits

- Annotations come from `data/processed/manifest.jsonl` and the splits `data/splits/{train,val,test}.jsonl` (Workstream A). Evaluation reads them through `--manifest` and never re-merges or edits labels.
- Size: 738 images of 448x448, split 446/146/146. Validation is used for model selection; the test split is touched only once, after the method is frozen.
- Overlapping annotations are adjudicated once, as recorded in `data/splits/split_metadata.json` (currently `LUKE`), and the same adjudication is used for training and evaluation.
- Annotations are stored as `(x, y)` pixel points and converted internally to `(row, col)`. A stride-8 cell is identified by `row = y // 8`, `col = x // 8`.

## 2. Metrics

Counts are obtained by summing the density map: `N = sum(D_out)`.

| Metric | Definition | Code |
|---|---|---|
| MAE | `mean(|N_pred - N_gt|)` over images | `metrics.mae` |
| RMSE | `sqrt(mean((N_pred - N_gt)^2))` | `metrics.rmse` |
| bias | `mean(N_pred - N_gt)`; negative means undercounting | `metrics.bias` |
| Dataset-level relative count error | `|sum(N_pred) - sum(N_gt)| / sum(N_gt)`, not the mean of per-image ratios | `metrics.relative_count_error` |
| Per-image relative error | `|N_pred - N_gt| / N_gt`; 0 when GT and prediction are both 0, otherwise NaN if GT is 0 | `metrics.image_relative_errors` |
| Tier metrics | MAE/RMSE/bias/relative error computed within each count tier | `tier_eval.evaluate_by_tier` |

Tiers are defined by the GT count: L0 = 0; L1 = 1-100; L2 = 101-500; L3 = 501-1000; L4 > 1000. Per-image relative error is undefined for empty images, so L0 contributes only count error (mostly background false positives), which is reported separately.

## 3. Output mask and its effect

The model produces an unmasked density `D` and a segmentation probability `S`. The final output used for counting is

```
D_out = D * 1[S >= 0.5]
```

Both `D` and `D_out` are kept during evaluation. `error_analysis.masking_effect_record` decomposes the difference per image and `summarize_masking_effect` aggregates it per tier:

- `fp_removed_mass`: density mass removed outside the GT support (background false positives eliminated);
- `true_suppressed_mass`: density mass removed inside the GT support (real targets suppressed);
- `mae_delta = masked_MAE - raw_MAE`: negative means masking reduces error;
- `over_suppression_rate = true_suppressed / GT_total`: share of true target mass eaten by the mask;
- `miss_rate`: GT mass still not covered after masking.

This table answers the question shared by Workstreams D and C: whether gains on L0/L1 come at the cost of suppressing real penguins on L3/L4. If L3/L4 show a positive `mae_delta` and an `over_suppression_rate` clearly above L1/L2, dense scenes are being undercounted by the mask. `--mask-threshold none` reproduces the soft-mask control.

## 4. Error-source decomposition

`error_analysis.counting_breakdown` builds a stride-8 GT support from the point annotations (optionally dilated with `--support-radius`) and splits predicted mass into:

| Quantity | Meaning |
|---|---|
| `tp_mass` | `sum(min(pred_cell, gt_cell))` inside GT cells; countable, matched mass |
| `excess_mass` | predicted mass above the GT count inside GT cells; duplicate responses in dense colonies |
| `bg_fp_mass` | all mass outside the GT support; responses on rock, shadow, snow or water |
| `missed_mass` | `max(GT_total - tp_mass, 0)`; missing counts |

Identity: `pred_total = tp_mass + excess_mass + bg_fp_mass`. The summary reports `bg_fp_rate`, `miss_rate`, `excess_rate` and `net_error`, so reduced background error can be distinguished from increased missed targets instead of relying on aggregate MAE alone.

## 5. Visual outputs

`scripts/evaluate.py` writes:

- `count_scatter.png`: per-image prediction vs. ground truth, colored by L0-L4, annotated with MAE/RMSE/relative error;
- `tier_metrics.png`: per-tier MAE/RMSE bars with sample counts;
- `count_histogram.png`: GT count distribution with tier boundaries, documenting sample size and tier imbalance;
- `masking_effect.png`: background vs. true-target mass removed by the mask (left) and MAE before/after masking (right);
- `figures/case_*.png` (`--case-figures N`): representative error cases selected per tier. Each panel shows input with point annotations, GT density, raw prediction `D`, segmentation support, masked `D_out`, and the signed error map. GT, prediction and masked prediction share one color scale, so the panels are paper-ready. Selection is implemented in `error_analysis.select_representative_cases`: per tier, take the images with the highest `bg_fp_rate`, `miss_rate` and `excess_rate`, then de-duplicate.

Qualitative inspection uses these perspectives: rocks and shadows (high `bg_fp_rate`), snow glare (weak scattered responses outside the support), zero-count/sparse images (L0/L1 false positives), and dense colonies (high `excess_rate` or `miss_rate`).

## 6. Ablations and multiple seeds

- The evaluator is model-agnostic: baseline, +DCB, +hard-negative and full models all export `D` and `S` through `predict.py` and are scored by the same `evaluate.py`.
- No tuning on the test split. Validation selects checkpoints and loss weights; the test split is evaluated once after the final configuration is frozen.
- Multi-seed results are aggregated with `scripts/aggregate_runs.py --glob "outputs/eval/*/metrics.json"` into mean +/- std. Each `metrics.json` records split, mask threshold, stride and support radius, and can be paired with `experiments/*/protocol.json` (commit, manifest hash) for exact reproduction.

## 7. Faster R-CNN reference baseline feasibility

`src/pygosnet/baselines/faster_rcnn.py` provides a feasibility check and a reference implementation; point annotations are never treated as real boxes.

Measured over all Labelbox exports (558 images with boxes, 137,187 boxes). Raw output is stored in `experiments/baselines/faster_rcnn_feasibility.json` and can be regenerated with `python -m pygosnet.baselines.faster_rcnn` from `data/raw`:

| Statistic | Value |
|---|---:|
| Median box size (longest side) | 6 px |
| p95 longest side | 8 px |
| Maximum longest side | 20 px |
| Share with longest side <= 4 px | 11.1% |
| Share with longest side <= 8 px | 97.3% |
| Distinct integer width-height pairs | 158 |

Conclusion: `suitable_as_primary_method = false`, `usable_as_reference_baseline = true`. The median object is 6 px, smaller than one stride-8 cell, and box sizes are highly discrete. A detector would need a high-resolution FPN and tiny anchors, and its output is not comparable with point-supervised density regression. Faster R-CNN is therefore recorded as a reference-baseline feasibility result only. If training is pursued, use `build_faster_rcnn(anchor_sizes=(4, 8, 16, 32, 64))` and state explicitly in the paper that it is a reference method, not a competing primary method.

## 8. Optional DEM / terrain extension: stopped

The data contain only 448x448 aerial tiles with pixel-level point annotations. There are no georeferenced coordinates (lat/lon or projection) and no DEM or slope raster for the same area, so elevation alignment, area-weighted counts and effort normalization cannot be performed. As agreed in the work split, this extension is stopped and no placeholder code is kept; `src/pygosnet/eco/` remains empty. To resume: obtain georeferenced source imagery plus a co-located DEM, and confirm whether tiles overlap or provide a mosaic. Only then implement `dem_align.py` and `terrain_stats.py`, and report correlations without causal claims.

## 9. Reproduction

```bash
# 1) export raw D and segmentation probabilities for validation (or the final test split)
python scripts/predict.py --checkpoint <run>/best.pt --manifest data/splits/val.jsonl \
    --project-root . --out-dir outputs/<run>_predictions --device cuda

# 2) metrics + tiers + error sources + mask effect + figures + qualitative cases
python scripts/evaluate.py --manifest data/splits/val.jsonl --pred-dir outputs/<run>_predictions \
    --use-seg --case-figures 1 --out-dir outputs/eval/<run> --project-root .

# 3) multi-seed / multi-configuration aggregation
python scripts/aggregate_runs.py --glob "outputs/eval/*/metrics.json" --out-dir outputs/eval/aggregated
```

Outputs: `metrics.json` (overall, overall_raw, tiers, error_sources, masking_effect), `per_image.csv`, `tier_metrics.csv`, `error_sources.csv`, `masking_effect.csv`, `case_gallery.csv`, and the PNGs listed above. `per_image.csv` contains both `raw_pred_count` and `pred_count`, so the mask effect can be audited image by image.

## 10. Tests

`tests/test_metrics.py` covers: counting and count preservation, MAE/RMSE/bias, dataset-level and per-image relative error, soft/hard/softmax masking, L0-L4 boundaries and tier metrics, the error-decomposition identity, the mask effect (separating FP removal from true-target suppression), representative case selection, Labelbox parsing, and the Faster R-CNN feasibility verdict.

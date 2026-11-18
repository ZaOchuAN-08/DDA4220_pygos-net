# Best-checkpoint evaluation on the validation split (Workstream D)

## Setup

- Checkpoints: `experiments/preliminary_epoch200_windows/checkpoints/baseline-best-epoch189.pt` and `plus_dcb-best-epoch158.pt` (best validation MAE+RMSE from epoch 101 onward).
- Data: `data/splits/val.jsonl`, 146 images of 448x448, 23,771 ground-truth penguins.
- Inference: raw density `D` and segmentation probability `S`; counts use `D_out = D * 1[S >= 0.5]` (threshold 0.5), stride 8, GT support dilation radius 1.
- Everything here was regenerated with the current evaluator. MAE/RMSE agree with `checkpoints/best-validation.md` to within 0.004; the difference comes from re-exported predictions and evaluator version, and does not change any conclusion.

Reproduce:

```bash
python scripts/predict.py --checkpoint <best.pt> --manifest data/splits/val.jsonl \
    --project-root . --out-dir outputs/best_checkpoint_eval/<run>_predictions --device cpu
python scripts/evaluate.py --manifest data/splits/val.jsonl --pred-dir outputs/best_checkpoint_eval/<run>_predictions \
    --use-seg --case-figures 3 --project-root . --out-dir experiments/best_checkpoint_eval/<run>
```

## Overall metrics

| Model | MAE | RMSE | Dataset-level relative count error | bias |
|---|---:|---:|---:|---:|
| Baseline (ep.189) | 20.240 | 39.689 | 2.94% | +4.8 |
| +DCB (ep.158) | 19.960 | 40.634 | 3.75% | +6.1 |

## Per-tier metrics

| Tier | n | Baseline MAE | Baseline RMSE | Baseline bias | +DCB MAE | +DCB RMSE | +DCB bias |
|---|---:|---:|---:|---:|---:|---:|---:|
| L0 | 35 | 1.88 | 5.75 | +1.88 | 2.09 | 5.36 | +2.09 |
| L1 | 53 | 12.42 | 20.26 | +7.12 | 11.95 | 19.15 | +6.89 |
| L2 | 43 | 29.46 | 42.40 | +10.39 | 27.96 | 41.73 | +11.47 |
| L3 | 12 | 68.02 | 101.00 | -11.49 | 69.04 | 105.77 | -1.84 |
| L4 | 3 | 49.29 | 49.55 | -17.83 | 58.91 | 61.92 | -6.07 |

## Output-mask effect (D <-> C)

| Tier | Baseline bg mass removed | Baseline true mass suppressed | Baseline true suppression rate | +DCB bg mass removed | +DCB true mass suppressed | +DCB true suppression rate |
|---|---:|---:|---:|---:|---:|---:|
| L0 | 9,483.7 | 0.0 | 0.00% | 9,107.1 | 0.0 | 0.00% |
| L1 | 12,854.6 | 34.6 | 1.72% | 12,285.0 | 26.4 | 1.31% |
| L2 | 7,221.7 | 42.0 | 0.44% | 7,033.7 | 25.7 | 0.27% |
| L3 | 1,053.5 | 7.6 | 0.09% | 1,019.1 | 7.7 | 0.09% |
| L4 | 174.9 | 1.3 | 0.03% | 169.0 | 0.7 | 0.02% |

True suppression rate = `true_suppressed_mass / GT_total` (`over_suppression_rate` in `masking_effect.csv`).

Conclusions:

1. Nearly all mass removed by the mask lies outside the GT support (about 30.8k and 29.6k), while removed true-target mass is only 0.25%-0.36% of GT. The L0/L1 improvement therefore does not come from suppressing real penguins in L3/L4.
2. L3/L4 remain undercounted (bias between -1.8 and -17.8), but their mask suppression rate is below 0.1%. Dense-scene undercounting originates in density regression, not in the output mask.
3. +DCB reduces L1/L2 MAE but increases L3/L4 RMSE, and leaves more residual background false positives (7.7% vs 6.3%). It undercounts L4 less severely (bias -6.1 vs -17.8).

## Files

| File | Content |
|---|---|
| `comparison.csv` | Overall, per-tier and mask-effect summary for both models |
| `tier_comparison.png` | Per-tier MAE/RMSE comparison of the two models |
| `baseline/`, `plus_dcb/` | Per-run `metrics.json`, `per_image.csv`, `tier_metrics.csv`, `error_sources.csv`, `masking_effect.csv`, `case_gallery.csv` and figures |
| `baseline/figures/`, `plus_dcb/figures/` | Representative cases per tier: input + points, GT density, raw prediction, segmentation support, masked prediction, signed error |
| `baseline/masking_effect.png`, `plus_dcb/masking_effect.png` | Background vs. true mass removed by the mask, and MAE before/after masking |

Qualitative cases are selected automatically by the highest `bg_fp_rate`, `miss_rate` and `excess_rate` per tier, supporting discussion of rocks/shadows (background false positives), sparse and zero-count images (L0/L1), and dense colonies (L2-L4 excess/miss).

## Qualitative error observations

Inspection of the generated panels (`--case-figures 3`) gives the following evidence, using baseline predictions:

| Category | Evidence | Figure |
|---|---|---|
| Zero-count / snow, ice and wet rock | 35 empty images average 272.8 raw counts but only 1.9 after masking. In this example the raw map scatters low-amplitude blobs over bright snow/ice and dark rock; the mask leaves 1 count. | `baseline/figures/case_01_L0_bg_fp_rate.png` (Jack, GT 0, raw 280, masked 1) |
| Sparse colony on dark rock / water | GT 7; the raw map blankets dark rock and water with responses (292 counts); masking keeps the small guano-covered cluster and yields 7. | `baseline/figures/case_03_L1_bg_fp_rate.png` |
| Dense colony | GT 1,461; the mask transmits almost all raw mass (1,428 to 1,405), so dense scenes are not collapsed by masking. Residual error is spatially mixed rather than a uniform undercount. | `baseline/figures/case_10_L4_bg_fp_rate.png` (Thomas) |
| Mid-density boundary case | GT 127, raw 296, masked 47. The mask removes 84% of raw mass, but only 9.6 units were true-target mass (7.5% of this image's GT); the remaining gap is missing segmentation support where penguins sit on snow/ice. Per-tier true suppression is 0.44% for L2, so this is a concentrated case, not a systematic trend. | `baseline/figures/case_06_L2_miss_rate.png` (Thomas) |
| Dense-cluster merging | Excess rate is 0.49 overall and 0.39 on L4: responses in colony cores exceed the point count, consistent with neighbouring birds merging into one density peak. | `baseline/figures/case_07_L2_excess_rate.png`, `case_09_L3_excess_rate.png` |

These observations are consistent with the mask-effect table: background false positives dominate the removed mass, while true-target suppression stays below 1.8% even at L1 and below 0.1% at L3/L4. Bright snow/ice and wet rock produce weak raw responses, but the support branch suppresses them before counting.

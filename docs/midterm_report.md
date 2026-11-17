# Midterm Report: Pygos-Net

**Course:** DDA4220 Deep Learning, Fall 2026
**Team:** [Member A], [Member B], [Member C], [Member D]
**Repository:** [URL] | **Commit:** [SHA]
**Date:** 17 November 2026

## 1. Introduction: Topic and Motivation

Counting penguins from aerial imagery is a core task in Antarctic wildlife monitoring, where repeated ground surveys are expensive and logistically difficult. The BAS benchmark curated by Qian et al. (2023) provides 448x448 aerial tiles covering approximately 140 colonies in the Antarctic Peninsula and South Shetland Islands, with point annotations for 135,279 penguins. The task is difficult for two reasons. First, individual penguins occupy only a few pixels, so a target spans roughly one stride-8 output cell. Second, the background is visually hostile: cliff shadows, rock surfaces, guano stains and snow glare produce responses that resemble penguins, and 61.2% of tiles contain at most 100 penguins, so false positives on sparse background dominate the overall error.

Detection-based counting is poorly suited to this setting because targets are smaller than typical anchor sizes and heavily overlapping in dense colonies. We therefore adopt point-supervised density-map estimation: each annotated point is rendered as a Gaussian kernel, the network predicts a downsampled density map, and the count is the sum of that map. This formulation preserves spatial information, handles overlap, and requires only point annotations.

Building on the VGG-19 dual-branch baseline of Qian et al., this project introduces two components. The **Dilated Context Block (DCB)** aggregates multi-scale context while preserving spatial detail, which is intended to reduce background false positives on heterogeneous terrain. The **hard-negative background loss (`Lbg`)** explicitly suppresses high predicted density outside the annotated density support. The project asks whether these two components improve sparse-background counting without degrading dense colonies.

**Objectives**

1. Replicate and benchmark the VGG-19 dual-branch density baseline and a Faster R-CNN detector on the BAS dataset.
2. Design and implement Pygos-Net with a DCB to address shadow and terrain artifacts.
3. Implement a hybrid loss combining `SL*` with hard-negative background suppression to reduce MAE on L0 and L1 tiles.
4. Evaluate counting accuracy alongside continuous density metrics across the density tiers, with MAE as the primary metric.
5. Construct an ecological evaluation pipeline linking predicted density to DEM elevation and slope.

Objective 5 is optional and is not required for the core contribution.

## 2. Related Works

Density-map estimation has become the standard formulation for crowd and animal counting because it converts point supervision into a dense regression target that tolerates overlap and scale variation. The BAS penguin benchmark of Qian et al. (2023) applies this formulation to aerial wildlife imagery and provides the VGG-19 dual-branch baseline reproduced here: a shared encoder, a stride-8 feature map, a segmentation head that predicts density support, and a density head that predicts non-negative density. The baseline also introduces segmentation-guided feature dampening and inference-time output masking, which we retain rather than claim as new contributions.

Our structural supervision follows Rong and Li (2021), who use SSIM (Wang et al., 2004) between predicted and ground-truth density maps to preserve local structure. This is particularly relevant for small, densely packed targets where per-pixel regression alone produces blurred predictions.

For context aggregation we build on dilated convolution (Yu and Koltun, 2016). Instead of a single large receptive field, the DCB runs parallel 3x3 convolutions with dilation 1, 2 and 4, concatenates the responses and projects them back to 512 channels. This keeps the stride-8 resolution unchanged, which matters because any further downsampling would erase targets that already occupy only one output cell.

Background suppression is related to hard-negative mining in detection and to background regularization in counting. Rather than weighting all background equally, the proposed `Lbg` selects the highest-response background cells and penalizes them more strongly, so that the loss focuses on the terrain that actually causes false positives. Detection-based counting, represented by Faster R-CNN, is used only as a reference baseline: its anchors must be adapted to 3x3 to 6x6 pixel targets, and its accuracy is expected to be limited by resolution rather than by annotation quality.

## 3. Project Plan and Progress

### 3.1 Experimental Plan and Methodology

**Pipeline.** The framework has five stages: annotation parsing, density-map generation, dual-branch prediction, combined loss optimization, and tier-stratified evaluation with output masking.

**Dataset.** Experiments use 738 non-overlapping 448x448 tiles with the official 3:1:1 split of 446 training, 146 validation and 146 test tiles, and 135,279 annotated penguins. Tiles are grouped into five density tiers so that sparse-background and dense-colony behavior can be separated:

| Tier | Count range | Role |
|---|---|---|
| L0 | 0 | Zero-count background |
| L1 | 1–100 | Ultra-sparse |
| L2 | 101–500 | Low density |
| L3 | 501–1000 | Moderate density |
| L4 | >1000 | High density |

L0 and L1 together account for 61.2% of the dataset but contain only a small fraction of penguins. Overall MAE is therefore dominated by sparse tiles, and an aggregate improvement can conceal undercounting in dense colonies. Every result is consequently reported per tier as well as overall.

**Density-map generation.** Point annotations are rendered as normalized isotropic Gaussian kernels with sigma = 4 input pixels. For each training crop, targets are regenerated after cropping and downsampled to stride-8 resolution, producing a 56x56 map for a 448x448 input. Each map is normalized so that its sum equals the annotated count, and count preservation is verified after cropping and downsampling. A threshold epsilon = 1e-3 defines the effective density support used for segmentation targets and hard-negative mining.

**Model.** The baseline is a VGG-19 encoder with a stride-8 feature map feeding two heads. The segmentation head predicts density support, which is used both for feature dampening during training and for output masking at inference. The density head produces non-negative density. The DCB is inserted before the heads and is controlled by a single `use_dcb` switch, so Baseline and Pygos-Net share all other components.

**Losses.** The objective combines masked structural loss, weighted segmentation cross-entropy and hard-negative background loss:

```
L = SL* + lambda_seg * Lseg + lambda_bg * Lbg
```

`SL*` uses SSIM on the masked density map. `Lbg` takes a background set `B` outside the truncated Gaussian support plus an annotation-tolerance margin, selects the top-K highest predicted cells `H` with `rho = 0.1`, and penalizes both diffuse and concentrated false responses with `eta = 0.5`. Empty tiles use all cells as background, `SL* = 0`, and segmentation and background losses remain active. Background supervision is applied to the density before output masking.

**Ablation.** The main experiment is a controlled 2x2 design:

| Configuration | DCB | Hard-Negative Loss |
|---|---:|---:|
| Baseline | no | no |
| +DCB | yes | no |
| +Hard-Negative | no | yes |
| Pygos-Net | yes | yes |

All configurations share the same split, training budget, augmentation and checkpoint rule. Hypotheses: H1, DCB reduces background false positives while maintaining or improving overall accuracy; H2, `Lbg` gives the largest improvement on L0/L1; H3, Pygos-Net gives the best overall trade-off without substantial L3/L4 degradation.

**Training protocol.** Random 256x256 crops and horizontal flips, Adam with learning rate 1e-5, weight decay 1e-4, batch size 16 and 600 epochs. Inference processes the full 448x448 image, producing a 56x56 density map whose sum is the predicted count. From epoch 101, validation runs every epoch and the checkpoint with the lowest validation MAE + RMSE is kept. The test split is untouched until model selection and hyperparameters are frozen.

### 3.2 Evaluation Metrics

MAE is the primary metric, with RMSE measuring sensitivity to large errors. We additionally report dataset-level relative count error and per-tier MAE/RMSE for L0–L4. Tier-level reporting is essential because it exposes whether a gain on sparse tiles is bought with undercounting in dense colonies.

Because continuous density metrics are also required, each tile is additionally summarized by the agreement between predicted and ground-truth density maps through per-tile correlation and SSIM. These density-level scores are reported per tier and are descriptive; MAE remains the primary criterion.

Beyond counting accuracy, the evaluation separates error sources. Each predicted density map is decomposed into matched mass, dense excess, background false positives and missed targets on the stride-8 grid, so an improvement can be attributed to background suppression rather than to a shift in overall calibration. Predictions are masked with the segmentation output at threshold 0.5, matching the inference definition. A single evaluation entry point is applied to all four configurations, and the test split is evaluated once after the hyperparameters are frozen.

Qualitative analysis compares input image, ground-truth density, predicted density, segmentation mask and final masked prediction for cliff shadows, rock surfaces, snow glare, sparse scenes and dense colonies. Before Faster R-CNN is used as a reference baseline, the annotations are audited: genuine bounding boxes are used if they exist, and point annotations are never treated as ground-truth boxes. The published results of Qian et al. are included as a second reference. Where computationally feasible, each configuration is trained with multiple seeds and reported as mean +/- standard deviation.

### 3.3 Current Progress, Successes, and Challenges

All Phase 1–3 deliverables are complete. The data pipeline converts Labelbox exports into density maps with verified count preservation; the baseline, DCB and loss modules are implemented with numerical and gradient tests; the four ablation configurations share a single training entry point; and the evaluation stack produces overall, tier-level and error-source tables together with comparison figures. The closed loop from data loading through backward pass, validation and checkpoint selection runs end-to-end, and the four preliminary ablation runs have been launched on the validation split. Phases 1–3 of the proposal timeline (Oct 12 – Nov 18) are therefore complete, and Phases 4–5 (Nov 19 – Dec 15) cover the full holdout evaluation, multi-seed runs and final reporting. The numbers below are inserted as each preliminary run finishes.

Preliminary runs follow the same conditions as the final ablation:

**P0 (Baseline, no DCB, no `Lbg`)**

```
Best epoch (min valid loss): [TODO] | valid_loss = [TODO] | valid_MAE = [TODO] | valid_RMSE = [TODO]
```

**P1 (Baseline + DCB)**

```
Best epoch (min valid loss): [TODO] | valid_loss = [TODO] | valid_MAE = [TODO] | valid_RMSE = [TODO]
```

**P2 (Baseline + Hard-Negative Loss)**

```
Best epoch (min valid loss): [TODO] | valid_loss = [TODO] | valid_MAE = [TODO] | valid_RMSE = [TODO]
```

**P3 (Pygos-Net, DCB + Hard-Negative Loss)**

```
Best epoch (min valid loss): [TODO] | valid_loss = [TODO] | valid_MAE = [TODO] | valid_RMSE = [TODO]
```

Selected loss weights: `lambda_seg = [TODO]`, `lambda_bg = [TODO]`.

**Challenges and next steps**

1. **Background false positives on L0/L1.** Shadow, rock and guano regions remain the main source of error. Next step: quantify the per-tier background false-positive rate for all four configurations and verify that DCB and `Lbg` reduce it rather than merely shifting calibration.
2. **Over-suppression of true targets.** Aggressive background suppression can lower L0/L1 error while removing legitimate density in dense colonies. Next step: report L0/L1 improvements jointly with L3/L4 bias and missed-target rates, and treat any dense-tier degradation as a failure of the full model regardless of overall MAE.
3. **Annotation uncertainty.** Points from multiple photo-interpreters may disagree on overlapping or ambiguous penguins, so small metric differences are not interpreted as meaningful. Next step: report the observed gap between reported and recounted totals and avoid over-interpreting differences below that scale.
4. **Extremely small targets.** Penguins span only a few pixels, which limits detection-based baselines and makes any additional downsampling harmful. Next step: keep the detector as a reference baseline with small anchors and high-resolution features, and report its outcome as a feasibility result rather than a competitive number.
5. **Class imbalance across tiers.** Because 61.2% of tiles are L0/L1, overall MAE can improve while the dense tiers worsen. Next step: make the tier table, not the aggregate MAE, the primary evidence for H2 and H3.
6. **Training cost and multi-seed coverage.** Four configurations at 600 epochs plus seed repeats exceed the time available for a single sweep. Next step: complete the main 2x2 ablation first, add seeds only for the configurations that support the main claim, and report mean +/- standard deviation where available.
7. **DEM availability.** The optional ecological analysis requires compatible georeferencing and elevation coverage. Next step: confirm data availability after the core ablation; if unavailable, document the decision and drop the extension.

**Planned work for the remaining period**

- Nov 19 – Nov 25: train Baseline and +DCB at the full budget; produce the first real overall and tier tables.
- Nov 26 – Dec 02: train +Hard-Negative and Pygos-Net; complete the 2x2 ablation; run the tier and error-source analysis.
- Dec 03 – Dec 05: multi-seed repeats where feasible; five qualitative panels; DEM go/no-go decision.
- Dec 06 – Dec 10: write the conference-style final paper.
- Dec 11 – Dec 15: reproducibility check, repository cleanup and final presentation.

Success at final submission requires a reliable baseline, quantified individual and combined effects of DCB and `Lbg`, and improved sparse-background performance without substantial dense-colony degradation.

**Reproducibility.** Every reported number is traceable to a configuration file, random seed, repository commit and checkpoint. Split indices, data notes and aggregate result tables are versioned, while raw imagery, Labelbox exports, checkpoints and large outputs are excluded from Git. Preparation, training, evaluation and test commands are documented in the repository README.

## 4. Team Contributions

| Member | Responsibility | Contributions | Paper sections |
|---|---|---|---|
| [Member A] | Data and training infrastructure | Official split and data documentation; Labelbox-to-density pipeline with count preservation; Dataset/DataLoader and augmentations; shared trainer with seeds, configuration, logging, checkpointing and validation-based model selection; `scripts/prepare_data.py`, `scripts/train.py`, `tests/test_data.py` | Dataset, preprocessing, reproducibility |
| [Member B] | Model architecture | VGG-19 dual-branch baseline; segmentation-guided dampening and output masking; DCB with `use_dcb` switch; shape, gradient, parameter and memory checks; `src/pygosnet/models/`, `tests/test_models.py` | Architecture, DCB, model figure |
| [Member C] | Loss and ablation | `SL*`, `Lseg` and `Lbg`; combined loss and empty-image handling; four ablation configurations; validation-based weight selection and run records; `src/pygosnet/losses/`, `configs/ablation/`, `tests/test_losses.py` | Loss design, ablation, hyperparameters |
| [Member D] | Evaluation and analysis | MAE/RMSE, relative count error and L0–L4 evaluation; error-source decomposition; five-panel visualization; Faster R-CNN feasibility; qualitative analysis and results figures; `src/pygosnet/evaluation/`, `src/pygosnet/baselines/faster_rcnn.py`, `tests/test_metrics.py` | Metrics, qualitative analysis, results figures |

**Collaboration.** A and C align the training loop, configuration files, seeds and run records. B and C align model outputs with loss inputs and keep the DCB and loss switches orthogonal. A and D align the split, count targets, tier definitions and evaluation interface. C and D jointly interpret whether L0/L1 gains come at the cost of L3/L4 undercounting. All members share experiment runs, the introduction and discussion sections, and the final presentation. Each contribution is traceable to the modules, configuration files and tests listed above; pull requests and commits are recorded in the repository history.

## 5. References

Qian, et al. (2023). BAS aerial penguin dataset and density-estimation benchmark. [TODO: complete bibliographic details.]

Rong, L., and Li, C. (2021). Structural supervision for crowd counting. [TODO: venue and full title.]

Wang, Z., Bovik, A. C., Sheikh, H. R., and Simoncelli, E. P. (2004). Image quality assessment: from error visibility to structural similarity. IEEE Transactions on Image Processing.

Yu, F., and Koltun, V. (2016). Multi-scale context aggregation by dilated convolutions. ICLR.

Simonyan, K., and Zisserman, A. (2015). Very deep convolutional networks for large-scale image recognition. ICLR.

[TODO: any additional references cited in the final report.]

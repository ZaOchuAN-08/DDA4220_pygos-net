# reproducibility notes

## Data and split

The paper/proposal specifies 738 tiles and a 446/146/146 train/validation/test split. The distributed Dryad archive does not include a per-image split list in its root. `scripts/prepare_data.py` therefore creates a fixed-seed random reconstruction with seed 42 and writes the exact selected filenames to JSONL. This is a reproducible local split, not a recovered original-author split.

The Labelbox export audit must be retained. The supplied files contain 753 records, 740 unique `External ID` values, 13 duplicate image IDs, and 2 annotation IDs with no corresponding PNG. The overlapping annotations are not identical. Choosing MAISIE for all overlaps yields 134,660 point annotations across the 738 matched images; choosing LUKE yields 134,767. Both differ from the proposal/paper total 135,279. The preparation script requires an explicit `--duplicate-source` choice (`MAISIE` or `LUKE`) and does not silently merge the labels. This count discrepancy should be raised with the group/instructor before claiming exact reproduction of the published annotation total.

## Targets

- Each Labelbox bounding box is converted to a point at `(left + width/2, top + height/2)`.
- A normalized isotropic Gaussian with sigma 4 input pixels is rendered around each point.
- Each point contributes one unit of density mass, including points near image borders.
- The 448x448 density map is sum-pooled into a 56x56 target (stride 8); its sum remains the image count.
- Segmentation support is `density_target > 1e-3` at stride-8 resolution.
- Training randomly crops 256x256 images and rebuilds the density target from points inside the crop. Horizontal flipping also transforms point coordinates.
- Validation uses full 448x448 images.

## Training protocol

The default configuration follows the proposal: Adam, learning rate 1e-5, weight decay 1e-4, batch size 16, 600 epochs. Starting at epoch 101, checkpoint selection minimizes validation MAE + RMSE. Test data must not be used for checkpoint selection or hyperparameter tuning.

## Shared interface with B and C

Model: `model(images) -> {"density": raw_density, "segmentation": segmentation_probabilities}`.

Loss: `criterion(outputs, batch) -> {"total": scalar, ...}`. `batch` contains `image`, `density`, `seg_mask`, `count`, `tier`, `image_id`.

The trainer applies the proposal's inference mask only for counting: `D_out = D * 1[S >= 0.5]`. The loss receives the raw density output so C can compute background loss before output masking.

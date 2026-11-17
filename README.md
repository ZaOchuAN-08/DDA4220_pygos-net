# Pygos-Net (DDA4220)

## Integration branch: prepare the committed raw data

The `integration/abcd-interfaces` branch includes the raw ZIPs, existing JSONL manifests, and ABCD interface repairs. After installing with `python -m pip install -e '.[dev]'`, run:

```bash
python scripts/prepare_images.py --report data/processed/image_preparation_report.json
python scripts/prepare_images.py --verify-only
python -m pytest -q
```

The preparation script restores all 738 PNGs at the paths referenced by the manifests, skips macOS metadata, and validates image dimensions, labels, and split consistency. It preserves the existing annotation choices and splits. No new dataset download or re-annotation is needed.

See [raw data setup and training steps](docs/raw_data_setup.md) and [integration runbook](docs/integration_runbook.md). The original preparation instructions below are for rebuilding data from the external combined archive; they are not needed for this branch's already committed manifests.


Multi-Scale Density Estimation for Antarctic Penguin Counting.

## Dataset
BAS aerial penguin dataset (Qian et al., 2023, Dryad). Data not distributed in this repo; see `data/README.md`.

## Quick Start
```bash
# 1. environment
conda env create -f environment.yml   # TODO

# 2. prepare data
bash scripts/prepare_data.sh          # TODO

# 3. train baseline
bash scripts/train_baseline.sh        # TODO

# 4. run 2x2 ablation
bash scripts/run_ablation.sh          # TODO

# 5. evaluate
bash scripts/eval_all.sh              # TODO

# Pygos-Net — Work A: Data & Training Infrastructure

This branch implements the A workstream only: data preparation/audit, point-to-density targets, point-aware augmentation, Dataset/DataLoader, generic training loop, checkpointing, configs, and tests. The VGG-19 model and project losses remain owned by B and C.

## 1. Environment

Python 3.10+ is recommended. From the repository root:

```bash
python -m venv .venv
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Install a PyTorch build compatible with your CUDA version if using a GPU. CPU can be used for tests and small debugging runs.

## 2. Prepare the data

First audit/extract the Dryad archive:

```bash
python scripts/prepare_data.py --archive /absolute/path/to/doi_10_5061_dryad_8931zcrv8__v20230306.zip
```

The script writes `data/processed/dataset_audit.json`. It detects 13 duplicated image annotations and stops before creating the training manifest. The group must explicitly agree whether to use the MAISIE or LUKE record for those overlaps. Then rerun, for example:

```bash
python scripts/prepare_data.py --archive /absolute/path/to/doi_10_5061_dryad_8931zcrv8__v20230306.zip --duplicate-source MAISIE
```

Use `--duplicate-source LUKE` instead only if that is the team's agreed annotation policy. The script creates the 738-row manifest and fixed 446/146/146 split with seed 42. This reproduces the published split sizes, not the authors' unavailable per-image assignment. See `data/README.md`.

## 3. Verify Work A

```bash
pytest -q
python scripts/smoke_test.py
```

The smoke test uses a tiny dummy model and dummy loss; it validates A's trainer/checkpoint mechanics without requiring B/C code.

## 4. Train after B and C merge their modules

The shared config expects these class paths:

- `pygosnet.models.pygosnet.PygosNet`
- `pygosnet.losses.combined.CombinedLoss`

The model must return raw density and segmentation probabilities. The loss must return a dictionary containing a scalar `total` tensor. See `docs/reproducibility.md` for the full interface.

```bash
python scripts/train.py --config configs/ablation/baseline.yaml
python scripts/train.py --config configs/ablation/plus_dcb.yaml
python scripts/train.py --config configs/ablation/plus_hardneg.yaml
python scripts/train.py --config configs/ablation/pygos_net.yaml
```

For a short debug run, append `--epochs 2`. Resume from a checkpoint with `--resume outputs/baseline/last.pt`.

## 5. Outputs

Each run writes `history.csv`, `last.pt`, and (from epoch 101 onward) `best.pt` under its configured output directory. Checkpoints include model, optimizer, epoch, best validation MAE+RMSE, and config.

## Ownership boundary

A does not implement the VGG-19 encoder/heads/DCB (B), the structural/segmentation/hard-negative losses (C), or the full reporting/visualization evaluation suite (D). A provides the data and training interfaces those modules plug into.

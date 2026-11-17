# ABCD integration check and runbook

Checked 17 November 2026 against main commit `9c3501b68e515be6ac2f981d40578ea14f2f44eb`.
Integration branch: `integration/abcd-interfaces`. This branch contains the latest main plus the interface repairs; main is not modified by publishing this branch.

## Findings and local fixes

1. A's YAML passed `use_hard_negative`, but C's constructor rejected it. Added a compatible background-term switch; disabled configurations now return differentiable zero for the background term. Existing background MSE is unchanged. **The switch name does not mean Top-K hard-negative is implemented.**
2. A's Dataset did not supply C's required `structural_mask`. It now returns a clone of `seg_mask`; both use the existing target definition `density > 1e-3`. These are GT labels, never model predictions.
3. Added CPU/CUDA/MPS selection and explicit `--device`, batch-size, worker, output-dir and offline-initialization overrides to the training CLI. Non-CUDA runs disable pinned memory. Short-run epoch overrides are saved in checkpoint configuration.
4. Added missing pandas/matplotlib dependencies for D's evaluator.
5. Added `scripts/predict.py` to load a trusted project checkpoint and export unmasked density plus segmentation probability as NPZ. It does not re-download pretrained initialization because the checkpoint supplies all weights.
6. Added `evaluate.py --manifest` to use A's adjudicated point manifest for GT, instead of independently reconstructing labels from potentially conflicting raw exports. Dataset and evaluator share `parent-directory/image-filename` keys. Probability masking is applied once using `--use-seg`.
7. Fixed a horizontal-flip edge case: a point at x=0 previously became x=W, was dropped during density generation, but was still included in count. Continuous reflection now keeps the result inside [0,W).

B's existing output interface already matched A/C and did not need changing:

- `density`: nonnegative, before final output mask.
- `segmentation`: sigmoid probability.
- `segmentation_logits`: raw logits for C's stable BCE.
- All maps are `[B,1,H/8,W/8]`; H/W must be divisible by 16.

## Validation completed

- Full pytest suite: **60 passed**, including all four DCB/background-toggle combinations through the actual Dataset/model/loss/Trainer, optimizer step, validation and checkpoint resume.
- CLI chain with generated images/points: train (256x256), validate (448x448), checkpoint, prediction export, evaluator CSV/JSON and plots.
- Baseline CLI train/validation on Apple M5 MPS: passed, batch 1.
- These are synthetic engineering checks, not BAS accuracy results, pretrained validation or convergence evidence.

## Still required before formal research experiments

- Actual image files and their local paths. The latest main already includes the adjudicated full/train/val/test JSONL manifests (738 images total). These labels were inspected; no real images were used in the engineering runs.
- Verify actual official pretrained weights load. Defaults remain pretrained=True; `--no-pretrained` is only for offline engineering checks.
- C's background term currently lacks Top-K selection, rho/eta and annotation-tolerance exclusion. It must be completed or described as simplified background MSE. Do not label the current four-way run as the complete proposal hard-negative ablation.
- C's segmentation weighting/reduction and SSIM settings still need source/proposal fidelity review before claiming strict replication. Defaults currently use equal foreground/background BCE weights, with normalization by total pixel weights.
- A's metadata explicitly says the split is a reconstructed fixed-seed split, not the original author image assignment. The selected LUKE overlap resolution gives 134,767 objects in both the audit and the uploaded manifest, rather than the proposal's 135,279. Document the actual dataset variant or reconcile it before interpreting paper comparisons. This check did not change these decisions.
- Input normalization is ImageNet normalization in A's code, unlike the archived author's dataset-specific constants. Agree on the protocol and keep it fixed across configurations.

## Run on this Mac

The isolated checkout is:
`/Users/snowmor/Documents/Codex/2026-11-16/cha/work/pygosnet-integration`

The existing Python environment is its sibling `../model-env`. It has an older editable checkout registered, so the commands here explicitly set `PYTHONPATH=src` to select this repaired source tree.

```bash
cd /Users/snowmor/Documents/Codex/2026-11-16/cha/work/pygosnet-integration
PYTHONPATH=src ../model-env/bin/python -m pytest -q
```

On another machine use Python >=3.10, create an environment and run `python -m pip install -e '.[dev]'` in the extracted project. Then ordinary `python scripts/...` commands use the repaired source. The verified environment is Python 3.12.14, torch 2.15.0 and torchvision 0.29.2.

### 1. Prepare actual data

A's preparation script expects a combined archive containing the image ZIPs and annotation JSONs. Use its actual path. LUKE is the overlap selection already recorded in main's metadata; this runbook preserves that choice.

```bash
PYTHONPATH=src ../model-env/bin/python scripts/prepare_data.py --archive /actual/path/to/combined_dataset.zip --duplicate-source LUKE
```

This creates `data/splits/train.jsonl`, `val.jsonl` and `test.jsonl`. Existing TXT split lists alone cannot start the Dataset. If A already prepared manifests, copy/reference the matching images and manifests rather than selecting a different split.

### 2. Short engineering run

```bash
PYTHONPATH=src ../model-env/bin/python scripts/train.py --config configs/ablation/baseline.yaml --device auto --epochs 2 --batch-size 1 --num-workers 0 --no-pretrained --output-dir outputs/baseline_debug
```

Use `--device mps` on a Mac terminal with MPS access or `--device cuda` on a configured NVIDIA machine. If the requested backend is unavailable, the script fails clearly. `auto` falls back to CPU when necessary. This short run is not a final experiment. The best-checkpoint rule starts at epoch 101, so use `last.pt` for debug inference; absence of `best.pt` after two epochs is expected.

### 3. Export and evaluate validation predictions

```bash
PYTHONPATH=src ../model-env/bin/python scripts/predict.py --checkpoint outputs/baseline_debug/last.pt --manifest data/splits/val.jsonl --project-root . --out-dir outputs/baseline_debug/predictions --device auto
PYTHONPATH=src ../model-env/bin/python scripts/evaluate.py --manifest data/splits/val.jsonl --pred-dir outputs/baseline_debug/predictions --use-seg --out-dir outputs/baseline_debug/evaluation
```

Results include per_image.csv, tier_metrics.csv, metrics.json and diagnostic plots. Do not use test data for debugging, tuning or checkpoint selection.

### 4. Formal training after integration and small-data overfit checks

```bash
PYTHONPATH=src ../model-env/bin/python scripts/train.py --config configs/ablation/baseline.yaml --device auto
PYTHONPATH=src ../model-env/bin/python scripts/train.py --config configs/ablation/plus_dcb.yaml --device auto
```

These default to pretrained=True, 600 epochs, batch 16. Measure actual memory/time first; if batch size is reduced, use and record the same protocol across compared runs. Share non-DCB initialization explicitly for controlled comparisons. The two background-enabled configurations should wait for C's proposal-level implementation/fidelity review.

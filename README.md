# Pygos-Net (DDA4220)

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

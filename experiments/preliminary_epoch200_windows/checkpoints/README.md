# Best validation checkpoints

The Windows CUDA training run selected checkpoints by minimizing validation MAE + RMSE from epoch 101. Each file contains the model, its configuration, and the saved Adam optimizer state.

| Model | Checkpoint | Validation MAE | Validation RMSE |
|---|---|---:|---:|
| Baseline | [Epoch 189](baseline-best-epoch189.pt) | 20.2421 | 39.6895 |
| +DCB | [Epoch 158](plus_dcb-best-epoch158.pt) | 19.9573 | 40.6293 |

The experiment summary reports the fixed epoch-200 comparison. These checkpoints reflect the training loop's validation selection. The +DCB checkpoint was saved before the epoch-161 interruption.

Use Git LFS to download the files after checking out `integration/abcd-interfaces`.

```sh
git lfs pull --include="experiments/preliminary_epoch200_windows/checkpoints/*.pt"
```

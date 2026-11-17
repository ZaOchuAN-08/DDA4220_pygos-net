# integration 分支：raw data 整理与运行

本说明适用于 `integration/abcd-interfaces`。该分支已合入 main 上传的四个原始图片 ZIP 和 JSON/JSONL 标注，并包含 ABCD 接口修复。不会修改 main。

## 第一步：环境

新电脑可以直接克隆这个分支（已有工作目录可正常 fetch/pull）：

```bash
git clone --branch integration/abcd-interfaces https://github.com/ZaOchuAN-08/DDA4220_pygos-net.git pygos-net-integration
cd pygos-net-integration
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

使用 Python 3.10 或更新版本。Windows 虚拟环境的激活命令为 `.venv\Scripts\activate`；NVIDIA 游戏本需使用适合其驱动的 CUDA 版 PyTorch。

## 第二步：从已有 ZIP 自动整理图片

```bash
python scripts/prepare_images.py --report data/processed/image_preparation_report.json
```

脚本会：

1. 读取现有 `data/processed/manifest.jsonl` 和 train/val/test JSONL，检查数量、计数、坐标、划分交叉及覆盖范围。
2. 从 `data/raw/Jack.zip`、`Luke.zip`、`Maisie.zip`、`Thomas.zip` 提取 manifest 实际引用的 PNG。
3. 忽略 `__MACOSX` 和 `._*` 元数据；不直接使用旧的双重嵌套解压目录。
4. 按 manifest 指定的位置生成 `data/raw/images/<site>/<file>.png`。
5. 检查全部 PNG 的可读取性和 448×448 尺寸。

**不会重新生成 manifest、重新划分数据或改选标注来源。** 保留仓库中的 LUKE 重叠标注选择。

可重复运行：已有且检查通过的图片会跳过。文件缺失、损坏、尺寸错误、ZIP 内文件名歧义或数据划分冲突会明确报错。

仅检查，不解压：

```bash
python scripts/prepare_images.py --verify-only
```

期望结果：738 张图片，train/val/test = 446/146/146，总标注数 134,767。按路径目录分组为 Jack 176、Luke 178、Maisie 250、Thomas 134；目录代表图片来源，不代表重新选择的标注来源。

原始 ZIP 已在仓库中，因此整理后的 `data/raw/images/` 和本机报告不会再次提交 Git，避免重复上传相同图片。换电脑后重新运行这一条命令即可恢复所需目录。

## 第三步：检查接口与训练

```bash
python -m pytest -q
python scripts/train.py --config configs/ablation/baseline.yaml --device auto --epochs 2 --batch-size 1 --num-workers 0 --no-pretrained --output-dir outputs/baseline_debug
```

这是随机初始化的短轮数工程检查，不是正式准确率实验。`auto` 依次选择 CUDA、MPS、CPU，也可显式指定 `--device cuda` 或 `--device mps`。两轮后使用 `last.pt`；正式协议从第 101 轮才开始保存最佳 checkpoint。

```bash
python scripts/predict.py --checkpoint outputs/baseline_debug/last.pt --manifest data/splits/val.jsonl --project-root . --out-dir outputs/baseline_debug/predictions --device auto
python scripts/evaluate.py --manifest data/splits/val.jsonl --pred-dir outputs/baseline_debug/predictions --use-seg --out-dir outputs/baseline_debug/evaluation
```

正式 Baseline/+DCB 实验应完成预训练加载与小样本拟合验证，去掉 `--no-pretrained`，采用统一训练协议。C 的背景项仍是简化平方惩罚，不是完整 Top-K hard-negative；完整四组创新消融仍待该项补齐。

## 已实际验证

- 四个实际 ZIP 成功生成全部 738 张图片，逐张验证尺寸和 PNG 完整性。
- 整理前后，现有 manifest 和 split 内容未改动；验证模式再次通过。
- 一个真实训练 batch 和一个真实验证 batch 的 density 总和与有效标注数量一致；标签字段可供 C 使用。
- 真实训练 batch 已完成 B 模型 forward、C loss、backward 和一次 Adam 更新（CPU、随机初始化）；这不构成准确率或收敛结论。
- 另有合成夹具覆盖 ZIP 元数据排除、重复运行、缺失图片、损坏/错误尺寸、重复文件名、路径与划分错误。

## 当前 Mac 已有环境

```bash
cd /Users/snowmor/Documents/Codex/2026-11-16/cha/work/pygosnet-integration
PYTHONPATH=src ../model-env/bin/python scripts/prepare_images.py --verify-only
```

图片已经在这台 Mac 的上述目录内整理完毕，不必再解压。该 Mac 的现有 Python 环境此前注册过旧代码，因此继续使用 `PYTHONPATH=src ../model-env/bin/python ...`，确保选择当前整合版。

完整测试套件结果：**69 项通过**（其中新增 raw-data 整理测试 9 项）。

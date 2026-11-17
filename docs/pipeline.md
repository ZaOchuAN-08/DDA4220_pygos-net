# 自动设备识别与测试 Pipeline

入口：`scripts/run_pipeline.py`。支持 Windows/Linux 的 NVIDIA CUDA、Mac 的 Apple MPS，以及 CPU。自动选择顺序为 **CUDA → MPS → CPU**，并实际分配一个小张量、执行计算，确认设备可用。手动指定不可用设备时明确报错；不会静默改用另一设备。

## 安装

在 `integration/abcd-interfaces` 项目目录、对应 Python 环境中：

```bash
python -m pip install -e '.[dev]'
```

CUDA 机器需要安装与驱动匹配的 CUDA 版 PyTorch。GPU 型号/显存、PyTorch CUDA build 和实际选择结果会写入环境记录。不会自动改驱动、换 PyTorch 或修改正在运行的训练。

## 1. 默认仅检查，不训练

```bash
python scripts/run_pipeline.py
```

等同于 `--mode check --device auto`，检查设备、依赖、全部图片、JSONL 标注和划分。缺少规范目录图片时，从仓库现有 ZIP 自动整理；不重写 manifest 或重新划分数据。该模式不会下载预训练权重，不启动模型训练或模型评估。

如果当前 CUDA 正忙，可显式选择 CPU 做环境/数据检查：

```bash
python scripts/run_pipeline.py --mode check --device cpu
```

## 2. 已有模型：自动预测 + 评估（主要测试入口）

```bash
python scripts/run_pipeline.py --mode evaluate --checkpoint outputs/your_run/last.pt
```

该命令不重新训练。自动识别 checkpoint 实际保存的轮数（例如 20、50 或 600），读取其模型配置，导出完整验证集预测，应用 `S>=0.5` 最终掩码，然后计算指标、分层/逐图误差和图表。不会重新下载预训练初始化，因为 checkpoint 已含全部模型权重。

手动选择设备或输出目录：

```bash
python scripts/run_pipeline.py --mode evaluate --checkpoint outputs/your_run/last.pt --device cuda --output-dir outputs/pipeline_cuda_evaluation
```

默认评估验证集，训练和调参过程中继续使用验证集。模型及超参数确定后，才能显式选择测试集：

```bash
python scripts/run_pipeline.py --mode evaluate --checkpoint outputs/your_run/last.pt --split test
```

默认从 checkpoint 配置读取对应 manifest；未配置 test 时使用 `data/splits/test.jsonl`。自定义标注清单可以传 `--manifest <JSONL>`，并用 `--split` 正确标记其用途。只加载可信的本项目 checkpoint。

## 3. 显式训练，然后自动预测与评估

仅当你希望启动新训练时使用：

```bash
python scripts/run_pipeline.py --mode train-eval --config configs/ablation/baseline.yaml --batch-size 1 --epochs 50
python scripts/run_pipeline.py --mode train-eval --config configs/ablation/plus_dcb.yaml --batch-size 1 --epochs 100
```

数字仅为用法示例，两条命令无需连续执行。对照实验应使用一致的预算和其他训练设置。

- **轮数不固定**：指定 `--epochs N` 时使用 N，否则沿用 YAML 的轮数；当前仓库原配置是 600。
- batch size 指定时覆盖 YAML，不指定则沿用 YAML；当前原配置是 16。RTX 4050 6 GB 可先显式使用 `--batch-size 1`；此 pipeline 不自动寻找最大 batch。
- 使用 `--device auto|cuda|mps|cpu` 选择设备，默认 auto。
- 默认 workers 为 0，便于跨系统运行；训练可以用 `--num-workers N` 设置。
- `--no-pretrained` 只适合明确需要随机初始化的工程测试，正式预训练实验不要加。
- 自动评估的是训练最终轮的 `last.pt`，不固定取第 20 轮，也不替换成中途最优轮。若希望评估 `best.pt`，使用 evaluate 模式并明确给出该文件。
- 不自动续训已有目录，避免把已有模型与新配置混用。恢复训练可使用原 `train.py --resume`；恢复后的 checkpoint 再交给 evaluate 模式。
- 保留已有研究方法：未实现 Top-K hard-negative，不会因为调用 pipeline 就补齐该算法。共享 head 初始化匹配和精确随机状态恢复仍属于后续可复现性改进。

## 输出与失败处理

每次默认创建一个带时间戳的新目录：

```text
outputs/pipeline/<时间戳>_<模式>/
├── pipeline_status.json     # running / complete / failed / interrupted
├── environment.json         # 实际设备、GPU 显存、Python/PyTorch 等
├── data_report.json         # 图片、标注和划分检查
├── prepare_images.txt       # 每阶段控制台日志
├── effective_config.yaml    # 仅 train-eval；不修改原 YAML
├── training/                # 仅 train-eval；history.csv 和 checkpoint
├── checkpoint_config.json   # 模型实际训练配置
├── predictions/             # 密度和分割概率 NPZ
├── evaluation/              # CSV / JSON / 图表
└── summary.json             # 实际 checkpoint 轮数、哈希与最终指标
```

check 模式只生成检查相关文件。用户显式给出的输出目录必须不存在；已有目录不会被覆盖。每阶段失败会立即停止，保留日志和错误状态。Ctrl+C 会终止当前子进程并记录中断。`--no-figures` 可跳过绘图；仍保留指标表。不会推送 GitHub，不会修改 main，也不会启动额外的背景训练。

在实际执行之前，用 `python scripts/run_pipeline.py --help` 检查所有参数。

## 验证范围

已在开发机器完成：

- CUDA/MPS/CPU 自动优先级、不可用设备、只检查不训练、任意轮数识别、配置预算覆盖、已有目录保护、失败中止与日志测试。
- 本机自动选择 MPS 的环境/数据检查。
- CPU 下一个真实训练 batch → checkpoint → 预测 → 指标的完整联调，以及已有 checkpoint 的独立评估。

RTX 4050 的真实 CUDA 执行需要在游戏本上验证，本次未连接或改动游戏本正在运行的训练。

完整仓库测试结果：85 项通过。开发联调结果不代表新的模型准确率实验。

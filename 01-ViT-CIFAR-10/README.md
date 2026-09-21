# 01 · Tiny ViT / CIFAR-10

状态：**七阶段实验运行与检查已全部完成**。主轨迹训练 50 轮，按最低验证 loss 选定第 19 轮 best；该模型 validation accuracy=65.58%，首次官方 test accuracy=**64.62%（6,462 / 10,000）**、test loss=1.023884。当前阅读入口为 [07-final.md](notes/07-final.md) 和[完整实验总结](notes/experiment-summary.md)，最后三个学习问题已作答，行列方向的纠正与补充解释已记录。此前的 [CPU/GPU 计时](notes/02-device-benchmark.md)与[结构图核对](notes/02-diagram-review.md)保留。

## 1. 问题 / Question

一张图如何变成分类 logits？loss 如何产生梯度？谁更新参数？模型记住训练数据与学会泛化有什么区别？训练中断后需要恢复哪些状态？

本实验以理解这些过程为目标，不设置 CIFAR-10 准确率排名目标。

## 2. 假设 / Hypothesis

- `backward()` 产生参数梯度，`optimizer.step()` 才更新本模型的参数。
- 关闭增强、dropout 和权重衰减后，小模型应能记住固定的 32 张训练图；失败时检查实现及优化设置。
- 训练与验证表现可能分离，应通过实际曲线判断，而不能只看训练准确率。
- 恢复训练需要模型与优化器状态；更严格的连续性检查还涉及随机状态与数据顺序。

前两项已验证：阶段 3 观察到 backward 产生梯度、step 更新参数；阶段 4 达到固定 32 图的记忆目标。阶段 5 观察到训练改善与验证波动并存，并验证了相同设备、软件、代码和数据条件下的 epoch 边界恢复一致性。当前只测试一个随机种子。

## 3. 实现 / Implementation

参考资料与设计入口：

- [用户提供的原始大纲](notes/outline-original.md)：原文归档，保留教学思路和上下文。
- [大纲审阅与已采纳调整](notes/outline-review.md)：R1 梯度演示修正、R2 SSD 资产路径、R3 恢复训练记录。
- [执行方案](PLAN.md)：固定配置、文件职责、五阶段预算与验收方法。

沿用原稿的五阶段学习路线，并按后续建议增加训练预算观察与最终评估，每阶段留下可阅读的观察结果和解释：

| 阶段 | 内容 | 要观察的证据 |
| --- | --- | --- |
| 1 | Dataset / DataLoader | 单样本与 batch 的区别、图像和标签 shape、样本网格 |
| 2 | Tiny ViT forward | patch、token、CLS 和 logits 的 shape |
| 3 | 单批训练一步 | backward 前后梯度，以及 step 前后参数变化 |
| 4 | 记住 32 张图 | 固定样本上的 loss 与 accuracy 曲线 |
| 5 | 完整 train / validation / checkpoint | 训练与验证曲线、模型选择、停止后恢复的检查 |
| 6 | 训练预算 20 → 50 | 已完成；train_eval 改善而 val loss 上升，best 仍为第 19 轮 |
| 7 | 最终评估与总结 | 已完成；冻结第 19 轮 best，test accuracy=64.62%，混淆矩阵与预测示例 |

阶段 1 已实现 [data.py](src/data.py) 和 [inspect_data.py](src/inspect_data.py)。阶段 2 已实现 [model.py](src/model.py) 和 [inspect_forward.py](src/inspect_forward.py)，模型使用基础 PyTorch 模块显式组合，并提供 [7 项模型检查](tests/test_model.py)。阶段 3 已实现 [inspect_step.py](src/inspect_step.py)，以显式五步训练主干完成一次更新，并单独对照梯度清零；阶段 4 已实现 [train.py](src/train.py) 的固定 batch 训练；阶段 5 使用 [train_full.py](src/train_full.py)、[evaluate.py](src/evaluate.py)、[checkpoint.py](src/checkpoint.py) 完成完整训练、验证和恢复，由 [compare_resume.py](src/compare_resume.py) 比较轨迹。

阶段 6 使用 [train_budget.py](src/train_budget.py) 延长预算并记录不影响训练状态的 train_eval；阶段 7 使用 [final_evaluate.py](src/final_evaluate.py) 固定模型并评估官方 test，由 [final_metrics.py](src/final_metrics.py) 从同一遍预测生成指标与图表。

## 4. 实验设置 / Experiment Setup

原稿提出的核心设置：

| 项目 | 设置 |
| --- | --- |
| 数据 | CIFAR-10，RGB，32×32，10 类 |
| 划分 | 官方 50,000 训练图进一步划为 45,000 train / 5,000 validation；官方 10,000 test 留到最终评估 |
| 模型 | 随机初始化 Tiny ViT；patch=4，embed_dim=128，depth=4，heads=4，mlp_ratio=4 |
| 损失 | CrossEntropyLoss，输入未经 softmax 的 logits |
| 优化器 | AdamW，学习率 3e-4；完整训练候选 weight_decay=1e-2，32 图记忆实验设为 0 |
| 第一版 | 显式 PyTorch 训练循环，单设备，float32 |
| 环境 | Python 3.12.3；torch 2.14.0+cu130；torchvision 0.29.0+cu130；matplotlib 3.11.2；tqdm 4.70.1 |
| 完整依赖 | [requirements-lock.txt](requirements-lock.txt)，安装后 `pip check` 通过 |
| 阶段 1 配置 | [01-data.json](configs/01-data.json)：seed=42，batch=128，worker=0，CPU 线程=4，ToTensor，无增强 |
| 阶段 2 配置 | [02-forward.json](configs/02-forward.json)：seed=42，B=2/128，CPU 线程=4；像素从 [0,1] 映射到 [-1,1] |
| 阶段 3 配置 | [03-step.json](configs/03-step.json)：seed=42，batch=128，FP32，确定性算法；AdamW lr=3e-4、weight_decay=0；主模型仅一次 step |
| 阶段 4 配置 | [04-overfit32.json](configs/04-overfit32.json)：固定 train 索引前 32 张，batch=32，最多 1000 步，每 10 步评估，连续三个评估点满足 accuracy=100%、loss≤0.05 时停止 |
| 阶段 5 配置 | [05-train.json](configs/05-train.json)：45k/5k，seed=42，batch=128，FP32，AdamW lr=3e-4、weight_decay=0.01，20 epoch；worker=0、pin_memory=True、无增强 |
| 阶段 6 配置 | [06-budget.json](configs/06-budget.json)：从 epoch 20 last 延长至总计 50 轮，训练配方不变；新增轮末 train_eval；设备变化显式记录并校验恢复 |
| 阶段 7 配置 | [07-final.json](configs/07-final.json)：固定第 19 轮 best 及 checkpoint 哈希；官方 test=10,000，batch=128、FP32，无增强、无更新，一遍推理 |
| 恢复检查配置 | [05-resume-smoke.json](configs/05-resume-smoke.json)：257/129 小子集先检查尾批与恢复，再在完整划分对照连续 6 轮和第五轮状态恢复后的第六轮 |
| 模型实际参数量 | 809,354；patch 投影、QKV/Linear、CLS 和位置编码采用截断正态初始化；bias=0，LayerNorm weight=1；详见配置、代码与结果摘要 |
| 数据划分 | 按类别分层，每类 train=4,500、validation=500；固定索引保存在 SSD，运行时记录 SHA-256 |
| 其余阶段配置 | 见 [PLAN.md](PLAN.md)，其中预算为计划值 |
| 代码版本 | 阶段 1–2 的结果在首次提交前生成，原运行保留代码快照与哈希；发布后新增运行记录 Git commit 和工作区变更 |

资产位置遵循[仓库约定](../README.md)：代码与小型记录进入实验目录，环境、数据及完整运行产物使用仓库外的 `LAB_ROOT`。原稿的相对路径示例已在执行入口按 R2 适配。

用户已为本实验指定 GPU，后续模型阶段持续沿用。实际设备保存在本机 `.local/device.json`，每次启动按 UUID 选择并复查占用，不重新询问或自动换卡。首次阶段 1–2 使用 CPU；现已完成 GPU forward 验证和六种 batch 的计时。环境首次 GPU 运行遇到 cuDNN 子库混用，已通过配套依赖升级解决，见[环境修复记录](notes/02-gpu-environment.md)。

CIFAR-10 的官方数据规模与图像尺寸已核对。[官方数据集说明](https://www.cs.toronto.edu/~kriz/cifar.html)

### 复现阶段 1

以下命令从**仓库根目录**运行。在本机加载已有环境脚本；其他机器需先按仓库约定配置外部资产目录与缓存变量。

```bash
source ./硬件环境配置信息/experiment-env.sh
cd 01-ViT-CIFAR-10

# 本机已完成安装；新环境第一次准备时执行。
bash scripts/setup_env.sh

# 数据已准备好时使用本地文件，每次创建新的运行目录。
bash scripts/inspect_data.sh

# 仅首次数据准备时需要 --download；由 torchvision 校验并解压官方归档。
# bash scripts/inspect_data.sh --download
```

也可逐步查看核心操作：

```python
image, label = dataset[i]                 # 一张图 [3,32,32] + 一个整数标签
images, labels = next(iter(train_loader)) # 一批图 [128,3,32,32] + 标签 [128]
```

上面两个 Python 片段用于解释已有 Dataset/DataLoader 对象，完整可运行入口为 shell 脚本。`inspect_data.sh` 会使用本实验独立解释器并隐藏 GPU，不依赖当前 shell 激活了哪个 Python 环境。

PyTorch 与 torchvision 采用配套的已发布版本；当前环境已经通过 GPU forward 检查。早期 CPU 观察所用旧版本仍记录在原结果中。[官方安装配对](https://pytorch.org/get-started/previous-versions/)

### 复现阶段 2

从仓库根目录开始，沿用已安装环境与本地数据：

```bash
source ./硬件环境配置信息/experiment-env.sh
cd 01-ViT-CIFAR-10
bash scripts/inspect_forward.sh
```

入口先运行 CPU 模型检查，然后在已指定 GPU 上输出实际 shape、patch 图与初始 logits。它只做 forward；使用 `bash scripts/inspect_forward.sh --cpu` 可在当前环境复现 CPU 观察。

运行不同 batch 的 CPU/GPU 对比：

```bash
bash scripts/benchmark_devices.sh
```

基准固定模型、输入、FP32 精度与 CPU 4 线程，分别测 CPU、GPU、GPU 含传输。每次输出独立运行记录，条件与解释见 [计时笔记](notes/02-device-benchmark.md)。

### 复现阶段 3

```bash
# 从仓库根目录开始
source ./硬件环境配置信息/experiment-env.sh
cd 01-ViT-CIFAR-10
bash scripts/inspect_step.sh
```

入口沿用已指定 GPU。每次从固定种子的初始模型开始，只对一个训练 batch 更新一次；另用初始权重副本对照清零与累积，不在副本上更新参数。完整观测张量保存在 SSD，解释与学习问题见 [阶段 3 笔记](notes/03-step.md)。

### 复现阶段 4

```bash
# 从仓库根目录开始
source ./硬件环境配置信息/experiment-env.sh
cd 01-ViT-CIFAR-10
bash scripts/overfit32.sh
```

入口沿用已指定 GPU，从 seed=42 重新初始化模型，反复训练预先固定的 32 张图。每次创建独立运行目录，保存每步 loss、评估指标、图表和最终模型权重。该权重文件仅用于加载模型，阶段 5 使用单独实现的完整 checkpoint 恢复训练。

### 复现阶段 5

```bash
# 从仓库根目录开始；先完成一轮资源与正确性检查。
source ./硬件环境配置信息/experiment-env.sh
cd 01-ViT-CIFAR-10
bash scripts/train_full.sh --until 1
```

每次运行新建 SSD 目录，并沿用已指定 GPU。后续连续六轮参考、从第五轮恢复、逐项比较以及继续到第 20 轮的完整命令见 [阶段 5 笔记](notes/05-training.md)。`--until 20` 表示总计完成 20 轮；恢复不覆盖源运行。关键统计与随机状态检查见 [test_training.py](tests/test_training.py)，精选结果导出见 [collect_training_results.py](scripts/collect_training_results.py)。

### 复现阶段 6

使用 [train_budget.sh](scripts/train_budget.sh) 从阶段 5 的第 20 轮 last 迁移，先 `--until 21` 检查，再由第 21 轮 last 继续 `--until 50`。完整命令、检查和解释见 [阶段 6 笔记](notes/06-budget.md)，范围见 [执行设计](notes/06-budget-plan.md)。原训练主干及阶段 5 恢复入口保留。

### 复现阶段 7

```bash
# 从仓库根目录开始，沿用已准备好的环境、数据和冻结的 best。
source ./硬件环境配置信息/experiment-env.sh
cd 01-ViT-CIFAR-10
bash scripts/final_evaluate.sh
```

入口先执行两项合成数据检查，再核对预先固定的模型选择、重测 validation，最后完整评估官方 test 一遍；完整记录位于外部资产目录。配置包含源运行 ID 和 best 哈希，不根据 test 结果选择模型。详情与结果导出命令见 [阶段 7 笔记](notes/07-final.md)。

## 5. 结果 / Results

### 阶段 1：数据观察

运行：`01-data-538ab412b224`。这是数据观察结果，没有训练指标或模型检查点。

| 观察项 | 实测结果 |
| --- | --- |
| 训练 / 验证样本 | 45,000 / 5,000，每类分别 4,500 / 500 |
| 单张图片 | `[3,32,32]`，float32 |
| 一个 batch | 图像 `[128,3,32,32]`，标签 `[128]`，标签为 int64 |
| batch 像素范围 | [0,1] |
| 一次完整训练集遍历 | 352 批，最后一批 72 张；本次仅观察首批，没有遍历训练 epoch |
| 正确性检查 | 划分互斥且完整、类别平衡、图片标签对应、shape/类型/范围、同种子首批一致：全部通过 |
| GPU / test | 未初始化 CUDA，未评估官方测试集 |

查看[完整观察摘要](results/01-data/summary.json)、[样本网格](results/01-data/batch-preview.png)与[阶段 1 学习笔记](notes/01-data.md)。图上的名称是真实标签，当前尚无模型预测。

![首个训练 batch 中的 16 张图片及真实标签](results/01-data/batch-preview.png)

### 阶段 2：Tiny ViT forward

运行：`02-forward-e4cef5b6ad62`。实测数据流：

```text
[2,3,32,32] → [2,128,8,8] → [2,64,128]
 → 拼接 CLS [2,65,128]
 → 加位置编码、四个 Transformer block [2,65,128]
 → 最终 LayerNorm、取 CLS [2,128]
 → Linear [2,10] logits
```

- 参数数：809,354；B=2 与 B=128 均通过检查。
- 单张图单独计算与在 batch 中计算的结果一致到检查容差；两个 batch 中相同前两张图片的输出在本次运行中最大绝对差为 0。
- forward 前后模型状态一致，全部参数 grad 为 None；没有训练准确率或 loss 结果。
- 7 项检查覆盖 patch 线性投影等价性、batch 独立性、图像影响 CLS 输出、block 参数独立性、forward 不修改状态、固定种子初始化和错误输入。

查看[阶段 2 学习笔记](notes/02-forward.md)、[完整摘要](results/02-forward/summary.json)、[patch 示意图](results/02-forward/patch-to-token.png)和[真实 logits](results/02-forward/first-image-logits.csv)。

### 阶段 2 补充：GPU 与 batch 规模

GPU forward：`02-forward-a82a1eb6235d`，见[结果](results/02-forward-gpu/README.md)。CPU/GPU 对比：`02-device-benchmark-8d295c88b727`，B=1/2/8/32/128/512 全部通过数值一致性检查，最大绝对差约 4.92e-7。

B=1 的 CPU/GPU 纯 forward 分别约 2.613/1.354 ms；B=128 约 130.107/1.455 ms；B=512 约 617.145/5.213 ms。CPU 使用 4 线程，未测反向传播；详见[完整表格与曲线](results/02-device-benchmark/README.md)。

### 阶段 3：单步训练

运行：`03-step-ea5732fd1e68`。同一批 128 张训练图，loss 从 **2.294613** 降到一次 step 后的 **2.119175**。

- backward 前梯度全为 None；backward 后 56 个参数张量的梯度均有限，而参数逐位不变。
- step 后 56 个参数张量均有元素改变；所有 AdamW 参数状态的 step 都为 1。
- 独立副本每次重新 forward/backward：每次清零得到 1、1、1 倍梯度，只在开始清零得到 1、2、3 倍梯度；全部参数逐元素检查通过。
- CrossEntropyLoss 与负 log 概率均值一致，logits 梯度与独立公式一致到容差。

查看[学习笔记](notes/03-step.md)、[结果摘要](results/03-step/README.md)和[梯度累积图](results/03-step/gradient-accumulation.png)。这是单批机制观察，没有完整 epoch、验证集或测试集指标。

### 阶段 4：记住 32 张训练图

运行：`04-overfit32-f206c27f589e`。初始 loss=2.318534、accuracy=9.375%；第一个记录到 accuracy=100% 的评估点是 step=40。step=130/140/150 连续满足 accuracy=100%、loss≤0.05，最终在 **150 步、loss=0.037204、32/32 正确**时停止。

- 输入固定为原 train 索引列表前 32 张，无增强、dropout 或权重衰减；子集没有 ship 类，不是类别平衡的评测集。
- 全部 loss、梯度及被检查的参数均有限，所有 AdamW 参数状态的 step 都为 150。
- 评估保持参数不变、没有创建梯度，保存并重新加载权重后的 logits 最大绝对差为 0。
- 本次只测训练所用的 32 张图，未评估 validation 或 test，不能据此推断泛化表现。

查看[阶段 4 学习笔记](notes/04-overfit32.md)、[指标和记录](results/04-overfit32/README.md)。

![固定 32 张训练图的曲线](results/04-overfit32/learning-curves.png)

### 阶段 5：完整 train / validation / checkpoint

最终运行：`05-train-f267ced4de42`。完整 train=45,000、validation=5,000，主轨迹完成 20 epoch、7,040 次参数更新。

| 模型位置 | Train loss | Train accuracy | Val loss | Val accuracy |
| --- | --- | --- | --- | --- |
| 第 1 轮 | 1.829942 | 31.93% | 1.646145 | 38.58% |
| 第 19 轮（best） | 0.749895 | 73.15% | 1.013452 | 65.58% |
| 第 20 轮（last） | 0.726002 | 74.09% | 1.020374 | 65.36% |

- best 按最低 val_loss 选择，重载后验证指标完全一致；last 保留第 20 轮的完整训练状态。
- 完整数据连续训练 6 轮，与从第 5 轮检查点重启后再训练第 6 轮，参数、AdamW 状态、随机状态、指标和样本顺序均完全一致；参数最大绝对差为 0。
- 三项关键检查通过：短尾批的样本加权统计、随机状态保存/恢复及受限加载、最佳参数快照独立性与差异检测。
- 主轨迹训练循环累计 292.29 秒、验证 14.58 秒，不含检查点写盘、启动和绘图；含首轮试跑及重复第 6 轮，共实际执行 22 个完整数据 epoch，另有 7 个小子集 epoch。
- 单种子观测，不代表跨设备或跨依赖版本逐位可复现。官方 test 始终未评估。

查看[结果与运行职责](results/05-train/README.md)、[完整恢复对照](results/05-train/resume-comparison.json)、[逐轮数据](results/05-train/epochs.csv)与[学习笔记](notes/05-training.md)。

![完整训练和验证曲线](results/05-train/learning-curves.png)

### 阶段 6：训练预算 20 → 50

最终运行：`06-budget-4b59a2a02b22`；先通过 `06-budget-56f9438e67a2` 第 21 轮试跑，再恢复至第 50 轮。新增 30 轮、10,560 次更新，总轨迹为 50 轮、17,600 次更新。

| Epoch | Train eval loss | Train eval accuracy | Val loss | Val accuracy |
| --- | ---: | ---: | ---: | ---: |
| 20 | 0.648338 | 77.24% | 1.020374 | 65.36% |
| 30 | 0.387491 | 86.55% | 1.164331 | 65.06% |
| 40 | 0.225757 | 91.80% | 1.545476 | 63.80% |
| 50 | 0.119099 | 95.93% | 1.751647 | 64.42% |

- 训练配方、数据划分和原训练主干保持一致；按用户指定更换设备，跨设备迁移先检查完整状态并复核历史模型验证指标。原运行与同设备恢复证据保留。
- 新增 train_eval 与 validation 使用同一个轮末模型；监测不改变参数、优化器或随机状态，三个专项检查通过。第 1–19 轮没有 train_eval，未伪造旧指标。
- best 仍为第 19 轮（val_loss=1.013452，val accuracy=65.58%）；最高验证准确率实际位于第 22 轮（65.68%），但没有改变预定的最低 val_loss 选择规则。
- 新增训练累计 446.09 秒、train_eval 196.45 秒、validation 21.86 秒；两次运行内部计时合计约 11.48 分钟。PyTorch 峰值已分配显存约 381.35 MiB。

查看[阶段 6 学习笔记](notes/06-budget.md)、[结果摘要](results/06-budget/README.md)、[逐轮数据](results/06-budget/epochs.csv)与[恢复检查](results/06-budget/resume-checks.json)。

![训练预算延长后的训练与验证曲线](results/06-budget/learning-curves.png)

### 阶段 7：官方测试集最终评估

运行：`07-final-643922676edf`。第 19 轮 best 的验证指标重测一致；首次评估官方 test 得到：

| 数据 | 样本数 | Loss | Accuracy | 正确数量 |
| --- | ---: | ---: | ---: | ---: |
| Validation（第 19 轮模型） | 5,000 | 1.013452 | 65.58% | 3,279 |
| Official test（同一模型） | 10,000 | **1.023884** | **64.62%** | **6,462** |

- 参数与源 checkpoint 保持不变，没有新增训练；官方 test 仅一遍 forward，79 批、尾批 16 张。所有图表复用同一遍预测。
- 混淆矩阵行是真实标签、列是预测标签；每行 1,000，总和 10,000，对角线和 6,462。最多的定向错误为 dog → cat（251 张）、cat → dog（170 张）、automobile → truck（161 张）。
- 每类 recall 最高为 ship=81.60%，最低为 cat=52.80%；总体 accuracy 不能替代分类别的观察。
- 两项专项检查及真实数据检查全部通过；测试循环约 1.48 秒，含取数、传输和预测收集，PyTorch 峰值已分配显存约 93.78 MiB。

查看[学习笔记](notes/07-final.md)、[精选结果](results/07-final/README.md)、[正确示例](results/07-final/examples-correct.png)、[错误示例](results/07-final/examples-incorrect.png)与[完整实验总结](notes/experiment-summary.md)。

![官方测试集混淆矩阵](results/07-final/confusion-matrix.png)

## 6. 分析 / Analysis

训练主干、小样本拟合、完整数据验证和 epoch 边界恢复均已取得实测证据。阶段 4 表明分类是否正确与真实类别概率大小是不同观察。阶段 5 中，第 19→20 轮训练指标继续改善，验证表现却略差，因此 best 保留第 19 轮。验证整体较初期改善，中间存在波动，不能凭单次波动推断继续训练一定恶化。

在阶段 5 中，train 指标来自训练过程中不断变化的参数，validation 来自该轮结束模型；没有用同一个最终模型额外遍历整个 train，因此两者不是完全相同条件下的评估。当时的恢复一致性检查只覆盖当前确定性设置、同 GPU/软件与 epoch 边界。取数等待约占训练循环的 44%，后续可单独测 worker 数量；当前结果不能把等待直接归因为 SSD 慢。

上述指标口径说明对应阶段 5。阶段 6 已新增固定轮末模型的 train_eval：它的准确率从 77.24% 升至 95.93%，而验证准确率约在 64%–65% 附近波动、验证 loss 持续上升，呈现过拟合。21–30、31–40、41–50 轮的平均 val_loss 分别为 1.094514、1.342833、1.666212，趋势不只来自单个端点。按预先指定的最低 val_loss 标准，额外 30 轮没有产生新的最佳模型；这不代表其他配方或模型的准确率上限。

阶段 7 的同一模型 val/test 准确率相差 0.96 个百分点；两者图片不同，validation 还参与了选择。测试结果是这次单种子、固定模型的最终报告，没有用于回头调参。错误示例中有较高 confidence 的错误判断；这些未经校准的 softmax 分数不能直接当作真实可靠率。混淆矩阵说明错误类别分布，本阶段未测具体特征依赖或分布外能力。

## 7. 学到的内容 / What I Learned

用户已正确回答阶段 1 的三个问题：batch 形状、标签一一对应和 batch size 不改变单图形状。阶段 2 的回答整体正确；位置编码通过 batch 维广播共享，labels 保存真实类别编号。用户结构图与模型一致，需修正一处 MLP 算术笔误并在流程中补最终 LayerNorm，见[核对记录](notes/02-diagram-review.md)。

阶段 3 中已讨论梯度清零、累加与多次更新的区别；用户答案保留在原笔记。阶段 4 的三个问题已回答并确认：accuracy 满分时 loss 仍可降低，训练图上的满分不能证明泛化，32 张图训练 150 步仍只见过 32 张不同图片。阶段 5 的学习重点是 train/validation 指标、best/last 选择，以及完整状态恢复。

阶段 6 提供了预算增加与过拟合的实测例子，解释 online train 与轮末 train_eval、验证 loss 与 accuracy 的不同趋势，以及预先固定模型选择规则的意义。用户已回答三个问题，核心理解正确；第一题补充了每批更新前预测与按样本数加权统计的细节，原回答和核对保存在阶段 6 笔记。

阶段 7 已提供最终测试、混淆矩阵、每类 recall / precision 与 confidence 的实测解释。相关三个问题已作答：第一题将行列方向读反，已明确解释；第二、三题的核心理解正确，并补充模型分数与实测正确率、调参与梯度训练的区别。全部阶段的概念、证据和工程问题汇总于[实验总结](notes/experiment-summary.md)。

## 8. 下一步 / Next Experiment

七阶段实验执行已收尾。先阅读 [07-final.md](notes/07-final.md) 和[实验总结](notes/experiment-summary.md)，查看最后三个问题的回答核对，尤其注意矩阵的行列方向。后续可按用户选择进入 Tiny CLIP；当前未启动新实验，也未根据 test 分数调参或增加训练。

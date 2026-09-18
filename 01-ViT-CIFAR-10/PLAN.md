# Tiny ViT / CIFAR-10 执行方案

状态：用户已同意 R1–R3，方案生效。阶段 1–4 均已完成运行并通过检查，当前学习入口为 [04-overfit32.md](notes/04-overfit32.md)；阶段 5 为后续任务。

## 1. 学习路线与范围

沿用[原始大纲](notes/outline-original.md)，落实[三项审阅调整](notes/outline-review.md)。按数据、模型 forward、单步更新、32 图记忆、完整训练的顺序推进。每阶段展示实测输出、解释关键代码，再让用户查看与理解。

第一版用基础 PyTorch 模块、float32 和显式训练循环。保持 32×32 输入与随机初始化；不引入预训练模型、AMP、compile、DDP、scheduler、梯度裁剪或训练器框架。梯度累积只作为阶段 3 的独立观察演示。

## 2. 文件与资产位置

仓库内逐阶段增加文件，表中后续入口不代表当前已实现：

| 路径 | 职责 / 阶段 |
| --- | --- |
| `README.md` | 状态、复现入口与八部分实验记录 |
| `PLAN.md` | 本执行方案 |
| `notes/outline-original.md` | 不改写的原始大纲 |
| `notes/outline-review.md` | 已采纳的修正依据 |
| `notes/01-data.md` | 阶段 1 的观察、解释和学习问题 |
| `notes/02-forward.md` | 阶段 2 的实测 shape、patch 图、模型解释与学习问题 |
| `configs/01-data.json` | 阶段 1 的固定配置 |
| `src/data.py` | CIFAR-10、划分索引与 DataLoader，阶段 1 |
| `src/inspect_data.py` | 查看样本、batch、完整性检查与图片，阶段 1 |
| `scripts/setup_env.sh` | 显式创建 SSD 环境并安装固定依赖 |
| `scripts/inspect_data.sh` | 加载目标解释器，创建独立运行并执行阶段 1 |
| `requirements*.txt` | 直接依赖与安装后锁定的依赖版本 |
| `results/` | 可公开的小型观察摘要和精选图片 |
| `src/model.py`、`src/inspect_forward.py` | 已实现阶段 2：模型与 shape 观察 |
| `configs/02-forward.json`、`scripts/inspect_forward.sh`、`tests/test_model.py` | 阶段 2 的配置、运行入口和模型检查 |
| `src/benchmark_devices.py`、`scripts/benchmark_devices.sh`、`configs/02-device-benchmark.json` | 已实现：固定 FP32 forward、不同 batch 的 CPU/GPU 计时 |
| `scripts/select_gpu.py` | 从本机选择记录读取 GPU 身份，检查占用 |
| `src/inspect_step.py`、`configs/03-step.json`、`scripts/inspect_step.sh` | 已实现阶段 3：单次参数更新、梯度与清零对照 |
| `src/run_record.py` | 阶段 3 起保存代码快照、依赖、Git 状态和本地运行元数据 |
| `notes/03-step.md`、`results/03-step/` | 阶段 3 的教学解释、学习问题与实测结果 |
| `src/train.py`、`src/evaluate.py` | 已实现阶段 4：固定 32 图训练与只读指标评估；阶段 5 的完整训练和恢复尚未实现 |
| `configs/04-overfit32.json`、`scripts/overfit32.sh` | 阶段 4 的固定配置与启动入口 |
| `notes/04-overfit32.md`、`results/04-overfit32/` | 阶段 4 曲线、逐图预测、解释与学习问题 |

下列位置均在仓库外，路径基于已加载的 `LAB_ROOT`：

```text
$LAB_ROOT/
├── envs/01-ViT-CIFAR-10/py312-cu130/
├── datasets/raw/cifar10/                    # 官方归档与解压后的数据
├── datasets/processed/cifar10/              # 划分索引和校验信息
└── runs/01-ViT-CIFAR-10/<run_id>/
    ├── config.json
    ├── metadata.local.json                 # 本机路径、设备、环境等
    ├── source/                             # 此次运行的代码与配置副本
    ├── logs/
    ├── metrics/
    ├── outputs/
    └── checkpoints/                        # 训练阶段按需创建
```

实验不下载模型。训练产生的模型保存在独立运行的 checkpoints 中。运行 ID 使用阶段名和随机 UUID；服务器时钟不可靠，时间戳仅记为机器时间，耗时采用单调时钟。

## 3. 固定数据与模型设计

### 数据

- 官方训练集 50,000 张；按类别分层，每类抽取 500 张做验证，得到 train=45,000、validation=5,000。划分 seed=42，保存完整索引与 SHA-256。
- 官方 test=10,000 张只在最终模型和流程固定后评估；阶段 1 下载归档时包含它，不创建 test DataLoader，不展示 test 样本或计算 test 指标。
- 阶段 1 只用 `ToTensor()`：HWC uint8 的原始像素转成 CHW float32，范围 [0,1]。不 resize，不做增强或标准化，便于直接对应原图。
- 阶段 2 已采用 `ToTensor()` 后 `Normalize((0.5,)*3, (0.5,)*3)`，映射到 [-1,1]，不依赖验证或测试集统计量。变更已在相应配置中明确记录。
- 基线不做数据增强，dropout=0；先理解训练，再根据实际曲线设计单变量改进。
- DataLoader 首轮 `num_workers=0`、`drop_last=False`；train shuffle，validation 不 shuffle。验证集使用独立数据视图，后续加增强时不共享可变 transform。

### Tiny ViT（阶段 2 实现）

| 项目 | 值 |
| --- | --- |
| 输入 / patch | 3×32×32 / 4×4，64 个图像 token |
| patch embedding | `Conv2d(3,128,kernel_size=4,stride=4)`，无 patch 重叠 |
| 序列 | 可学习 CLS `[1,1,128]` + 可学习位置编码 `[1,65,128]` |
| Encoder | 4 个独立初始化的 block，4 heads，每头 32 维 |
| block | Pre-LayerNorm → MultiheadAttention → residual → Pre-LayerNorm → MLP → residual |
| MLP | Linear(128,512) → GELU → Linear(512,128) |
| 输出 | 最终 LayerNorm 后取 CLS，Linear(128,10)，返回 logits |
| 初始化 | seed=42；Conv/Linear、QKV、CLS/位置编码用截断正态 mean=0、std=0.02、范围 [-0.04,0.04]；bias=0，LayerNorm weight=1、eps=1e-6；各 block 参数独立 |

模型代码注明关键 shape，使用 `batch_first=True`。观察入口展示中间 shape；正式 forward 返回 logits，训练循环保持简明。

## 4. 五阶段预算与验收

所有阈值都是验收目标，不是已取得的结果。

| 阶段 | 初始预算 | 验收与停止位置 |
| --- | --- | --- |
| 1 数据 | CPU，4 个计算线程，batch=128，worker=0；查看一个训练 batch | 45k/5k 索引互斥且覆盖 50k；每类 4500/500；图像 `[128,3,32,32]`、标签 `[128]`；类型/范围/图像标签对应正确；同种子首批可复现；展示 16 张图后停下 |
| 2 forward | 首轮 CPU 已完成；后续使用指定 GPU，B=2/128；补充 B=1/2/8/32/128/512 的 CPU/GPU 计时；不反传 | 逐段 shape 匹配，输出有限值 `[B,10]`；记录实际参数量；解释 patch、位置编码与 CLS |
| 3 单步 | 已运行：固定 batch=128，AdamW lr=3e-4、weight_decay=0；仅一次 step | 清零后 grad=None；backward 后梯度有限，权重未变；step 后权重发生变化；完成 R1 对照 |
| 4 32 图记忆 | 从 train 索引固定取 32 张；batch=32，lr=3e-4，weight_decay=0，dropout=0；最多 1000 step | 每 10 step 对固定 32 图评估；目标 accuracy=100% 且 loss≤0.05，连续 3 次满足可停止；不达标先诊断，不自动无限延长 |
| 5 完整训练 | 单张用户指定 GPU；batch=128，lr=3e-4，weight_decay=1e-2，seed=42；先 1 epoch 检查，再最多 20 epoch | 保存四条曲线、best/last checkpoint；完成 5 epoch 中断后从第 6 epoch 恢复的对照；解释实测曲线，不设置最终准确率门槛 |

阶段 4 固定使用保存的 train 索引列表前 32 项，未读取 validation 图片，未重新挑选子集。实测在 step=130/140/150 连续满足目标并停止，最终 loss=0.037204、accuracy=100%；完整记录见阶段 4 笔记。阶段 5 若未观察到明显过拟合，如实记录；后续可单独设计缩小训练集的诊断，不擅自替换完整训练结果。

用户已指定本实验 GPU，身份保存于本机 `.local/device.json`，后续阶段持续沿用，不重复询问。模型运行入口调用 `scripts/select_gpu.py` 检查身份与占用；阶段 1 数据读取及正确性单元检查保留 CPU，阶段 2–5 模型计算默认 GPU。CPU/GPU 基准仅作为阶段 2 补充，未提前执行反向传播。

## 5. 梯度观察修正（R1）

每次训练 step 保持清晰顺序：`zero_grad(set_to_none=True)` → forward → loss → backward → step。

独立的梯度累积演示固定 batch、固定权重、关闭随机操作；每次重新 forward、计算 loss，再 backward。A 组每次清零；B 组只在开始时清零，中途不 step。比较同一梯度与单次梯度的 1、2、3 倍，记录数值误差。演示后清零，不把该实验代码混入正式训练。

## 6. 评估与 checkpoint（R3）

- 训练每批 `model.train()` 下更新；验证 `model.eval()` 配合 `torch.inference_mode()`，验证阶段不更新参数。
- loss 按样本数加权汇总；accuracy 为正确数量 / 实际样本数，不对 batch accuracy 简单平均。
- 每 epoch 保存训练期间统计的四个指标。解释 train 指标来自该轮不断变化的参数，validation 来自该轮结束模型；若需要严格比较，再额外用 eval 模式评估训练集并单独命名指标。
- `best` 依据最低 val_loss，平局保留较早 epoch；`last` 用于恢复。测试集不参与模型选择。
- 在 epoch 边界保存 model、optimizer、completed_epoch、global_step、best_val_loss、配置、数据划分 hash、Python/NumPy/PyTorch CPU 与可见 CUDA 随机状态、DataLoader generator 状态及环境和代码标识。
- 先支持从完整 epoch 边界恢复；中途退出从最近完整 epoch 重新开始，不宣称支持任意 batch 精确续跑。恢复到新 run，记录 parent_run，不覆盖源 checkpoint。
- 核验：连续训练 6 epoch 与相同初态训练 5 epoch 后重启继续到第 6 epoch。先在小训练子集上验证协议；控制随机操作与数据顺序，比较参数、优化器步数和指标并记录误差。GPU 上使用确定性设置时若操作不支持，明确记录限制，不声称跨设备逐位相同。
- 保存先写同目录临时文件再原子替换；RNG 状态编码为张量与基本容器，验证可用 `weights_only=True` 加载。以后引入 scheduler/AMP 时才扩展其状态。

## 7. 环境、资源和运行证据

环境使用 Python 3.12，当前为 PyTorch 2.14.0 / torchvision 0.29.0 的 cu130 配对，直接依赖固定版本，并记录所有解析版本。早期 CPU 观察使用 2.13.0 / 0.28.0；首次 GPU 运行发现 cuDNN 子库版本冲突，已通过配套升级修复，见[环境记录](notes/02-gpu-environment.md)。

安装只使用目标环境的 `python -m pip`，缓存与临时目录位于 SSD。CUDA wheel 同样能执行 CPU 阶段；已在指定 GPU 上验证真实图片 forward 和六种 batch 的数值一致性。阶段 1 明确隐藏 CUDA 设备。

每次运行保存配置、种子、数据校验和、命令、依赖版本、代码文件快照与哈希、Git 状态；首次提交前标记 `unversioned`。完整记录位于 SSD，公开结果只导出指定字段，避免复制本机路径和设备信息。

性能计时使用单调时钟；后续 GPU 计时在边界同步。先测真实显存峰值和吞吐，再调整 worker 等参数；不因机器有四张卡就启用多卡。

## 8. 当前交付边界

阶段 1–4 已完成运行与检查，阶段 4 停在 32 图记忆结果和学习问题；阶段 5 尚未开始。阶段 1–4 的代码、笔记和精选结果纳入仓库版本管理；完整运行产物保存在仓库外。

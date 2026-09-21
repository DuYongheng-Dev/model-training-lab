# Tiny ViT / CIFAR-10 · 七阶段实验总结

**实验执行已完成。** 使用基础 PyTorch 实现并训练 809,354 参数的 Tiny ViT；主轨迹训练 50 轮，按最低验证 loss 选出第 19 轮模型，官方 test 准确率为 **64.62%**。最后三个学习问题已作答并核对：矩阵行列方向的错误已指出，并补充了 confidence 与 accuracy、调参与梯度训练的区别；具体见阶段 7 笔记。

## 1. 问题

本实验的目标是把数据、Transformer forward、loss、梯度、参数更新、验证、检查点恢复和最终测试连成一条能实际运行、解释和复现的流程。不以 CIFAR-10 准确率排名为目标。

## 2. 假设与证据

| 最初需要验证的认识 | 得到的证据 |
| --- | --- |
| backward 计算梯度，step 更新参数 | 单步实验中 backward 后 56 个参数张量不变，step 后均有元素变化 |
| 小模型应能记住少量固定训练图片 | 固定 32 图在 150 步达到 accuracy=100%、loss=0.037204 |
| 训练表现改善不必然意味着泛化改善 | 第 20→50 轮 train_eval 准确率提高，val loss 从 1.020374 升至 1.751647 |
| 完整恢复需要权重之外的训练状态 | 同设备连续第 6 轮与从第 5 轮恢复后的第 6 轮，参数、AdamW、随机状态、指标和数据顺序完全一致 |
| 一个总体分数不足以描述全部错误 | Test 总体准确率 64.62%，不同类别 recall 从 52.8% 到 81.6%，并有明显的 dog → cat 混淆 |

## 3. 实现：从图片到参数更新

```text
Dataset：一张 [3,32,32] 图 + 一个类别编号
    ↓ DataLoader
images [B,3,32,32]，labels [B]
    ↓ 4×4 patch 投影
64 个 patch tokens，每个 128 维
    ↓ 加 CLS 和位置编码
[B,65,128]
    ↓ 4 个 Transformer blocks + 最终 LayerNorm
取 CLS → Linear
    ↓
logits [B,10]
    ↓ CrossEntropyLoss(logits, labels)
loss → backward → .grad → AdamW.step → 更新参数
```

核心源码：[model.py](../src/model.py)、[train_full.py](../src/train_full.py)、[checkpoint.py](../src/checkpoint.py)。训练预算扩展复用原主干；最终评估单独加载冻结模型，不建立优化器。

## 4. 设置与预算

- CIFAR-10：从官方 50,000 张训练图分层划分 train=45,000、validation=5,000，每类分别 4,500/500；官方 test=10,000，最后阶段才评估。
- Tiny ViT：输入 32×32、patch=4、embed_dim=128、depth=4、heads=4、mlp_ratio=4、10 类，随机初始化；没有预训练。
- 完整训练：seed=42、batch=128、FP32、AdamW lr=3e-4、weight_decay=0.01，无增强、dropout 或 scheduler；像素归一化到 [-1,1]。
- 主轨迹 50 epoch，每轮 352 次更新，总计 17,600 次。阶段 5 包含额外的首轮试跑和恢复对照，实际另执行 2 个完整数据 epoch 及 7 个小子集 epoch；这些检查不混入主轨迹。早期单步与 32 图实验也单独记录。
- 阶段 6 新增 30 轮没有重复训练其中某轮。阶段 7 没有训练，固定模型对 10,000 张官方 test 做一遍推理。
- 运行前检查用户指定资源；所有环境、数据、完整日志、预测和检查点使用仓库外的 `LAB_ROOT`。Git 保存源码、配置、依赖锁和精选结果，完整运行保留命令、代码快照、哈希及未提交改动。

## 5. 七阶段结果

| 阶段 | 完成的观察 | 阅读入口 |
| --- | --- | --- |
| 1 数据 | `[128,3,32,32]` 与 `[128]`；固定分层划分与尾批 | [01-data.md](01-data.md) |
| 2 forward | 64 patches → 65 tokens → `[B,10]` logits；补充 GPU 与 batch 计时 | [02-forward.md](02-forward.md) |
| 3 单步 | 参数、梯度、清零、累积与 optimizer.step 的关系 | [03-step.md](03-step.md) |
| 4 小样本拟合 | 32 图、150 步、100% accuracy、loss=0.037204 | [04-overfit32.md](04-overfit32.md) |
| 5 完整训练 | 20 轮；best=19，val accuracy=65.58%；恢复对照通过 | [05-training.md](05-training.md) |
| 6 预算扩展 | 50 轮；轮末 train accuracy=95.93%，val accuracy=64.42%，val loss=1.751647 | [06-budget.md](06-budget.md) |
| 7 最终评估 | 第 19 轮 best：test loss=1.023884，accuracy=64.62%，6,462/10,000 正确 | [07-final.md](07-final.md) |

阶段 6 的第 50 轮验证准确率 64.42% 与阶段 7 的测试准确率 64.62% 来自**不同 checkpoint、不同数据集**，不能把它们当成同一模型的 val/test 对照。最终同模型对照是第 19 轮的 val=65.58%、test=64.62%。

## 6. 分析、问题与边界

### 拟合、过拟合与模型选择

记住 32 张图验证了学习过程能工作，不能证明泛化。扩大到完整训练后，增加轮数持续改善训练集，却使验证 loss 上升。best 仍为第 19 轮，last 为第 50 轮；最高 val accuracy 曾出现在第 22 轮，但我们保持预定的最低 val_loss 选择规则。

online train 按样本数汇总各 batch 更新前的预测，反映这一轮训练过程，统计成本小；train_eval 用同一个轮末模型重测训练集，更适合与 validation 比较。第 6 阶段专门增加这项观察，并检查它没有改变训练状态或随机序列。

### 最终模型的错误

最终 test 的主要定向混淆为 dog → cat（251/1000）、cat → dog（170/1000）和 automobile → truck（161/1000）。船类 recall=81.6%，猫类 recall=52.8%。混淆矩阵描述了错误分布，尚不能证明模型依赖某种具体图像特征。

错误示例也可能具有较高 softmax 分数，例如 ship → truck 的 #2 样本分数约 78.7%。我们没有做置信度校准或分布外测试，不能把这些分数当作真实可靠率。

### 保留的问题与工程经验

- 原大纲的梯度累积演示按已同意的调整改为每次重新 forward/backward；训练资产迁至外部 SSD，checkpoint 补充完整随机状态和数据索引。
- 早期 GPU 环境遇到 cuDNN 子库混用，配套依赖修复后通过 GPU 检查，过程保留于 [环境记录](02-gpu-environment.md)。
- 按用户指定在 epoch 20 后更换了物理 GPU。加载状态及历史模型验证检查通过，但没有执行两张卡完整后续轨迹的逐位对照。原有同设备恢复证据与跨设备验证范围分开记录。
- 训练阶段的取数等待占比约 44%，尚未单独调优 worker；这不能直接归因为 SSD 慢。记录的训练时间包含取数、传输和检查，不是纯 GPU 性能。
- 只使用一个种子与一种完整训练配方；结果不能代表 ViT 架构上限。没有以 test 结果选模型、改超参数或继续加训。

## 7. 已讨论的概念

| 层面 | 本实验给出的解释和证据 |
| --- | --- |
| 数据 | Dataset 提供样本；DataLoader 组 batch；shuffle 控制顺序；train 用于更新、val 用于选择、test 留作最终报告 |
| 模型 | patch 投影、CLS、位置编码与 batch 维广播；logits 是分类得分，labels 是真实类别编号 |
| 训练 | loss 连接预测与标签；backward 产生梯度；step 更新参数；梯度累积与连续多次更新不同 |
| 优化 | learning rate 控制更新尺度；AdamW 保留梯度历史的移动统计，weight decay 参与参数衰减；epoch 与 step 是不同计数 |
| 泛化 | 训练拟合与验证效果可能分离；accuracy 和 loss 衡量不同方面；best 与 last 服务于不同目的 |
| 工程 | 权重文件与完整训练 checkpoint 的区别；种子是起点，继续训练还需恢复随机状态和 DataLoader 状态 |
| 评估 | 总体 accuracy、每类 recall / precision、混淆矩阵与 confidence；最后三道题已作答并核对，行列方向错误已指出 |

用户原始回答保留在各阶段笔记；本总结不替代这些学习过程，也不把尚未确认的修正记为已掌握。

## 8. 后续

本轮 ViT 实验按预定路线收尾。阶段 7 的回答核对已记录，可先回看矩阵行列方向与测试集参与模型选择的解释，再按用户选择开展下一实验。原建议中的 Tiny CLIP 尚未创建或启动；若开展，可复用数据加载、梯度、优化器和检查点知识，进一步学习图文编码、相似度矩阵与对比损失。

# 实验大纲审阅

状态：用户已同意 R1–R3，三项调整全部采纳，落实于 [执行方案](../PLAN.md)。原稿继续原样保留。

## 原稿归档

[outline-original.md](outline-original.md) 是用户提供的 AI 大纲，按原始字节复制，未改写正文；临时目录中的源文件保留。

- SHA-256：`fcd396ee2ef7e8632929175ede12e837e5852f61033ae74703eb5c21462919ec`
- 原稿中的命令是教学示例，不代表已在本实验运行。
- 执行时同时遵循仓库的存储、资源指定和实验记录约定。

## 可以保留的主体

- 学习训练过程：数据 → forward → backward → 参数更新 → 泛化评估 → 保存与恢复。
- CIFAR-10 保持 32×32；从官方训练集划出 45,000 训练样本和 5,000 验证样本，最终评估前封存官方测试集。
- Tiny ViT 随机初始化：patch 4、维度 128、4 层、4 个注意力头、MLP 比例 4、10 类。
- 用可读的 PyTorch 模块和显式训练循环；先观察单步，再记住 32 张图，最后运行完整训练。
- 第一版保持简单，不引入预训练模型、训练器框架、多卡和混合精度。

## 已采纳的三项调整

| 编号 | 原稿位置 | 问题或不足 | 建议 |
| --- | --- | --- | --- |
| R1 | 第七节 `zero_grad()` | 对同一次 forward 产生的 `loss` 连续执行三次 `backward()`，这个模型默认会在第二次反传时报计算图已释放的错误 | 每次重新 forward、计算 loss、backward；固定输入和参数且关闭随机操作，分别观察清零与不清零的梯度 |
| R2 | 第一、十六、十七节路径示例 | `./data`、项目内 `.venv`、checkpoints 等与已确定的仓库外 SSD 存储约定不一致 | 执行方案统一使用 `$LAB_ROOT` 下的环境、数据及运行目录；仓库保存代码、配置、笔记和精选结果 |
| R3 | 第十三节 checkpoint | 原示例足以介绍模型与优化器恢复，但不足以验证中断前后是否沿着同一随机训练过程继续 | 补充配置、数据划分标识、随机数及 DataLoader generator 状态；先限定在 epoch 边界恢复，再做连续训练与中断恢复的对照 |

### R1 的建议演示方式

以下教学设计已采纳，将在阶段 3 实现和运行；当前不标记为已验证。

1. 固定一个 batch，暂时不调用 `optimizer.step()`，关闭 dropout 等随机操作。
2. A 组每次先 `zero_grad(set_to_none=True)`，再重新 forward、计算 loss、backward。
3. B 组仅开始时清零，每次同样重新 forward、计算 loss、backward。
4. 比较同一参数的梯度：B 组的梯度应在数值误差范围内接近单次梯度的 1、2、3 倍。
5. 演示结束后清零。正式训练仍然每批清零；这段观察不引入梯度累积训练方案。

PyTorch 说明 backward 会将梯度累加到叶子张量，默认不保留用于反向传播的图。这里通过重新 forward 建立新图，不需要 `retain_graph=True`。[PyTorch backward 文档](https://docs.pytorch.org/docs/stable/generated/torch.autograd.backward.html)

### R3 的适用范围

保存模型与优化器状态是恢复训练的基础；若要比较连续运行和中断恢复，还需控制采样顺序及其他随机来源。补充状态不等于承诺跨设备或跨版本逐位一致。[PyTorch 保存与加载教程](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)、[PyTorch 可复现性说明](https://docs.pytorch.org/docs/stable/notes/randomness.html)

## 实验解释边界

- 记住 32 张训练图是正确性检查；实际能否在预算内达到目标需要运行后判断。未达到时也要排查学习率、初始化等因素。
- 完整训练未必在预定轮数内出现验证 loss 上升；如实解释实际曲线，不把预想曲线当结果。
- 本实验为教学规模 ViT，不能将结果解读为复现原论文准确率。[ViT 原论文](https://arxiv.org/abs/2010.11929)

# 阶段 7 · 最终测试与误差分析

状态：**最后阶段的运行、检查与结果整理已完成**。七阶段实验流程已闭环，三个学习问题已作答；第一题的行列方向错误与其余补充解释见第 8 节。运行：`07-final-643922676edf`；方案见 [07-final-plan.md](07-final-plan.md)。

## 1. 最终结果

使用 epoch 1–50 中最低 validation loss 对应的 **第 19 轮 best**。选择规则和 checkpoint SHA-256 在测试推理前已固定，配置见 [07-final.json](../configs/07-final.json)。

| 数据集 | 图片数 | 平均交叉熵 loss | Accuracy | 正确数量 |
| --- | ---: | ---: | ---: | ---: |
| Validation（加载后重测） | 5,000 | 1.013452 | 65.58% | 3,279 |
| 官方 test（首次评估） | 10,000 | **1.023884** | **64.62%** | **6,462** |

验证与测试准确率相差 **0.96 个百分点**。它们包含不同图片，而且 validation 参与了模型选择，不能要求两个分数完全相同。本结果对应一次固定配方、一个种子和一个已选定模型，不能据此推断换数据分布后的表现。

本阶段没有训练：没有 optimizer、backward 或参数更新。官方 test 完整做了一遍 forward，79 批，最后一批 16 张；图表全部复用这次预测。未对第 50 轮 last 或其他 checkpoint 计算 test 分数，也没有根据 test 结果调整配置。

## 2. 核心代码在做什么？

```python
model.load_state_dict(best_checkpoint['model'])
model.requires_grad_(False)
model.eval()

with torch.inference_mode():
    logits = model(images)
    losses = cross_entropy(logits, labels, reduction='none')
    predictions = logits.argmax(dim=1)
    confidence = logits.softmax(dim=1).max(dim=1).values
```

这是核心逻辑示意，完整入口为 [final_evaluate.py](../src/final_evaluate.py) 和 [final_metrics.py](../src/final_metrics.py)。交叉熵仍然直接接收 logits；softmax 只用于展示 confidence。总 loss 是 10,000 张图片 loss 的平均值，accuracy 是正确总数 / 10,000。

这里的 `eval()` 与 `inference_mode()` 配合使用：前者设置模块的评估行为，后者关闭梯度记录及相关推理开销。数据来自 test，才使它成为测试集评估；调用 `eval()` 本身不会选择数据集。

## 3. 如何读混淆矩阵？

![官方测试集混淆矩阵](../results/07-final/confusion-matrix.png)

- **行：真实类别。列：预测类别。**
- 对角线表示预测正确；其他格子表示具体错成了哪一类。
- 左图是图片数量，右图按每行归一化为百分比。每个真实类别正好有 1,000 张，因此每一行合计为 1,000 或 100%。
- 对角线数量合计 6,462，除以总数 10,000，得到总体 accuracy=64.62%。

例如，`dog` 行、`cat` 列的 **251** 表示：1,000 张真实的狗图中，有 251 张被判成猫，即 **25.1%**。反方向 `cat → dog` 是 170 张，两种错误不要求对称。

本次数量最多的五种有方向的错误：

| 真实 → 预测 | 数量 | 占该真实类别 |
| --- | ---: | ---: |
| dog → cat | 251 | 25.1% |
| cat → dog | 170 | 17.0% |
| automobile → truck | 161 | 16.1% |
| deer → bird | 130 | 13.0% |
| horse → deer | 106 | 10.6% |

这些是完整测试集的实测频率。它们能定位错误类型，但不能单凭混淆矩阵断定模型是依赖背景、形状或颜色作出判断；本阶段没有做因果归因实验。

### Recall 和 precision 的分母不同

以猫类为例：

- **Recall（召回率）：**真实的 1,000 张猫图里，认对 528 张，`528 / 1000 = 52.80%`。
- **Precision（精确率）：**模型总共把 1,183 张图判成猫，其中 528 张真是猫，`528 / 1183 ≈ 44.63%`。

前者问“真正的猫找到了多少”，后者问“说是猫的判断有多少正确”。本次每类 recall 最高的是 ship（81.60%），最低的是 cat（52.80%）；bird 为 52.90%。所以总体 64.62% 并不意味着每个类别都识别得一样好。完整表见 [per-class.csv](../results/07-final/per-class.csv)。

## 4. 正确与错误示例，以及 confidence

下面各取官方测试索引顺序中最早出现的 10 张正确、10 张错误图片。这一规则在评估前固定；图片用于帮助理解，错误频率仍以完整混淆矩阵为准。

![预测正确的 10 张图](../results/07-final/examples-correct.png)

![预测错误的 10 张图](../results/07-final/examples-incorrect.png)

标题的 `true` 是真实标签，`pred` 是预测类别，括号内为该预测类别的 softmax 分数。

例如测试索引 **#2**：真实是 ship，模型预测为 truck，confidence 约 **78.7%**，但判断是错的。这说明模型输出的高分不能保证正确；未经校准的 softmax 分数不能直接解释为真实世界中同类判断的可靠正确率。相反，正确示例 #8 的 cat 分数只有约 43.9%，仍可因为高于其余九类而预测正确。

## 5. 正确性、资源与记录

- 在合成数据上完成两项检查：尾批样本加权、混淆矩阵方向、recall / precision、confidence、参数和梯度不变，以及示例排序与零分母处理；均通过，未使用 test 做这些检查。
- 载入的模型参数与 best 完全匹配；测试前重测 validation，loss 与原记录差为 0，正确数相同。
- 核对官方 test_batch、类别元数据与原划分哈希；测试索引完整且不重复，混淆矩阵数量一致。
- 评估后所有参数不变、梯度为空，源 checkpoint 哈希不变。没有新增训练更新，也没有改变已有依赖或训练代码。
- 测试循环约 **1.48 秒**，含取数、传输、模型推理、softmax 和预测收集；PyTorch 峰值已分配显存约 **93.78 MiB**。运行内部计时约 5.71 秒，包括模型加载、验证复核和绘图等，不包括前置检查、进程导入及记录初始化之前的时间。
- 当前耗时不是纯 GPU 延迟或吞吐基准。全部完整预测、日志与来源设备信息保存在外部运行目录；公开目录只保留精选结果。

查看[结果目录](../results/07-final/README.md)、[模型选择清单](../results/07-final/model-selection.json)、[预检查](../results/07-final/preflight.json)和[完整汇总](../results/07-final/summary.json)。本机运行入口为 `.local/latest-final-run.json`。

## 6. 复现入口

从仓库根目录开始。本机加载本地环境，其他机器按仓库约定准备自己的外部资产目录、对应版本环境与 GPU 指定。

```bash
source ./硬件环境配置信息/experiment-env.sh
cd 01-ViT-CIFAR-10

# 固定配置包含源运行 ID、选定 epoch 和 checkpoint 哈希。
# 自动先运行合成数据检查，再验证已选模型并评估 test。
bash scripts/final_evaluate.sh

# 导出本次已有结果；新运行需替换 run ID。已有结果目录时拒绝覆盖。
"$LAB_ENVS_DIR/01-ViT-CIFAR-10/py312-cu130/bin/python" scripts/collect_final_results.py \
  --run "$LAB_RUNS_DIR/01-ViT-CIFAR-10/07-final-643922676edf"
```

每次启动会生成独立运行目录。这里的复现入口用于核验同一冻结模型的结果；本次实际只评估了一遍官方 test。看到 test 分数后，不继续围绕该分数选 checkpoint 或调参。

## 7. 最后三个学习问题

1. 混淆矩阵中 `dog` 行、`cat` 列为 251，它表示什么？为什么不一定等于 `cat` 行、`dog` 列的数值？
//表示把251张cat图错误识别成了dog，当然不等于`cat` 行、`dog` 列的数值
2. 模型把一张船图以约 78.7% 的 confidence 判成卡车，为什么仍然会错？这个分数与整个测试集的 accuracy 有什么区别？
//模型输出的高分不能保证正确，未经校准的 softmax 分数不能直接解释为真实世界中同类判断的可靠正确率；这个分数和整个测试集的accuracy区别在于，整个测试集的accuracy是10类图片共10000张的准确率，而这个分数仅仅只是一张图的confidence
3. 如果看过 test=64.62% 后，又尝试多个学习率并挑 test 最高的配置，还能把同一 test 当成未参与选择的最终评估吗？为什么？
//不行，因为这样操作后test实际已经间接参与了训练，成了validation_set而非test_set
七阶段运行已完成。下一份阅读材料是[完整实验总结](experiment-summary.md)；Tiny CLIP 仍是后续选题，当前未启动。

## 8. 用户回答与核对

### 第一题：原回答

> 表示把251张dog图错误识别成了cat，当然不等于`cat` 行、`dog` 列的数值



### 第二题：原回答

> 模型输出的高分不能保证正确，未经校准的 softmax 分数不能直接解释为真实世界中同类判断的可靠正确率；这个分数和整个测试集的accuracy区别在于，整个测试集的accuracy是10类图片共10000张的准确率，而这个分数仅仅只是一张图的confidence

核心理解正确。除了单张与整组的粒度差异，还要区分**模型分数**和**依据真实标签统计的正确比例**：

- confidence=78.7%：模型对这张图的预测类别给出的 softmax 分数，只靠模型输出就能计算，不需要真实标签。
- test accuracy=64.62%：把每张预测与真实标签比较后，得到 6,462 / 10,000 的实测正确比例。

即使只看一张图，也可以记录是否预测正确（0 或 1）；错误样本 #2 的正确性为 0，同时 confidence 仍可以是 78.7%。confidence 不是该图的实测正确率。

### 第三题：原回答

> 不行，因为这样操作后test实际已经间接参与了训练，成了validation_set而非test_set

结论正确。更精确地说，test **参与了调参和模型选择**，因而被当作验证集使用。即使从未对 test 执行 backward，只要根据它的分数选择学习率、checkpoint 或其他方案，最终选出的模型就已经受 test 信息影响。

此时它不再是独立于选择过程的最终评估依据。数据文件的名字即使仍叫 test，也不会恢复这种独立性。


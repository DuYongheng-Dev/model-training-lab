## 阶段 6：Training Budget / Plateau

按我们刚才说的做：

```text
从 epoch 20 的 last checkpoint
继续训练到 epoch 50

保持不变：
模型不变
lr = 3e-4
AdamW 不变
weight decay 不变
无 augmentation
无 scheduler
```

这一阶段只回答：

> **什么都不改，只增加训练预算，这个模型还能继续学多少？**

重点观察：

```text
train loss / accuracy

epoch-end train_eval loss / accuracy
（建议补上）

val loss / accuracy

best epoch
```

你最终应该会区分三种情况：

```text
① train、val 都继续改善
→ 之前预算还不够

② train 继续改善，val 基本不动
→ 泛化进入平台，继续死磕意义降低

③ train 继续改善，val 越来越差
→ 明显 overfitting
```

到这里，你对“训练多久”就已经有实际经验了。

------

# 然后直接做最后一个阶段

## 阶段 7：Final Evaluation / 实验收尾

这个阶段甚至不需要继续训练。

### 第一步：选定 best checkpoint

不是用 epoch 50 的模型，而是：

```text
epoch 1~50

↓

找最低 val_loss

↓

best checkpoint
```

这一次你就真正把第五阶段学到的：

```text
best ≠ last
```

用起来了。

------

### 第二步：第一次使用 CIFAR-10 官方 test

你一直没碰：

```text
10,000 test images
```

现在才拿出来。

流程：

```text
best checkpoint
↓
model.eval()
↓
torch.inference_mode()
↓
10,000 test
↓
test loss
test accuracy
```

然后结束。

关键点是：

> **看到 test 结果以后，不再为了提高 test accuracy 回头改模型。**

否则 test 又会逐渐变成 validation。

------

### 第三步：做一个很小的误差分析

不用搞很复杂。

我建议只生成：

```text
Confusion Matrix 10×10
```

你可能看到：

```text
cat ↔ dog
automobile ↔ truck
deer ↔ horse
```

之类的混淆。

目的不是研究 CIFAR-10。

而是第一次意识到：

> 一个 `65% accuracy` 并不能告诉我模型究竟错在哪里。

以后做 CLIP / VLM / 数据集项目，这个认知很重要：

```text
单个总分
≠
完整模型能力
```

除此之外最多再展示：

```text
10 张预测正确图片
10 张预测错误图片
预测类别 + 真实类别 + confidence
```

就够了。

------

# 然后写一份实验总结，ViT 正式毕业

最终 README / 学习报告建议回答这些问题，而不是继续调准确率：

### 数据层

你能解释：

```text
Dataset
DataLoader
batch
shuffle
train / val / test
```

### 模型层

你能解释：

```text
32×32 RGB
↓
4×4 patch
↓
64 patch tokens
↓
128-d embedding
↓
CLS + position embedding
↓
Transformer blocks
↓
Linear classifier
↓
logits
```

以及：

```text
[B,65,128]
```

是什么意思。

### 训练层

你能解释：

```text
logits
↓
CrossEntropyLoss
↓
backward
↓
.grad
↓
AdamW
↓
parameter update
```

并知道：

```text
loss.backward()
不修改参数

optimizer.step()
才修改参数
```

### 优化层

至少理解：

```text
learning rate
AdamW
weight decay
epoch
step
training budget
plateau
```

不要求你手推 AdamW。

### 泛化层

你能解释：

```text
fit
generalization
overfitting
train / validation gap
loss vs accuracy
best vs last
```

### 工程层

你能解释：

```text
model weights
≠
training checkpoint

checkpoint =
model
+ optimizer
+ progress
+ RNG state
+ DataLoader generator state
...
```

如果这些你都能讲出来：

> **ViT 实验已经完成使命。**

------

# 有一个实验我会列为“可选”，但不建议现在深入

如果阶段 6 到 epoch 50 明显 plateau，你大概率会很自然地产生一个问题：

> “是不是应该降 learning rate？”

这时候你可以做一个**唯一的额外对照**：

```text
实验 A
constant lr = 3e-4

vs

实验 B
cosine learning-rate decay
```

除此之外完全一样。

目的是只建立：

> **learning rate schedule 为什么存在**

这个概念。

但我甚至不认为它是进入 CLIP 前的必修。

因为如果你继续加：

```text
RandomCrop
RandomFlip
RandAugment
Mixup
CutMix
Dropout
label smoothing
scheduler
更大模型
```

你马上就会从：

> “理解模型训练”

变成：

> “CIFAR-10 ViT 调参工程”。

这已经偏离你的目标。

------

# 我建议你的 ViT 最终路线就停在这里

```text
1 Dataset / DataLoader
        ↓
2 Tiny ViT Forward
        ↓
3 One Batch One Step
        ↓
4 Overfit 32 Images
        ↓
5 Full Train / Validation / Checkpoint
        ↓
6 Training Budget
20 → 50 epochs
        ↓
7 Best Checkpoint
        ↓
Official Test
        ↓
Confusion Matrix
        ↓
实验总结
        ↓
================
ViT 毕业
================
        ↓
Tiny CLIP
```

阶段 7 不应该再花很多时间，可能一两个小时就够。

------

## 为什么我建议在这里停

因为你现在已经从最开始的：

> “我知道 Transformer 大概是什么。”

走到了：

> “我亲手训练了一个 Transformer-based vision model，并且知道数据、forward、loss、gradient、optimizer、validation、generalization 和 checkpoint 是怎么连起来的。”

再继续提高 CIFAR-10 accuracy，对你进入 CLIP 的**边际学习收益已经开始明显下降**。

反而进入 CLIP 后，你会第一次遇见真正的新东西：

```text
一个 image encoder
+
一个 text encoder

↓

image embedding
text embedding

↓

similarity matrix

↓

contrastive loss
```

而你前面学过的：

```text
Dataset
DataLoader
batch
forward
loss
backward
gradient
AdamW
epoch
train / val
checkpoint
```

全部可以直接复用。

那时候你会非常明显地感觉：

> **CLIP 并不是又重新学一套训练体系，只是模型输出和 loss 换了。**

这才是你做这次 Tiny ViT 实验最主要的意义。
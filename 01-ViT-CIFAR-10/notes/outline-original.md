非常适合，而且你现在这个目标定义得很准确：**不是“学会调用 ViT”，而是亲自经历一次神经网络到底怎么被训练出来。**

对你后面学 CLIP / SigLIP 来说，这一步价值很高。因为 CLIP 看起来复杂，但底层训练循环仍然是：

```text
batch
↓
forward
↓
得到输出
↓
计算 loss
↓
backward
↓
得到 gradient
↓
optimizer.step()
↓
参数发生变化
↓
下一批数据
```

区别只是 CIFAR-10 的 `CrossEntropyLoss` 后面会换成 image-text contrastive loss。

我建议你把这个实验做成一个**“可拆开观察的 ViT 实验室”**，而不是让 Codex 一口气帮你生成一个训练项目然后 `python train.py` 跑完。

------

# 一、你实际上只需要下载两样东西

## 1. 数据集：CIFAR-10

就用它，不要再找别的数据集。

CIFAR-10：

- 60,000 张图片
- 32×32
- RGB 三通道
- 10 个类别
- 官方划分：
  - 50,000 train
  - 10,000 test

类别是：

```text
airplane
automobile
bird
cat
deer
dog
frog
horse
ship
truck
```

官方数据说明如此。([多伦多大学计算机系](https://www.cs.toronto.edu/~kriz/cifar.html?usg=ALkJrhjqbhW2lLxLo8EmqNS-tbK0aT96JQ&utm_source=chatgpt.com))

你甚至不用手动下载。

PyTorch：

```python
from torchvision.datasets import CIFAR10

dataset = CIFAR10(
    root="./data",
    train=True,
    download=True,
)
```

第一次运行会自动下载。Torchvision 官方就提供了 `CIFAR10(..., download=True)`。([PyTorch Documentation](https://docs.pytorch.org/vision/0.11/datasets.html?utm_source=chatgpt.com))

[CIFAR-10 官方数据集页面](https://www.cs.toronto.edu/~kriz/cifar.html?utm_source=chatgpt.com)

------

## 2. 模型：不要下载

这是这个实验最重要的决定。

**自己写一个 Tiny ViT，随机初始化。**

不要第一遍使用：

```python
torchvision.models.vit_b_16()
```

不要：

```python
timm.create_model(...)
```

不要 HuggingFace。

不要 pretrained。

因为你现在就是要看：

```text
图片
↓
patch
↓
token
↓
Transformer
↓
CLS token
↓
Linear
↓
10 logits
```

而不是体验：

```python
model = create_model(...)
```

ViT 原始思想本身就是把图像切成固定大小 patch，再将 patch 当成 Transformer 的 token。([arxiv.org](https://arxiv.org/abs/2010.11929?utm_source=chatgpt.com))

------

# 二、我建议你的 Tiny ViT 长这样

CIFAR-10 图像只有：

```text
3 × 32 × 32
```

所以千万别照搬原版 ViT-B/16。

用：

```text
image_size     = 32
patch_size     = 4

embed_dim      = 128
depth          = 4
num_heads      = 4
mlp_ratio      = 4

num_classes    = 10
```

于是：

```text
32 × 32 image

切成 4 × 4 patch

↓
8 × 8 patches

↓
64 patches

↓
64 image tokens

+ 1 CLS token

↓
65 tokens

每个 token 128 维

↓
[B, 65, 128]

Transformer Encoder

↓

[B, 65, 128]

取 CLS token

↓

[B, 128]

Linear(128 → 10)

↓

[B, 10]

logits
```

这张 shape 流程你一定要理解。

以后 CLIP 的 image encoder，你看到的本质仍然是这个东西。

------

# 三、你的完整数据流到底是什么

你最开始写的是：

```text
Dataset
 ↓
DataLoader
 ↓
batch image
 ↓
ViT
 ↓
logits
 ↓
CrossEntropyLoss
 ↓
loss.backward()
 ↓
optimizer.step()
 ↓
validation
 ↓
accuracy
```

把它扩展以后，真正发生的是：

```text
CIFAR10 Dataset
        │
        │ __getitem__(i)
        ↓
(image, label)
        │
        │ DataLoader 拼 batch
        ↓
images: [B, 3, 32, 32]
labels: [B]
        │
        ↓
     TinyViT
        │
        ↓
logits: [B, 10]
        │
        ↓
CrossEntropyLoss(logits, labels)
        │
        ↓
 scalar loss
        │
        ↓
 loss.backward()
        │
        ↓
每个 Parameter 的 .grad
        │
        ↓
 optimizer.step()
        │
        ↓
模型 Parameter 改变
```

这就是整个训练过程最核心的一条链。

------

# 四、这里最应该亲眼观察的是 4 个东西

例如某一个 batch：

```python
images, labels = next(iter(train_loader))
```

先看：

```python
print(images.shape)
print(labels.shape)
```

应该类似：

```text
images: [128, 3, 32, 32]
labels: [128]
```

------

然后：

```python
logits = model(images)
```

看：

```text
logits.shape
```

得到：

```text
[128, 10]
```

注意：

**logits 不是概率。**

比如可能是：

```text
cat:

[
  0.28,
 -0.91,
  0.14,
  1.23,
 -0.41,
 ...
]
```

不要自己做：

```python
softmax(logits)
```

再传给 CrossEntropyLoss。

PyTorch 的 `CrossEntropyLoss` 就是直接接收**未经归一化的 logits**。([PyTorch Documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html?utm_source=chatgpt.com))

所以：

```python
loss = criterion(logits, labels)
```

------

# 五、真正值得你停下来看的地方：backward

这是这次实验最重要的一步。

训练之前：

```python
parameter.grad
```

通常是：

```text
None
```

然后：

```python
loss.backward()
```

再：

```python
print(model.head.weight.grad)
```

突然就出现 tensor 了。

比如：

```text
tensor([
 [ 0.0003, -0.0008, ...],
 ...
])
```

这就是：

# gradient

你可以专门打印：

```python
grad_norm = model.head.weight.grad.norm().item()

print("gradient norm:", grad_norm)
```

此时要建立一个很重要的认识：

```text
forward
```

只是：

> 根据当前参数做了一次计算。

而：

```text
loss.backward()
```

是：

> 根据 loss，沿计算图反向算出了每个参数应该往哪个方向调整。

但注意：

## `backward()` 没改参数。

这是很多初学者容易模糊的地方。

------

# 六、然后亲眼看 optimizer.step() 改参数

强烈建议第一次这么玩：

```python
weight_before = model.head.weight.detach().clone()

optimizer.zero_grad()

logits = model(images)

loss = criterion(logits, labels)

loss.backward()

weight_after_backward = model.head.weight.detach().clone()

optimizer.step()

weight_after_step = model.head.weight.detach().clone()
```

然后比较：

```python
print(
    torch.equal(
        weight_before,
        weight_after_backward
    )
)
```

应该：

```text
True
```

也就是说：

```text
loss.backward()
```

**没有改 weight。**

再：

```python
print(
    torch.equal(
        weight_before,
        weight_after_step
    )
)
```

应该：

```text
False
```

这一下你就真正理解：

```text
loss
↓
backward
↓
gradient
↓
optimizer
↓
parameter update
```

了。

------

# 七、`zero_grad()` 也一定要故意实验一次

标准训练：

```python
optimizer.zero_grad()

logits = model(images)

loss = criterion(logits, labels)

loss.backward()

optimizer.step()
```

PyTorch 的梯度默认会**累积**。

如果连续调用：

```python
loss.backward()
loss.backward()
loss.backward()
```

没有清零：

```text
parameter.grad
```

会一直累加。PyTorch 官方 autograd 教程也专门强调这一点。([PyTorch Documentation](https://docs.pytorch.org/tutorials/beginner/introyt/autogradyt_tutorial.html?utm_source=chatgpt.com))

所以你甚至应该故意：

```text
实验 A：正常 zero_grad
实验 B：删掉 zero_grad
```

看看 gradient norm 怎么变。

这样以后你看到：

```text
gradient accumulation
```

就不会觉得这是神秘机制。

你会立刻知道：

> 原来就是故意不 zero gradient。

------

# 八、不要直接开始正式训练，按 5 个实验做

我非常建议按这个顺序。

## 实验 1：Single Batch Anatomy

目标：

**只训练一步。**

不要 epoch。

不要 validation。

就一个 batch。

观察：

```text
image shape

↓

patch embedding shape

↓

token shape

↓

CLS token shape

↓

logits

↓

loss

↓

grad = None

↓

backward()

↓

grad 出现

↓

optimizer.step()

↓

weight 改变
```

最好模型 `forward()` 里临时支持：

```python
debug=True
```

输出：

```text
input            [B,3,32,32]

patch embedding  [B,128,8,8]

flatten          [B,64,128]

+ CLS            [B,65,128]

transformer      [B,65,128]

CLS              [B,128]

logits           [B,10]
```

这是整个实验最重要的一次运行。

------

# 九、实验 2：故意让模型记住 32 张图

这其实比直接跑 CIFAR-10 重要。

只取：

```text
32 张
```

或者：

```text
64 张
```

训练。

关闭：

```text
data augmentation
dropout
weight decay
```

疯狂训练这些数据。

目标不是泛化。

目标就是：

```text
train accuracy → 接近 100%
train loss → 接近 0
```

如果一个有百万级参数的模型连几十张 CIFAR 图片都记不住：

**训练 pipeline 大概率有 bug。**

这个实验在实际模型研发里也非常重要，通常叫：

> overfit a tiny batch / tiny dataset

这时候你第一次真正理解：

# overfitting

因为模型干的事情就是：

> 把几十张训练数据背下来了。

------

# 十、实验 3：真正建立 train / validation

这里不要直接使用 CIFAR 官方的 test 当 validation。

建议：

```text
原始 50,000 train

↓

45,000 train
5,000 validation

官方 10,000 test
```

test 暂时封存。

因为：

```text
train
```

用于修改模型参数。

```text
validation
```

用于每个 epoch 判断模型泛化。

```text
test
```

应该最后才看。

于是：

```text
train loader
↓
forward
↓
loss
↓
backward
↓
step
```

但是：

```text
validation loader
↓
forward
↓
loss / accuracy
```

**没有：**

```text
backward
optimizer.step
```

验证时：

```python
model.eval()

with torch.no_grad():
    ...
```

或者：

```python
with torch.inference_mode():
    ...
```

PyTorch 也明确区分了正常 autograd 与 no-grad 模式。([PyTorch Documentation](https://docs.pytorch.org/docs/stable/notes/autograd.html?utm_source=chatgpt.com))

------

# 十一、这时候你终于可以理解 epoch

假设 DataLoader：

```python
for images, labels in train_loader:
```

每跑一次循环只是：

# 一个 batch

而：

```python
for epoch in range(num_epochs):

    for images, labels in train_loader:
        ...
```

内部整个：

```python
train_loader
```

被遍历一次，才叫：

# 一个 epoch

所以关系其实是：

```text
epoch 1

batch 1
 ↓
step

batch 2
 ↓
step

batch 3
 ↓
step

...

最后一个 batch
 ↓
step

↓

validation

↓

epoch 2
```

所以：

> optimizer.step() 的频率通常是 batch 级别。

而：

> validation 通常是 epoch 级别。

这两个时间尺度一定不要混起来。

------

# 十二、实验 4：亲眼看 overfitting 曲线

必须记录四条：

```text
train_loss
val_loss
train_accuracy
val_accuracy
```

然后 matplotlib 画两个图：

```text
Loss

train ─────────
val   ─────────
```

和：

```text
Accuracy

train ─────────
val   ─────────
```

很可能看到类似：

```text
epoch

1        train ↓   val ↓
5        train ↓   val ↓
10       train ↓   val →
20       train ↓   val ↑
```

注意这种情况：

```text
train loss ↓↓↓↓↓

val loss ↓↓↓
          ↑↑↑
```

这才叫：

# overfitting

不是：

> “模型训练准确率很高。”

而是：

> 模型在训练数据上越来越好，但对没见过的数据并没有同步变好。

------

# 十三、实验 5：checkpoint

这一步也必须亲手经历。

不要只保存：

```python
model.state_dict()
```

为了“恢复训练”，建议：

```python
torch.save({
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "train_loss": train_loss,
    "val_loss": val_loss,
    "val_acc": val_acc,
}, "checkpoint.pt")
```

然后故意：

```text
训练 5 epoch

↓

停止程序

↓

重新启动

↓

load checkpoint

↓

从 epoch 6 接着跑
```

PyTorch 官方也建议恢复训练时同时保存 model state 和 optimizer state，因为 optimizer 本身也包含训练状态。([PyTorch Documentation](https://docs.pytorch.org/tutorials/beginner/saving_loading_models?utm_source=chatgpt.com))

你就会理解：

```text
模型文件
```

和：

```text
训练 checkpoint
```

其实不是完全一回事。

------

# 十四、第一版 optimizer 用什么？

直接：

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=1e-2
)
```

就行。

AdamW 是 Transformer/ViT 很常见的选择；PyTorch 原生支持。([PyTorch Documentation](https://docs.pytorch.org/docs/stable/optim.aliases.html?utm_source=chatgpt.com))

但是：

### Tiny Dataset Overfit 实验

先：

```python
weight_decay=0
```

甚至：

```text
dropout=0
```

因为你就是故意让模型记忆。

------

# 十五、第一轮实验千万不要加入这些东西

让 Codex **不要**给你加：

```text
AMP
mixed precision

torch.compile

gradient accumulation

gradient clipping

warmup

cosine scheduler

DDP

multi-GPU

WandB

Hydra

Lightning

HuggingFace Trainer

timm

预训练模型
```

不是因为这些东西没用。

而是：

> 它们会把你正在学习的训练主干遮住。

你的第一版代码越“笨”，反而越好。

------

# 十六、你的项目目录我建议这么设计

```text
vit-cifar10/
│
├── data/
│
├── checkpoints/
│
├── outputs/
│
│   ├── metrics.csv
│   └── curves.png
│
├── src/
│   ├── data.py
│   ├── model.py
│   ├── inspect_step.py
│   ├── train.py
│   └── evaluate.py
│
├── requirements.txt
└── README.md
```

其中最重要的甚至不是：

```text
train.py
```

而是：

```text
inspect_step.py
```

它专门做：

```text
取一个 batch
↓
forward
↓
打印 shape
↓
计算 loss
↓
打印 backward 前 grad
↓
backward
↓
打印 grad norm
↓
step
↓
比较 weight before / after
```

这个文件对你的学习价值非常高。

------

# 十七、环境只需要这些

第一轮：

```text
Python
PyTorch
torchvision
matplotlib
tqdm
```

比如：

```bash
python -m venv .venv
```

然后激活虚拟环境，再安装。

PyTorch 如果使用 NVIDIA GPU，**不要随便抄一个 CUDA 安装命令**，直接去官方安装选择器，根据 Windows/Linux、pip 和你的 GPU 环境生成当前适用命令。([PyTorch Documentation](https://docs.pytorch.org/get-started/locally/?utm_source=chatgpt.com))

[PyTorch 官方安装页面](https://pytorch.org/get-started/locally/?utm_source=chatgpt.com)

安装以后第一件事情：

```python
import torch

print(torch.__version__)
print(torch.cuda.is_available())

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
```

但其实这种 Tiny ViT：

> CPU 都能完成实验。

GPU 只是让你舒服一点。

------

# 十八、你目前应该掌握到什么程度再开始？

不需要先把所有数学补完。

你只需要保证下面这些概念大概明白。

### Python / PyTorch

至少知道：

```text
class
function
for loop
tensor
shape
```

以及：

```python
nn.Module
nn.Linear
model.parameters()
```

------

### 神经网络

知道：

```text
参数 parameter

输入 x

模型 f(x; θ)

输出 prediction

loss

gradient
```

不需要手推复杂导数。

但必须知道：

```text
θ
 ↓
forward
 ↓
prediction
 ↓
loss
 ↓
∂loss / ∂θ
 ↓
更新 θ
```

------

### Transformer

你现在已经懂基础 Transformer 的话基本够了。

ViT 前需要真正搞清：

```text
token
embedding
self-attention
multi-head attention
MLP
residual
LayerNorm
position embedding
```

ViT 新增加的核心只是：

```text
image
↓
patch
↓
patch embedding
↓
image token
```

所以 ViT 其实特别适合你作为第一次完整模型训练。

------

# 十九、让 Codex 帮你，但不要让它替你学习

这是我最想提醒你的地方。

如果你直接跟 Codex 说：

> 帮我写一个 CIFAR-10 ViT 训练项目。

它很可能十分钟给你一套漂亮代码：

```text
config
logger
trainer
scheduler
augmentation
checkpoint
mixed precision
...
```

然后你成功跑起来：

```text
Epoch 20 Acc 72.6%
```

但你的知识增长可能只有：

> 我成功训练了 ViT。

而不是：

> 我理解模型怎么被训练。

所以你应该把 Codex 当：

# 实验助手

而不是：

# 项目外包工程师

------

# 二十、我建议你第一条 Codex prompt 直接这样写

```text
我要学习神经网络训练的底层流程，而不是追求 CIFAR-10 准确率。

请和我一起从零实现一个用于 CIFAR-10 的 Tiny Vision Transformer。

学习目标是让我完整观察：

Dataset
DataLoader
batch
forward
logits
CrossEntropyLoss
loss.backward()
gradient
optimizer.step()
epoch
train/validation
overfitting
checkpoint

要求：

1. 使用 PyTorch。
2. CIFAR-10 使用 torchvision 自动下载。
3. 不使用预训练模型。
4. 不使用 timm。
5. 不使用 HuggingFace Trainer。
6. 不使用 PyTorch Lightning。
7. 不使用高级 Trainer 封装。
8. 不使用 AMP、torch.compile、DDP 等优化。
9. ViT 自己实现，但 MultiheadAttention / TransformerEncoder 等基础 PyTorch 模块可以使用。
10. 图像保持 32×32，不 resize 到 224×224。
11. patch_size=4。
12. 模型保持很小，例如 embed_dim=128、depth=4、num_heads=4。
13. 所有重要 tensor 都注明 shape。
14. 显式写出训练循环：
    optimizer.zero_grad()
    logits = model(images)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()
15. train / validation 分开实现。
16. validation 中不得 backward。
17. 保存 checkpoint，包括 model 和 optimizer state。
18. 输出 train_loss、val_loss、train_acc、val_acc 曲线。
19. 设置随机种子。
20. 代码优先考虑可读性，不考虑工程封装和性能优化。

不要一次把整个项目写完。

第一步只完成：
- 创建项目
- 安装依赖
- 下载 CIFAR-10
- 建立 DataLoader
- 取一个 batch
- 打印 image / label shape
- 展示几张图片

完成后停下来，让我先运行和理解。
```

这一句：

> **完成后停下来。**

很重要。

------

# 二十一、然后依次给 Codex 五个任务

整个过程不要超过这条路径：

```text
① Dataset / DataLoader
        ↓
② Tiny ViT forward
        ↓
③ single batch backward
        ↓
④ tiny dataset overfit
        ↓
⑤ full train / val / checkpoint
```

### Task 1

```text
只做 Dataset/DataLoader。

我要搞清楚：
dataset[i]
和
next(iter(dataloader))
分别返回什么。
```

------

### Task 2

```text
实现 TinyViT。

只测试 forward。

不要训练。

我要看到每个阶段 tensor shape。
```

------

### Task 3

```text
现在只拿一个 batch 做一次训练 step。

我要分别观察：
loss
backward 前 gradient
backward 后 gradient
optimizer.step 前 parameter
optimizer.step 后 parameter。
```

------

### Task 4

```text
现在只取 32 张训练图片。

尝试让 TinyViT 把它们完全记住。

不要数据增强。

画 loss 和 accuracy。

我要观察 overfitting。
```

------

### Task 5

```text
现在再做正式的 train / validation loop。

加入 checkpoint 和 resume training。
```

这样你几乎不可能被 Codex 带着“跳过理解阶段”。

------

# 二十二、做到什么程度，我会认为这个 ViT 实验毕业了？

不是准确率达到多少。

而是我突然问你下面任何一个问题，你能够自己解释：

> `Dataset` 和 `DataLoader` 有什么区别？

你能答。

> 一个 batch 是什么？

能答。

> `[128,3,32,32]` 是什么意思？

能答。

> ViT 为什么从 `[B,3,32,32]` 变成 `[B,64,128]`？

能答。

> logits 是什么？

能答。

> 为什么 CrossEntropyLoss 前面不 softmax？

能答。

> `loss.backward()` 修改模型了吗？

你马上说：

> 没有，它计算 gradient。

> 谁修改模型？

你说：

```python
optimizer.step()
```

> gradient 放在哪里？

你知道：

```python
parameter.grad
```

> 为什么需要 zero_grad？

你知道 PyTorch 梯度默认累积。

> 一个 epoch 是什么？

你知道是整个 train dataset 被遍历一次。

> validation 为什么没有 backward？

你知道 validation 不参与参数更新。

> overfitting 是什么？

你能看 train/val 曲线解释。

> checkpoint 为什么还要保存 optimizer？

你能解释是为了恢复训练状态。

如果这些全能回答：

**你的第一次模型训练就成功了。**

------

# 二十三、然后你会发现 CLIP 突然没那么神秘

现在你的 ViT 是：

```text
image

↓

ViT

↓

image representation

↓

Linear

↓

10 logits

↓

CrossEntropyLoss
```

下一阶段 CLIP 大概变成：

```text
                 Image
                   ↓
                  ViT
                   ↓
             image embedding
                   ↓
                 [D]
                   │
                   │ similarity
                   │
                 [D]
                   ↑
             text embedding
                   ↑
             Text Transformer
                   ↑
                  Text
```

于是原来的：

```python
classifier = Linear(D, 10)
```

消失。

变成：

```text
image embedding

和

text embedding

对齐
```

训练循环却还是：

```text
forward

↓

loss

↓

backward

↓

gradient

↓

optimizer.step
```

所以你现在学的：

```text
Dataset
DataLoader
forward
loss
backward
optimizer
epoch
checkpoint
```

**100% 继续存在。**

SigLIP 又是在这个基础上改变 image-text 配对的 loss 设计。

所以学习路线我建议就是：

```text
基础 Transformer
        ↓
Tiny ViT + CIFAR-10
        ↓
ViT patch / token / CLS 真正理解
        ↓
Tiny CLIP
        ↓
image encoder + text encoder
        ↓
contrastive learning
        ↓
CLIP
        ↓
SigLIP
        ↓
现代 VLM
```

而不是：

```text
看完 ViT 理论
↓
看 CLIP 理论
↓
看 SigLIP 理论
↓
看 VLM 理论
```

那样很容易一直处于“我好像都懂，但是让我训练一个模型就不会”的状态。

**你现在这个 CIFAR-10 Tiny ViT 实验，恰好是在跨过这道坎。**

PyTorch 官方的 beginner 教程本身也是按 `Datasets/DataLoaders → Build Model → Autograd → Optimization → Save/Load` 这条完整 workflow 来组织的，和你现在设定的目标高度一致。([PyTorch Documentation](https://docs.pytorch.org/tutorials/beginner/basics/intro?utm_source=chatgpt.com))

Codex CLI 本身也很适合这种本地迭代式实验，它就是在你的本机项目目录中运行的 coding agent。([GitHub](https://github.com/openai/codex?utm_source=chatgpt.com))

[OpenAI Codex CLI 官方仓库](https://github.com/openai/codex?utm_source=chatgpt.com)

如果按上面路线做，**第一阶段不要让 Codex 写 `train.py`。先只做到 DataLoader + 一个 batch。** 这是我最建议你从今天开始的第一步。
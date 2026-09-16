"""教学版 Tiny ViT：图像 -> patch tokens -> Transformer -> CLS -> logits。"""

import torch
from torch import nn


class TransformerBlock(nn.Module):
    """两个 Pre-LayerNorm 子层；输入和输出都是 [B,N,D]。"""

    def __init__(self, embed_dim, num_heads, mlp_ratio, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * mlp_ratio, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, tokens):
        normalized = self.norm1(tokens)                    # [B,N,D]
        attended, _ = self.attention(
            normalized, normalized, normalized, need_weights=False
        )                                                  # [B,N,D]
        tokens = tokens + self.attention_dropout(attended)  # 第一条 residual
        tokens = tokens + self.mlp(self.norm2(tokens))      # 第二条 residual
        return tokens


class TinyViT(nn.Module):
    def __init__(self, image_size=32, patch_size=4, embed_dim=128, depth=4,
                 num_heads=4, mlp_ratio=4, num_classes=10, dropout=0.0):
        super().__init__()
        dimensions = (image_size, patch_size, embed_dim, depth, num_heads, mlp_ratio, num_classes)
        if any(not isinstance(value, int) or value <= 0 for value in dimensions):
            raise ValueError('Model dimensions must be positive integers')
        if image_size % patch_size:
            raise ValueError('image_size must be divisible by patch_size')
        if embed_dim % num_heads:
            raise ValueError('embed_dim must be divisible by num_heads')
        if not 0 <= dropout < 1:
            raise ValueError('dropout must be in [0,1)')
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        # 每个 3×4×4 patch 的 48 个像素值，共用同一个可学习的 48 -> 128 投影。
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim))
        self.position_embedding = nn.Parameter(torch.empty(1, self.num_patches + 1, embed_dim))
        self.embedding_dropout = nn.Dropout(dropout)
        # 每轮调用构造器：四个不同的 block，不是把同一个模块引用四次。
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, num_classes)

        self.apply(self._initialize_module)
        nn.init.trunc_normal_(self.cls_token, std=0.02, a=-0.04, b=0.04)
        nn.init.trunc_normal_(self.position_embedding, std=0.02, a=-0.04, b=0.04)

    @staticmethod
    def _initialize_module(module):
        # 显式初始化，便于复现；不是复现原论文的全部训练细节。
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(module.weight, std=0.02, a=-0.04, b=0.04)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            # PyTorch 将 Q/K/V 投影合并存入 in_proj_weight。
            nn.init.trunc_normal_(module.in_proj_weight, std=0.02, a=-0.04, b=0.04)
            nn.init.zeros_(module.in_proj_bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, images, *, trace=None):
        """始终返回 logits；可选 trace 字典只记录 shape，不保留中间张量。"""
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, self.image_size, self.image_size):
            raise ValueError(f'Expected [B,3,{self.image_size},{self.image_size}] images')

        def record(name, tensor):
            if trace is not None:
                trace[name] = list(tensor.shape)

        record('input', images)                             # [B,3,32,32]
        patches = self.patch_embed(images)
        record('patch_embedding', patches)                  # [B,128,8,8]
        tokens = patches.flatten(2).transpose(1, 2)
        record('patch_tokens', tokens)                      # [B,64,128]
        cls = self.cls_token.expand(images.shape[0], -1, -1)
        tokens = torch.cat((cls, tokens), dim=1)
        record('with_cls', tokens)                          # [B,65,128]
        tokens = self.embedding_dropout(tokens + self.position_embedding)
        record('with_position', tokens)                     # [B,65,128]
        for index, block in enumerate(self.blocks, start=1):
            tokens = block(tokens)
            record(f'block_{index}', tokens)                # [B,65,128]
        tokens = self.norm(tokens)
        record('final_norm', tokens)                        # [B,65,128]
        image_representation = tokens[:, 0]
        record('cls_representation', image_representation)  # [B,128]
        logits = self.head(image_representation)
        record('logits', logits)                            # [B,10]，未做 softmax
        return logits

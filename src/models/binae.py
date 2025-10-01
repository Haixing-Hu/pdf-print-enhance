#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BinModel - 二值化增强模型

基于 Vision Transformer 的文档图像增强模型
原始代码来自: https://github.com/dali92002/DocEnTR
"""

import torch
from torch import nn
import torch.nn.functional as F
from vit_pytorch.vit import Transformer


class BinModel(nn.Module):
    """
    自编码器模型，用于以图像到图像翻译方式增强图像。
    此代码基于 vit-pytorch: https://github.com/lucidrains/vit-pytorch

    Args:
        encoder (model): 定义的编码器，这里是 ViT
        decoder_dim (int): 解码器维度（嵌入大小）
        decoder_depth (int): 解码器层数
        decoder_heads (int): 解码器注意力头数
        decoder_dim_head (int): 解码器头维度
    """

    def __init__(
        self,
        *,
        encoder,
        decoder_dim,
        decoder_depth=1,
        decoder_heads=8,
        decoder_dim_head=64
    ):
        super().__init__()
        # 从 ViT 编码器提取超参数和函数
        self.encoder = encoder
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]
        self.to_patch, self.patch_to_emb = encoder.to_patch_embedding[:2]
        pixel_values_per_patch = self.patch_to_emb.weight.shape[-1]

        # 定义解码器
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder = Transformer(
            dim=decoder_dim,
            depth=decoder_depth,
            heads=decoder_heads,
            dim_head=decoder_dim_head,
            mlp_dim=decoder_dim * 4
        )
        self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)
        self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)

    def forward(self, img, gt_img):
        """
        前向传播

        Args:
            img: 输入图像
            gt_img: ground truth 图像（训练时使用，推理时可以是虚拟值）

        Returns:
            loss: 损失值
            patches: 输入图像的补丁
            pred_pixel_values: 预测的像素值
        """
        # 获取补丁及其数量
        patches = self.to_patch(img)
        _, num_patches, *_ = patches.shape

        # 将像素补丁投影到 tokens 并添加位置编码
        tokens = self.patch_to_emb(patches)
        tokens = tokens + self.encoder.pos_embedding[:, 1:(num_patches + 1)]

        # 通过编码器编码 tokens
        encoded_tokens = self.encoder.transformer(tokens)

        # 如果编码器和解码器维度不同，进行投影
        decoder_tokens = self.enc_to_dec(encoded_tokens)

        # 通过解码器解码 tokens
        decoded_tokens = self.decoder(decoder_tokens)

        # 将 tokens 投影到像素
        pred_pixel_values = self.to_pixels(decoded_tokens)

        # 计算与 ground truth 的损失
        gt_patches = self.to_patch(gt_img)
        loss = F.mse_loss(pred_pixel_values, gt_patches)

        return loss, patches, pred_pixel_values


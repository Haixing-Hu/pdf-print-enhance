#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DocEnTR 图像增强器

基于 DocEnTR 模型的文档图像增强实现
"""

import os
from pathlib import Path
from typing import Optional, Union
import logging

import numpy as np
import torch
import cv2
from tqdm import tqdm
from vit_pytorch import ViT
from einops import rearrange

from ..models import BinModel
from .image_utils import (
    split_image,
    merge_patches,
    pad_image,
    normalize_patch,
    denormalize_patch,
    binarize_image
)

logger = logging.getLogger(__name__)


class DocEnTREnhancer:
    """DocEnTR 文档图像增强器"""

    # 模型配置
    MODEL_CONFIGS = {
        'base': {
            'ENCODERLAYERS': 6,
            'ENCODERHEADS': 8,
            'ENCODERDIM': 768,
            'patch_size': 8,
        },
        'large': {
            'ENCODERLAYERS': 12,
            'ENCODERHEADS': 16,
            'ENCODERDIM': 1024,
            'patch_size': 16,
        },
    }

    # 默认参数
    DEFAULT_PATCH_SIZE = 256
    DEFAULT_THRESHOLD = 0.5
    DEFAULT_MODEL_SIZE = 'base'

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_size: str = 'base',
        device: Optional[Union[str, torch.device]] = None,
        patch_size: int = DEFAULT_PATCH_SIZE,
        threshold: float = DEFAULT_THRESHOLD,
        verbose: bool = False,
        logger: Optional[logging.Logger] = None
    ):
        """
        初始化增强器

        Args:
            model_path: 模型权重路径
            model_size: 模型大小 ('base' 或 'large')
            device: 计算设备 ('cpu', 'cuda' 或 torch.device 对象)
            patch_size: 图像块大小（默认 256）
            threshold: 二值化阈值（默认 0.5）
            verbose: 是否显示详细输出
            logger: 日志记录器
        """
        if model_size not in self.MODEL_CONFIGS:
            raise ValueError(f"不支持的模型大小: {model_size}，支持: {list(self.MODEL_CONFIGS.keys())}")

        self.model_size = model_size
        self.patch_size = patch_size
        self.threshold = threshold
        self.verbose = verbose
        self.logger = logger or logging.getLogger(__name__)

        # 设置日志级别
        if verbose:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)

        # 确定设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        # 创建模型
        self._create_model()

        # 加载权重
        if model_path and os.path.exists(model_path):
            self._load_weights(model_path)
        else:
            if model_path:
                self.logger.warning(f"模型权重文件不存在: {model_path}")
            self.logger.warning("使用随机初始化的权重（效果会很差）")
            self.logger.warning("请下载预训练权重以获得最佳效果")

        self.model.eval()

    def _create_model(self):
        """创建模型"""
        config = self.MODEL_CONFIGS[self.model_size]

        self.logger.info(f"正在初始化 DocEnTR-{self.model_size.upper()} 模型...")
        self.logger.info(f"  设备: {self.device}")
        self.logger.info(f"  图像块大小: {self.patch_size}x{self.patch_size}")
        self.logger.info(f"  二值化阈值: {self.threshold}")

        # 创建 ViT 编码器
        vit = ViT(
            image_size=(self.patch_size, self.patch_size),
            patch_size=config['patch_size'],
            num_classes=1000,
            dim=config['ENCODERDIM'],
            depth=config['ENCODERLAYERS'],
            heads=config['ENCODERHEADS'],
            mlp_dim=2048,
        )

        # 创建 BinModel
        self.model = BinModel(
            encoder=vit,
            decoder_dim=config['ENCODERDIM'],
            decoder_depth=config['ENCODERLAYERS'],
            decoder_heads=config['ENCODERHEADS'],
        )

        self.model.to(self.device)

    def _load_weights(self, model_path: str):
        """加载模型权重"""
        self.logger.info(f"正在加载模型权重: {model_path}")
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.logger.info("✓ 模型权重加载成功")
        except Exception as e:
            self.logger.error(f"✗ 加载模型权重失败: {e}")
            raise

    def enhance(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        show_progress: bool = True
    ) -> None:
        """
        增强单张图像

        Args:
            input_path: 输入图像路径
            output_path: 输出图像路径
            show_progress: 是否显示进度条
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            raise FileNotFoundError(f"输入图像不存在: {input_path}")

        self.logger.info(f"输入: {input_path}")
        self.logger.info(f"输出: {output_path}")

        # 创建输出目录
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 读取图像
        image = cv2.imread(str(input_path)) / 255.0
        if image is None:
            raise ValueError(f"无法读取图像: {input_path}")

        orig_h, orig_w = image.shape[0], image.shape[1]
        self.logger.info(f"原始尺寸: {orig_w}x{orig_h}")

        # 填充图像
        padded_image, (orig_h, orig_w) = pad_image(image, self.patch_size)
        h, w = padded_image.shape[0], padded_image.shape[1]

        # 分割图像
        patches = split_image(padded_image, self.patch_size)
        num_patches = len(patches)
        self.logger.info(f"分割成 {num_patches} 个 {self.patch_size}x{self.patch_size} 的块")

        # 预处理
        self.logger.debug("正在预处理图像块...")
        preprocessed_patches = [normalize_patch(p) for p in patches]

        # 处理每个块
        self.logger.info("正在增强图像...")
        result_patches = []

        config = self.MODEL_CONFIGS[self.model_size]
        patch_size_model = config['patch_size']

        with torch.no_grad():
            iterator = tqdm(
                preprocessed_patches,
                desc="增强进度",
                disable=not show_progress or self.verbose
            )

            for patch in iterator:
                # 转换为 tensor
                patch_tensor = torch.from_numpy(patch.astype('float32'))
                patch_tensor = patch_tensor.view(1, 3, self.patch_size, self.patch_size).to(self.device)

                # 创建虚拟 ground truth（推理时不使用）
                dummy_gt = torch.rand_like(patch_tensor).to(self.device)

                # 模型推理
                _, _, pred_pixel_values = self.model(patch_tensor, dummy_gt)

                # 重建图像块
                rec_image = torch.squeeze(
                    rearrange(
                        pred_pixel_values,
                        'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                        p1=patch_size_model,
                        p2=patch_size_model,
                        h=self.patch_size // patch_size_model,
                    )
                )

                # 转换为 numpy
                output = rec_image.cpu().numpy()
                output = np.transpose(output, (1, 2, 0))

                # 反归一化
                output = denormalize_patch(output)
                result_patches.append(output)

        # 合并图像块
        self.logger.info("正在合并图像块...")
        clean_image = merge_patches(result_patches, h, w, self.patch_size)

        # 裁剪到原始尺寸
        clean_image = clean_image[:orig_h, :orig_w, :]

        # 二值化
        clean_image = binarize_image(clean_image, self.threshold)

        # 保存
        cv2.imwrite(str(output_path), clean_image.astype(np.uint8))
        self.logger.info(f"✓ 已保存到: {output_path}")

    @staticmethod
    def find_model_weights(model_size: str = 'base', models_dir: Optional[Union[str, Path]] = None) -> Optional[str]:
        """
        自动查找模型权重文件

        Args:
            model_size: 模型大小（'base' 或 'large'）
            models_dir: 模型目录（默认为项目根目录的 models 文件夹）

        Returns:
            模型权重路径或 None
        """
        if models_dir is None:
            # 默认查找项目根目录的 models 文件夹
            script_dir = Path(__file__).parent.parent.parent
            models_dir = script_dir / 'models'
        else:
            models_dir = Path(models_dir)

        if not models_dir.exists():
            return None

        # 查找匹配的权重文件
        patterns = {
            'base': ['*base*8*.pt*', '*hdibco*base*8*.pt*'],
            'large': ['*large*16*.pt*', '*hdibco*large*16*.pt*']
        }

        for pattern in patterns.get(model_size, []):
            matches = list(models_dir.glob(pattern))
            if matches:
                return str(matches[0])

        return None


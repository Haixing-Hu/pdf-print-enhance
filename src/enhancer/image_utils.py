#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像处理工具

提供图像分割、合并、归一化等功能
"""

import numpy as np
from typing import List


# ImageNet 归一化参数
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def split_image(image: np.ndarray, patch_size: int = 256) -> List[np.ndarray]:
    """
    将图像分割成固定大小的块

    Args:
        image: 输入图像 [H, W, C]
        patch_size: 块的大小

    Returns:
        图像块列表
    """
    patches = []
    h, w = image.shape[0], image.shape[1]

    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch = image[i:i + patch_size, j:j + patch_size, :]
            patches.append(patch)

    return patches


def merge_patches(patches: List[np.ndarray], height: int, width: int, patch_size: int = 256) -> np.ndarray:
    """
    将图像块合并成完整图像

    Args:
        patches: 图像块列表
        height: 目标图像高度
        width: 目标图像宽度
        patch_size: 块的大小

    Returns:
        合并后的图像 [H, W, C]
    """
    image = np.zeros((height, width, 3))
    idx = 0

    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            if idx < len(patches):
                image[i:i + patch_size, j:j + patch_size, :] = patches[idx]
                idx += 1

    return image


def pad_image(image: np.ndarray, patch_size: int = 256) -> tuple[np.ndarray, tuple[int, int]]:
    """
    填充图像使其尺寸可被 patch_size 整除

    Args:
        image: 输入图像 [H, W, C]
        patch_size: 块的大小

    Returns:
        (填充后的图像, (原始高度, 原始宽度))
    """
    orig_h, orig_w = image.shape[0], image.shape[1]

    # 计算填充后的尺寸
    h = ((orig_h // patch_size) + 1) * patch_size
    w = ((orig_w // patch_size) + 1) * patch_size

    # 创建填充图像（白色背景）
    padded_image = np.ones((h, w, 3))
    padded_image[:orig_h, :orig_w, :] = image

    return padded_image, (orig_h, orig_w)


def normalize_patch(patch: np.ndarray, mean: List[float] = None, std: List[float] = None) -> np.ndarray:
    """
    归一化图像块

    Args:
        patch: 输入图像块 [H, W, 3]
        mean: 均值列表（默认使用 ImageNet 均值）
        std: 标准差列表（默认使用 ImageNet 标准差）

    Returns:
        归一化后的图像块 [3, H, W]
    """
    if mean is None:
        mean = IMAGENET_MEAN
    if std is None:
        std = IMAGENET_STD

    normalized = np.zeros([3, *patch.shape[:-1]])
    for i in range(3):
        normalized[i] = (patch[:, :, i] - mean[i]) / std[i]

    return normalized


def denormalize_patch(patch: np.ndarray, mean: List[float] = None, std: List[float] = None) -> np.ndarray:
    """
    反归一化图像块

    Args:
        patch: 归一化的图像块 [3, H, W]
        mean: 均值列表（默认使用 ImageNet 均值）
        std: 标准差列表（默认使用 ImageNet 标准差）

    Returns:
        反归一化后的图像块 [H, W, 3]
    """
    if mean is None:
        mean = IMAGENET_MEAN
    if std is None:
        std = IMAGENET_STD

    result = patch.copy()
    for ch in range(3):
        result[:, :, ch] = (result[:, :, ch] * std[ch]) + mean[ch]

    result = np.clip(result, 0, 1)
    return result


def binarize_image(image: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    二值化图像

    Args:
        image: 输入图像 [H, W, C]，值范围 [0, 1]
        threshold: 二值化阈值

    Returns:
        二值化后的图像，值范围 [0, 255]
    """
    return (image > threshold) * 255


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型权重下载脚本

此脚本用于下载文档增强模型的预训练权重。
由于真实的 DocEnTR 模型权重可能需要从特定来源获取，
这里提供了一个通用的下载框架，可以根据实际情况修改下载链接。

使用方法:
    python download_model.py
"""

import os
import sys
import requests
from tqdm import tqdm
import hashlib


# 模型配置
# 来源: https://github.com/dali92002/DocEnTR
MODEL_CONFIGS = {
    'docentr_hdibco2012_base': {
        'name': 'DocEnTR-Base (H-DIBCO 2012) - Patch 8x8',
        'url': 'https://drive.google.com/uc?id=1ZOjnqxeg2620x4qLeqHCCmF3wozPnSDv',
        'filename': 'docentr_hdibco2012_base_8x8.pth',
        'md5': None,
        'psnr': 22.29,
    },
    'docentr_hdibco2012_large': {
        'name': 'DocEnTR-Large (H-DIBCO 2012) - Patch 16x16',
        'url': 'https://drive.google.com/uc?id=1h1bdMg7fvoQv4N5dY9T03c92_ViCMlSM',
        'filename': 'docentr_hdibco2012_large_16x16.pth',
        'md5': None,
        'psnr': 22.04,
    },
    'docentr_dibco2017_base': {
        'name': 'DocEnTR-Base (DIBCO 2017) - Patch 8x8',
        'url': 'https://drive.google.com/uc?id=1zz0aFPNFctjNTVRpng4Lh-1X-Ms6Zroa',
        'filename': 'docentr_dibco2017_base_8x8.pth',
        'md5': None,
        'psnr': 19.11,
    },
    'docentr_dibco2017_large': {
        'name': 'DocEnTR-Large (DIBCO 2017) - Patch 16x16',
        'url': 'https://drive.google.com/uc?id=1Qz8um2nwAMla2AgRnaKEc2JNQnb1w2nH',
        'filename': 'docentr_dibco2017_large_16x16.pth',
        'md5': None,
        'psnr': 18.85,
    },
    'docentr_hdibco2018_base': {
        'name': 'DocEnTR-Base (H-DIBCO 2018) - Patch 8x8',
        'url': 'https://drive.google.com/uc?id=1CpvS9ahZolRz2sJ4PHobofOXVJBMWwD0',
        'filename': 'docentr_hdibco2018_base_8x8.pth',
        'md5': None,
        'psnr': 19.46,
    },
    'docentr_hdibco2018_large': {
        'name': 'DocEnTR-Large (H-DIBCO 2018) - Patch 16x16',
        'url': 'https://drive.google.com/uc?id=1uIzdGMGshX-sxCdWU7CwRE7CJ4b4z5o-',
        'filename': 'docentr_hdibco2018_large_16x16.pth',
        'md5': None,
        'psnr': 19.47,
    },
}

# 模型存储目录
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')


def calculate_md5(file_path, chunk_size=8192):
    """计算文件的 MD5 值"""
    md5_hash = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def download_file_from_google_drive(file_id, dest_path):
    """
    从 Google Drive 下载文件

    Args:
        file_id: Google Drive 文件 ID
        dest_path: 目标保存路径

    Returns:
        bool: 下载是否成功
    """
    try:
        print(f"正在从 Google Drive 下载 (ID: {file_id})...")

        # 创建目标目录
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        # 使用 gdown 库下载（更可靠）
        try:
            import gdown
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, dest_path, quiet=False)
            return True
        except ImportError:
            print("提示: 安装 gdown 可以更可靠地下载 Google Drive 文件")
            print("      运行: pip install gdown")
            print("正在使用备用方法下载...")

        # 备用方法：直接使用 requests
        session = requests.Session()

        # 第一次请求
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        response = session.get(url, stream=True, timeout=30)

        # 检查是否有病毒扫描警告
        token = None
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                token = value
                break

        # 如果有警告，获取确认 token
        if token:
            params = {'id': file_id, 'confirm': token, 'export': 'download'}
            url = "https://drive.google.com/uc"
            response = session.get(url, params=params, stream=True, timeout=60)

        # 如果还是没有真实文件，尝试从响应中提取确认链接
        if response.headers.get('content-type', '').startswith('text/html'):
            # 从 HTML 中提取下载链接
            import re
            content = response.content.decode('utf-8')
            match = re.search(r'href="/uc\?export=download&amp;confirm=([^&]+)&amp;id=' + file_id, content)
            if match:
                confirm = match.group(1)
                params = {'id': file_id, 'confirm': confirm, 'export': 'download'}
                url = "https://drive.google.com/uc"
                response = session.get(url, params=params, stream=True, timeout=60)

        response.raise_for_status()

        # 获取文件大小
        total_size = int(response.headers.get('content-length', 0))

        if total_size == 0:
            print("⚠️  警告: 无法获取文件大小，可能下载失败")
            print("建议:")
            print("  1. 安装 gdown: pip install gdown")
            print("  2. 或手动下载模型文件到 models/ 目录")
            return False

        # 下载文件并显示进度条
        downloaded = 0
        with open(dest_path, 'wb') as f, tqdm(
            desc=os.path.basename(dest_path),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for chunk in response.iter_content(chunk_size=32768):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    progress_bar.update(len(chunk))

        # 验证下载的文件大小
        actual_size = os.path.getsize(dest_path)
        if actual_size < 1024 * 1024:  # 小于 1MB
            print(f"⚠️  警告: 下载的文件太小 ({actual_size} bytes)，可能下载失败")
            print("建议:")
            print("  1. 安装 gdown: pip install gdown")
            print("  2. 或手动下载: " + f"https://drive.google.com/file/d/{file_id}/view")
            os.remove(dest_path)
            return False

        return True

    except requests.exceptions.RequestException as e:
        print(f"下载失败: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False
    except Exception as e:
        print(f"发生错误: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False


def download_file(url, dest_path, expected_md5=None):
    """
    下载文件并显示进度条

    Args:
        url: 下载链接
        dest_path: 目标保存路径
        expected_md5: 期望的 MD5 值，用于校验

    Returns:
        bool: 下载是否成功
    """
    try:
        # 检查是否是 Google Drive 链接
        if 'drive.google.com' in url:
            # 提取文件 ID
            if 'id=' in url:
                file_id = url.split('id=')[1].split('&')[0]
            else:
                print("错误: 无法解析 Google Drive 文件 ID")
                return False

            success = download_file_from_google_drive(file_id, dest_path)
        else:
            # 普通 HTTP 下载
            print(f"正在从 {url} 下载...")

            # 发送 HEAD 请求获取文件大小
            response = requests.head(url, allow_redirects=True, timeout=10)
            total_size = int(response.headers.get('content-length', 0))

            # 开始下载
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            # 创建目标目录
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)

            # 下载文件并显示进度条
            with open(dest_path, 'wb') as f, tqdm(
                desc=os.path.basename(dest_path),
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as progress_bar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        progress_bar.update(len(chunk))

            success = True

        # 验证 MD5
        if success and expected_md5:
            print("正在验证文件完整性...")
            actual_md5 = calculate_md5(dest_path)
            if actual_md5.lower() != expected_md5.lower():
                print(f"错误: MD5 校验失败!")
                print(f"  期望值: {expected_md5}")
                print(f"  实际值: {actual_md5}")
                os.remove(dest_path)
                return False
            print("文件校验通过 ✓")

        return success

    except requests.exceptions.RequestException as e:
        print(f"下载失败: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False
    except Exception as e:
        print(f"发生错误: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False


def download_model(model_key):
    """
    下载指定的模型

    Args:
        model_key: 模型配置的键名

    Returns:
        bool: 是否成功下载
    """
    if model_key not in MODEL_CONFIGS:
        print(f"错误: 未找到模型配置 '{model_key}'")
        print(f"可用的模型: {', '.join(MODEL_CONFIGS.keys())}")
        return False

    config = MODEL_CONFIGS[model_key]
    dest_path = os.path.join(MODELS_DIR, config['filename'])

    # 检查文件是否已存在
    if os.path.exists(dest_path):
        print(f"模型文件已存在: {dest_path}")

        # 如果有 MD5，验证现有文件
        if config['md5']:
            print("正在验证现有文件...")
            actual_md5 = calculate_md5(dest_path)
            if actual_md5.lower() == config['md5'].lower():
                print("现有文件校验通过 ✓")
                return True
            else:
                print("现有文件已损坏，将重新下载...")
                os.remove(dest_path)
        else:
            response = input("是否重新下载? (y/N): ")
            if response.lower() != 'y':
                return True

    # 下载模型
    print(f"\n正在下载 {config['name']}...")
    success = download_file(config['url'], dest_path, config['md5'])

    if success:
        print(f"✓ 模型已成功下载到: {dest_path}")
        file_size = os.path.getsize(dest_path) / (1024 * 1024)  # MB
        print(f"  文件大小: {file_size:.2f} MB")
        return True
    else:
        print(f"✗ 模型下载失败")
        return False


def main():
    """主函数"""
    print("=" * 60)
    print("文档增强模型下载工具")
    print("=" * 60)
    print()

    # 创建模型目录
    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"模型存储目录: {MODELS_DIR}")
    print()

    # 显示提示信息
    print("✨ DocEnTR 模型下载工具")
    print()
    print("📚 模型来源: https://github.com/dali92002/DocEnTR")
    print("   论文: DocEnTr: An End-to-End Document Image Enhancement Transformer")
    print("   会议: ICPR 2022")
    print()
    print("📌 可用模型:")
    print("   本工具提供 6 个预训练的 DocEnTR 模型，分别在不同的")
    print("   DIBCO 数据集上训练，适用于文档图像增强（二值化）任务。")
    print()
    print("   - Base 模型: 使用 8x8 patch size")
    print("   - Large 模型: 使用 16x16 patch size")
    print()
    print("⚠️  注意事项:")
    print("   1. 这些模型是基于 Transformer 架构，与本项目的 U-Net 不兼容")
    print("   2. 要使用这些模型，需要克隆 DocEnTR 的代码:")
    print("      git clone https://github.com/dali92002/DocEnTR")
    print("   3. 下载的权重文件应配合 DocEnTR 的代码使用")
    print()
    print("💡 推荐选择:")
    print("   对于一般文档增强任务，推荐下载 H-DIBCO 2012 的模型，")
    print("   因为它的 PSNR 最高（22.29 for Base, 22.04 for Large）")
    print()

    # 列出所有可用的模型
    print("可下载的模型:")
    for i, (key, config) in enumerate(MODEL_CONFIGS.items(), 1):
        status = "✓" if os.path.exists(os.path.join(MODELS_DIR, config['filename'])) else "✗"
        psnr_info = f"PSNR: {config['psnr']}" if 'psnr' in config else ""
        print(f"  {i}. [{status}] {config['name']}")
        if psnr_info:
            print(f"      {psnr_info}")
    print()

    # 询问用户要下载哪个模型
    choice = input("请选择要下载的模型编号 (1-{}, 或按 Enter 下载所有): ".format(len(MODEL_CONFIGS)))

    if choice.strip() == '':
        # 下载所有模型
        print("\n开始下载所有模型...")
        success_count = 0
        for key in MODEL_CONFIGS.keys():
            if download_model(key):
                success_count += 1
            print()

        print(f"完成! 成功下载 {success_count}/{len(MODEL_CONFIGS)} 个模型")
    else:
        # 下载指定模型
        try:
            index = int(choice) - 1
            model_keys = list(MODEL_CONFIGS.keys())
            if 0 <= index < len(model_keys):
                download_model(model_keys[index])
            else:
                print("错误: 无效的选择")
                sys.exit(1)
        except ValueError:
            print("错误: 请输入有效的数字")
            sys.exit(1)


if __name__ == '__main__':
    main()


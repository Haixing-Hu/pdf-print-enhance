#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pdf_enhance.py - PDF 文档增强主程序

完整的 PDF 增强工作流：
1. 提取 PDF 每一页为图片
2. 使用 DocEnTR 模型增强每一页
3. 将增强后的页面合并为新的 PDF

作者: PDF Print Enhance
版本: 1.0.0
"""

import os
import sys
import argparse
import time
from datetime import datetime
import tempfile
import shutil
from pathlib import Path
from typing import Optional, List
import logging

try:
    from PIL import Image
except ImportError:
    print("错误: 需要安装 Pillow 库")
    print("安装命令: pip install Pillow")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("错误: 需要安装 tqdm 库")
    print("安装命令: pip install tqdm")
    sys.exit(1)

# 导入本地模块
from .pdf_to_images import pdf_to_images, get_pdf_page_count
from .enhancer import DocEnTREnhancer

__version__ = "1.0.0"

# 配置日志
def setup_logging(log_file: Optional[str] = None) -> logging.Logger:
    """
    设置日志系统

    Args:
        log_file: 日志文件路径（可选）

    Returns:
        配置好的 logger
    """
    logger = logging.getLogger('pdf_enhance')
    logger.setLevel(logging.DEBUG)

    # 清除现有的 handlers
    logger.handlers.clear()

    # 控制台输出（只显示 INFO 及以上级别）
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # 文件输出（显示所有级别，包括 DEBUG）
    if log_file is None:
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'pdf_enhance_{timestamp}.log'

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    logger.info(f"日志文件: {log_file}")

    return logger


class PDFEnhancer:
    """PDF 文档增强器"""

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_size: str = 'base',
        dpi: int = 300,
        device: Optional[str] = None,
        verbose: bool = False,
        logger: Optional[logging.Logger] = None
    ):
        """
        初始化 PDF 增强器

        Args:
            model_path: DocEnTR 模型权重路径
            model_size: 模型大小 ('base' 或 'large')
            dpi: PDF 转图片的分辨率
            device: 计算设备 ('cpu' 或 'cuda')
            verbose: 是否显示详细输出
            logger: 日志记录器
        """
        self.model_size = model_size
        self.dpi = dpi
        self.verbose = verbose
        self.logger = logger or logging.getLogger('pdf_enhance')

        # 如果未指定模型路径，尝试自动查找
        if model_path is None:
            model_path = DocEnTREnhancer.find_model_weights(model_size)
            if model_path:
                self.logger.info(f"✓ 自动找到模型: {Path(model_path).name}")
            else:
                self.logger.warning(f"⚠️  未找到 {model_size} 模型权重")
                self.logger.warning("   将使用随机初始化的权重（效果会很差）")
                self.logger.warning("   请运行: python download_model.py")

        # 创建图像增强器
        self.enhancer = DocEnTREnhancer(
            model_path=model_path,
            model_size=model_size,
            device=device,
            verbose=verbose,
            logger=self.logger
        )

    def enhance_pdf(
        self,
        input_pdf: str,
        output_pdf: Optional[str] = None,
        temp_dir: Optional[str] = None,
        keep_temp: bool = False
    ) -> str:
        """
        增强 PDF 文件

        Args:
            input_pdf: 输入 PDF 文件路径
            output_pdf: 输出 PDF 文件路径（可选，默认为输入文件名 + _enhanced）
            temp_dir: 临时文件目录（可选，默认使用系统临时目录）
            keep_temp: 是否保留临时文件

        Returns:
            输出 PDF 文件路径

        Raises:
            FileNotFoundError: 输入文件不存在
            ValueError: 输入参数无效
            RuntimeError: 处理过程中出错
        """
        input_pdf = Path(input_pdf)
        if not input_pdf.exists():
            raise FileNotFoundError(f"输入 PDF 文件不存在: {input_pdf}")

        if not input_pdf.is_file():
            raise ValueError(f"不是有效的文件: {input_pdf}")

        if input_pdf.suffix.lower() != '.pdf':
            raise ValueError(f"输入文件必须是 PDF 格式: {input_pdf}")

        # 确定输出文件路径
        if output_pdf is None:
            output_pdf = input_pdf.parent / f"{input_pdf.stem}_enhanced.pdf"
        else:
            output_pdf = Path(output_pdf)

        # 创建输出目录
        output_pdf.parent.mkdir(parents=True, exist_ok=True)

        # 打印处理信息
        print("=" * 70)
        print(f"输入文件: {input_pdf}")
        print(f"输出文件: {output_pdf}")
        print(f"分辨率: {self.dpi} DPI")
        print(f"模型: DocEnTR-{self.model_size.upper()}")
        print("=" * 70)

        # 获取 PDF 页数
        try:
            total_pages = get_pdf_page_count(str(input_pdf))
            print(f"PDF 页数: {total_pages}")
        except Exception as e:
            raise RuntimeError(f"无法读取 PDF 信息: {e}")

        # 记录开始时间
        start_time = time.time()
        self.self.logger.debug(f"开始处理 PDF: {input_pdf}")
        self.self.logger.debug(f"输出文件: {output_pdf}")

        # 创建临时目录
        if temp_dir is None:
            temp_dir = Path(tempfile.mkdtemp(prefix="pdf_enhance_"))
            cleanup_temp = not keep_temp
        else:
            temp_dir = Path(temp_dir)
            temp_dir.mkdir(parents=True, exist_ok=True)
            cleanup_temp = False

        try:
            self.logger.info(f"临时目录: {temp_dir}")
            self.logger.debug(f"清理临时文件: {'是' if cleanup_temp else '否'}")
            print()

            # 步骤 1: 将 PDF 转换为图片
            print("步骤 1/3: 提取 PDF 页面为图片")
            print("-" * 70)
            self.logger.info("=" * 50)
            self.logger.info("步骤 1: 提取 PDF 页面")
            self.logger.info("=" * 50)

            step1_start = time.time()
            original_dir = temp_dir / "original"
            original_dir.mkdir(exist_ok=True)
            self.logger.debug(f"原始图片目录: {original_dir}")

            original_images = pdf_to_images(
                pdf_path=str(input_pdf),
                output_dir=str(original_dir),
                dpi=self.dpi,
                quality=95,
                fmt='png',
                verbose=self.verbose,
                logger=self.logger
            )

            if not original_images:
                raise RuntimeError("未能提取任何图片")

            step1_time = time.time() - step1_start
            step1_avg = step1_time / len(original_images) if original_images else 0

            print(f"✓ 成功提取 {len(original_images)} 页")
            print(f"  耗时: {step1_time:.2f}秒, 平均每页: {step1_avg:.2f}秒\n")

            self.logger.info(f"提取完成: {len(original_images)} 页")
            self.logger.info(f"步骤 1 耗时: {step1_time:.2f}秒")
            self.logger.info(f"步骤 1 平均每页: {step1_avg:.2f}秒")

            # 步骤 2: 增强每一页
            print("步骤 2/3: 增强每一页图像")
            print("-" * 70)
            self.logger.info("=" * 50)
            self.logger.info("步骤 2: 增强图像")
            self.logger.info("=" * 50)

            step2_start = time.time()
            enhanced_dir = temp_dir / "enhanced"
            enhanced_dir.mkdir(exist_ok=True)
            self.logger.debug(f"增强图片目录: {enhanced_dir}")

            enhanced_images = []
            failed_pages = []

            # 使用进度条
            with tqdm(total=len(original_images), desc="增强进度", unit="页", ncols=80) as pbar:
                for i, img_path in enumerate(original_images):
                    img_path = Path(img_path)
                    output_path = enhanced_dir / f"page_{i+1:04d}_enhanced.png"
                    page_start = time.time()

                    try:
                        self.logger.debug(f"增强第 {i+1} 页: {img_path.name}")
                        self.enhancer.enhance(
                            input_path=img_path,
                            output_path=output_path,
                            show_progress=False
                        )
                        enhanced_images.append(str(output_path))
                        page_time = time.time() - page_start
                        self.logger.debug(f"  第 {i+1} 页完成，耗时: {page_time:.2f}秒")

                    except Exception as e:
                        self.logger.error(f"✗ 增强第 {i+1} 页失败: {e}")
                        self.logger.warning(f"  使用原始图片: {img_path.name}")
                        enhanced_images.append(str(img_path))
                        failed_pages.append(i+1)

                    pbar.update(1)

            step2_time = time.time() - step2_start
            step2_avg = step2_time / len(original_images) if original_images else 0

            print(f"✓ 成功增强 {len(enhanced_images) - len(failed_pages)}/{len(enhanced_images)} 页")
            if failed_pages:
                print(f"  失败页面: {failed_pages}")
            print(f"  耗时: {step2_time:.2f}秒, 平均每页: {step2_avg:.2f}秒\n")

            self.logger.info(f"增强完成: {len(enhanced_images) - len(failed_pages)}/{len(enhanced_images)} 页成功")
            if failed_pages:
                self.logger.warning(f"失败页面: {failed_pages}")
            self.logger.info(f"步骤 2 耗时: {step2_time:.2f}秒")
            self.logger.info(f"步骤 2 平均每页: {step2_avg:.2f}秒")

            # 步骤 3: 合并为 PDF
            print("步骤 3/3: 合并页面为 PDF")
            print("-" * 70)
            self.logger.info("=" * 50)
            self.logger.info("步骤 3: 合并为 PDF")
            self.logger.info("=" * 50)

            step3_start = time.time()

            # 使用进度条显示合并进度
            with tqdm(total=1, desc="合并进度", unit="文件", ncols=80) as pbar:
                self._images_to_pdf(enhanced_images, str(output_pdf))
                pbar.update(1)

            step3_time = time.time() - step3_start

            print(f"✓ PDF 已保存到: {output_pdf}")
            print(f"  耗时: {step3_time:.2f}秒\n")

            self.logger.info(f"合并完成: {output_pdf}")
            self.logger.info(f"步骤 3 耗时: {step3_time:.2f}秒")

            # 计算总耗时
            end_time = time.time()
            total_time = end_time - start_time

            # 打印统计信息
            print("=" * 70)
            print("✓ 处理完成！")
            print("=" * 70)
            print(f"总页数: {len(enhanced_images)}")
            print(f"成功增强: {len(enhanced_images) - len(failed_pages)} 页")
            if failed_pages:
                print(f"失败页面: {len(failed_pages)} 页")
            print()
            print("各步骤耗时:")
            print(f"  步骤 1 (提取): {step1_time:.2f}秒 (平均每页: {step1_avg:.2f}秒)")
            print(f"  步骤 2 (增强): {step2_time:.2f}秒 (平均每页: {step2_avg:.2f}秒)")
            print(f"  步骤 3 (合并): {step3_time:.2f}秒")
            print()
            print(f"总耗时: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")
            print(f"平均每页: {total_time/len(enhanced_images):.2f}秒")
            print()
            print(f"输出文件: {output_pdf}")

            # 文件大小信息
            if output_pdf.exists():
                file_size_mb = output_pdf.stat().st_size / (1024 * 1024)
                print(f"文件大小: {file_size_mb:.2f} MB")
                self.logger.info(f"输出文件大小: {file_size_mb:.2f} MB")

            print("=" * 70)

            # 记录最终统计到日志
            self.logger.info("=" * 50)
            self.logger.info("处理完成统计")
            self.logger.info("=" * 50)
            self.logger.info(f"总页数: {len(enhanced_images)}")
            self.logger.info(f"成功页数: {len(enhanced_images) - len(failed_pages)}")
            self.logger.info(f"失败页数: {len(failed_pages)}")
            self.logger.info(f"步骤 1 耗时: {step1_time:.2f}秒")
            self.logger.info(f"步骤 2 耗时: {step2_time:.2f}秒")
            self.logger.info(f"步骤 3 耗时: {step3_time:.2f}秒")
            self.logger.info(f"总耗时: {total_time:.2f}秒")
            self.logger.info(f"平均每页: {total_time/len(enhanced_images):.2f}秒")

            return str(output_pdf)

        finally:
            # 清理临时文件
            if cleanup_temp and temp_dir.exists():
                self.logger.info(f"正在清理临时文件: {temp_dir}")
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    self.logger.warning(f"清理临时文件失败: {e}")

    @staticmethod
    def _images_to_pdf(image_paths: List[str], output_pdf: str) -> None:
        """
        将图片列表合并为 PDF 文件

        Args:
            image_paths: 图片路径列表
            output_pdf: 输出 PDF 文件路径
        """
        if not image_paths:
            raise ValueError("图片列表为空")

        # 打开所有图片
        images = []
        for img_path in image_paths:
            try:
                img = Image.open(img_path)
                # 转换为 RGB（PDF 不支持透明度）
                if img.mode in ('RGBA', 'LA', 'P'):
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    if img.mode in ('RGBA', 'LA'):
                        background.paste(img, mask=img.split()[-1])
                    else:
                        background.paste(img)
                    img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')

                images.append(img)
            except Exception as e:
                self.logger.warning(f"无法打开图片 {img_path}: {e}")

        if not images:
            raise RuntimeError("没有有效的图片可以转换为 PDF")

        # 保存为 PDF
        first_image = images[0]
        other_images = images[1:] if len(images) > 1 else []

        first_image.save(
            output_pdf,
            "PDF",
            resolution=100.0,
            save_all=True,
            append_images=other_images
        )


def main():
    """命令行入口函数"""
    parser = argparse.ArgumentParser(
        description=f"pdf_enhance.py v{__version__} - PDF 文档增强工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本使用（自动查找模型）
  python -m src.pdf_enhance document.pdf

  # 指定输出文件
  python -m src.pdf_enhance document.pdf -o enhanced.pdf

  # 使用 large 模型
  python -m src.pdf_enhance document.pdf --model-size large

  # 指定模型权重
  python -m src.pdf_enhance document.pdf --model models/xxx.pth

  # 高分辨率处理
  python -m src.pdf_enhance document.pdf --dpi 600

  # 详细输出模式
  python -m src.pdf_enhance document.pdf -v

功能特性:
  ✓ 完整的 PDF 增强工作流
  ✓ 自动提取、增强、合并
  ✓ 支持多页 PDF
  ✓ 可自定义分辨率和模型
  ✓ 自动清理临时文件
        """
    )

    parser.add_argument(
        'input_pdf',
        help='输入的 PDF 文件路径'
    )

    parser.add_argument(
        '-o', '--output',
        dest='output_pdf',
        help='输出 PDF 文件路径（默认为输入文件名 + _enhanced）'
    )

    parser.add_argument(
        '--model',
        help='模型权重文件路径（可选，默认自动查找）'
    )

    parser.add_argument(
        '--model-size',
        choices=['base', 'large'],
        default='base',
        help='模型大小（默认: base）'
    )

    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='PDF 转图片的分辨率 DPI（默认: 300）'
    )

    parser.add_argument(
        '--cpu',
        action='store_true',
        help='强制使用 CPU'
    )

    parser.add_argument(
        '--temp-dir',
        help='临时文件目录（可选）'
    )

    parser.add_argument(
        '--keep-temp',
        action='store_true',
        help='保留临时文件'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='显示详细输出'
    )

    parser.add_argument(
        '--log-file',
        help='日志文件路径（默认: logs/pdf_enhance_YYYYMMDD_HHMMSS.log）'
    )

    parser.add_argument(
        '--version',
        action='version',
        version=f'pdf_enhance.py v{__version__}'
    )

    args = parser.parse_args()

    try:
        # 设置日志系统
        log_file = args.log_file if hasattr(args, 'log_file') else None
        logger = setup_logging(log_file)

        logger.info("=" * 70)
        logger.info(f"PDF 文档增强工具 v{__version__}")
        logger.info("=" * 70)
        logger.debug(f"命令行参数: {vars(args)}")

        # 创建增强器
        enhancer = PDFEnhancer(
            model_path=args.model,
            model_size=args.model_size,
            dpi=args.dpi,
            device='cpu' if args.cpu else None,
            verbose=args.verbose,
            logger=logger
        )

        # 执行增强
        output_file = enhancer.enhance_pdf(
            input_pdf=args.input_pdf,
            output_pdf=args.output_pdf,
            temp_dir=args.temp_dir,
            keep_temp=args.keep_temp
        )

        logger.info("程序正常结束")
        return 0

    except KeyboardInterrupt:
        print("\n\n操作已取消")
        logger.warning("用户中断操作")
        return 130
    except Exception as e:
        logger.error(f"\n✗ 错误: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
            logger.exception("详细错误信息:")
        return 1


if __name__ == "__main__":
    sys.exit(main())


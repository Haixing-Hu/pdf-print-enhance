#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pdf_to_images.py - 将PDF文件的每一页提取为PNG图片

该模块提供了将PDF文件转换为图片的功能，支持：
- 自定义输出目录
- 自定义DPI和图片质量
- 支持PNG、JPEG格式
- 可作为命令行工具使用，也可作为模块导入

作者: PDF Print Enhance
版本: 1.0.0
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Optional, List, Tuple
import logging
import multiprocessing

try:
    from pdf2image import convert_from_path, pdfinfo_from_path
    from pdf2image.exceptions import PDFInfoNotInstalledError, PDFPageCountError
except ImportError:
    print("错误: 需要安装 pdf2image 库")
    print("安装命令: pip install pdf2image")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("错误: 需要安装 Pillow 库")
    print("安装命令: pip install Pillow")
    sys.exit(1)

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # 如果没有tqdm，提供一个简单的替代
    class tqdm:
        def __init__(self, iterable=None, total=None, desc=None, unit=None, **kwargs):
            self.iterable = iterable
            self.total = total
            self.n = 0

        def __iter__(self):
            for item in self.iterable:
                yield item
                self.n += 1

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def update(self, n=1):
            self.n += n


# 脚本版本
__version__ = "1.0.0"

# 默认配置
DEFAULT_DPI = 300
DEFAULT_QUALITY = 95
DEFAULT_FORMAT = "png"

# 线程配置常量
THREAD_COUNT_MIN_CPU_FOR_DUAL_THREAD = 3  # 至少3核才使用2线程
THREAD_COUNT_MIN_CPU_FOR_MULTI_THREAD = 5  # 至少5核才使用多线程
THREAD_COUNT_MAX_CORES_THRESHOLD = 8  # 8核以下使用不同策略
THREAD_COUNT_MAX_LIMIT = 16  # 最多使用的线程数上限
THREAD_COUNT_CPU_RATIO = 1  # 线程数占CPU核心数的比例（0.5=一半, 0.75=3/4, 1.0=全部）

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


def get_optimal_thread_count() -> int:
    """
    根据CPU核心数获取最优线程数

    策略：
    - 1-2核: 使用1个线程
    - 3-4核: 使用2个线程
    - 5-8核: 使用核心数的一半（至少3个）
    - 8核以上: 使用核心数 * CPU比例（但不超过最大限制）

    Returns:
        最优线程数
    """
    try:
        cpu_count = multiprocessing.cpu_count()

        if cpu_count < THREAD_COUNT_MIN_CPU_FOR_DUAL_THREAD:
            # 1-2核: 使用1个线程
            return 1
        elif cpu_count < THREAD_COUNT_MIN_CPU_FOR_MULTI_THREAD:
            # 3-4核: 使用2个线程
            return 2
        elif cpu_count <= THREAD_COUNT_MAX_CORES_THRESHOLD:
            # 5-8核: 使用核心数的一半（至少3个）
            return max(3, cpu_count // 2)
        else:
            # 8核以上: 使用核心数 * 比例，但不超过最大限制
            optimal_count = int(cpu_count * THREAD_COUNT_CPU_RATIO)
            return min(THREAD_COUNT_MAX_LIMIT, max(4, optimal_count))
    except:
        # 如果无法获取CPU核心数，默认使用2个线程
        return 2


class PDFToImagesConverter:
    """PDF转图片转换器类"""

    def __init__(
        self,
        dpi: int = DEFAULT_DPI,
        quality: int = DEFAULT_QUALITY,
        fmt: str = DEFAULT_FORMAT,
        verbose: bool = False,
        logger: Optional[logging.Logger] = None
    ):
        """
        初始化转换器

        Args:
            dpi: 图片分辨率DPI (72-2400)
            quality: 图片质量 (1-100)
            fmt: 图片格式 ('png' 或 'jpeg')
            verbose: 是否显示详细输出
            logger: 日志记录器
        """
        self.dpi = self._validate_dpi(dpi)
        self.quality = self._validate_quality(quality)
        self.format = self._validate_format(fmt)
        self.verbose = verbose
        self.logger = logger or logging.getLogger(__name__)

        if self.verbose:
            self.self.logger.setLevel(logging.DEBUG)

    @staticmethod
    def _validate_dpi(dpi: int) -> int:
        """验证DPI值"""
        if not isinstance(dpi, int) or dpi < 72 or dpi > 2400:
            raise ValueError(f"DPI值必须是72-2400之间的整数，当前值: {dpi}")
        return dpi

    @staticmethod
    def _validate_quality(quality: int) -> int:
        """验证质量值"""
        if not isinstance(quality, int) or quality < 1 or quality > 100:
            raise ValueError(f"质量值必须是1-100之间的整数，当前值: {quality}")
        return quality

    @staticmethod
    def _validate_format(fmt: str) -> str:
        """验证格式"""
        fmt_lower = fmt.lower()
        if fmt_lower not in ['png', 'jpeg', 'jpg']:
            raise ValueError(f"图片格式必须是 'png'、'jpeg' 或 'jpg'，当前值: {fmt}")
        # 统一使用 'PNG' 和 'JPEG'
        return 'PNG' if fmt_lower == 'png' else 'JPEG'

    def convert(
        self,
        pdf_path: str,
        output_dir: Optional[str] = None,
        filename_prefix: Optional[str] = None
    ) -> List[str]:
        """
        将PDF文件转换为图片

        Args:
            pdf_path: PDF文件路径
            output_dir: 输出目录（可选，默认为PDF所在目录）
            filename_prefix: 输出文件名前缀（可选，默认使用PDF文件名）

        Returns:
            生成的图片文件路径列表

        Raises:
            FileNotFoundError: PDF文件不存在
            ValueError: 输入参数无效
            RuntimeError: 转换过程中出错
        """
        # 验证输入文件
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")

        if not pdf_path.is_file():
            raise ValueError(f"不是有效的文件: {pdf_path}")

        if pdf_path.suffix.lower() != '.pdf':
            raise ValueError(f"输入文件必须是PDF格式: {pdf_path}")

        # 确定输出目录
        if output_dir is None:
            output_dir = pdf_path.parent
        else:
            output_dir = Path(output_dir)

        # 创建输出目录
        output_dir.mkdir(parents=True, exist_ok=True)

        # 确定文件名前缀
        if filename_prefix is None:
            filename_prefix = pdf_path.stem

        self.logger.info("开始将PDF转换为图片...")
        self.logger.info(f"输入文件: {pdf_path}")
        self.logger.info(f"输出目录: {output_dir}")
        self.logger.info(f"图片格式: {self.format}")
        self.logger.info(f"分辨率: {self.dpi} DPI")
        self.logger.info(f"质量: {self.quality}")

        try:
            # 获取PDF页数
            if self.verbose:
                self.logger.debug("正在获取PDF信息...")

            pdf_info = pdfinfo_from_path(str(pdf_path))
            total_pages = pdf_info.get("Pages", 0)

            if total_pages == 0:
                raise RuntimeError("无法获取PDF页数")

            self.logger.info(f"PDF页数: {total_pages}")

            # 获取最优线程数
            thread_count = get_optimal_thread_count()
            cpu_count = multiprocessing.cpu_count()
            self.logger.info(f"CPU核心数: {cpu_count}, 使用线程数: {thread_count}")

            # 保存图片
            output_files = []
            page_num_width = len(str(total_pages))  # 计算页码宽度

            # 记录开始时间
            start_time = time.time()

            # 使用进度条逐页转换
            progress_bar = tqdm(
                range(1, total_pages + 1),
                desc="正在转换",
                unit="页",
                ncols=80,
                disable=self.verbose  # 详细模式下禁用进度条
            )

            for page_num in progress_bar:
                # 逐页转换PDF，使用多线程加速
                images = convert_from_path(
                    str(pdf_path),
                    dpi=self.dpi,
                    fmt=self.format.lower(),
                    first_page=page_num,
                    last_page=page_num,
                    thread_count=thread_count
                )

                if not images:
                    self.logger.warning(f"警告: 第{page_num}页转换失败，跳过")
                    continue

                image = images[0]  # 每次只转换一页

                # 生成文件名，格式: prefix_page_001.png
                page_num_str = str(page_num).zfill(page_num_width)
                ext = 'png' if self.format == 'PNG' else 'jpg'
                output_filename = f"{filename_prefix}_page_{page_num_str}.{ext}"
                output_path = output_dir / output_filename

                # 保存图片
                if self.format == 'JPEG':
                    # JPEG不支持透明度，转换为RGB
                    if image.mode in ('RGBA', 'LA', 'P'):
                        background = Image.new('RGB', image.size, (255, 255, 255))
                        if image.mode == 'P':
                            image = image.convert('RGBA')
                        background.paste(image, mask=image.split()[-1] if image.mode in ('RGBA', 'LA') else None)
                        image = background
                    image.save(output_path, self.format, quality=self.quality, optimize=True)
                else:
                    image.save(output_path, self.format, optimize=True)

                output_files.append(str(output_path))

                if self.verbose:
                    file_size = output_path.stat().st_size / 1024  # KB
                    self.logger.debug(f"  生成: {output_filename} ({file_size:.1f} KB)")

            # 计算总耗时
            end_time = time.time()
            total_time = end_time - start_time
            avg_time_per_page = total_time / len(output_files) if output_files else 0

            self.logger.info(f"✓ 转换成功完成")
            self.logger.info(f"生成的图片文件数量: {len(output_files)}")
            self.logger.info(f"总耗时: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")
            self.logger.info(f"平均每页: {avg_time_per_page:.2f}秒")

            return output_files

        except PDFInfoNotInstalledError:
            raise RuntimeError(
                "错误: 需要安装 poppler-utils\n"
                "安装方法:\n"
                "  macOS:   brew install poppler\n"
                "  Ubuntu:  apt-get install poppler-utils\n"
                "  CentOS:  yum install poppler-utils"
            )
        except PDFPageCountError as e:
            raise RuntimeError(f"无法读取PDF文件: {e}")
        except Exception as e:
            raise RuntimeError(f"转换失败: {e}")


def pdf_to_images(
    pdf_path: str,
    output_dir: Optional[str] = None,
    dpi: int = DEFAULT_DPI,
    quality: int = DEFAULT_QUALITY,
    fmt: str = DEFAULT_FORMAT,
    verbose: bool = False,
    logger: Optional[logging.Logger] = None
) -> List[str]:
    """
    将PDF文件转换为图片（便捷函数）

    这是一个便捷的包装函数，用于快速调用转换功能。

    Args:
        pdf_path: PDF文件路径
        output_dir: 输出目录（可选，默认为PDF所在目录）
        dpi: 图片分辨率DPI，默认300 (72-2400)
        quality: 图片质量，默认95 (1-100)
        fmt: 图片格式，默认'png' ('png'、'jpeg'、'jpg')
        verbose: 是否显示详细输出，默认False

    Returns:
        生成的图片文件路径列表

    Raises:
        FileNotFoundError: PDF文件不存在
        ValueError: 输入参数无效
        RuntimeError: 转换过程中出错

    Example:
        >>> # 基本使用
        >>> files = pdf_to_images("document.pdf")
        >>>
        >>> # 指定输出目录和参数
        >>> files = pdf_to_images(
        ...     "document.pdf",
        ...     output_dir="/path/to/output",
        ...     dpi=600,
        ...     quality=90,
        ...     fmt="jpeg"
        ... )
    """
    converter = PDFToImagesConverter(dpi=dpi, quality=quality, fmt=fmt, verbose=verbose, logger=logger)
    return converter.convert(pdf_path, output_dir)


def get_pdf_page_count(pdf_path: str) -> int:
    """
    获取PDF文件的页数

    Args:
        pdf_path: PDF文件路径

    Returns:
        PDF页数

    Raises:
        FileNotFoundError: PDF文件不存在
        RuntimeError: 无法读取PDF文件
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")

    try:
        from pdf2image import pdfinfo_from_path
        info = pdfinfo_from_path(str(pdf_path))
        return info.get("Pages", 0)
    except Exception as e:
        raise RuntimeError(f"无法读取PDF信息: {e}")


def main():
    """命令行入口函数"""
    parser = argparse.ArgumentParser(
        description=f"pdf_to_images.py v{__version__} - 将PDF文件的每一页提取为PNG图片",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本转换
  python pdf_to_images.py document.pdf

  # 指定输出目录
  python pdf_to_images.py document.pdf -o /path/to/output

  # 高分辨率转换
  python pdf_to_images.py --dpi 600 document.pdf

  # 转换为JPEG格式
  python pdf_to_images.py --format jpeg --quality 90 document.pdf

  # 详细输出模式
  python pdf_to_images.py -v document.pdf

功能特性:
  ✓ 高质量图片输出
  ✓ 自动页码编号
  ✓ 支持多种图片格式
  ✓ 可自定义分辨率和质量
  ✓ 智能文件命名
        """
    )

    parser.add_argument(
        'pdf_file',
        help='输入的PDF文件路径'
    )

    parser.add_argument(
        '-o', '--output',
        dest='output_dir',
        help='输出图片的目录（默认为PDF文件所在目录）'
    )

    parser.add_argument(
        '--dpi',
        type=int,
        default=DEFAULT_DPI,
        help=f'图片分辨率DPI (默认: {DEFAULT_DPI})'
    )

    parser.add_argument(
        '--quality',
        type=int,
        default=DEFAULT_QUALITY,
        help=f'图片质量 1-100 (默认: {DEFAULT_QUALITY})'
    )

    parser.add_argument(
        '--format',
        choices=['png', 'jpeg', 'jpg'],
        default=DEFAULT_FORMAT,
        help=f'图片格式 (默认: {DEFAULT_FORMAT})'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='显示详细输出'
    )

    parser.add_argument(
        '--version',
        action='version',
        version=f'pdf_to_images.py v{__version__}'
    )

    args = parser.parse_args()

    try:
        # 执行转换
        output_files = pdf_to_images(
            pdf_path=args.pdf_file,
            output_dir=args.output_dir,
            dpi=args.dpi,
            quality=args.quality,
            fmt=args.format,
            verbose=args.verbose
        )

        # 显示结果摘要
        print()
        print("转换完成！")
        if args.output_dir:
            print(f"输出目录: {args.output_dir}")

        if output_files:
            print(f"示例文件: {Path(output_files[0]).name}")
            if len(output_files) > 1:
                print(f"          {Path(output_files[-1]).name}")

        return 0

    except Exception as e:
        self.logger.error(f"✗ {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())


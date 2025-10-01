# PDF 打印增强工具

完整的 PDF 文档增强解决方案 - 自动提取、增强和重组 PDF 文档，使其更适合打印和阅读。

## 🌟 主要功能

✅ **完整工作流** - 一键完成 PDF 提取 → 增强 → 重组
✅ **进度显示** - 每个步骤都有实时进度条
✅ **详细计时** - 完整的性能统计和计时信息
✅ **完整日志** - 自动保存到文件，便于调试
✅ **临时文件管理** - 所有中间文件在临时目录，自动清理
✅ **模块化设计** - 清晰的代码结构，易于维护
✅ **独立运行** - 无需依赖外部仓库

## 🚀 快速开始

### 1. 环境准备

```bash
# 激活虚拟环境
source .venv/bin/activate

# 一次性安装所有依赖
pip install -r requirements.txt

# 安装系统级别的 poppler（PDF 处理必需）
# macOS:
brew install poppler

# Ubuntu/Debian:
sudo apt-get install poppler-utils

# CentOS/RHEL:
sudo yum install poppler-utils

# Windows:
# 方法 1: 使用 conda（推荐）
conda install -c conda-forge poppler

# 方法 2: 使用 chocolatey
choco install poppler

# 方法 3: 手动下载
# 1. 下载: https://github.com/oschwartz10612/poppler-windows/releases
# 2. 解压到 C:\poppler
# 3. 添加 C:\poppler\bin 到系统 PATH
```

### 2. 下载模型

```bash
# 下载 DocEnTR 预训练模型
python download_model.py
```

### 3. 一行命令使用

```bash
# 基本使用（自动查找模型、自动命名输出）
python -m src.pdf_enhance input.pdf

# 输出: input_enhanced.pdf
# 日志: logs/pdf_enhance_YYYYMMDD_HHMMSS.log
```

### 4. 输出示例

```
======================================================================
PDF 文档增强工具 v1.0.0
======================================================================
输入文件: document.pdf
输出文件: document_enhanced.pdf
分辨率: 300 DPI
模型: DocEnTR-BASE
======================================================================
PDF 页数: 32
日志文件: logs/pdf_enhance_20251001_143022.log

步骤 1/3: 提取 PDF 页面为图片
----------------------------------------------------------------------
正在转换: 100%|████████████████| 32/32 [00:45<00:00, 1.41秒/页]
✓ 成功提取 32 页
  耗时: 45.12秒, 平均每页: 1.41秒

步骤 2/3: 增强每一页图像
----------------------------------------------------------------------
增强进度: 100%|████████████████| 32/32 [04:23<00:00, 8.23秒/页]
✓ 成功增强 32/32 页
  耗时: 263.36秒, 平均每页: 8.23秒

步骤 3/3: 合并页面为 PDF
----------------------------------------------------------------------
合并进度: 100%|████████████████| 1/1 [00:02<00:00, 2.15秒/文件]
✓ PDF 已保存到: document_enhanced.pdf
  耗时: 2.15秒

======================================================================
✓ 处理完成！
======================================================================
总页数: 32
成功增强: 32 页

各步骤耗时:
  步骤 1 (提取): 45.12秒 (平均每页: 1.41秒)
  步骤 2 (增强): 263.36秒 (平均每页: 8.23秒)
  步骤 3 (合并): 2.15秒

总耗时: 310.63秒 (5.18分钟)
平均每页: 9.71秒

输出文件: document_enhanced.pdf
文件大小: 12.45 MB
======================================================================
```

## 📖 使用指南

### 基本命令

```bash
# 指定输出文件
python -m src.pdf_enhance input.pdf -o output.pdf

# 使用 large 模型（更好效果，但更慢）
python -m src.pdf_enhance input.pdf --model-size large

# 高分辨率处理（600 DPI）
python -m src.pdf_enhance input.pdf --dpi 600

# 详细输出模式
python -m src.pdf_enhance input.pdf -v
```

### 调试选项

```bash
# 保留临时文件（不自动删除）
python -m src.pdf_enhance input.pdf --keep-temp

# 指定临时文件目录
python -m src.pdf_enhance input.pdf --temp-dir /path/to/temp

# 指定日志文件位置
python -m src.pdf_enhance input.pdf --log-file my_log.log

# 强制使用 CPU（不使用 GPU）
python -m src.pdf_enhance input.pdf --cpu
```

### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `input_pdf` | 输入 PDF 文件 | 必需 |
| `-o, --output` | 输出 PDF 文件 | `输入_enhanced.pdf` |
| `--model` | 模型权重路径 | 自动查找 |
| `--model-size` | 模型大小（base/large） | `base` |
| `--dpi` | 图片分辨率 | `300` |
| `--cpu` | 强制使用 CPU | `False` |
| `--temp-dir` | 临时文件目录 | 系统临时目录 |
| `--keep-temp` | 保留临时文件 | `False` |
| `--log-file` | 日志文件路径 | `logs/pdf_enhance_*.log` |
| `-v, --verbose` | 详细输出 | `False` |

### Python API 使用

除了命令行，也可以在 Python 代码中使用：

```python
from src import PDFEnhancer

# 创建增强器
enhancer = PDFEnhancer(
    model_size='base',  # 或 'large'
    dpi=300,
    verbose=False
)

# 增强 PDF
output_file = enhancer.enhance_pdf(
    input_pdf='input.pdf',
    output_pdf='output.pdf'
)

print(f"增强后的文件: {output_file}")
```

#### 仅增强单张图片

```python
from src import DocEnTREnhancer

# 创建图像增强器
enhancer = DocEnTREnhancer(model_size='base')

# 增强单张图片
enhancer.enhance(
    input_path='input.png',
    output_path='output.png'
)
```

#### 仅转换 PDF 为图片

```python
from src import pdf_to_images

# 转换 PDF 为图片
images = pdf_to_images(
    pdf_path='document.pdf',
    output_dir='./output',
    dpi=300,
    fmt='png'
)

print(f"生成了 {len(images)} 张图片")
```

### 批量处理

```bash
# 创建批处理脚本
for pdf in *.pdf; do
    echo "处理: $pdf"
    python -m src.pdf_enhance "$pdf"
done
```

或使用 Python:

```python
from pathlib import Path
from src import PDFEnhancer

enhancer = PDFEnhancer(model_size='base', dpi=300)

for pdf_file in Path('./pdfs').glob('*.pdf'):
    print(f"处理: {pdf_file}")
    output_file = pdf_file.parent / f"{pdf_file.stem}_enhanced.pdf"

    try:
        enhancer.enhance_pdf(
            input_pdf=str(pdf_file),
            output_pdf=str(output_file)
        )
        print(f"✓ 完成: {output_file}")
    except Exception as e:
        print(f"✗ 失败: {e}")
```

## 📁 项目结构

```
pdf-print-enhance/
├── src/                         # 源代码目录 ⭐
│   ├── __init__.py
│   ├── pdf_to_images.py        # PDF 转图片
│   ├── pdf_enhance.py          # 主程序入口 ⭐
│   ├── models/                 # 模型定义
│   │   ├── __init__.py
│   │   └── binae.py           # BinModel
│   └── enhancer/               # 增强器模块
│       ├── __init__.py
│       ├── docentr.py         # DocEnTR 增强器
│       └── image_utils.py     # 图像处理工具
│
├── models/                      # 模型权重
│   ├── docentr_hdibco2012_base_8x8.pth
│   └── docentr_hdibco2012_large_16x16.pth
│
├── logs/                        # 日志文件 ⭐
│   └── pdf_enhance_*.log       # 自动生成
│
├── requirements.txt             # Python 依赖 ⭐
├── download_model.py            # 模型下载脚本
├── test_pdf_enhance.py          # 测试脚本
├── .gitignore                   # Git 忽略规则
└── README.md                    # 本文件
```

## 🎯 核心功能

### 1. PDF 增强工作流

程序执行以下三个步骤：

**步骤 1: 提取 PDF 页面**
- 使用 `pdf2image` 将 PDF 每一页转换为高清图片
- 支持自定义 DPI（默认 300）
- 自动优化线程数以提高速度
- 实时进度条显示

**步骤 2: 增强每一页**
- 使用 DocEnTR 深度学习模型增强图像
- 自动分块处理大图片（256x256 块）
- 支持 GPU 加速（如果可用）
- 自动二值化输出
- 详细的每页计时

**步骤 3: 合并为 PDF**
- 将增强后的图片合并为新的 PDF
- 保持原始页面顺序
- 自动处理图片格式转换

### 2. DocEnTR 深度学习增强

基于 Vision Transformer 的文档图像增强模型：

- **智能去重影** - 自动识别并去除扫描重影
- **高质量二值化** - 保持文字完整性和清晰度
- **自适应处理** - 针对不同质量的文档自动优化
- **端到端训练** - 从输入到输出全流程优化

### 3. 进度显示与日志

- **实时进度条** - 每个步骤独立的进度条
- **详细计时** - 总耗时和平均每页耗时
- **完整日志** - 自动保存到 `logs/` 目录
- **DEBUG 信息** - 包含所有操作的详细记录

### 4. 临时文件管理

所有中间文件都在临时目录中：

```
/tmp/pdf_enhance_XXXXXX/        # 临时根目录
├── original/                    # 提取的原始图片
└── enhanced/                    # 增强后的图片
```

默认自动清理，可选保留（`--keep-temp`）。

## 📊 性能参考

基于不同配置的性能参考（单页平均耗时）：

| 配置 | DPI | 模型 | GPU | 每页耗时 | 适用场景 |
|------|-----|------|-----|----------|---------|
| 快速 | 150 | base | 是 | ~3秒 | 快速预览 |
| 标准 | 300 | base | 是 | ~8秒 | 一般文档 |
| 高质量 | 300 | large | 是 | ~15秒 | 重要文档 |
| 超高清 | 600 | large | 是 | ~30秒 | 出版印刷 |
| CPU模式 | 300 | base | 否 | ~45秒 | 无GPU环境 |

*注：实际性能取决于硬件配置和文档复杂度*

## 🔧 故障排除

### 模型未找到

```
⚠️  未找到 base 模型权重
```

**解决方案**：
```bash
python download_model.py
```

### CUDA 内存不足

```
RuntimeError: CUDA out of memory
```

**解决方案**：
```bash
# 使用 CPU 模式
python -m src.pdf_enhance input.pdf --cpu
```

### poppler 未安装

```
错误: 需要安装 poppler-utils
```

**解决方案**：
```bash
# macOS
brew install poppler

# Ubuntu/Debian
sudo apt-get install poppler-utils

# CentOS/RHEL
sudo yum install poppler-utils
```

### 依赖库缺失

```
ImportError: No module named 'xxx'
```

**解决方案**：
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### 查看日志

日志文件自动保存在 `logs/` 目录：

```bash
# 查看最新的日志文件
ls -lt logs/ | head -n 1

# 实时查看日志（在另一个终端）
tail -f logs/pdf_enhance_*.log

# 查找错误信息
cat logs/pdf_enhance_*.log | grep ERROR
```

### 调试模式

```bash
# 1. 使用详细输出模式
python -m src.pdf_enhance input.pdf -v

# 2. 保留临时文件检查中间结果
python -m src.pdf_enhance input.pdf --keep-temp --temp-dir ./debug_temp

# 3. 检查日志文件
cat logs/pdf_enhance_*.log
```

## 📚 技术架构

### 核心组件

1. **PDFToImagesConverter** (`pdf_to_images.py`)
   - PDF 到图片转换
   - 多线程优化
   - 支持多种图片格式

2. **DocEnTREnhancer** (`enhancer/docentr.py`)
   - 图像增强核心
   - 基于 Vision Transformer
   - 分块处理大图片

3. **BinModel** (`models/binae.py`)
   - 深度学习模型
   - 自编码器架构
   - 支持 GPU 加速

4. **PDFEnhancer** (`pdf_enhance.py`)
   - 主工作流程序
   - 临时文件管理
   - 进度显示和日志

### 依赖关系

```
pdf_enhance.py (主程序)
├── pdf_to_images.py (PDF 转换)
└── enhancer/
    ├── docentr.py (增强器)
    │   ├── image_utils.py (图像处理)
    │   └── models/binae.py (深度学习模型)
    └── image_utils.py
```

### 技术栈

**核心依赖**：
- PyTorch (>=2.6.0) - 深度学习框架
- vit-pytorch (0.24.3) - Vision Transformer（严格版本）
- einops (0.3.2) - 张量操作
- opencv-python (4.5.4.60) - 图像处理
- pdf2image - PDF 转换
- tqdm - 进度显示

**系统依赖**：
- poppler-utils - PDF 处理（系统级别）

## 🧪 测试

运行测试脚本：

```bash
# 基本测试（检查模型）
python test_pdf_enhance.py

# 完整测试（包含图像增强）
python test_pdf_enhance.py --full
```

## ⚠️ 重要提醒

### 必须使用虚拟环境

```bash
source .venv/bin/activate  # 每次使用前都要执行
```

### vit-pytorch 版本要求

- 必须使用 `vit-pytorch==0.24.3`
- 其他版本与预训练权重不兼容
- 已在 `requirements.txt` 中严格指定

### 临时文件位置

所有中间文件都在临时目录中（不是在当前目录）：
- 默认：系统临时目录（如 `/tmp/pdf_enhance_XXXXXX/`）
- 自定义：使用 `--temp-dir` 参数指定
- 自动清理：处理完成后自动删除
- 调试：使用 `--keep-temp` 保留临时文件

## 📝 版本历史

### v1.0.0 (2025-10-01)

- ✅ 完整的 PDF 增强工作流
- ✅ 模块化代码结构（独立于外部仓库）
- ✅ 支持 base 和 large 模型
- ✅ 详细的进度显示（每个步骤独立进度条）
- ✅ 完整的日志系统（自动保存到文件）
- ✅ 每个步骤的详细计时统计
- ✅ 自动临时文件管理
- ✅ 完善的错误处理
- ✅ 完整的文档和测试

## 📄 许可证

MIT License

DocEnTR 模型来自：https://github.com/dali92002/DocEnTR

## 🔗 相关链接

- DocEnTR GitHub: https://github.com/dali92002/DocEnTR
- DocEnTR 论文: DocEnTr: An End-to-End Document Image Enhancement Transformer (ICPR 2022)

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

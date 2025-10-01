#!/bin/bash
# PDF转图片运行脚本
# 使用方法: ./run.sh pdf/document.pdf [其他参数]

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 使用虚拟环境的Python直接运行
"$SCRIPT_DIR/.venv/bin/python" "$SCRIPT_DIR/src/pdf_to_images.py" "$@"


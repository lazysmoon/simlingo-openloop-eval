#!/bin/bash
# SimLingo 开环评估 - Conda 环境一键安装脚本
# 用法:
#   1. 将 simlingo_packed.tar.gz 和本脚本放在同一目录
#   2. bash install_env.sh [conda安装路径]
#   示例: bash install_env.sh /home/user/miniconda3
#         bash install_env.sh  （自动检测 conda 路径）

set -e

CONDA_BASE="${1:-$(conda info --base 2>/dev/null)}"
PACK_FILE="simlingo_packed.tar.gz"

if [ -z "$CONDA_BASE" ]; then
    echo "[错误] 未检测到 conda，请指定 conda 安装路径："
    echo "  bash install_env.sh /path/to/miniconda3"
    exit 1
fi

if [ ! -f "$PACK_FILE" ]; then
    echo "[错误] 未找到 $PACK_FILE，请确保压缩包和本脚本在同一目录"
    exit 1
fi

ENV_DIR="$CONDA_BASE/envs/simlingo"

if [ -d "$ENV_DIR" ]; then
    echo "[警告] 环境 simlingo 已存在于 $ENV_DIR"
    read -p "是否覆盖？(y/N): " confirm
    if [ "$confirm" != "y" ]; then
        echo "已取消"
        exit 0
    fi
    rm -rf "$ENV_DIR"
fi

echo "============================================"
echo " simlingo 环境安装"
echo "============================================"
echo ""

echo "[1/3] 解压环境到 $ENV_DIR ..."
mkdir -p "$ENV_DIR"
tar -xzf "$PACK_FILE" -C "$ENV_DIR"

echo "[2/3] 修复环境路径（conda-unpack）..."
eval "$(conda shell.bash hook)"
conda activate simlingo
conda-unpack

echo "[3/3] 设置环境变量 ..."
export CUDA_HOME="$ENV_DIR"
export HF_ENDPOINT=https://hf-mirror.com

echo ""
echo "[验证] 检查 PyTorch + CUDA ..."
python -c "
import torch
print(f'  PyTorch 版本: {torch.__version__}')
print(f'  CUDA 可用:    {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU 设备:     {torch.cuda.get_device_name(0)}')
"

echo ""
echo "============================================"
echo "  安装完成！"
echo ""
echo "  使用方式："
echo "    conda activate simlingo"
echo "    export CUDA_HOME=\$CONDA_PREFIX"
echo ""
echo "  后续每次使用前记得设置："
echo "    export HF_ENDPOINT=https://hf-mirror.com"
echo "============================================"

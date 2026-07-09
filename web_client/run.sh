#!/usr/bin/env bash
#
# 启动 ABot-World 实时可交互世界模型 Gradio 服务
#
# 用法:
#   bash web_client/run.sh              # 默认 GPU 0
#   CUDA_ID=0 bash web_client/run.sh    # 指定 GPU
#   DEBUG=1 bash web_client/run.sh      # 前端调试模式（不加载模型）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_ID:-0}"

# 前端调试模式：不加载模型，只运行 UI
# export ABOTWORLD_DEBUG_FRONTEND=1
export ABOTWORLD_DEBUG_FRONTEND=0

echo "=== ABot-World Streaming UI ==="
echo "  GPU:  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  Root: $PROJECT_ROOT"
echo "================================="

cd "$PROJECT_ROOT"
python web_client/app.py

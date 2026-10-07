#!/usr/bin/env bash
# =============================================================================
# 长 rollout 课程训练：后台启动 train.py，自动 nohup + 写日志 + 打印进度提示。
# 退出 SSH 后训练仍在运行（nohup 守护）。
#
# 用法：
#   bash train.sh            # 在任意位置运行均可（脚本会自己 cd 到 landemu/）
#
# 训练完成后，跑评估 / 推理 / 诊断：
#   python run_pipeline.py --skip train
# =============================================================================
set -euo pipefail

# 脚本所在目录（= landemu/），确保从正确路径启动
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# 激活 conda 环境（没有 conda 则用当前已激活的 python）
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate landemu
fi

LOG="train_$(date +%Y%m%d_%H%M%S).log"

echo "=========================================================="
echo " 长 rollout 课程训练启动"
echo " 目录: $(pwd)"
echo " 环境: $(which python)"
echo " 日志: $LOG"
echo "=========================================================="

# 后台启动训练（nohup 守护，断连不中断）
nohup python train.py > "$LOG" 2>&1 &
echo "已后台启动，PID=$!"
echo ""
echo "实时看进度（每 epoch 一行 K/loss/v1/v30）："
echo "  tail -f $LOG"

# -*- coding: utf-8 -*-
"""
一键 pipeline：按顺序跑 训练(train) → 评估(evaluate) → 预测(predict)。

用法：
  python run_pipeline.py                       # 三步全跑
  python run_pipeline.py --skip train          # 跳过训练，只跑评估 + 预测
  python run_pipeline.py --skip train,evaluate # 只跑预测

每步用当前 conda 环境的 python（sys.executable）执行，任一步失败即停（不会带着坏权重往下走）。
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

STEPS = [
    ("train", "train.py"),
    ("evaluate", "evaluate.py"),
    ("predict", "predict.py"),
    ("diagnose", "diagnose_freerun.py"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", default="", help="要跳过的步骤，逗号分隔：train,evaluate,predict")
    args = ap.parse_args()

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}

    for name, script in STEPS:
        if name in skip:
            print(f"\n===== [skip] {name} =====", flush=True)
            continue
        print(f"\n===== [{name}] {script} =====", flush=True)
        subprocess.run([sys.executable, str(HERE / script)], check=True)


if __name__ == "__main__":
    main()

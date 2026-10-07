# -*- coding: utf-8 -*-
"""
长 rollout 课程 smoke 测试。

与 train.py 唯一区别：临时覆盖 config 的 rollout_schedule / epochs / batches，
把课程「压缩」到几个 epoch 内快速爬到 K=30/60，验证新 lr=3e-4 + grad_clip=0.5 下
长 rollout 的 val 30 天 free-run RMSE（v30）不崩，再决定是否跑满 300 epoch 的完整课程。

所有输出（pack 缓存 / 权重 / config 快照 / 日志）都落到新实验目录：
    cache/experiments/long_rollout_cldas_smoke/
不会覆盖现有 default / cross_forcing_era5 的权重。

用法：
    python smoke_rollout.py
"""
import config
import train

# 新实验目录 + 与诊断基线（default/cldas）一致的数据源
config.EXPERIMENT = "long_rollout_cldas_smoke"
config.SOURCE = "cldas"

tc = config.cfg.train
# 压缩课程：epoch 0 K=1 → 1~2 K=7 → 3~4 K=30 → 5 K=60（K=120 太慢，留给完整跑）
tc.epochs = 6
tc.rollout_schedule = {0: 1, 1: 7, 3: 30, 5: 60}
# 砍采样规模，保证 smoke 在几分钟内跑完（完整训练用默认 2048/400）
tc.batch_size = 1024
tc.batches_per_epoch = 30
# 强制 CPU 单线程（tiny MLP 实测最快的组合，见 开发问题总结.md 第 7 节）
tc.device = "cpu"
# lr / grad_clip 沿用 config.py 已改好的 3e-4 / 0.5（长 rollout 配套），此处不再覆盖

if __name__ == "__main__":
    print(f"[smoke] experiment={config.EXPERIMENT}  source={config.SOURCE}")
    print(f"[smoke] rollout_schedule={tc.rollout_schedule}  epochs={tc.epochs}  "
          f"batch={tc.batch_size}x{tc.batches_per_epoch}  lr={tc.lr}  grad_clip={tc.grad_clip}")
    train.main()

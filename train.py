# -*- coding: utf-8 -*-
"""
训练显式状态算子 DeltaOperator（或隐状态 LSTM 基线）：rollout 课程式训练（K: 1 -> 7）。

模型切换：config.py 的 ModelConfig.version —— "delta"（默认）或 "lstm"。
  - delta：单步递推 state_t=clamp(state_{t-1} + MLP([state_{t-1},F_t,static]))，rollout K 步训练。
    时间约定：forcing[t] 表示从 state[t-1] 推进到 state[t] 这一时段（第 t 天）的强迫。
  - lstm ：lookback 窗口 [t-L, t) 真值 -> 预测 t，teacher-forcing 训练（不做多步 rollout）。

性能要点：数据一次性预转 torch tensor（零拷贝）；delta 每 batch 先 gather 整个 rollout 窗口
再纯张量 K 步展开；lstm 每 batch 用一次高级索引 gather lookback 窗口。
"""
import math
import os
import random
import sys
import time

# 强制单线程（仅 CPU 生效；控制底层 OpenMP/MKL，小算子多线程反而拖慢）
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
# 允许 torch 与 matplotlib 等库各自的 OpenMP 运行时共存（避免 libiomp5md.dll 重复初始化冲突）
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import config
from data import build
from model import build_model, free_run


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def k_for_epoch(epoch, schedule):
    k = 1
    for thr, val in sorted(schedule.items()):
        if epoch >= thr:
            k = val
    return k


def main():
    tc = config.cfg.train
    mc = config.cfg.model
    set_seed(tc.seed)
    device = torch.device(config.resolve_device(tc.device))
    print("device =", device)
    # 小算子（11→128→…→2）在 CPU 上多线程 OpenMP 会因 spin-wait 严重拖慢，单线程反而快（仅 CPU 生效）。
    if device.type == "cpu":
        torch.set_num_threads(1)
    pack = build(config.SOURCE)
    # 预转 torch（from_numpy 零拷贝，与 numpy 共享内存）
    target = torch.from_numpy(pack["target"]).to(device)    # (N,T,2)
    forcing = torch.from_numpy(pack["forcing"]).to(device)  # (N,T,F)
    static = torch.from_numpy(pack["static"]).to(device)    # (N,S)
    n, T, _ = target.shape
    (tr_s, tr_e), (va_s, va_e) = pack["split"]["train"], pack["split"]["val"]

    model = build_model(mc).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)

    # 实验配置快照：记录本次实验的关键配置，便于复现
    from dataclasses import asdict
    import json as _json
    os.makedirs(config.exp_dir(), exist_ok=True)
    snapshot = {
        "source": config.SOURCE,
        "experiment": config.EXPERIMENT,
        "strict_rootzone": config.STRICT_ROOTZONE,
        "model": asdict(mc),
        "train": asdict(tc),
    }
    with open(config.exp_dir() / "config.json", "w", encoding="utf-8") as _f:
        _json.dump(snapshot, _f, ensure_ascii=False, indent=2)
    rng = np.random.default_rng(tc.seed)
    L = mc.lookback
    arange_lb = torch.arange(L, device=device) - L          # [-L, ..., -1]

    use_tqdm = sys.stderr.isatty()
    pbar = tqdm(range(tc.epochs), desc=f"train[{mc.version}]",
                disable=not use_tqdm,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}")
    for epoch in pbar:
        t0e = time.time()
        model.train()
        k = k_for_epoch(epoch, tc.rollout_schedule)
        total = 0.0

        if mc.version == "lstm":
            # LSTM：lookback 窗口真值 -> 预测 t0（teacher forcing）
            for _ in range(tc.batches_per_epoch):
                sites = torch.from_numpy(rng.integers(0, n, tc.batch_size).astype(np.int64)).to(device)
                t0 = torch.from_numpy(rng.integers(tr_s + L, tr_e, tc.batch_size).astype(np.int64)).to(device)
                idx = t0[:, None] + arange_lb[None, :]              # (B, L) 历史窗口 [t0-L, t0)
                swin = target[sites[:, None], idx, :]               # (B, L, 2)
                fwin = forcing[sites[:, None], idx, :]              # (B, L, F)
                pred, _ = model(swin, fwin, static[sites])
                loss = F.mse_loss(pred, target[sites, t0])
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
                opt.step()
                total += loss.item()
        else:
            # delta：rollout K 步课程训练（时间约定：state_{t-1} + F_t -> state_t，与 free_run 一致）
            arange_k = torch.arange(k, device=device)
            for _ in range(tc.batches_per_epoch):
                sites = torch.from_numpy(rng.integers(0, n, tc.batch_size).astype(np.int64)).to(device)
                t0 = torch.from_numpy(rng.integers(tr_s + 1, tr_e - k + 1, tc.batch_size).astype(np.int64)).to(device)
                idx = t0[:, None] + arange_k[None, :]               # (B, K) = [t0 .. t0+k-1]
                fwin = forcing[sites[:, None], idx, :]              # (B, K, F)  F_{t0}..F_{t0+k-1}
                twin = target[sites[:, None], idx, :]               # (B, K, 2)  SM_{t0}..SM_{t0+k-1}
                state = target[sites, t0 - 1]                       # (B, 2) 初始状态 SM_{t0-1}
                sb = static[sites]
                loss = 0.0
                for s in range(k):
                    state, delta = model(state, fwin[:, s, :], sb)
                    loss = loss + F.mse_loss(state, twin[:, s, :])
                    if tc.lambda_delta > 0:
                        loss = loss + tc.lambda_delta * (delta ** 2).mean()
                loss = loss / k
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
                opt.step()
                total += loss.item()

        # 验证：one-step + 30 天 free-run（随机子集）
        model.eval()
        with torch.no_grad():
            m = 256
            vsites = torch.from_numpy(rng.integers(0, n, m).astype(np.int64)).to(device)
            if mc.version == "lstm":
                vt0 = torch.from_numpy(rng.integers(va_s + L, va_e, m).astype(np.int64)).to(device)
                idx = vt0[:, None] + arange_lb[None, :]
                pred, _ = model(target[vsites[:, None], idx, :],
                                forcing[vsites[:, None], idx, :], static[vsites])
                onestep_rmse = torch.sqrt(F.mse_loss(pred, target[vsites, vt0])).item()
            else:
                vt0 = torch.from_numpy(rng.integers(va_s + 1, va_e, m).astype(np.int64)).to(device)
                state = target[vsites, vt0 - 1]
                pred, _ = model(state, forcing[vsites, vt0], static[vsites])
                onestep_rmse = torch.sqrt(F.mse_loss(pred, target[vsites, vt0])).item()

            # 30 天 free-run（统一起点 va_s，随机子集站点）
            m2 = 128
            rsites = torch.from_numpy(rng.integers(0, n, m2).astype(np.int64)).to(device)
            pred30 = free_run(model, mc, target[rsites], forcing[rsites], static[rsites], va_s, 30)
            roll_rmse = torch.sqrt(F.mse_loss(pred30, target[rsites, va_s:va_s + 30])).item()

        kinfo = str(k) if mc.version == "delta" else "-"
        dt = time.time() - t0e
        if use_tqdm:
            pbar.set_postfix(K=kinfo,
                             loss=f"{total / tc.batches_per_epoch:.5e}",
                             v1=f"{onestep_rmse:.5f}",
                             v30=f"{roll_rmse:.5f}",
                             ep=f"{dt:.1f}s")
        else:
            print(f"epoch {epoch:3d}  K={kinfo:>2}  loss={total / tc.batches_per_epoch:.5e}  "
                  f"v1={onestep_rmse:.5f}  v30={roll_rmse:.5f}  ({dt:.1f}s)", flush=True)
        if (epoch + 1) % 5 == 0 or epoch == tc.epochs - 1:
            os.makedirs(config.exp_weights_dir(), exist_ok=True)
            torch.save(model.state_dict(), config.exp_weights_dir() / f"model_{mc.version}_{config.SOURCE}.pt")

    print("saved ->", config.exp_weights_dir() / f"model_{mc.version}_{config.SOURCE}.pt")


if __name__ == "__main__":
    main()

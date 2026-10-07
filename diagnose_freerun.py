# -*- coding: utf-8 -*-
"""
free-run 劣化诊断（对应 free_run改进方向.md 的 ⑤「诊断先行」）。

回答两个问题：
  Q1 漂移是「系统性 bias」还是「方差爆炸」？
     —— 证据有三条：
        (a) 逐日域均值误差 bias(t)：是否随时间单调增长（系统性漂移）；
        (b) 逐日跨站点误差标准差 spread(t)：是否随时间发散（方差爆炸）；
        (c) 逐站点 bias 的符号一致性：大多数站点同号 => 系统性，而非个别站点跑飞。
  Q2 表层是否「降水响应过强、回不到回落曲线」？
     —— 超级历元分析（superposed epoch）：以降水事件为锚点，比较表层真值 vs 预测的
        逐日轨迹（事件前 1 天 ~ 后 5 天），量化降水后被推高的幅度与回落是否滞后。

用法（默认对应 free_run改进方向.md 的 default/cldas 实验）：
  python diagnose_freerun.py
  python diagnose_freerun.py --experiment cross_forcing_era5 --source era5
  python diagnose_freerun.py --precip 3.0          # 降水事件阈值 mm/day
"""
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse

import numpy as np
import torch
import torch.nn.functional as F

import config
from data import build, rootzone_to_0_100
from model import build_model, free_run

VAR_NAMES = ["surface", "rootzone"]


def _precip_idx(source):
    """返回 forcing 里降水变量所在的列索引（era5 的 tp / cldas 的 PRCP 均为第 2 列）。"""
    spec = config.FORCING_ERA5 if source == "era5" else config.FORCING_CLDAS
    for i, (_sub, varname, _u) in enumerate(spec):
        if varname in ("tp", "PRCP"):
            return i
    return 2


def _flat_metrics(pred, truth):
    """展平后的整体指标：rmse / mae / bias / R / R²。"""
    p = np.asarray(pred).reshape(-1).astype(np.float64)
    t = np.asarray(truth).reshape(-1).astype(np.float64)
    m = np.isfinite(p) & np.isfinite(t)
    p, t = p[m], t[m]
    err = p - t
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    pm, tm = p - p.mean(), t - t.mean()
    denom = np.sqrt(np.sum(pm ** 2) * np.sum(tm ** 2)) + 1e-12
    r = float(np.sum(pm * tm) / denom)
    r2 = 1.0 - float(np.sum(err ** 2)) / (float(np.sum(tm ** 2)) + 1e-12)
    return rmse, mae, bias, r, r2


def one_step(model, mc, target_t, forcing_t, static_t, te_s, te_e):
    """delta 版 one-step：用真值 state_{t-1} + F_t 预测 t，返回预测 (N,T,2)。"""
    n = target_t.shape[0]
    pred = torch.empty((n, te_e - te_s, 2), device=target_t.device, dtype=target_t.dtype)
    with torch.no_grad():
        for t in range(te_s, te_e):
            pred[:, t - te_s, :], _ = model(target_t[:, t - 1], forcing_t[:, t], static_t)
    return pred.cpu().numpy()


def superposed_epoch(data, truth, precip, thresh, lags):
    """以降水事件（precip>thresh 的 (N,T) 掩膜）为锚点，返回各 lag 的均值轨迹与事件数。"""
    T = data.shape[1]
    E = precip > thresh
    n_events = int(E.sum())
    pred_means, truth_means = [], []
    idx0 = np.arange(T)
    for lag in lags:
        idx = idx0 + lag
        valid = (idx >= 0) & (idx < T)
        idx_c = np.clip(idx, 0, T - 1)
        m = E & valid[None, :]
        cnt = m.sum()
        if cnt == 0:
            pred_means.append(np.nan)
            truth_means.append(np.nan)
            continue
        pred_means.append(float((data[:, idx_c] * m).sum() / cnt))
        truth_means.append(float((truth[:, idx_c] * m).sum() / cnt))
    return np.array(pred_means), np.array(truth_means), n_events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default=config.EXPERIMENT, help="实验标识（cache/experiments/<exp>）")
    ap.add_argument("--source", default=config.SOURCE, help="训练源 era5/cldas（决定权重与 pack 缓存）")
    ap.add_argument("--precip", type=float, default=3.0, help="降水事件阈值 (mm/day)")
    args = ap.parse_args()

    config.EXPERIMENT = args.experiment
    config.SOURCE = args.source

    mc = config.cfg.model
    print("experiment =", config.EXPERIMENT, " source =", config.SOURCE, " weights =", config.ckpt_path())

    pack = build(config.SOURCE)
    target = pack["target"]                       # (N,T,2)，rootzone 已是 5-100cm（若 strict）
    forcing = pack["forcing"]                     # (N,T,6)
    static = pack["static"]
    n, T, _ = target.shape
    (te_s, te_e) = pack["split"]["test"]
    pi = _precip_idx(config.SOURCE)

    target_t = torch.from_numpy(np.ascontiguousarray(target)).float()
    forcing_t = torch.from_numpy(np.ascontiguousarray(forcing)).float()
    static_t = torch.from_numpy(np.ascontiguousarray(static)).float()

    model = build_model(mc)
    model.load_state_dict(torch.load(config.ckpt_path(), map_location="cpu", weights_only=True))
    model.eval()

    pred1 = one_step(model, mc, target_t, forcing_t, static_t, te_s, te_e)
    truth = np.ascontiguousarray(target[:, te_s:te_e])

    with torch.no_grad():
        fr = free_run(model, mc, target_t, forcing_t, static_t, te_s, te_e - te_s).cpu().numpy()

    fr_rep, truth_rep, pred1_rep = fr.copy(), truth.copy(), pred1.copy()
    if config.STRICT_ROOTZONE:
        fr_rep = rootzone_to_0_100(fr_rep)
        truth_rep = rootzone_to_0_100(truth_rep)
        pred1_rep = rootzone_to_0_100(pred1_rep)

    T_test = te_e - te_s
    days = np.arange(T_test)

    print(f"\n================ one-step vs free-run 整体指标（test {config.SPLIT_YEARS['test'][0]}） ================")
    for j, name in enumerate(VAR_NAMES):
        o = _flat_metrics(pred1_rep[:, :, j], truth_rep[:, :, j])
        f = _flat_metrics(fr_rep[:, :, j], truth_rep[:, :, j])
        print(f"  {name:8s}  one-step: RMSE={o[0]:.5f} bias={o[2]:+.5f} R={o[3]:.4f} R²={o[4]:.4f}")
        print(f"  {'':8s}  free-run: RMSE={f[0]:.5f} bias={f[2]:+.5f} R={f[3]:.4f} R²={f[4]:.4f}")

    # ---- Q1：逐日 bias / spread 演化 + 符号一致性 + 漂移率 ----
    print("\n================ Q1 漂移诊断：bias(t) / spread(t) / 符号一致性 ================")
    os_bias = {}
    for j, name in enumerate(VAR_NAMES):
        err = fr_rep[:, :, j] - truth_rep[:, :, j]           # (N,T)
        bias_t = err.mean(axis=0)                             # (T,)
        spread_t = err.std(axis=0)                            # (T,)
        truth_std = truth_rep[:, :, j].std(axis=0).mean()

        snap = (0, 90, 180, 270, T_test - 1)
        bias_snap = " ".join(f"{bias_t[t]:+.4f}" for t in snap)
        spread_snap = " ".join(f"{spread_t[t]:.4f}" for t in snap)

        # spread 是否随时间发散（早段 vs 晚段）
        spread_early = spread_t[:60].mean()
        spread_late = spread_t[-60:].mean()
        # 逐站点 bias 的符号一致性
        site_bias = err.mean(axis=1)                          # (N,)
        frac_pos = float((site_bias > 0).mean())
        frac_neg = float((site_bias < 0).mean())

        os_bias[name] = float((pred1_rep[:, :, j] - truth_rep[:, :, j]).mean())
        slope = float(np.polyfit(days, bias_t, 1)[0])

        print(f"  {name:8s}")
        print(f"     bias(t)    [{'/'.join(str(t) for t in snap)}] = {bias_snap}")
        print(f"     spread(t)  同上 = {spread_snap}")
        print(f"     spread 早/晚段均值 = {spread_early:.4f} / {spread_late:.4f}  (比值 {spread_late/(spread_early+1e-12):.2f}x)；"
              f"  全年 spread 均值={spread_t.mean():.4f} vs 真值 std≈{truth_std:.4f}")
        print(f"     逐站点 bias 符号：正 {frac_pos*100:.1f}% / 负 {frac_neg*100:.1f}%  "
              f"（若单侧 >70% => 系统性同号漂移）")
        print(f"     线性漂移率={slope:+.6f}/day (×366≈{slope*366:+.4f})；one-step 每步 bias={os_bias[name]:+.6f}")

    # ---- Q2：降水响应（superposed epoch）----
    print(f"\n================ Q2 降水响应（阈值 >{args.precip} mm/day，超级历元合成，表层） ================")
    precip = forcing[:, te_s:te_e, pi]
    lags = [-1, 0, 1, 2, 3, 4, 5]
    pm, tm, n_events = superposed_epoch(fr[:, :, 0], truth[:, :, 0], precip, args.precip, lags)
    print(f"  降水事件数（站点×天） = {n_events}")
    if n_events > 0:
        lag_err = pm - tm
        jump_p = pm[1] - pm[0]
        jump_t = tm[1] - tm[0]
        print("  lag        -1       0      +1      +2      +3      +4      +5")
        print("  pred   " + "  ".join(f"{v:.4f}" for v in pm))
        print("  truth  " + "  ".join(f"{v:.4f}" for v in tm))
        print("  err    " + "  ".join(f"{v:+.4f}" for v in lag_err))
        print(f"  事件当天增量：pred={jump_p:+.4f}  truth={jump_t:+.4f}  "
              f"=> 降水响应{'过强' if jump_p > jump_t else '并不强（增量略小于真值）'}")
        print(f"  事件前基线误差 lag-1={lag_err[0]:+.4f}；事件后误差 lag0={lag_err[1]:+.4f} -> lag5={lag_err[-1]:+.4f}  "
              f"=> {'回不到回落曲线：表层只升不降（排水不足）' if lag_err[-1] > lag_err[1] else '回落正常'}")

    _plot(args, fr_rep, truth_rep, fr, truth, precip, args.precip, days, lags, n_events)


def _plot(args, fr_rep, truth_rep, fr, truth, precip, thresh, days, lags, n_events):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    # 左上/右上：bias(t) 与 spread(t) 演化
    for j, name in enumerate(VAR_NAMES):
        ax = axes[0, j]
        err = fr_rep[:, :, j] - truth_rep[:, :, j]
        bias_t = err.mean(axis=0)
        spread_t = err.std(axis=0)
        ax.plot(days, bias_t, lw=1.2, color="tab:red", label="bias(t)")
        ax.fill_between(days, bias_t - spread_t, bias_t + spread_t, color="tab:red", alpha=0.15,
                        label="+/- spread(t)")
        c = np.polyfit(days, bias_t, 1)
        ax.plot(days, c[0] * days + c[1], lw=1.0, ls="--", color="k",
                label=f"linear drift {c[0]:+.4f}/day")
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_title(f"{name}: bias(t) & spread(t)")
        ax.set_xlabel("day of test year")
        ax.set_ylabel("error (m3/m3)")
        ax.legend(fontsize=8)

    # 左下：降水事件合成轨迹（表层）
    ax = axes[1, 0]
    pm, tm, _ = superposed_epoch(fr[:, :, 0], truth[:, :, 0], precip, thresh, lags)
    ax.plot(lags, tm, "o-", lw=1.5, color="tab:blue", label="SMAP truth")
    ax.plot(lags, pm, "s-", lw=1.5, color="tab:red", label="free-run")
    ax.axvline(0, color="gray", lw=0.8, ls=":")
    ax.set_title(f"surface precip composite (events={n_events}, >{thresh} mm/day)")
    ax.set_xlabel("lag relative to precip day")
    ax.set_ylabel("surface SM (m3/m3)")
    ax.legend(fontsize=8)

    # 右下：降水事件后的误差随 lag 演化
    ax = axes[1, 1]
    ax.plot(lags, pm - tm, "o-", lw=1.5, color="tab:purple")
    ax.axhline(0, color="gray", lw=0.8)
    ax.axvline(0, color="gray", lw=0.8, ls=":")
    ax.set_title("surface error vs lag after precip (pred - truth)")
    ax.set_xlabel("lag relative to precip day")
    ax.set_ylabel("error (m3/m3)")

    fig.suptitle(f"free-run diagnosis  [{args.experiment} / {args.source}]", y=1.0)
    fig.tight_layout()
    import os as _os
    _os.makedirs(config.exp_eval_dir(), exist_ok=True)
    out = config.exp_eval_dir() / "diagnose_freerun.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("\nsaved ->", out)


if __name__ == "__main__":
    main()

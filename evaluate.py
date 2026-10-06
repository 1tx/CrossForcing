# -*- coding: utf-8 -*-
"""
评估（支持 delta / lstm 两种模型，见 config.ModelConfig.version）：
  1. one-step：用真值状态预测下一步（diagnostic）
  2. free-run：从 2019-12-31 状态起，仅用 forcing 滚完 2020（"模拟陆面模式"的正确评测）
指标：RMSE / MAE / bias / 相关系数 R，分表层、根区。

注意：自回归循环必须包 torch.no_grad()，否则会构建 366 步计算图导致内存耗尽。
"""
import os

# 强制单线程（仅 CPU 生效；在 import numpy/torch 之前设置）
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
# 允许 torch 与 matplotlib 等库各自的 OpenMP 运行时共存（避免 libiomp5md.dll 重复初始化冲突）
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn.functional as F

import config
from data import build, rootzone_to_0_100
from model import build_model, free_run


def metrics(pred, truth):
    p = pred.reshape(-1)
    t = truth.reshape(-1)
    m = torch.isfinite(p) & torch.isfinite(t)
    p, t = p[m], t[m]
    rmse = torch.sqrt(F.mse_loss(p, t)).item()
    mae = torch.mean(torch.abs(p - t)).item()
    bias = (p.mean() - t.mean()).item()
    vp = p - p.mean()
    vt = t - t.mean()
    r = (vp * vt).sum() / (vp.norm() * vt.norm() + 1e-12)
    return rmse, mae, bias, r.item()


def site_metrics(pred, truth):
    """逐站点计算指标。输入 (N,T) 单变量，返回 (N,) 的 rmse/mae/bias/R/R2。"""
    p = pred.astype(np.float64)
    t = truth.astype(np.float64)
    diff = p - t
    rmse = np.sqrt(np.nanmean(diff ** 2, axis=1))
    mae = np.nanmean(np.abs(diff), axis=1)
    bias = np.nanmean(p, axis=1) - np.nanmean(t, axis=1)
    pm = p - np.nanmean(p, axis=1, keepdims=True)
    tm = t - np.nanmean(t, axis=1, keepdims=True)
    denom = np.sqrt(np.nansum(pm ** 2, axis=1)) * np.sqrt(np.nansum(tm ** 2, axis=1)) + 1e-12
    r = np.nansum(pm * tm, axis=1) / denom
    ss_res = np.nansum(diff ** 2, axis=1)
    ss_tot = np.nansum(tm ** 2, axis=1) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    return rmse, mae, bias, r, r2


def write_site_metrics_xlsx(pred1, fr, truth, ys, xs, out_path):
    """把 one-step 与 free-run 两套每站点指标写成一个 xlsx（两个 sheet：one_step / free_run）。"""
    try:
        import pandas as pd
    except ImportError:
        print("跳过每站点指标导出：缺少 pandas，请 `pip install pandas openpyxl`")
        return

    lat = config.GRID_LAT0 + ys * config.GRID_RES
    lon = config.GRID_LON0 + xs * config.GRID_RES
    var_names = ["surface", "rootzone"]
    metric_names = ["rmse", "mae", "bias", "R", "R2"]

    def build_df(pred):
        cols = {"ys": ys, "xs": xs, "lat": lat, "lon": lon}
        for j, vn in enumerate(var_names):
            rmse, mae, bias, r, r2 = site_metrics(pred[:, :, j], truth[:, :, j])
            for mn, v in zip(metric_names, [rmse, mae, bias, r, r2]):
                cols[f"{vn}_{mn}"] = v
        return pd.DataFrame(cols)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        build_df(pred1).to_excel(writer, sheet_name="one_step", index=False)
        build_df(fr).to_excel(writer, sheet_name="free_run", index=False)
    print("saved site metrics ->", out_path)


def plot_freerun(fr_pred, truth, out_path, version):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        dm_pred = fr_pred.mean(axis=0)   # (T,2)
        dm_tru = truth.mean(axis=0)
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        for j, name in enumerate(["surface", "rootzone"]):
            ax = axes[j]
            ax.plot(dm_pred[:, j], lw=1.0, label="free-run")
            ax.plot(dm_tru[:, j], lw=1.0, alpha=0.7, label="SMAP L4")
            ax.set_ylabel(f"{name} (m3/m3)")
            ax.legend()
        axes[1].set_xlabel("day of 2020")
        fig.suptitle(f"Domain-mean free-run vs SMAP L4 (test 2020, {version})")
        fig.tight_layout()
        fig.savefig(out_path, dpi=110)
        plt.close(fig)
        print("saved plot ->", out_path)
    except Exception as e:
        print("plot skipped:", e)


def main():
    tc = config.cfg.train
    mc = config.cfg.model
    device = torch.device(config.resolve_device(tc.device))
    print("device =", device)

    pack = build(config.SOURCE)
    target = pack["target"]
    forcing = pack["forcing"]
    ys = pack["ys"]
    xs = pack["xs"]
    static_t = torch.from_numpy(np.ascontiguousarray(pack["static"])).float().to(device)
    n, T, _ = target.shape
    (sp_s, sp_e) = pack["split"]["spinup"]
    (te_s, te_e) = pack["split"]["test"]
    L = mc.lookback

    model = build_model(mc).to(device)
    model.load_state_dict(torch.load(config.ckpt_path(),
                                     map_location=device, weights_only=True))
    model.eval()

    # 预载到 device（含 lstm 所需的 L 天历史），绝对时间索引与 free_run 一致
    t_dev = torch.from_numpy(np.ascontiguousarray(target)).float().to(device)
    f_dev = torch.from_numpy(np.ascontiguousarray(forcing)).float().to(device)
    arange_lb = torch.arange(L, device=device) - L          # [-L, ..., -1]

    # ---- one-step ----
    err = np.zeros((2, 4))
    pred1 = torch.empty((n, te_e - te_s, 2), device=device)   # 累积逐日 one-step 预测 (N,T,2)
    with torch.no_grad():
        if mc.version == "lstm":
            # 用真值 lookback 窗口 [t-L, t) 预测 t
            for t in range(te_s, te_e):
                idx = t + arange_lb
                pred, _ = model(t_dev[:, idx, :], f_dev[:, idx, :], static_t)
                truth = t_dev[:, t]
                pred1[:, t - te_s, :] = pred
                for j in range(2):
                    err[j] += np.array(metrics(pred[:, j], truth[:, j]))
            err /= (te_e - te_s)
        else:
            # 用 (state_{t-1}, F_t) 预测 t（与 free_run 一致：state_{t-1} + F_t -> state_t）
            for t in range(te_s, te_e):
                pred, _ = model(t_dev[:, t - 1], f_dev[:, t], static_t)
                truth = t_dev[:, t]
                pred1[:, t - te_s, :] = pred
                for j in range(2):
                    err[j] += np.array(metrics(pred[:, j], truth[:, j]))
            err /= (te_e - te_s)

    scope = "5-100cm" if config.STRICT_ROOTZONE else "0-100cm"
    print(f"=== one-step (test 2020, {mc.version}, rootzone={scope}) ===")
    for j, name in enumerate(["surface", "rootzone"]):
        print(f"  {name:8s}: RMSE={err[j, 0]:.5f}  MAE={err[j, 1]:.5f}  bias={err[j, 2]:.5f}  R={err[j, 3]:.4f}")

    # ---- free-run ----
    with torch.no_grad():
        fr_gpu = free_run(model, mc, t_dev, f_dev, static_t, te_s, te_e - te_s)
    fr = fr_gpu.cpu().numpy()
    truth = np.ascontiguousarray(target[:, te_s:te_e])
    # 严格质量平衡：把 5-100cm 口径还原成 SMAP 0-100cm 口径再对比
    if config.STRICT_ROOTZONE:
        fr = rootzone_to_0_100(fr)
        truth = rootzone_to_0_100(truth)

    print(f"=== free-run (test 2020, 自 2019-12-31 滚动, {mc.version}) ===")
    for j, name in enumerate(["surface", "rootzone"]):
        p = torch.from_numpy(np.ascontiguousarray(fr[:, :, j]))
        t = torch.from_numpy(np.ascontiguousarray(truth[:, :, j]))
        rmse, mae, bias, r = metrics(p, t)
        print(f"  {name:8s}: RMSE={rmse:.5f}  MAE={mae:.5f}  bias={bias:.5f}  R={r:.4f}")

    os.makedirs(config.exp_eval_dir(), exist_ok=True)
    plot_freerun(fr, truth, str(config.exp_eval_dir() / f"freerun_2020_{mc.version}.png"), mc.version)

    # ---- 每站点指标 -> Excel（one-step 与 free-run 两套，统一还原到 SMAP 0-100cm 口径）----
    pred1_np = pred1.cpu().numpy()
    if config.STRICT_ROOTZONE:
        pred1_np = rootzone_to_0_100(pred1_np)
    year = config.SPLIT_YEARS["test"][0]
    write_site_metrics_xlsx(pred1_np, fr, truth, ys, xs,
                            str(config.exp_eval_dir() / f"site_metrics_{year}.xlsx"))


if __name__ == "__main__":
    main()

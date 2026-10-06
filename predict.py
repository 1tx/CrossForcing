# -*- coding: utf-8 -*-
"""
推理（支持 delta / lstm）：加载训练好的模型，对 test 年做 free-run，把逐日预测重构回完整 160x200 网格并保存。

输出（都在 config.exp_predict_dir()）：
  freerun_<version>_<source>_<year>_flat.npz   展平预测 (N,T,2)，含 ys/xs
  freerun_<version>_<source>_<year>_grid.nc    完整网格 (T,160,200,2)，掩膜外 -9999.0，含 lat/lon/time
  freerun_<version>_<source>_<year>_map.png    表层/根区年均空间分布图

用法：
  python predict.py                       # version/source 取 config，权重取对应 checkpoint
  python predict.py --source cldas        # cross-forcing：用 cldas 驱动，权重仍取 config.SOURCE 的
  python predict.py --ckpt <路径>         # 显式指定权重文件
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
# 允许 torch 与 matplotlib 等库各自的 OpenMP 运行时共存（避免 libiomp5md.dll 重复初始化冲突）
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import numpy as np
import torch
import torch.nn.functional as F

import config
from data import build, rootzone_to_0_100
from model import build_model, free_run


def _metrics(pred, truth):
    p = torch.from_numpy(np.ascontiguousarray(pred)).reshape(-1)
    t = torch.from_numpy(np.ascontiguousarray(truth)).reshape(-1)
    m = torch.isfinite(p) & torch.isfinite(t)
    p, t = p[m], t[m]
    rmse = torch.sqrt(F.mse_loss(p, t)).item()
    mae = torch.mean(torch.abs(p - t)).item()
    bias = (p.mean() - t.mean()).item()
    vp = p - p.mean()
    vt = t - t.mean()
    r = (vp * vt).sum() / (vp.norm() * vt.norm() + 1e-12)
    return rmse, mae, bias, r.item()


def _to_grid(pred_flat, mask, ys, xs):
    """(N,T,2) -> (T,H,W,2)，掩膜外填 NaN。"""
    T = pred_flat.shape[1]
    H, W = mask.shape
    out = np.full((T, H, W, 2), np.nan, dtype=np.float32)
    out[:, ys, xs, :] = pred_flat.transpose(1, 0, 2)   # (N,T,2) -> (T,N,2) 对齐索引
    return out


def _save_netcdf(grid, time_idx, out_path, lat, lon):
    from netCDF4 import Dataset
    T, H, W, V = grid.shape
    ds = Dataset(str(out_path), "w", format="NETCDF4")
    ds.createDimension("time", T)
    ds.createDimension("lat", H)
    ds.createDimension("lon", W)
    t = ds.createVariable("time", "i4", ("time",))
    t.units = f"days since {config.TARGET_START_YEAR}-01-01 00:00:00"
    t[:] = time_idx
    ds.createVariable("lat", "f4", ("lat",))[:] = lat
    ds.createVariable("lon", "f4", ("lon",))[:] = lon
    names = ["sm_surface", "sm_rootzone"]
    for v in range(V):
        var = ds.createVariable(names[v], "f4", ("time", "lat", "lon"),
                                fill_value=np.float32(-9999.0))
        var.units = "m3 m-3"
        data = grid[..., v].copy()
        data[~np.isfinite(data)] = -9999.0
        var[:] = data
    ds.close()


def _plot_maps(grid, out_path, lat, lon):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ann = np.nanmean(grid, axis=0)  # (H,W,2)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for j, name in enumerate(["surface", "rootzone"]):
            ax = axes[j]
            im = ax.pcolormesh(lon, lat, ann[..., j], cmap="viridis", shading="auto")
            ax.set_title(name)
            ax.set_xlabel("lon")
            ax.set_ylabel("lat")
            fig.colorbar(im, ax=ax, label="m3/m3")
        fig.suptitle("annual-mean free-run soil moisture")
        fig.tight_layout()
        fig.savefig(out_path, dpi=110)
        plt.close(fig)
        print("saved map ->", out_path)
    except Exception as e:
        print("map plot skipped:", e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=config.SOURCE,
                    help="驱动 forcing：era5 | cldas（默认 config.SOURCE）")
    ap.add_argument("--ckpt", default=None,
                    help="显式指定权重文件；缺省取 config 对应 checkpoint")
    ap.add_argument("--year", type=int, default=None, help="仅用于输出文件名，默认取 test 年")
    args = ap.parse_args()

    mc = config.cfg.model
    device = torch.device(config.resolve_device(config.cfg.train.device))
    print("device =", device)

    pack = build(args.source)
    target = pack["target"]
    forcing = pack["forcing"]
    mask = pack["mask"]
    ys = pack["ys"]
    xs = pack["xs"]
    n, T, _ = target.shape
    (sp_s, sp_e) = pack["split"]["spinup"]
    (te_s, te_e) = pack["split"]["test"]
    static_t = torch.from_numpy(np.ascontiguousarray(pack["static"])).float().to(device)

    ckpt = args.ckpt or str(config.ckpt_path(config.SOURCE))
    model = build_model(mc).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    print("weights <-", ckpt)

    # free-run：预载到 device（含 lstm 所需历史），绝对时间索引
    t_dev = torch.from_numpy(np.ascontiguousarray(target)).float().to(device)
    f_dev = torch.from_numpy(np.ascontiguousarray(forcing)).float().to(device)
    with torch.no_grad():
        fr_gpu = free_run(model, mc, t_dev, f_dev, static_t, te_s, te_e - te_s)
    fr = fr_gpu.cpu().numpy()

    # 严格质量平衡：把 5-100cm 口径还原成 SMAP 0-100cm 口径（指标与 netCDF 输出均用还原后口径）
    truth = target[:, te_s:te_e]
    if config.STRICT_ROOTZONE:
        fr = rootzone_to_0_100(fr)
        truth = rootzone_to_0_100(truth)

    year = args.year if args.year is not None else config.SPLIT_YEARS["test"][0]
    base = f"freerun_{mc.version}_{args.source}_{year}"
    os.makedirs(config.exp_predict_dir(), exist_ok=True)
    np.savez(config.exp_predict_dir() / f"{base}_flat.npz", fr=fr, ys=ys, xs=xs)
    print("saved ->", config.exp_predict_dir() / f"{base}_flat.npz")

    # 指标（对照 SMAP 真值）
    print(f"=== free-run (test {year}, {mc.version}) ===")
    for j, name in enumerate(["surface", "rootzone"]):
        rmse, mae, bias, r = _metrics(fr[:, :, j], truth[:, :, j])
        print(f"  {name:8s}: RMSE={rmse:.5f}  MAE={mae:.5f}  bias={bias:.5f}  R={r:.4f}")

    # 完整网格 + netCDF
    grid = _to_grid(fr, mask, ys, xs)
    lat = config.GRID_LAT0 + np.arange(grid.shape[1]) * config.GRID_RES
    lon = config.GRID_LON0 + np.arange(grid.shape[2]) * config.GRID_RES
    time_idx = te_s + np.arange(te_e - te_s, dtype=np.int32)
    _save_netcdf(grid, time_idx, config.exp_predict_dir() / f"{base}_grid.nc", lat, lon)
    print("saved ->", config.exp_predict_dir() / f"{base}_grid.nc")

    _plot_maps(grid, config.exp_predict_dir() / f"{base}_map.png", lat, lon)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
随机选几个站点，画真值（SMAP L4）vs free-run 模拟值的时间序列图（test 年）。

用法（手动指定实验，不读 config.py 里的 EXPERIMENT/SOURCE）：
  python plot_time_series.py                            # 默认 default 实验 + cldas
  python plot_time_series.py --experiment default --source cldas
  python plot_time_series.py --experiment cross_forcing_era5 --source era5 --sites 4
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse

import numpy as np
import torch

import config
from data import build, rootzone_to_0_100
from model import build_model, free_run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="cross_forcing_era5", help="实验标识（cache/experiments/<exp>）")
    ap.add_argument("--source", default="era5", help="训练源 era5/cldas（决定权重文件名与 pack 缓存）")
    ap.add_argument("--sites", type=int, default=6, help="随机选的站点数")
    ap.add_argument("--seed", type=int, default=0, help="随机种子")
    args = ap.parse_args()

    # 手动指定实验（覆盖 config 模块的全局值，不修改 config.py 文件本身）
    config.EXPERIMENT = args.experiment
    config.SOURCE = args.source

    mc = config.cfg.model
    tc = config.cfg.train
    device = torch.device(config.resolve_device(tc.device))
    print("device     =", device)
    print("experiment =", config.EXPERIMENT)
    print("source     =", config.SOURCE)
    print("weights    =", config.ckpt_path())
    print("pack cache =", config.exp_pack_dir() / f"pack_{config.SOURCE}{'_strict' if config.STRICT_ROOTZONE else ''}.npz")

    pack = build(config.SOURCE)
    target = pack["target"]
    forcing = pack["forcing"]
    ys = pack["ys"]
    xs = pack["xs"]
    static_t = torch.from_numpy(np.ascontiguousarray(pack["static"])).float().to(device)
    n, T, _ = target.shape
    (te_s, te_e) = pack["split"]["test"]

    model = build_model(mc).to(device)
    model.load_state_dict(torch.load(config.ckpt_path(), map_location=device, weights_only=True))
    model.eval()
    print("weights <-", config.ckpt_path())

    t_dev = torch.from_numpy(np.ascontiguousarray(target)).float().to(device)
    f_dev = torch.from_numpy(np.ascontiguousarray(forcing)).float().to(device)

    # free-run 滚完 test 年
    with torch.no_grad():
        fr_gpu = free_run(model, mc, t_dev, f_dev, static_t, te_s, te_e - te_s)
    fr = fr_gpu.cpu().numpy()                            # (N, T, 2)
    truth = np.ascontiguousarray(target[:, te_s:te_e])   # (N, T, 2)

    # 统一还原到 SMAP 0-100cm 口径
    if config.STRICT_ROOTZONE:
        fr = rootzone_to_0_100(fr)
        truth = rootzone_to_0_100(truth)

    # 随机选站点
    rng = np.random.default_rng(args.seed)
    k = min(args.sites, n)
    idx = rng.choice(n, size=k, replace=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    var_names = ["surface", "rootzone"]
    days = np.arange(te_e - te_s)
    fig, axes = plt.subplots(k, 2, figsize=(14, 2.1 * k), squeeze=False)

    for r, i in enumerate(idx):
        lat = config.GRID_LAT0 + ys[i] * config.GRID_RES
        lon = config.GRID_LON0 + xs[i] * config.GRID_RES
        for c, vn in enumerate(var_names):
            ax = axes[r, c]
            ax.plot(days, truth[i, :, c], lw=1.0, color="tab:blue", label="SMAP L4 (truth)")
            ax.plot(days, fr[i, :, c], lw=1.0, color="tab:red", alpha=0.8, label="free-run")
            ax.set_title(f"site #{i}  {vn}  (lat {lat:.2f}, lon {lon:.2f})", fontsize=9)
            ax.set_ylabel("m3/m3")
            if r == 0:
                ax.legend(fontsize=8, loc="upper right")
    for ax in axes[-1]:
        ax.set_xlabel("day of 2020")

    fig.suptitle(f"Truth vs free-run at {k} random sites (test 2020, {mc.version})", y=1.0)
    fig.tight_layout()

    os.makedirs(config.exp_eval_dir(), exist_ok=True)
    out = config.exp_eval_dir() / "time_series_samples.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("saved ->", out)


if __name__ == "__main__":
    main()

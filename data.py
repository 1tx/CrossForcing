# -*- coding: utf-8 -*-
"""
数据装载与打包（不做预归一化，保留原始物理单位；量纲统一交给模型内部的 norm 层）。

流程：
  1. 按文件名顺序拼接各源（均为逐日、同一 160x200 网格，填充值 -9999.0）
  2. 切片到目标时段 2016-2020
  3. 计算陆地有效掩膜 mask
  4. 抽取有效站点 -> target[N,T,2] / forcing[N,T,F] / static[N,S]
  5. 缓存到 config.exp_pack_dir()/pack_<source>.npz（按实验分目录）

用法：
  from data import build
  pack = build("era5")          # 或 "cldas"
"""
import os
import re
import glob
import datetime
import numpy as np
from netCDF4 import Dataset
from pathlib import Path

import config


def _ym(name):
    """从文件名解析 (year, month)。兼容 ERA5LAND_2016_01_... 与 CLDAS...-201601_... 两种命名。"""
    m = re.search(r"20\d{2}[_-]?(\d{2})", name)
    if not m:
        raise ValueError(f"无法从文件名解析年月: {name}")
    year = int(m.group(0)[:4])
    month = int(m.group(1))
    return year, month


def _load_var(dirpath, varname, y_lo, y_hi):
    """读取某变量目录下所有 monthly nc，按时间拼接，切片到 [y_lo, y_hi]。返回 (T, lat, lon) float32。"""
    files = sorted(glob.glob(str(Path(dirpath) / "*.nc")))
    arrs = []
    for f in files:
        y, m = _ym(os.path.basename(f))
        if not (y_lo <= y <= y_hi):
            continue
        ds = Dataset(f)
        v = ds.variables[varname]
        a = np.asarray(v[:]).astype(np.float32)
        fill = getattr(v, "_FillValue", None)
        ds.close()
        if fill is not None:
            a[a == fill] = np.nan
        a[np.abs(a) > 1e30] = np.nan
        arrs.append(a)
    return np.concatenate(arrs, axis=0)  # (T, lat, lon)


def _day_split():
    """各段在 2016-01-01 起算的日索引 [start, end)。"""
    d0 = datetime.date(config.TARGET_START_YEAR, 1, 1)
    split = {}
    for k, (y0, y1) in config.SPLIT_YEARS.items():
        s = (datetime.date(y0, 1, 1) - d0).days
        e = (datetime.date(y1 + 1, 1, 1) - d0).days
        split[k] = (s, e)
    return split


def _load_target():
    y_lo, y_hi = config.TARGET_START_YEAR, config.TARGET_END_YEAR
    layers = []
    for varname, _u, _d in config.TARGET_VARS:
        layers.append(_load_var(config.SMAP_DIR / varname, varname, y_lo, y_hi))
    target = np.stack(layers, axis=-1)  # (T, lat, lon, 2)
    # 物理有效区间：土壤体积含水量 θ ∈ [0, 1.0] m3/m3。
    # 实测 sm_rootzone 在 2016-06 有 ~1.7 万像元被污染(1.19~99.49)，必须剔除。
    target[(target < 0.0) | (target > 1.0)] = np.nan
    return target


def _forcing_spec(source):
    return config.FORCING_ERA5 if source == "era5" else config.FORCING_CLDAS


# 变量级尺度转换（按 netCDF 变量名匹配，只作用于命中的变量）：
#   CLDAS 的 PRCP 是"1 小时降水量日平均"(mm/hr)，需 ×24 转为日总量(mm)。
_FORCING_SCALE = {"PRCP": 24.0}


def _load_forcing(source):
    y_lo, y_hi = config.TARGET_START_YEAR, config.TARGET_END_YEAR
    base = config.ERA5_DIR if source == "era5" else config.CLDAS_DIR
    layers = []
    for sub, varname, _u in _forcing_spec(source):
        a = _load_var(base / sub, varname, y_lo, y_hi)
        if varname in _FORCING_SCALE:
            a = a * _FORCING_SCALE[varname]
        layers.append(a)
    return np.stack(layers, axis=-1)  # (T, lat, lon, F)


def _load_static():
    layers = []
    for varname, _u, _d in config.STATIC_VARS:
        f = config.STATIC_DIR / f"{varname}_05cm.nc"
        ds = Dataset(f)
        v = ds.variables[varname]
        a = np.asarray(v[:]).astype(np.float32)
        fill = getattr(v, "_FillValue", None)
        ds.close()
        if fill is not None:
            a[a == fill] = np.nan
        a[np.abs(a) > 1e30] = np.nan
        layers.append(a)
    return np.stack(layers, axis=-1)  # (lat, lon, S)


def _fill_nan_time(arr):
    """对每个站点、每个变量沿时间维线性插值填 NaN（首尾用最近有效值外推）。arr: (N, T, V)。"""
    n, t, v = arr.shape
    x = np.arange(t)
    out = arr.copy()
    for i in range(n):
        for j in range(v):
            y = arr[i, :, j]
            m = np.isfinite(y)
            if m.all() or m.sum() == 0:
                continue
            out[i, :, j] = np.interp(x, x[m], y[m])
    return out


def rootzone_to_5_100(arr):
    """严格质量平衡·正向：把 (...,2) 的 [surface, rootzone(0-100cm)] 转成 [surface, rootzone(5-100cm)]。
    水量守恒 1000·R = 50·S + 950·Z ⇒ Z = (1000·R − 50·S)/950，再 clamp 到 [0,1] 兜底（个别 S>R 的异常像元）。
    """
    s = arr[..., 0]
    r = arr[..., 1]
    z = (1000.0 * r - 50.0 * s) / 950.0
    z = np.clip(z, 0.0, 1.0)
    out = arr.copy()
    out[..., 1] = z
    return out


def rootzone_to_0_100(arr):
    """严格质量平衡·反向：把 (...,2) 的 [surface, rootzone(5-100cm)] 还原成 [surface, rootzone(0-100cm)]（SMAP 口径）。
    R = (50·S + 950·Z)/1000，是深度加权平均，自动落在 [0,1]，无需 clamp。
    """
    s = arr[..., 0]
    z = arr[..., 1]
    out = arr.copy()
    out[..., 1] = (50.0 * s + 950.0 * z) / 1000.0
    return out


def build(source="era5", rebuild=False):
    os.makedirs(config.exp_pack_dir(), exist_ok=True)
    suffix = "_strict" if config.STRICT_ROOTZONE else ""
    cache_file = config.exp_pack_dir() / f"pack_{source}{suffix}.npz"
    if cache_file.exists() and not rebuild:
        z = np.load(cache_file)
        return {"target": z["target"], "forcing": z["forcing"], "static": z["static"],
                "mask": z["mask"], "ys": z["ys"], "xs": z["xs"], "split": _day_split()}

    target = _load_target()            # (T,160,200,2)
    forcing = _load_forcing(source)    # (T,160,200,F)
    static = _load_static()            # (160,200,S)

    t_valid = np.mean(np.isfinite(target), axis=0)   # (160,200,2)
    f_valid = np.mean(np.isfinite(forcing), axis=0)  # (160,200,F)
    mask = (t_valid.min(-1) >= config.MASK_MIN_VALID_FRAC) \
        & (f_valid.min(-1) >= config.FORCING_MIN_VALID_FRAC) \
        & np.isfinite(static).all(-1)

    ys, xs = np.where(mask)
    n = len(ys)
    target_s = np.ascontiguousarray(target[:, ys, xs, :].transpose(1, 0, 2))    # (N,T,2)
    forcing_s = np.ascontiguousarray(forcing[:, ys, xs, :].transpose(1, 0, 2))  # (N,T,F)
    static_s = np.ascontiguousarray(static[ys, xs, :])                           # (N,S)
    del target, forcing, static

    # 填掉有效站点内残余的个别缺测日（时间维线性插值）
    target_s = _fill_nan_time(target_s)
    forcing_s = _fill_nan_time(forcing_s)
    print(f"[data] 插值后残余 NaN: target={int(np.isnan(target_s).sum())}  forcing={int(np.isnan(forcing_s).sum())}")

    # 严格质量平衡：把根区目标从 0-100cm 扣除为 5-100cm（消除两层重叠）
    if config.STRICT_ROOTZONE:
        target_s = rootzone_to_5_100(target_s)
        print("[data] STRICT_ROOTZONE=True：根区目标已从 0-100cm 扣除为 5-100cm")

    split = _day_split()

    print(f"[data] source={source}  陆地有效站点 N={n}  (占比 {n / (160 * 200) * 100:.1f}%)")
    print(f"[data] target 范围 (m3/m3):  surf [{np.nanmin(target_s[..., 0]):.3f}, {np.nanmax(target_s[..., 0]):.3f}]  "
          f"root [{np.nanmin(target_s[..., 1]):.3f}, {np.nanmax(target_s[..., 1]):.3f}]")
    for i, (sub, vn, u) in enumerate(_forcing_spec(source)):
        print(f"[data] forcing[{i}] {vn:8s} [{np.nanmin(forcing_s[..., i]):.5g}, {np.nanmax(forcing_s[..., i]):.5g}] {u}")
    for i, (vn, u, _d) in enumerate(config.STATIC_VARS):
        print(f"[data] static[{i}]  {vn:5s} [{np.nanmin(static_s[..., i]):.5g}, {np.nanmax(static_s[..., i]):.5g}] {u}")
    for k, (s, e) in split.items():
        print(f"[data] split {k:6s}: days [{s}, {e})  共 {e - s} 天")

    np.savez(cache_file, target=target_s, forcing=forcing_s, static=static_s,
             mask=mask, ys=ys, xs=xs)
    return {"target": target_s, "forcing": forcing_s, "static": static_s,
            "mask": mask, "ys": ys, "xs": xs, "split": split}


if __name__ == "__main__":
    build("era5")

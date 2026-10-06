# -*- coding: utf-8 -*-
"""
将 ERA5-Land 逐小时数据聚合为逐日数据。

聚合规则：
  - 降水 tp      -> 日求和 (mm/日)      [小时率 mm/hr x 24h 求和]
  - 其余变量     -> 日平均 (mean)

缺失值处理：按各变量自带的 _FillValue 掩膜后聚合；
输出统一使用 _FillValue = -9999.0。

输出：Northeast_China_era5land_daily/<subfolder>/ERA5LAND_YYYY_MM_<var>_daily.nc
"""
from netCDF4 import Dataset, num2date
from datetime import datetime
import numpy as np
import glob
import os
import warnings

warnings.filterwarnings("ignore")

SRC = "Northeast_China_era5land"
DST = "Northeast_China_era5land_daily"

# 变量配置：子目录名 -> (netCDF 变量名, 聚合方式 'mean'/'sum', 输出单位覆盖[None=沿用源单位])
VARS = {
    "10m_wind_speed": ("wind_speed", "mean", None),
    "2m_temperature": ("t2m", "mean", None),
    "precipitation": ("tp", "sum", "mm/day"),   # 小时率求和后为日总量
    "specific_humidity": ("Q", "mean", None),
    "surface_solar_radiation_downwards_w_m2": ("ssrd", "mean", None),
    "surface_thermal_radiation_downwards_w_m2": ("strd", "mean", None),
}

FILL_OUT = -9999.0


def mask_invalid(a, fill_value):
    """把填充值/异常大值转为 NaN。"""
    a = a.astype(np.float64)
    invalid = (a == fill_value) | (a > 1e30)
    a[invalid] = np.nan
    return a


def aggregate_month(src_path, var_name, mode):
    ds = Dataset(src_path, "r")
    tv = ds.variables["time"]
    t = np.asarray(tv[:])
    tunits = tv.units
    lat = ds.variables["latitude"][:]
    lon = ds.variables["longitude"][:]
    var = ds.variables[var_name]
    a = np.asarray(var[:])                      # (n_hours, lat, lon)
    fill = var._FillValue
    attrs = {k: var.getncattr(k) for k in var.ncattrs()
             if k not in ("_FillValue", "missing_value")}
    ds.close()

    n_hours = a.shape[0]
    n_days = n_hours // 24
    if n_hours % 24 != 0:
        raise RuntimeError(f"{src_path}: 小时数 {n_hours} 不是 24 的整数倍")

    a = a[: n_days * 24].reshape(n_days, 24, a.shape[1], a.shape[2])
    a = mask_invalid(a, fill)

    with np.errstate(invalid="ignore"):
        if mode == "sum":
            daily = np.nansum(a, axis=1)        # (n_days, lat, lon)
        else:
            daily = np.nanmean(a, axis=1)

    # 全时段缺失(整日为 NaN)的像元 -> 输出填充值
    all_missing = np.all(np.isnan(a), axis=1)   # (n_days, lat, lon)
    daily[all_missing] = FILL_OUT
    daily = np.where(np.isnan(daily), FILL_OUT, daily).astype(np.float32)

    # 逐日时间戳：由源文件时间推算每月绝对日期（避免每月从 0 起算）
    dates = num2date(t[: n_days * 24], tunits,
                     only_use_cftime_datetimes=False,
                     only_use_python_datetimes=True)
    day_offsets = []
    for d in range(n_days):
        dt = dates[d * 24]                      # 每天的第一个小时
        day0 = datetime(dt.year, dt.month, dt.day)
        day_offsets.append((day0 - datetime(2010, 1, 1)).days * 24.0)
    day_offsets = np.array(day_offsets, dtype="f8")

    return daily, day_offsets, lat, lon, attrs


def write_daily(dst_path, daily, day_offsets, lat, lon, attrs, var_name, units_override=None):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    ds = Dataset(dst_path, "w")
    ds.createDimension("time", len(day_offsets))
    ds.createDimension("latitude", lat.size)
    ds.createDimension("longitude", lon.size)

    t = ds.createVariable("time", "f8", ("time",))
    t.units = "hours since 2010-01-01 00:00:00"
    t.calendar = "gregorian"
    t.standard_name = "time"
    t[:] = day_offsets

    la = ds.createVariable("latitude", "f4", ("latitude",))
    la.units = "degrees_north"
    la.standard_name = "latitude"
    la[:] = lat

    lo = ds.createVariable("longitude", "f4", ("longitude",))
    lo.units = "degrees_east"
    lo.standard_name = "longitude"
    lo[:] = lon

    v = ds.createVariable(var_name, "f4", ("time", "latitude", "longitude"),
                          fill_value=FILL_OUT)
    v[:] = daily
    for k, val in attrs.items():
        try:
            v.setncattr(k, val)
        except Exception:
            pass
    # _FillValue 已在 createVariable(fill_value=...) 时设定，无需再赋值
    v.missing_value = FILL_OUT
    if units_override is not None:
        v.units = units_override

    ds.close()


def main():
    total_files = 0
    for sub, (var_name, mode, units_override) in VARS.items():
        src_files = sorted(glob.glob(os.path.join(SRC, sub, "*.nc")))
        print(f"\n=== {sub}  ({var_name}, mode={mode})  n_files={len(src_files)} ===")
        for i, sf in enumerate(src_files, 1):
            daily, day_offsets, lat, lon, attrs = aggregate_month(sf, var_name, mode)
            base = os.path.basename(sf).replace("_cut.nc", "_daily.nc")
            # 文件名里的变量片段统一换成 var_name
            base = base.replace("total_precipitation_m_hr", var_name)
            dst = os.path.join(DST, sub, base)
            write_daily(dst, daily, day_offsets, lat, lon, attrs, var_name, units_override)
            if i % 12 == 0 or i == len(src_files):
                print(f"  [{i}/{len(src_files)}] {os.path.basename(sf)} -> {len(day_offsets)} days")
            total_files += 1
    print(f"\n完成：共聚合 {total_files} 个文件。")


if __name__ == "__main__":
    main()

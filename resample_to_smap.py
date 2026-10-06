# -*- coding: utf-8 -*-
"""
将 ERA5-Land(日尺度)、CLDAS、SoilGrids 重采样/聚合到 SMAP 的 0.1° 网格。

方法：
  - ERA5-Land / CLDAS : 双线性插值 (scipy RegularGridInterpolator)
  - SoilGrids (250m)  : 面积加权聚合 (每个 SMAP 格点内所有 250m 像元取平均)

坐标基准：SMAP 的 lat/lon (115.05~134.95°E, 38.05~53.95°N, 200×160)。
输出填充值统一为 -9999.0。
"""
from netCDF4 import Dataset
from scipy.interpolate import RegularGridInterpolator
import numpy as np
import glob
import os
import warnings

warnings.filterwarnings("ignore")

SMAP_REF = "Northeast_China_SMAPL4/sm_surface/SMAP_L4_SM_gph_201601_sm_surface_cut.nc"
FILL = -9999.0

ERA5_SRC = "Northeast_China_era5land_daily"
ERA5_DST = "Northeast_China_era5land_daily_smap"
CLDAS_SRC = "CLDAS"
CLDAS_DST = "CLDAS_smap"
STATIC_SRC = "Northeast_China_static/soilgrids"
STATIC_DST = "Northeast_China_static_smap"

ERA5_VARS = {
    "10m_wind_speed": "wind_speed",
    "2m_temperature": "t2m",
    "precipitation": "tp",
    "specific_humidity": "Q",
    "surface_solar_radiation_downwards_w_m2": "ssrd",
    "surface_thermal_radiation_downwards_w_m2": "strd",
}
# CLDAS 子目录 -> (变量名, 填充值)
CLDAS_VARS = {
    "PRE": ("PRCP", -999.0),
    "SHU": ("QAIR", -999.0),
    "SSRA": ("SWDN", -999.0),
    "TMP": ("TAIR", -999.0),
    "WIN": ("WIND", -999.0),
    "tstr": ("strd", -32767.0),
}
# SoilGrids 文件 -> (变量名, 描述)
SOIL_FILES = {
    "clay_05cm_250m.nc": ("clay", "Clay content (0-5 cm)"),
    "sand_05cm_250m.nc": ("sand", "Sand content (0-5 cm)"),
    "silt_05cm_250m.nc": ("silt", "Silt content (0-5 cm)"),
}


def load_smap_grid():
    ds = Dataset(SMAP_REF)
    lat = np.asarray(ds.variables["lat"][:]).astype(np.float64)
    lon = np.asarray(ds.variables["lon"][:]).astype(np.float64)
    ds.close()
    return lat, lon


def regrid_bilinear(data, src_lat, src_lon, dst_lat, dst_lon, fill_value):
    """data: (time, lat, lon) -> (time, dst_lat, dst_lon)，双线性插值。"""
    work = data.astype(np.float64)
    work[work == fill_value] = np.nan
    work[np.abs(work) > 1e30] = np.nan
    if src_lat[1] < src_lat[0]:            # 纬度降序 -> 升序
        src_lat = src_lat[::-1].copy()
        work = work[:, ::-1, :]
    if src_lon[1] < src_lon[0]:
        src_lon = src_lon[::-1].copy()
        work = work[:, :, ::-1]

    vals = np.moveaxis(work, 0, -1)        # (lat, lon, time)
    interp = RegularGridInterpolator(
        (src_lat, src_lon), vals, method="linear",
        bounds_error=False, fill_value=np.nan)
    latg, longg = np.meshgrid(dst_lat, dst_lon, indexing="ij")
    pts = np.column_stack([latg.ravel(), longg.ravel()])
    out = interp(pts)                       # (npts, time)
    out = out.reshape(len(dst_lat), len(dst_lon), -1)
    out = np.moveaxis(out, -1, 0)           # (time, lat, lon)
    out = np.where(np.isnan(out), FILL, out).astype(np.float32)
    return out


def write_time_latlon_nc(out_path, data, var_name, attrs, time_vals, time_attrs, lat, lon):
    """写出 NetCDF：time 沿用源，lat/lon 用 SMAP 网格。"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ds = Dataset(out_path, "w")
    ds.createDimension("time", data.shape[0])
    ds.createDimension("latitude", lat.size)
    ds.createDimension("longitude", lon.size)

    t = ds.createVariable("time", "f8", ("time",))
    for a, val in time_attrs.items():
        try:
            t.setncattr(a, val)
        except Exception:
            pass
    t[:] = np.asarray(time_vals)

    la = ds.createVariable("latitude", "f8", ("latitude",))
    la.units = "degrees_north"; la.standard_name = "latitude"; la[:] = lat
    lo = ds.createVariable("longitude", "f8", ("longitude",))
    lo.units = "degrees_east"; lo.standard_name = "longitude"; lo[:] = lon

    v = ds.createVariable(var_name, "f4", ("time", "latitude", "longitude"),
                          fill_value=FILL)
    v[:] = data
    for k, val in attrs.items():
        if k in ("_FillValue", "missing_value"):
            continue
        try:
            v.setncattr(k, val)
        except Exception:
            pass
    v.missing_value = FILL
    ds.setncattr("note", "regridded to SMAP 0.1 deg grid via bilinear interpolation")
    ds.close()


def process_era5land():
    lat, lon = load_smap_grid()
    for sub, var_name in ERA5_VARS.items():
        files = sorted(glob.glob(os.path.join(ERA5_SRC, sub, "*.nc")))
        print(f"\n=== ERA5 {sub} ({var_name})  n={len(files)} ===")
        for i, f in enumerate(files, 1):
            ds = Dataset(f)
            src_lat = np.asarray(ds.variables["latitude"][:]).astype(np.float64)
            src_lon = np.asarray(ds.variables["longitude"][:]).astype(np.float64)
            v = ds.variables[var_name]
            fill = v._FillValue
            data = np.asarray(v[:])
            attrs = {k: v.getncattr(k) for k in v.ncattrs()}
            time_vals = np.asarray(ds.variables["time"][:])
            time_attrs = {a: ds.variables["time"].getncattr(a)
                          for a in ds.variables["time"].ncattrs()}
            ds.close()

            out = regrid_bilinear(data, src_lat, src_lon, lat, lon, fill)
            dst = os.path.join(ERA5_DST, sub, os.path.basename(f))
            write_time_latlon_nc(dst, out, var_name, attrs, time_vals, time_attrs, lat, lon)
            if i % 44 == 0 or i == len(files):
                print(f"  [{i}/{len(files)}]")


def process_cldas():
    lat, lon = load_smap_grid()
    for sub, (var_name, fill) in CLDAS_VARS.items():
        files = sorted(glob.glob(os.path.join(CLDAS_SRC, sub, "*.nc")))
        print(f"\n=== CLDAS {sub} ({var_name})  n={len(files)} ===")
        for i, f in enumerate(files, 1):
            ds = Dataset(f)
            src_lat = np.asarray(ds.variables["latitude"][:]).astype(np.float64)
            src_lon = np.asarray(ds.variables["longitude"][:]).astype(np.float64)
            v = ds.variables[var_name]
            data = np.asarray(v[:])
            attrs = {k: v.getncattr(k) for k in v.ncattrs()}
            time_vals = np.asarray(ds.variables["time"][:])
            time_attrs = {a: ds.variables["time"].getncattr(a)
                          for a in ds.variables["time"].ncattrs()}
            ds.close()

            out = regrid_bilinear(data, src_lat, src_lon, lat, lon, fill)
            dst = os.path.join(CLDAS_DST, sub, os.path.basename(f))
            write_time_latlon_nc(dst, out, var_name, attrs, time_vals, time_attrs, lat, lon)
            if i % 44 == 0 or i == len(files):
                print(f"  [{i}/{len(files)}]")


def aggregate_soilgrids(data, src_lat, src_lon, dst_lat, dst_lon, fill_value):
    """250m -> 0.1°，每个 SMAP 格点内所有有效像元取平均。"""
    lat_edges = np.empty(len(dst_lat) + 1)
    lat_edges[0] = dst_lat[0] - 0.05
    lat_edges[1:-1] = (dst_lat[1:] + dst_lat[:-1]) / 2.0
    lat_edges[-1] = dst_lat[-1] + 0.05
    lon_edges = np.empty(len(dst_lon) + 1)
    lon_edges[0] = dst_lon[0] - 0.05
    lon_edges[1:-1] = (dst_lon[1:] + dst_lon[:-1]) / 2.0
    lon_edges[-1] = dst_lon[-1] + 0.05

    lat_idx = np.digitize(src_lat, lat_edges) - 1
    lon_idx = np.digitize(src_lon, lon_edges) - 1
    valid = data != fill_value

    n_smap = len(dst_lat) * len(dst_lon)
    sums = np.zeros(n_smap, dtype=np.float64)
    cnts = np.zeros(n_smap, dtype=np.int64)
    chunk = 512
    nlat = len(src_lat)
    for i0 in range(0, nlat, chunk):
        i1 = min(i0 + chunk, nlat)
        sub = data[i0:i1].astype(np.float64)
        mask = valid[i0:i1]
        li = lat_idx[i0:i1]
        in_lat = (li >= 0) & (li < len(dst_lat))
        in_lon = (lon_idx >= 0) & (lon_idx < len(dst_lon))
        fi = li[:, None] * len(dst_lon) + lon_idx[None, :]   # (chunk, nlon)
        keep = mask & in_lat[:, None] & in_lon[None, :]
        f = fi[keep].ravel()
        s = sub[keep].ravel()
        sums += np.bincount(f, weights=s, minlength=n_smap)
        cnts += np.bincount(f, minlength=n_smap)

    mean = np.full(n_smap, FILL, dtype=np.float32)
    m = cnts > 0
    mean[m] = (sums[m] / cnts[m]).astype(np.float32)
    return mean.reshape(len(dst_lat), len(dst_lon))


def process_static():
    lat, lon = load_smap_grid()
    for fn, (var_name, long_name) in SOIL_FILES.items():
        f = os.path.join(STATIC_SRC, fn)
        ds = Dataset(f)
        src_lat = np.asarray(ds.variables["lat"][:]).astype(np.float64)
        src_lon = np.asarray(ds.variables["lon"][:]).astype(np.float64)
        data = np.asarray(ds.variables["Band1"][:])
        fill = int(ds.variables["Band1"]._FillValue)
        ds.close()
        print(f"\n=== STATIC {fn} -> {var_name} (aggregate 250m -> 0.1deg) ===")
        out = aggregate_soilgrids(data, src_lat, src_lon, lat, lon, fill)

        os.makedirs(STATIC_DST, exist_ok=True)
        dst = os.path.join(STATIC_DST, f"{var_name}_05cm.nc")
        ds = Dataset(dst, "w")
        ds.createDimension("latitude", lat.size)
        ds.createDimension("longitude", lon.size)
        la = ds.createVariable("latitude", "f8", ("latitude",))
        la.units = "degrees_north"; la.standard_name = "latitude"; la[:] = lat
        lo = ds.createVariable("longitude", "f8", ("longitude",))
        lo.units = "degrees_east"; lo.standard_name = "longitude"; lo[:] = lon
        v = ds.createVariable(var_name, "f4", ("latitude", "longitude"), fill_value=FILL)
        v[:] = out
        v.long_name = long_name
        v.units = "g/kg"
        v.missing_value = FILL
        ds.setncattr("note", "aggregated from SoilGrids 250m to SMAP 0.1 deg grid (mean)")
        ds.close()
        print(f"  written {dst}")


def main():
    print("SMAP 参考网格：200 x 160")
    process_era5land()
    process_cldas()
    process_static()
    print("\n全部完成。")


if __name__ == "__main__":
    main()

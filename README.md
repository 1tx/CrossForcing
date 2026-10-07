# LandEmu — 东北 AI 陆面模式（土壤水逐步预测）

用 ERA5-Land（或 CLDAS）作为驱动 forcing、SoilGrids 作为静态土壤属性，训练一个**显式状态神经算子**
`DeltaOperator`，在东北 0.1° 网格上逐步预测 SMAP L4 的表层（0–5 cm）与根区（0–100 cm）土壤水。

核心递推方程：

```
state_t = clamp( state_{t-1} + MLP([state_{t-1}, forcing_t, static]) , 0, 1 )
```

即：网络输出的是**增量 ΔSM**，再与当前状态相加并做物理裁剪（土壤体积含水量 θ ∈ [0,1] m³/m³）。
状态始终以原始物理单位递推，可保存、可重启、便于后续内嵌水量守恒约束。

---

## 目录

1. [目录结构](#1-目录结构)
2. [数据](#2-数据)
3. [环境安装](#3-环境安装)
4. [快速开始](#4-快速开始)
5. [GPU / CPU 切换](#5-gpu--cpu-切换)
6. [配置文件 `config.py`](#6-配置文件-configpy)
7. [数据装载 `data.py`](#7-数据装载-datapy)
8. [模型 `model.py`](#8-模型-modelpy)
9. [训练 `train.py`（含采样规则与训练规则）](#9-训练-trainpy)
10. [评估 `evaluate.py`](#10-评估-evaluatepy)
11. [推理 `predict.py`](#11-推理-predictpy)
12. [切换驱动数据集](#12-切换驱动数据集)
13. [输出文件说明](#13-输出文件说明)
14. [设计决策与已知限制](#14-设计决策与已知限制)
15. [常见问题](#15-常见问题)

---

## 1. 目录结构

```
CrossForcing/                  # 项目根目录（ROOT，config.py 位于 landemu/ 下，取上一级）
├── landemu/                   # 本项目代码
│   ├── config.py              # 统一配置（路径/变量/实验标识/时间划分/模型/训练超参）
│   ├── data.py                # 数据装载 → 过滤 → 掩膜 → 抽站点 → 插值 → 缓存
│   ├── model.py               # DeltaOperator + LSTMState
│   ├── train.py               # rollout 课程训练
│   ├── evaluate.py            # one-step + free-run 评测 + 域均值时序图
│   ├── predict.py             # 推理：全网格地图输出并保存
│   ├── resample_to_smap.py    # 历史脚本（重采样，数据已对齐，无需运行）
│   └── aggregate_era5land_daily.py  # 历史脚本（逐时→逐日，数据已对齐，无需运行）
├── data/                      # 数据根目录（DATA_DIR = ROOT/data）
│   ├── Northeast_China_SMAPL4/    # 目标：sm_surface / sm_rootzone
│   ├── Northeast_China_era5land/  # 主 forcing（已逐日）
│   ├── CLDAS/                     # 跨 forcing 验证
│   └── Northeast_China_static/    # 静态土壤质地（SoilGrids 0–5 cm）
└── cache/                     # 预处理缓存 / 模型权重 / 输出（自动生成）
    └── experiments/
        └── <EXPERIMENT>/      # 每个实验一个目录（改 config.EXPERIMENT 切换）
            ├── pack/          # 预处理数据缓存 pack_<source>[_strict].npz
            ├── weights/       # 模型权重 model_<version>_<source>.pt
            ├── eval/          # 评估图 freerun_2020_<version>.png
            ├── predict/       # 预测 npz/nc/png
            └── config.json    # 实验配置快照（训练时自动生成）
```

---

## 2. 数据

| 目录 | 作用 | 时间 | 网格 |
|---|---|---|---|
| `Northeast_China_SMAPL4` | 目标：`sm_surface`(0-5cm) / `sm_rootzone`(0-100cm) | 2016-01 ~ 2020-12 逐日 | 160×200 |
| `Northeast_China_era5land` | 主 forcing：气温/风速/降水/比湿/短波/长波，共 6 变量 | 逐日 | 160×200 |
| `CLDAS` | 跨 forcing 验证：同 6 个物理量（变量名不同） | 逐日 | 160×200 |
| `Northeast_China_static` | 静态质地：`clay`/`sand`/`silt`（0-5cm） | 静态 | 160×200 |

四个数据源已在同一 0.1° 网格（lat 38.05~53.95，lon 115.05~134.95，160×200）、同一逐日尺度对齐，
填充值统一为 `-9999.0`。详细变量名/单位/连续性见 `数据源汇总.md`；数据质量异常与修复见 `开发问题总结.md`。

---

## 3. 环境安装

```bash
conda create -n landemu python=3.10 -y
conda run -n landemu python -m pip install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu121
conda run -n landemu python -m pip install numpy netCDF4 scipy matplotlib tqdm pandas openpyxl
```

自检（应输出 `True`，且 numpy 转换不报错）：

```bash
conda run -n landemu python -c "import torch; print(torch.cuda.is_available()); print(torch.tensor([1.0]).numpy())"
```

> 说明：torch 必须与 numpy/netCDF4 的 ABI 匹配（本项目统一 numpy 2.x 生态 + torch 2.5.1）。见 `开发问题总结.md` 第 0 节。

---

## 4. 快速开始

```bash
conda activate landemu
cd landemu

python run_pipeline.py  # 一键跑完 训练→评估→预测（已训练好可加 --skip train 跳过训练）
python train.py        # 训练，权重存到 ../cache/experiments/<EXPERIMENT>/weights/model_<version>_<source>.pt
python evaluate.py     # one-step + free-run 评测，出图 ../cache/experiments/<EXPERIMENT>/eval/freerun_2020_<version>.png
python predict.py      # 推理：free-run 全网格地图 + 保存 netCDF/npz/PNG
```

---

## 5. GPU / CPU 切换

`config.py` 里 `TrainConfig.device` 默认是 `"auto"`：

- `"auto"` —— 自动检测：有 CUDA 就用 `cuda`，否则用 `cpu`（推荐，无需手动改）。
- `"cuda"` / `"cpu"` —— 显式指定。
- 三个脚本启动时都会打印 `device = cuda` 或 `device = cpu` 供确认。

```python
# config.py, TrainConfig
device: str = "auto"
```

**重要提示**：当前网络是极小的 MLP（11→128→…→2），GPU 的 kernel 启动开销与 CPU↔GPU 传输开销往往**大于**算力收益，
因此小模型在 GPU 上未必比 CPU 快（CPU 单线程是之前实测最快的组合）。GPU 支持是为后续升级到**空间模型/大网络**
准备的。想切回 CPU，把 `device` 改为 `"cpu"` 即可。代码层面已做了两处针对设备的优化：

1. `torch.set_num_threads(1)` 只在 `device.type == "cpu"` 时执行（CPU 单线程对小算子更快）。
2. 评估/推理把 test 窗口数据**一次性预载到设备**（`t_win`/`f_win`），避免自回归循环里逐日做 H2D 拷贝。

---

## 6. 配置文件 `config.py`

统一配置入口，其他模块不写死路径与超参，做实验只改这里。

### 6.1 模块级常量

| 常量 | 说明 |
|---|---|
| `ROOT` | 项目根目录（`landemu/` 的上一级），由本文件位置向上取两级自动推导 |
| `DATA_DIR` | 数据根目录 `ROOT/data`，四个数据源都在其下 |
| `SMAP_DIR` / `ERA5_DIR` / `CLDAS_DIR` / `STATIC_DIR` | 四个数据源目录（均在 `DATA_DIR` 下） |
| `CACHE_DIR` | 缓存/输出目录 `ROOT/cache` |
| `FILL_VALUE` | 填充值 `-9999.0` |
| `SOURCE` | 主驱动数据集：`"era5"` 或 `"cldas"`（切换数据集只改这里，见第 12 节） |
| `EXPERIMENT` | 实验标识（默认 `"default"`）；做多实验时改它，输入缓存与输出自动落到 `cache/experiments/<EXPERIMENT>/` 下，实现隔离 |
| `GRID_LAT0` / `GRID_LON0` / `GRID_RES` | 网格起点纬度 38.05、起点经度 115.05、分辨率 0.1°（用于重建经纬度坐标） |
| `TARGET_VARS` | 目标变量列表 `(变量名, 单位, 描述)`，共 2 个：表层 + 根区 |
| `FORCING_ERA5` | ERA5 主 forcing 列表 `(子目录名, netCDF 变量名, 单位)`，共 6 个 |
| `FORCING_CLDAS` | CLDAS 跨 forcing 列表，同样 6 个物理量（变量名不同） |
| `STATIC_VARS` | 静态质地列表 `(变量名, 单位, 描述)`，共 3 个：clay/sand/silt |
| `TARGET_START_YEAR` / `TARGET_END_YEAR` | 目标时段 2016–2020 |
| `SPLIT_YEARS` | 四段时间划分 `train(2016-17) / val(2018) / spinup(2019) / test(2020)` |
| `MASK_MIN_VALID_FRAC` | 目标有效天数占比阈值 `0.5`，低于则掩掉 |
| `FORCING_MIN_VALID_FRAC` | forcing 有效天数占比阈值 `0.8` |
| `NORMALIZE` | 是否预归一化，`False`（量纲统一交给模型内部 norm 层） |
| `STRICT_ROOTZONE` | 严格质量平衡开关 `False`：`True` 时把根区目标从 0-100cm 扣除为 5-100cm（见第 14 节） |

### 6.2 `ModelConfig`（dataclass）

| 字段 | 默认 | 说明 |
|---|---|---|
| `version` | `"delta"` | 模型版本：`"delta"`（显式状态算子）\| `"lstm"`（隐状态 LSTM 基线） |
| `lookback` | `7` | 仅 `"lstm"` 使用：lookback 窗口长度（历史天数） |
| `n_state` | `2` | 状态维数（表层 + 根区） |
| `n_forcing` | `len(FORCING_ERA5)` = 6 | forcing 变量数 |
| `n_static` | `len(STATIC_VARS)` = 3 | 静态变量数 |
| `hidden` | `128` | 隐层宽度 |
| `n_layers` | `3` | 隐层数量 |
| `activation` | `"relu"` | 激活：`relu` / `tanh` |
| `norm` | `"layer"` | 内部归一化：`layer` / `batch` / `none` |
| `input_norm` | `True` | 是否对拼接后的输入先做一次归一化（处理异构量纲） |
| `state_clip` | `(0.0, 1.0)` | 状态物理裁剪区间（m³/m³）；`None` 则不裁剪 |

### 6.3 `TrainConfig`（dataclass）

| 字段 | 默认 | 说明 |
|---|---|---|
| `seed` | `0` | 随机种子 |
| `epochs` | `1000` | 训练轮数 |
| `batch_size` | `2048` | 每步采样站点数（有放回随机采样） |
| `batches_per_epoch` | `400` | 每 epoch 采样批次数 |
| `lr` | `3e-4` | 学习率（长 rollout 课程配套更低 LR） |
| `weight_decay` | `1e-5` | Adam 权重衰减 |
| `grad_clip` | `0.5` | 梯度裁剪阈值（长 rollout 课程配套更强裁剪） |
| `rollout_schedule` | `{0:1, 100:7, 300:30, 500:60, 800:120}` | 长 rollout 课程：epoch 阈值→展开步长 K（详见 9.2） |
| `lambda_delta` | `0.0` | ΔSM 正则系数（默认 0；>0 会趋向持久化） |
| `noise_sigma` | `0.01` | 状态注入噪声 σ（m³/m³，零均值高斯）；0=关闭 |
| `teacher_init` | `True` | 每步用真值初始状态（当前实现恒为真值起点） |
| `device` | `"auto"` | 计算设备：`auto` / `cuda` / `cpu` |

### 6.4 `Config` 与全局实例

`Config` 组合 `model` 与 `train` 两个子配置；模块底部 `cfg = Config()` 生成全局单例，
其余模块通过 `config.cfg.model` / `config.cfg.train` 访问。

### 6.5 函数

#### 实验目录 helper

- `exp_dir()`：当前实验根目录 `cache/experiments/<EXPERIMENT>/`。
- `exp_pack_dir()` / `exp_weights_dir()` / `exp_eval_dir()` / `exp_predict_dir()`：实验目录下的
  `pack` / `weights` / `eval` / `predict` 子目录。
- 均返回 `pathlib.Path`；数据缓存、权重、评估图、预测输出统一通过这些函数定位，切实验只改 `EXPERIMENT`。

#### `ckpt_path(source=None, version=None)`

- **功能**：按训练源与模型版本返回当前实验的 checkpoint 路径。
- **实现**：`exp_weights_dir() / f"model_{version}_{source}.pt"`。
- **返回**：`pathlib.Path`。

#### `resolve_device(pref=None)`

- **功能**：把 `device` 配置解析成可用设备字符串。
- **实现**：`pref` 缺省取 `cfg.train.device`；若为 `"auto"` 则惰性 `import torch` 并检测
  `torch.cuda.is_available()`，可用返回 `"cuda"`，否则 `"cpu"`；非 `"auto"` 直接原样返回。
  惰性 import 使 `config.py` 本身不依赖 torch，导入顺序更安全。
- **返回**：字符串 `"cuda"` 或 `"cpu"`。

---

## 7. 数据装载 `data.py`

整体流程：**按文件名拼接各源 → 切片到目标时段 → 算陆地掩膜 → 抽有效站点 → 时间插值 → 缓存到 npz**。

> **标签（target）与缓存**：标签即 SMAP 土壤水 `target`，与 `forcing`、`static` **一起打包进 `pack_*.npz`**，
> 不是单独读取；`build()` 返回的 `dict` 里 `target(N,T,2)` 就是训练/评估的监督目标。
>
> **异常值只读处理、不改原始文件**：所有清洗（`_FillValue`→NaN、超大值→NaN、物理区间 θ∈[0,1]→NaN、
> 时间维插值、根区质量平衡）都在**加载到内存后**完成，原始 netCDF 文件**始终只读、从不回写**；
> 处理后的干净数据才固化进 `pack_*.npz`。
>
> 关键约定：有效站点被展平成 `(N, …)` 一维索引序列，`N`≈27403（占 160×200 的 ~85.6%），
> 同时保留 `mask`（布尔网格）、`ys`/`xs`（有效像元的行/列索引），供推理时反铺回完整网格。

### 7.1 `_ym(name)`

- **功能**：从文件名解析 `(year, month)`。
- **实现**：用正则 `20\d{2}[_-]?(\d{2})` 匹配，兼容 `ERA5LAND_2016_01_*_daily.nc` 与
  `CLDAS...-201601_*` 两种命名。解析失败抛 `ValueError`。
- **返回**：`(year, month)`。

### 7.2 `_load_var(dirpath, varname, y_lo, y_hi)`

- **功能**：读取某变量目录下所有 monthly netCDF，按时间拼接并切片到年份区间。
- **实现**：`glob` 目录下 `*.nc` → 逐个 `_ym` 判断年份是否落在 `[y_lo, y_hi]` → 用 `netCDF4.Dataset`
  读变量转 `float32` → 把 `_FillValue` 与绝对值 >1e30 的值置为 NaN → 沿时间轴 `np.concatenate`。
- **返回**：`(T, lat, lon)` 的 `float32` 数组。

### 7.3 `_day_split()`

- **功能**：计算各段时间段在 2016-01-01 起算的日索引 `[start, end)`。
- **实现**：以 `TARGET_START_YEAR` 的 1 月 1 日为 `d0`，对 `SPLIT_YEARS` 的每段，
  用 `(date(y,1,1)-d0).days` 与 `(date(y1+1,1,1)-d0).days` 求起止天数。
- **返回**：`{"train": (s,e), "val": (s,e), "spinup": (s,e), "test": (s,e)}`。

### 7.4 `_load_target()`

- **功能**：装载 SMAP 表层 + 根区两层土壤水。
- **实现**：对 `TARGET_VARS` 每个变量调 `_load_var`，`np.stack(..., axis=-1)` 得到 `(T,lat,lon,2)`；
  再做物理有效区间过滤 `θ ∈ [0,1]`，把越界值（含 2016-06 被污染的根区 1.19~99.49）置为 NaN。
- **返回**：`(T, 160, 200, 2)` 的 `float32` 数组。

### 7.5 `_forcing_spec(source)`

- **功能**：按数据源返回 forcing 变量列表。
- **实现**：`source == "era5"` 返回 `FORCING_ERA5`，否则返回 `FORCING_CLDAS`。
- **返回**：列表 `[(子目录, 变量名, 单位), ...]`。

### 7.6 `_load_forcing(source)`

- **功能**：装载主 forcing（6 变量）。
- **实现**：遍历 `_forcing_spec(source)`，对每个变量调 `_load_var` 读取；若变量名命中模块常量
  `_FORCING_SCALE = {"PRCP": 24.0}` 则乘上尺度（CLDAS 的 `PRCP` 是 mm/hr 日平均，×24 转日总量；
  ERA5 的 `tp` 已是日累计，不受影响）；最后 `np.stack` 到最后一维。
- **返回**：`(T, 160, 200, 6)` 的 `float32` 数组。

### 7.7 `_load_static()`

- **功能**：装载 SoilGrids 静态质地。
- **实现**：对 `STATIC_VARS` 的 clay/sand/silt，读 `Northeast_China_static/<var>_05cm.nc`，
  处理 `_FillValue` 与超大值，`np.stack` 到最后一维。
- **返回**：`(160, 200, 3)` 的 `float32` 数组。

### 7.8 `_fill_nan_time(arr)`

- **功能**：对每个站点、每个变量沿时间维线性插值填 NaN。
- **实现**：对 `(N,T,V)` 数组逐站点、逐变量用 `np.interp(x, x[m], y[m])` 在时间维插值；
  首尾用最近有效值（`np.interp` 对区间外的 x 返回端点值）。若某变量全有效或全 NaN 则跳过。
- **返回**：填好的同形状数组。

### 7.9 `rootzone_to_5_100(arr)`

- **功能**：严格质量平衡·正向，把 `(…,2)` 的 `[surface, rootzone(0-100cm)]` 转成 `[surface, rootzone(5-100cm)]`。
- **实现**：`Z = (1000·R − 50·S) / 950`，再 `clamp(Z, 0, 1)` 兜底（个别 S>R 的异常像元会越界）。
- **返回**：同形状数组（第 1 通道被替换为 5-100cm 根区）。

### 7.10 `rootzone_to_0_100(arr)`

- **功能**：严格质量平衡·反向，把 `[surface, rootzone(5-100cm)]` 还原成 `[surface, rootzone(0-100cm)]`（SMAP 口径）。
- **实现**：`R = (50·S + 950·Z) / 1000`，是深度加权平均，自动落在 `[0,1]`，无需 clamp。
- **返回**：同形状数组（第 1 通道还原为 0-100cm 根区）。

### 7.11 `build(source="era5", rebuild=False)`

- **功能**：主数据管线入口，产出训练/评估/推理用的打包数据。
- **实现**：
  1. 若缓存 `experiments/<EXPERIMENT>/pack/pack_<source>[_strict].npz` 存在且 `rebuild=False`，直接 `np.load` 返回（**命中缓存，秒级**）。
  2. 否则 `_load_target` / `_load_forcing` / `_load_static` 装载三块数据。
  3. 计算陆地掩膜 `mask`：先对每个像元算目标/forcing 各自在时间维的有效（非 NaN）天数占比
     `t_valid = mean(isfinite(target), axis=0)`、`f_valid = mean(isfinite(forcing), axis=0)`；同时满足
     ① `t_valid` 在 2 个目标变量里的最小值 ≥ `MASK_MIN_VALID_FRAC`(0.5)，
     ② `f_valid` 在 6 个 forcing 变量里的最小值 ≥ `FORCING_MIN_VALID_FRAC`(0.8)，
     ③ 3 个 static 变量全有限 —— 才判为陆地有效像元。
  4. `np.where(mask)` 得到 `ys, xs`，用高级索引抽出站点并 `transpose` 成 `(N,T,…)`，`ascontiguousarray` 保证内存连续。
  5. `_fill_nan_time` 填掉站点内残余缺测日。
  6. 若 `STRICT_ROOTZONE=True`，`rootzone_to_5_100` 把根区目标从 0-100cm 扣除为 5-100cm。
  7. 打印诊断信息，`np.savez` 缓存 `target/forcing/static/mask/ys/xs`（strict 与非 strict 用不同文件名）。
- **返回**：`dict`，键 `target(N,T,2)`、`forcing(N,T,F)`、`static(N,S)`、`mask`、`ys`、`xs`、`split`。

  `pack_*.npz` 缓存内容（`split` 为纯日期计算、每次现算，不入缓存）：

  | 键 | 形状 | 含义 |
  |---|---|---|
  | `target` | `(N,T,2)` | 标签：SMAP 表层(0-5cm) + 根区(0-100cm 或 5-100cm)土壤水 |
  | `forcing` | `(N,T,6)` | 6 个大气 forcing 变量 |
  | `static` | `(N,3)` | clay / sand / silt 静态质地 |
  | `mask` | `(160,200)` | 陆地有效掩膜（布尔） |
  | `ys` / `xs` | `(N,)` | 有效像元的行 / 列索引 |

---

## 8. 模型 `model.py`

### 8.1 `_act(name)`

- **功能**：返回激活函数层。
- **实现**：`"relu"` → `nn.ReLU()`，否则 `nn.Tanh()`。

### 8.2 `_norm(dim, kind)`

- **功能**：返回归一化层。
- **实现**：`"layer"` → `nn.LayerNorm(dim)`；`"batch"` → `nn.BatchNorm1d(dim)`；否则 `nn.Identity()`。

### 8.3 `DeltaOperator(nn.Module)`

显式状态神经算子（主模型）。

- **`__init__(mc=None)`**
  - 输入维 `in_dim = n_state + n_forcing + n_static = 2 + 6 + 3 = 11`。
  - 若 `input_norm=True`，先加一个 `_norm(in_dim, norm)` 处理异构量纲。
  - 依次堆 `n_layers` 个 `Linear(d→hidden) + norm + act`，最后 `Linear(hidden→n_state)` 输出增量。
  - 保存 `state_clip`。

- **`forward(state, forcing, static)`**
  - `x = cat([state, forcing, static], dim=-1)` → `delta = net(x)`。
  - `nxt = state + delta`，若 `state_clip` 非空则 `clamp(nxt, lo, hi)`。
  - **返回** `(nxt, delta)`，`delta` 供 `lambda_delta` 正则使用。

### 8.4 `LSTMState(nn.Module)`

隐状态 LSTM 基线（baseline），输入 lookback 窗口预测下一时刻增量。

> 已接入：把 `config.ModelConfig.version` 改为 `"lstm"` 即可切换（见第 6.2 节）。`LSTMState` 需要 lookback 窗口
> （历史多天状态+forcing）输入，与 `DeltaOperator` 的单步递推接口不同；训练时按窗口采样、free-run 时滑窗递推，
> 这些差异已由 `train.py`/`evaluate.py`/`predict.py` 的 `version` 分支与 `model.free_run` 统一处理。

- **`__init__(mc=None, lookback=7)`**：LSTM 输入为 `(B, L, D)`，故只支持 LayerNorm/None；两层 LSTM + 线性头。
- **`forward(states, forcings, static)`**：把 `static` 沿时间维广播拼到每个窗口时刻，
  `LayerNorm → LSTM → head(out[:, -1])` 得到增量，`nxt = states[:, -1] + delta` 后裁剪。

---

## 9. 训练 `train.py`

rollout 课程式训练：`K` 从 1 逐步增大到 7，让模型先学会单步再学会多步自回归。
本节先给出**采样规则**与**训练规则**（训练的核心设计），再逐函数说明。

### 9.1 采样规则（数据 → 训练样本）

预处理后数据组织为展平张量 `target(N,T,2)` / `forcing(N,T,F)` / `static(N,S)`，
`N`≈27403 个有效像元、`T`=1827 天（2016–2020），每个站点是一条 5 年逐日序列。
训练样本由「站点 × 时间」两维**随机采样**构造，每 batch 步骤如下：

| 步骤 | 规则 | 代码位置 |
|---|---|---|
| ① 站点采样 | `sites = rng.integers(0, N, batch_size)`，**有放回**随机抽 `batch_size=2048` 个站点 | `train.py` 主循环 |
| ② 时间采样 | 每样本独立抽起始日 `t0 ~ U[tr_s+1, tr_e-K+1)`，`tr_s=0`、`tr_e=731`（train 段 2016–2017 的日索引） | `train.py` 主循环 |
| ③ 窗口切片 | `idx = t0 + [0..K-1]`；forcing 取 `idx`（K 天），target 取 `idx`（K 天） | `train.py` 主循环 |
| ④ 起点状态 | `state = target[sites, t0-1]`，恒取真值（teacher forcing 起点） | `train.py` 主循环 |

要点：

- 采样是**随机的、有放回的**：每 epoch 共 `batches_per_epoch × batch_size = 400×2048 = 819200` 个
  `(站点, 起点)` 样本，**不遍历**全量样本（全量规模为 N×天数 量级），属标准小批量随机梯度采样；
  因此某些站点/时间点会被重复抽到。
- 时间起点上界 `tr_e-K+1`、下界 `tr_s+1`，保证 rollout 窗口与起点状态 `t0-1` 均不越出 train 段边界。
- 随机性由 `numpy.random.default_rng(seed)` 独立控制，与 torch 权重初始化分开，保证可复现。

### 9.2 训练规则（rollout 课程 + 损失）

**损失函数**（对 K 步 rollout 逐步累加后平均）：

```
L = (1/K) · Σ_{s=0}^{K-1} [ MSE( ŷ_{t+s}, y_{t+s} ) + λ_δ · mean(Δ_s²) ]
```

其中 `ŷ` 为预测状态、`y` 为真值、`Δ_s = ŷ_{t+s} − ŷ_{t+s-1}` 是第 s 步网络输出的增量，
`λ_δ = lambda_delta`（默认 0，>0 时趋向持久化）。

**具体规则**：

| 项 | 规则 |
|---|---|
| 递推 | `state ← clamp(state + MLP([state, forcing, static]), 0, 1)`，窗口内部用上一步预测继续（自回归） |
| 起点 | teacher forcing：每步 rollout 起点恒为真值 `y_{t0-1}`（`teacher_init=True`） |
| rollout 课程 | K 随 epoch 增大：`{0:1, 100:7, 300:30, 500:60, 800:120}`（K=1→7→30→60→120），先单步后多步，缓解误差累积 |
| 优化器 | Adam，`lr=3e-4`，`weight_decay=1e-5` |
| 梯度 | `clip_grad_norm_(max_norm=0.5)`（长 rollout 深展开需更强裁剪防梯度爆炸） |
| 物理约束 | 每步输出后 `clamp(θ, 0, 1)`（土壤体积含水量有效区间） |
| 噪声注入 | `noise_sigma>0` 时每步输入状态加零均值高斯噪声 ε~N(0,σ²)，扰动输入但递推轨迹保持干净（方案 A） |
| 量纲 | 不做预归一化，模型内部 `LayerNorm` 统一异构量纲 |
| 验证 | 每 epoch 结束：随机抽 256 站点算 one-step RMSE、128 站点算 30 天 rollout RMSE |
| 保存 | 每 5 个 epoch 及最后存一次权重到 `experiments/<EXPERIMENT>/weights/model_<version>_<source>.pt` |

**长 rollout 课程实现**（`config.py` 的 `rollout_schedule` + `train.py` 的 delta 分支）：

1. **课程表**：`rollout_schedule = {0:1, 100:7, 300:30, 500:60, 800:120}`，按 epoch 把展开步长 K 从 1 分阶段拉到 120；`k_for_epoch(epoch, schedule)` 遍历阈值查表得到当前 K。
2. **K 步展开（整条图反传 BPTT）**：每个 batch 先按 K 采样窗口，再自回归展开 K 步，梯度穿过整条 K 步计算图：
   ```python
   idx = t0[:, None] + arange(k)                 # (B,K) 采样 K 个时间点
   fwin = forcing[sites[:, None], idx, :]        # (B,K,F)
   twin = target[sites[:, None], idx, :]         # (B,K,2)
   state = target[sites, t0 - 1]                 # teacher-forcing 起点真值
   loss = 0.0
   for s in range(k):
       state, delta = model(state, fwin[:, s, :], static[sites])
       loss = loss + F.mse_loss(state, twin[:, s, :])
   loss = loss / k                               # 对 K 平均，避免长 rollout 放大 loss 量级
   ```
3. **配套稳定化**：`lr=3e-4`（更低）+ `grad_clip=0.5`（更强），防止深展开梯度爆炸。
4. **采样上界随 K 收缩**：`t0 ∈ [tr_s+1, tr_e-K+1)`，保证 K 步窗口与起点 `t0-1` 不越出训练段（K=120 时 `tr_e-K+1=612`，仍有充足样本）。
5. **成本**：单 epoch 耗时近似随 K 线性增长；K=30/60/120 约 1.5/3/6 分钟/epoch（CPU 全配置 2048×400），完整 1000 epoch 课程约 40 小时。

**噪声注入（③，缓解 train/test 分布偏移）**：free-run 每步输入是模型自己的预测（带误差），训练时对每步输入状态加零均值高斯噪声 ε~N(0,σ²)，让模型对带误差输入鲁棒。采用「干净递推」方案 A——噪声只扰动输入、不进入递推轨迹，避免长 K 下随机游走（σ√K）累积漂移：

   ```python
   for s in range(k):
       state_in = state + torch.randn_like(state) * tc.noise_sigma   # 仅扰动输入
       _, delta = model(state_in, fwin[:, s, :], sb)
       state = state + delta                                         # 干净递推（含 clamp）
       loss = loss + F.mse_loss(state, twin[:, s, :])
   ```
   默认 `noise_sigma=0.01`（已开启）；0 关闭。噪声是零均值随机扰动，只提升对随机误差的鲁棒性，不针对系统性偏湿（后者靠 ① 长 rollout 与 ④ 通量守恒）。

> **LSTM 版本的训练规则**（`version="lstm"` 时）：不做 rollout 课程，改为 teacher-forcing 的 lookback 采样——
> 每 batch 随机抽站点与起点 `t0`（`t0 ∈ [tr_s+L, tr_e)`），用真值窗口 `[t0-L, t0)` 的状态+forcing 喂 LSTM，
> 预测 `t0` 时刻状态，损失为单步 `MSE(pred, y_t0)`。验证同样抽随机窗口做 one-step，并用 `model.free_run` 做 30 天滑窗 free-run。

### 9.3 `set_seed(seed)`

- **功能**：固定随机性。
- **实现**：分别 `random.seed`、`np.random.seed`、`torch.manual_seed`。

### 9.4 `k_for_epoch(epoch, schedule)`

- **功能**：根据 epoch 查表得到 rollout 步长 K。
- **实现**：遍历 `schedule` 的 `(阈值, K)`（按阈值排序），取 `epoch >= 阈值` 的最大 K；默认 K=1。
  例如 `{0:1, 100:7, 300:30, 500:60, 800:120}`：epoch 0~99→K=1、100~299→K=7、300~499→K=30、500~799→K=60、≥800→K=120。

### 9.5 `main()`

训练主流程（以下为 delta 分支；lstm 分支的训练见 9.2 末尾）：

1. **设备与数据**：`resolve_device` 选设备；`build(SOURCE)` 载数据；`torch.from_numpy(...).to(device)`
   把 `target/forcing/static` 零拷贝预转为张量。
2. **模型与优化器**：`build_model(mc)`（按 `version` 选 `DeltaOperator` / `LSTMState`）+ Adam。
3. **每 epoch**：
   - `k = k_for_epoch(...)`，预生成 `arange(k)` 供时间索引。
   - 循环 `batches_per_epoch` 次：
     - 用 numpy RNG 随机采 `batch_size` 个站点 `sites` 和起始时间 `t0`（`t0 ∈ [tr_s+1, tr_e-k+1)`）。
     - `idx = t0[:,None] + arange(k)` 得到每个样本的 `K` 个时间点；高级索引 gather 出
       forcing 窗口 `fwin(B,K,F)` 与 target 窗口 `twin(B,K,2)`。
     - `state = target[sites, t0-1]`（真值起点），K 步 rollout：`state, delta = model(state, fwin[:,s], sb)`，
       累加 `MSE(state, twin[:,s])`，可选加 `lambda_delta * mean(delta²)`。
     - 损失对 K 取均值，反向传播 + 梯度裁剪 + 更新。
   - **验证**：`model.eval()` + `no_grad()`，随机抽子集做 one-step RMSE 与 30 天 rollout RMSE。
   - 每 5 个 epoch（及最后一个）保存 `state_dict` 到 `experiments/<EXPERIMENT>/weights/model_<version>_<SOURCE>.pt`。

> 性能要点：先一次性 gather 整个 rollout 窗口再在窗口上做纯张量展开，避免逐步 numpy 索引的 Python 开销；
> CPU 下 `torch.set_num_threads(1)` 避免小算子多线程 spin-wait 拖慢。

> 进度显示：用 `tqdm` 显示 epoch 级进度条、剩余时间 ETA，以及每 epoch 的 train_loss / val 指标 / 耗时。

---

## 10. 评估 `evaluate.py`

两种评测 + 域均值时序图 + 每站点指标 Excel。指标：RMSE / MAE / bias / R / R²，分表层、根区。

### 10.1 `metrics(pred, truth)`

- **功能**：计算四个指标。
- **实现**：先 `reshape(-1)` 展平，`isfinite` 掩掉无效值；`RMSE=sqrt(MSE)`、`MAE=mean|p-t|`、
  `bias=mean(p)-mean(t)`、`R = Σ(vp·vt)/(‖vp‖·‖vt‖+ε)`。
- **返回**：`(rmse, mae, bias, r)`。

`site_metrics(pred, truth)`：逐站点指标（向量化，沿时间维）。输入 `(N,T)` 单变量，返回 5 个 `(N,)`
数组 rmse/mae/bias/R(Pearson)/R²（`R² = 1 - SS_res/SS_tot`）。

`write_site_metrics_xlsx(pred1, fr, truth, ys, xs, out_path)`：把 one-step 与 free-run 两套每站点指标
写成一个 xlsx（两个 sheet `one_step`/`free_run`），每行一个站点，列含 `ys/xs/lat/lon` 与
`surface/rootzone` 各自的 rmse/mae/bias/R/R²；缺 pandas/openpyxl 时打印提示并跳过。

### 10.2 `plot_freerun(fr_pred, truth, out_path)`

- **功能**：画域均值时间序列对比图（free-run vs SMAP 真值）。
- **实现**：对 `(N,T,2)` 沿站点维取 `mean(axis=0)` 得 `(T,2)`；上下两个子图分别画表层/根区；
  `matplotlib.use("Agg")` 后台出图，异常吞掉只打印跳过。

### 10.3 `main()`

1. 载数据与模型权重（`build_model` + `config.ckpt_path()`）。
2. **one-step**：把 target/forcing 全量预载到设备 `t_dev/f_dev`，逐日喂模型（delta 用单步、lstm 用 lookback 窗口），
   与真值比对，累加指标后按天数平均（STRICT 时 rootzone 为 5-100cm 口径，标题注明）。
3. **free-run**：调用 `model.free_run`，从 `te_s`（2020-01-01）起仅靠 forcing 自回归滚完 2020（delta 单步递推、
   lstm 滑窗递推），结果一次性 `.cpu().numpy()`。若 `STRICT_ROOTZONE=True`，把 fr 与 truth 一起
   `rootzone_to_0_100` 还原成 SMAP 0-100cm 口径再对比。
4. 打印两组指标，`plot_freerun` 出图到 `experiments/<EXPERIMENT>/eval/freerun_2020_<version>.png`。
5. 逐站点计算 one-step 与 free-run 两套指标（统一还原到 SMAP 0-100cm 口径），`write_site_metrics_xlsx`
   导出到 `experiments/<EXPERIMENT>/eval/site_metrics_<year>.xlsx`。

> 注意：自回归循环必须包 `torch.no_grad()`，否则 366 步会构建超大计算图导致内存耗尽。

---

## 11. 推理 `predict.py`

加载训练好的模型做 free-run，把预测**反铺回完整 160×200 网格**并保存（这是"输出全部地图 + 保存结果"的入口）。

### 11.1 `_metrics(pred, truth)`

- **功能**：与 `evaluate.metrics` 等价，但接受 numpy 输入（内部 `ascontiguousarray` 后转张量），
  供展平结果与真值在 CPU 上算指标。

### 11.2 `_to_grid(pred_flat, mask, ys, xs)`

- **功能**：把展平预测 `(N,T,2)` 反铺成完整网格 `(T,H,W,2)`。
- **实现**：`np.full` 预分配 NaN 网格，再用 `ys/xs` 索引把站点预测写回对应行列；掩膜外保持 NaN。
- **返回**：`(T, 160, 200, 2)` 数组。

### 11.3 `_save_netcdf(grid, time_idx, out_path, lat, lon)`

- **功能**：把网格预测写成 netCDF。
- **实现**：`netCDF4.Dataset` 建 `time/lat/lon` 三维与 `sm_surface/sm_rootzone` 两个变量；
  时间单位 `days since 2016-01-01`；NaN 写为 `-9999.0`（`fill_value`）。
- **返回**：无（落盘）。

### 11.4 `_plot_maps(grid, out_path, lat, lon)`

- **功能**：画年均空间分布图。
- **实现**：`np.nanmean(grid, axis=0)` 得 `(H,W,2)` 年均值，`pcolormesh(lon, lat, …)` 画表层/根区两张，
  共用颜色条；异常吞掉只打印跳过。

### 11.5 `main()`

1. 解析参数 `--source`（默认 `config.SOURCE`）、`--ckpt`（默认 `config.ckpt_path(config.SOURCE)`）、`--year`。
2. 载数据与权重（`build_model`），`no_grad()` 下 `model.free_run` 滚完 test 段。
3. 若 `STRICT_ROOTZONE=True`，把 fr 与 truth 一起 `rootzone_to_0_100` 还原成 SMAP 0-100cm 口径。
4. 输出三件套（文件名含 version）：
   - `freerun_<version>_<source>_<year>_flat.npz` —— 展平预测 + `ys/xs`；
   - `freerun_<version>_<source>_<year>_grid.nc` —— 完整网格 netCDF（含坐标与时间）；
   - `freerun_<version>_<source>_<year>_map.png` —— 年均空间分布图。
5. 打印 free-run 四项指标（还原后口径）。

---

## 12. 切换驱动数据集

数据装载由 `data.build(source)` 决定，`source` 只支持 `"era5"` / `"cldas"`。两者都是 6 个 forcing 变量，
模型输入维不变，**无需改网络结构**。

- **整条管线换源**：改 `config.py` 的 `SOURCE = "cldas"`，`train.py` / `evaluate.py` / `predict.py` 都会改用 CLDAS。
  首次运行会在 `experiments/<EXPERIMENT>/pack/` 构建缓存 `pack_cldas.npz`（几十秒到几分钟），之后秒级加载。
  CLDAS 的 `PRCP` 会在 `_load_forcing` 里自动 ×24 转日总量。
- **cross-forcing（era5 权重 + cldas 驱动）**，不改 config，直接：
  ```bash
  python predict.py --source cldas
  ```
  权重默认仍取 `config.SOURCE`（era5）训练出的 checkpoint；显式指定用 `--ckpt <路径>`。
- **注意**：checkpoint 文件名含 version 与 source（`model_<version>_<source>.pt`，如 `model_delta_era5.pt` / `model_lstm_era5.pt`），
  换源或换模型重训**不会互相覆盖**（同实验目录下按 `model_<version>_<source>.pt` 区分；不同实验按 `EXPERIMENT` 目录隔离）。

---

## 13. 输出文件说明

以下文件均位于 `cache/experiments/<EXPERIMENT>/` 对应子目录下（`<EXPERIMENT>` 为当前实验标识）：

| 文件 | 生成者 | 说明 |
|---|---|---|
| `pack/pack_<source>[_strict].npz` | `data.py` | 预处理缓存：target/forcing/static/mask/ys/xs |
| `weights/model_<version>_<source>.pt` | `train.py` | 模型权重 |
| `config.json` | `train.py` | 实验配置快照（source/experiment/model/train 超参） |
| `eval/freerun_2020_<version>.png` | `evaluate.py` | 域均值时序图 |
| `eval/site_metrics_<year>.xlsx` | `evaluate.py` | 每站点指标（`one_step`/`free_run` 两个 sheet，含 RMSE/MAE/bias/R/R² 与坐标） |
| `predict/freerun_<version>_<source>_<year>_flat.npz` | `predict.py` | 展平预测 `(N,T,2)` + ys/xs |
| `predict/freerun_<version>_<source>_<year>_grid.nc` | `predict.py` | 完整网格逐日预测（netCDF） |
| `predict/freerun_<version>_<source>_<year>_map.png` | `predict.py` | 年均空间分布图 |

---

## 14. 设计决策与已知限制

- **不做预归一化**：输入保持原始物理单位，量纲统一放在模型内部 `LayerNorm/BatchNorm`（`ModelConfig.norm`）。
- **状态以原始 m³/m³ 递推**：显式物理状态，可保存重启；每步后物理裁剪 `clamp(θ, 0, 1)`。
- **两层深度是包含关系**：`sm_rootzone(0-100cm)` 包含 `sm_surface(0-5cm)`，默认按"两层耦合水库"简化处理（重叠的
  0-5cm 被 surface 与 rootzone 各算一次）。开启 `STRICT_ROOTZONE=True` 时做严格质量平衡，把根区重定义为 5-100cm：
  正向 `Z=(1000R−50S)/950`（训练口径）、反向 `R=(50S+950Z)/1000`（输出还原成 SMAP 口径），见第 6.1 节。
  注意：物理自洽像元（S≤R 且 Z∈[0,1]）正反向精确互逆；个别不自洽像元（S>R，或根区饱和而表层偏低使 Z>1）
  会在正向被 clamp 到 [0,1]，反向还原与原始 SMAP 略有出入。
- **`lambda_delta` 默认 0**：原始单位下 ΔSM 很小（~1e-3），开正则会过度鼓励持久化。
- **长 rollout 课程（K=1→7→30→60→120）**：`rollout_schedule` 分阶段把展开步长从 1 拉到 120，配更低 LR（3e-4）+ 更强 grad_clip（0.5）以稳定深展开。诊断与 smoke 验证见 `free_run改进方向.md` 与 `smoke_rollout.py`。注意 K 越大单 epoch 越慢（K=120 约 6 min/epoch，完整 1000 epoch 约 40 h CPU）。
- **Static 仅 0-5cm 质地**：根区(0-100cm)质地以 0-5cm 为代理，是当前限制。
- **`teacher_init` 未真正实现分支**：训练始终用真值起点，`false` 分支（用上一步预测继续）尚未落地。

---

## 15. 常见问题

**Q：`torch.from_numpy` 报 `Numpy is not available`？**
A：torch/numpy/netCDF4 的 ABI 不匹配，按第 3 节用 numpy 2.x 生态的 conda 环境重装。

**Q：训练/评估很慢（CPU 100%）？**
A：小算子在 CPU 多线程下会 spin-wait 拖慢，代码已自动 `set_num_threads(1)`（仅 CPU）。若仍异常慢，多为后台进程抢资源或热降频。

**Q：free-run 段错误 / 内存耗尽？**
A：自回归循环必须包 `torch.no_grad()`（已处理）；加载权重用 `weights_only=True`（已处理）。

**Q：为什么 GPU 上没变快？**
A：当前 MLP 极小，GPU 传输/启动开销大于算力收益，属正常。GPU 支持是为后续大模型准备，可把 `device` 改回 `"cpu"`。

**Q：`_ym` 解析文件名报错？**
A：文件名不符合 `20xx_MM` / `20xx-MM` 两种格式；确认 netCDF 命名与 `data.py` 约定一致。

**Q：`LSTMState` 怎么启用？**
A：把 `config.ModelConfig.version` 改为 `"lstm"`（默认 `"delta"`）。`train.py`/`evaluate.py`/`predict.py` 会按
   version 分支：训练用 lookback 窗口采样、free-run 用 `model.free_run` 滑窗递推；`lookback` 可调窗口长度。

**Q：`STRICT_ROOTZONE` 是什么、要不要开？**
A：它是"严格质量平衡"开关（默认 `False`）。`True` 时把根区目标从 SMAP 的 0-100cm 扣除为 5-100cm（消除两层重叠），
   训练/评估内部用 5-100cm 口径，`predict.py`/`evaluate.py` 输出与对比前再反向还原成 SMAP 0-100cm 口径。
   注意：切换开关后缓存文件名不同（`pack_<source>_strict.npz`），会自动重建；个别物理不自洽像元（S>R 或 Z>1）
   正向扣除时会被 clamp 到 [0,1]（有损），物理自洽像元则精确互逆。

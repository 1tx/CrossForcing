# -*- coding: utf-8 -*-
"""
统一配置文件 —— 后续所有实验从这里改，data.py / model.py / train.py / evaluate.py 不写死任何路径与超参。

约定：
  * 四个数据源已在同一 0.1° 网格 (lat 38.05~53.95, lon 115.05~134.95, 160x200)、同一逐日尺度、填充值统一 -9999.0。
  * 目标 = SMAP L4 表层 + 根区土壤水（两层状态）。
  * 主 forcing = ERA5-Land；CLDAS 用于 cross-forcing 验证。
"""
from dataclasses import dataclass, field
from pathlib import Path

# 数据根目录：ROOT = 本文件上一级（CrossForcing/）；四个数据源在 ROOT/data/ 下，cache 在 ROOT/cache/ 下。
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

# ---------------------------------------------------------------------------
# 数据路径
# ---------------------------------------------------------------------------
SMAP_DIR   = DATA_DIR / "Northeast_China_SMAPL4"    # 目标：sm_surface / sm_rootzone
ERA5_DIR   = DATA_DIR / "Northeast_China_era5land"  # 主 forcing（已逐日）
CLDAS_DIR  = DATA_DIR / "CLDAS"                     # 跨 forcing 验证
STATIC_DIR = DATA_DIR / "Northeast_China_static"    # 静态土壤质地
CACHE_DIR  = ROOT / "cache"                         # 预处理缓存（自动生成）

FILL_VALUE = -9999.0

# 主驱动数据集："era5" 或 "cldas"。切换训练/评估/推理的 forcing 只需改这一处。
# 两者都是 6 个 forcing 变量，模型输入维度不变，无需改网络结构。
SOURCE = "cldas"

# 实验标识：做多实验时改这里，所有输入缓存与输出文件会自动落到
# cache/experiments/<EXPERIMENT>/ 下（pack/weights/eval/predict 分子目录），实现实验间隔离。
EXPERIMENT = "long_rollout_cldas"

# 网格坐标（四源已对齐同一 160x200 网格：lat 38.05~53.95，lon 115.05~134.95）
GRID_LAT0 = 38.05
GRID_LON0 = 115.05
GRID_RES  = 0.1      # 度/格

# ---------------------------------------------------------------------------
# 目标变量（两层土壤水状态）
#   SMAP L4：sm_surface(0-5cm) 是 sm_rootzone(0-100cm) 的子集。
#   第一版按"两层耦合水库"简化处理；如需严格质量平衡，可把根区定义为 5-100cm。
# ---------------------------------------------------------------------------
TARGET_VARS = [
    ("sm_surface",  "m3 m-3", "表层 0-5 cm"),
    ("sm_rootzone", "m3 m-3", "根区 0-100 cm"),
]

# 严格质量平衡（深度扣除）：True 时把根区目标从 0-100cm 重定义为 5-100cm（扣除表层 0-5cm），
# 消除两层重叠、便于水量守恒。训练/评估内部用 5-100cm 口径，输出/对比时反向还原成 SMAP 0-100cm 口径。
# 正反向均为深度加权的线性可逆变换：
#   正向 Z = (1000·R − 50·S)/950     反向 R = (50·S + 950·Z)/1000
# 注意：物理自洽像元（S≤R 且 Z∈[0,1]）正反向精确互逆；个别不自洽像元（S>R 使 Z<0，或根区饱和而表层偏低使 Z>1）
#       正向扣除后 Z 会越界，clamp 到 [0,1] 会引入有损，反向还原后与原始 SMAP 略有出入。
STRICT_ROOTZONE = True

# ---------------------------------------------------------------------------
# 主 forcing：ERA5-Land（子目录名, netCDF 变量名, 单位/含义）
# ---------------------------------------------------------------------------
FORCING_ERA5 = [
    ("2m_temperature",                           "t2m",        "K"),
    ("10m_wind_speed",                           "wind_speed", "m/s"),
    ("precipitation",                            "tp",         "mm/day 日累计"),
    ("specific_humidity",                        "Q",          "kg/kg"),
    ("surface_solar_radiation_downwards_w_m2",   "ssrd",       "W/m2 下行短波"),
    ("surface_thermal_radiation_downwards_w_m2", "strd",       "W/m2 下行长波"),
]

# 跨 forcing 验证：CLDAS（子目录名, netCDF 变量名, 单位/含义）
# 注意：CLDAS 的 PRCP 单位是 mm/hr（1 小时降水量日平均），做 cross-forcing 时需先 ×24 转日总量。
FORCING_CLDAS = [
    ("TMP",  "TAIR", "K"),
    ("WIN",  "WIND", "m/s"),
    ("PRE",  "PRCP", "mm/hr (需×24 转日量)"),
    ("SHU",  "QAIR", "kg/kg"),
    ("SSRA", "SWDN", "W/m2 下行短波"),
    ("tstr", "strd", "W/m2 下行长波"),
]

# ---------------------------------------------------------------------------
# 静态变量（SoilGrids, 0-5cm）
# ---------------------------------------------------------------------------
STATIC_VARS = [
    ("clay", "g/kg", "黏粒含量"),
    ("sand", "g/kg", "砂粒含量"),
    ("silt", "g/kg", "粉粒含量"),
]

# ---------------------------------------------------------------------------
# 时间划分：目标时段 2016-2020，四段 train / val / spinup / test
#   train(2016-17) 学参数；val(2018) 选模型；spinup(2019) 只滚动不计分；test(2020) free-run。
# ---------------------------------------------------------------------------
TARGET_START_YEAR = 2016
TARGET_END_YEAR   = 2020

SPLIT_YEARS = {
    "train":  (2016, 2017),
    "val":    (2018, 2018),
    "spinup": (2019, 2019),
    "test":   (2020, 2020),
}

# ---------------------------------------------------------------------------
# 掩膜：判定陆地有效像元
# ---------------------------------------------------------------------------
MASK_MIN_VALID_FRAC = 0.5    # 目标时段内有效天数占比 >= 该值才保留
FORCING_MIN_VALID_FRAC = 0.8  # forcing 有效天数占比阈值（用于剔除数据缺口）

# ---------------------------------------------------------------------------
# 归一化：不做预归一化（输入保持原始物理单位）。
# 量纲差异统一在模型内部用 LayerNorm / BatchNorm 处理（见 ModelConfig.norm）。
# ---------------------------------------------------------------------------
NORMALIZE = False

# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    version:    str = "delta"   # 模型版本："delta"（显式状态算子）| "lstm"（隐状态 LSTM 基线）
    lookback:   int = 7         # 仅 "lstm" 使用：lookback 窗口长度（历史天数）
    n_state:    int = 2         # 表层 + 根区
    n_forcing:  int = len(FORCING_ERA5)
    n_static:   int = len(STATIC_VARS)
    hidden:     int = 128
    n_layers:   int = 3
    activation: str = "relu"
    norm:        str = "layer"  # 模型内部归一化："layer" | "batch" | "none"
    input_norm:  bool = True    # 对拼接后的输入先做一次归一化（处理异构量纲）
    # 物理约束：对状态的裁剪上下限（原始 m3/m3；None=不裁剪）
    state_clip: tuple = (0.0, 1.0)

# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    seed:          int = 0
    epochs:        int = 300
    batch_size:    int = 2048      # 每步采样多少个站点
    batches_per_epoch: int = 400    # 每 epoch 采样批次数
    lr:            float = 3e-4      # 长 rollout 课程配套更低 LR（原 1e-3；K≥30 需更稳，见 free_run改进方向.md ①）
    weight_decay:  float = 1e-5
    grad_clip:     float = 0.5       # 更强梯度裁剪（原 1.0；长 rollout 梯度易爆）
    # rollout 课程：epoch 阈值 -> 展开步长 K。
    #   0~11 K=1 → 12~59 K=7 → 60~139 K=30 → 140~219 K=60 → 220~299 K=120
    #   注意：K 越大每 epoch 越慢（K=30 实测 ~60 min/epoch，K=120 约为其 4 倍），且长 rollout 更易震荡，
    #   务必配合低 LR + 强 grad_clip；建议先小试（如只到 K=30），确认不爆再往上加。
    rollout_schedule: dict = field(default_factory=lambda: {0: 1, 12: 7, 60: 30, 140: 60, 220: 120})
    lambda_delta:  float = 0.0     # ΔSM 正则（原始单位下增量本就很小，默认 0，开启则趋向持久化）
    # 训练期每步用真值初始状态（teacher forcing 起点）；false 则用上一步预测继续
    teacher_init:  bool = True
    device:        str = "auto"    # "auto"=有 CUDA 用 cuda，否则 cpu；也可显式写 "cuda"/"cpu"


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


cfg = Config()


def exp_dir():
    """当前实验根目录：cache/experiments/<EXPERIMENT>/。"""
    return CACHE_DIR / "experiments" / EXPERIMENT


def exp_pack_dir():
    return exp_dir() / "pack"


def exp_weights_dir():
    return exp_dir() / "weights"


def exp_eval_dir():
    return exp_dir() / "eval"


def exp_predict_dir():
    return exp_dir() / "predict"


def ckpt_path(source=None, version=None):
    """按训练源与模型版本返回当前实验的 checkpoint 路径。"""
    source = source or SOURCE
    version = version or cfg.model.version
    return exp_weights_dir() / f"model_{version}_{source}.pt"


def resolve_device(pref=None):
    """把 device 配置解析成可用设备字符串；"auto" 时优先 CUDA，否则回退 cpu。"""
    pref = pref or cfg.train.device
    if pref and pref != "auto":
        return pref
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"

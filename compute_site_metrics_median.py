# -*- coding: utf-8 -*-
"""
读取每站点指标 Excel（site_metrics_<year>.xlsx），计算各指标的中位数，
输出一份 markdown 汇总文档到同目录（默认 default 实验）。

用法：
  python compute_site_metrics_median.py
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def main():
    excel = ROOT / "cache" / "experiments" / "default" / "eval" / "site_metrics_2020.xlsx"
    if not excel.exists():
        print("未找到", excel)
        sys.exit(1)

    out = excel.with_name("site_metrics_median_2020.md")

    var_names = ["surface", "rootzone"]
    metric_names = [("rmse", "RMSE"), ("mae", "MAE"), ("bias", "bias"), ("R", "R"), ("R2", "R²")]

    dfs = {sheet: pd.read_excel(excel, sheet_name=sheet) for sheet in ["one_step", "free_run"]}
    n = len(dfs["free_run"])

    lines = []
    lines.append("# 每站点指标中位数（default · cldas · test 2020）")
    lines.append("")
    lines.append(f"- 站点数 N = {n}")
    lines.append("- 每个站点、每个变量在 2020 全年的指标，取所有站点的中位数（surface=0-5cm，rootzone=0-100cm）。")
    lines.append("")

    for sheet in ["one_step", "free_run"]:
        df = dfs[sheet]
        lines.append(f"## {sheet}")
        lines.append("")
        lines.append("| 变量 | " + " | ".join(label for _, label in metric_names) + " |")
        lines.append("|---|" + "---|" * len(metric_names))
        for vn in var_names:
            cells = [vn]
            for key, _label in metric_names:
                cells.append(f"{df[f'{vn}_{key}'].median():.5f}")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    text = "\n".join(lines) + "\n"
    out.write_text(text, encoding="utf-8")
    print(text)
    print("saved ->", out)


if __name__ == "__main__":
    main()

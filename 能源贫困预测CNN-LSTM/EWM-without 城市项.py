# -*- coding: utf-8 -*-
"""
Entropy Weight Method (EWM) for Energy Poverty Index (EPI) - No Urban Components
- Compute ONE global set of weights on the whole dataset (NOT year-by-year).
- Keep the ORIGINAL row order exactly as in the input file (no sorting).
- Exclude urban-specific indicators to build a sensitivity EPI for later SHAP rerun.

Output:
- EPI_entropy_no_urban (higher = more severe energy poverty)
"""

import numpy as np
import pandas as pd

# ========= 1) 配置：列名 =========
AREA_COL = "AREA"
YEAR_COL = "YEAR"

# ========= 2) 原始指标列表 =========
INDICATORS = [
    "人均用电量（千瓦时/人）",
    "人均天然气消费量（立方米/人）",
    "城市天然气渗透率（%）",
    "城市人均天然气供应量（立方米）",
    "非火力发电占比",
    "国有经济电力、蒸汽、热水生产和供应业固定资产投资（亿元）",
    "城镇居民平均每百户电冰箱拥有量（台）",
    "城镇居民平均每百户空调拥有量（台）",
    "农村居民平均每百户抽油烟机拥有量（台）",
    "农村太阳能热水器人均覆盖面积（平方米）",
    "居民生活二氧化硫人均排放量（吨）",
]

# 方向：'-' 表示越大越不贫困（需要反向），'+' 表示越大越贫困（正向）
DIRECTIONS = {
    "人均用电量（千瓦时/人）": "-",
    "人均天然气消费量（立方米/人）": "-",
    "城市天然气渗透率（%）": "-",
    "城市人均天然气供应量（立方米）": "-",
    "非火力发电占比": "-",
    "国有经济电力、蒸汽、热水生产和供应业固定资产投资（亿元）": "-",
    "城镇居民平均每百户电冰箱拥有量（台）": "-",
    "城镇居民平均每百户空调拥有量（台）": "-",
    "农村居民平均每百户抽油烟机拥有量（台）": "-",
    "农村太阳能热水器人均覆盖面积（平方米）": "-",
    "居民生活二氧化硫人均排放量（吨）": "+",
}

# ========= 3) 定义要剔除的城市项（方案1） =========
URBAN_SPECIFIC_INDICATORS = [
    "城市天然气渗透率（%）",
    "城市人均天然气供应量（立方米）",
    "城镇居民平均每百户电冰箱拥有量（台）",
    "城镇居民平均每百户空调拥有量（台）",
]

# 最终用于计算“去城市项EPI”的指标
INDICATORS_NO_URBAN = [c for c in INDICATORS if c not in URBAN_SPECIFIC_INDICATORS]


# ========= 4) 归一化（Min-Max，统一为“贫困越大越差”方向） =========
def minmax_normalize_poverty_oriented(df: pd.DataFrame, cols: list, directions: dict) -> pd.DataFrame:
    """
    返回 Z（0-1），且 Z 越大表示越贫困（越差）。
    '-' 指标：Z = (max - x)/(max - min)
    '+' 指标：Z = (x - min)/(max - min)
    """
    Z = pd.DataFrame(index=df.index)

    for c in cols:
        x = pd.to_numeric(df[c], errors="coerce")

        # 全空列保护
        if x.notna().sum() == 0:
            Z[c] = np.nan
            continue

        xmin = np.nanmin(x.values)
        xmax = np.nanmax(x.values)

        # 常数列或全空列：没有区分度，设为0
        if np.isnan(xmin) or np.isnan(xmax) or np.isclose(xmax, xmin):
            Z[c] = 0.0
            continue

        if directions[c] == "+":
            Z[c] = (x - xmin) / (xmax - xmin)
        elif directions[c] == "-":
            Z[c] = (xmax - x) / (xmax - xmin)
        else:
            raise ValueError(f"Unknown direction for {c}: {directions[c]} (use '+' or '-')")

    return Z


# ========= 5) 熵权法：由Z计算权重，并得到EPI =========
def entropy_weight_method(Z: pd.DataFrame, epsilon: float = 1e-12):
    """
    输入：Z (n x m) 的贫困向归一化矩阵，范围[0,1]
    输出：weights（m,），EPI（n,）
    """
    X = Z.fillna(0.0).to_numpy(dtype=float)

    n, m = X.shape
    if n < 2:
        raise ValueError("样本数n太小，无法进行熵权法（至少需要2行）。")
    if m < 1:
        raise ValueError("指标数m不能为空。")

    # p_ij = x_ij / sum_i x_ij
    col_sums = X.sum(axis=0)

    P = np.zeros_like(X)
    for j in range(m):
        # 某列全0时给均匀分布，避免除零
        if np.isclose(col_sums[j], 0.0):
            P[:, j] = 1.0 / n
        else:
            P[:, j] = X[:, j] / col_sums[j]

    # 熵值
    k = 1.0 / np.log(n)
    P_safe = np.clip(P, epsilon, 1.0)  # 防止ln(0)
    E = -k * np.sum(P_safe * np.log(P_safe), axis=0)

    # 差异系数 & 权重
    D = 1.0 - E
    if np.isclose(D.sum(), 0.0):
        W = np.ones(m) / m
    else:
        W = D / D.sum()

    weights = pd.Series(W, index=Z.columns, name="weight")

    # EPI = sum_j w_j * z_ij (越大越贫困)
    EPI = (Z.fillna(0.0) * weights).sum(axis=1)
    EPI.name = "EPI_entropy_no_urban"

    return weights, EPI


# ========= 6) 主函数：读取 -> 归一化 -> 权重 -> 指数 -> 导出（不改顺序） =========
def compute_epi_entropy_no_urban(df: pd.DataFrame, export_path: str = "EPI_entropy_no_urban_output.xlsx"):
    # 检查列
    required_cols = [AREA_COL, YEAR_COL] + INDICATORS_NO_URBAN
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"数据缺少以下列：{missing}")

    # 保留原始顺序：绝不 sort / reset / groupby 改顺序
    base = df[[AREA_COL, YEAR_COL]].copy()

    # 1) Min-Max归一化（统一为“贫困向”）
    Z = minmax_normalize_poverty_oriented(df, INDICATORS_NO_URBAN, DIRECTIONS)

    # 2) 熵权
    weights, epi = entropy_weight_method(Z)

    # 3) 输出（保持原始行顺序）
    score_df = base.copy()
    score_df["EPI_entropy_no_urban"] = epi.values

    weights_df = weights.reset_index().rename(columns={"index": "indicator"})
    weights_df["is_urban_specific"] = False  # 此版本已剔除城市项

    normalized_df = pd.concat([base, Z], axis=1)
    normalized_df["EPI_entropy_no_urban"] = epi.values

    # 4) 导出
    with pd.ExcelWriter(export_path, engine="openpyxl") as writer:
        score_df.to_excel(writer, sheet_name="EPI_score_no_urban", index=False)
        weights_df.to_excel(writer, sheet_name="Weights_no_urban", index=False)
        normalized_df.to_excel(writer, sheet_name="Normalized_Z_no_urban", index=False)

    return score_df, weights_df, normalized_df


# ========= 7) 运行入口 =========
if __name__ == "__main__":
    data_path = "EP数据(2).xlsx"  # 改成你的文件名/路径

    if data_path.lower().endswith(".csv"):
        df0 = pd.read_csv(data_path, encoding="utf-8-sig")
    else:
        df0 = pd.read_excel(data_path)

    score_df, weights_df, norm_df = compute_epi_entropy_no_urban(
        df0,
        export_path="EPI_entropy_no_urban_output.xlsx"
    )

    print("Done. Saved to EPI_entropy_no_urban_output.xlsx")
    print("\n[Used indicators (no urban-specific components)]")
    for c in INDICATORS_NO_URBAN:
        print("-", c)

    print("\n[Head of EPI scores]")
    print(score_df.head(10))

    print("\n[Top weights]")
    print(weights_df.sort_values("weight", ascending=False).head(5))
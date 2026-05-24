# -*- coding: utf-8 -*-
"""
Entropy Weight Method (EWM) for Energy Poverty Index (EPI)
- Compute ONE global set of weights on the whole dataset (NOT year-by-year).
- Keep the ORIGINAL row order exactly as in the input file (no sorting).

Assumption (consistent with your direction table):
- Direction '-' : higher value -> less energy poverty (protective / reduces poverty)
- Direction '+' : higher value -> more energy poverty (risk / increases poverty)

We build a "poverty-oriented" normalized matrix Z in [0,1]:
- For '-' indicators (higher -> less poverty): reverse min-max => Z = (max - x)/(max - min)
- For '+' indicators (higher -> more poverty): normal min-max  => Z = (x - min)/(max - min)

Then entropy weights are computed from Z, and EPI = sum_j w_j * Z_ij.
So: Higher EPI => more severe energy poverty.
"""

import numpy as np
import pandas as pd

# ========= 1) 配置：列名 =========
AREA_COL = "AREA"
YEAR_COL = "YEAR"

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


# ========= 2) 归一化（Min-Max，统一为“贫困越大越差”方向） =========
def minmax_normalize_poverty_oriented(df: pd.DataFrame, cols: list, directions: dict) -> pd.DataFrame:
    """
    返回 Z（0-1），且 Z 越大表示越贫困（越差）。
    '-' 指标：Z = (max - x)/(max - min)
    '+' 指标：Z = (x - min)/(max - min)
    """
    Z = pd.DataFrame(index=df.index)

    for c in cols:
        x = pd.to_numeric(df[c], errors="coerce")
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

    # 缺失值保留为 NaN（你也可选择填补）
    return Z


# ========= 3) 熵权法：由Z计算权重，并得到EPI =========
def entropy_weight_method(Z: pd.DataFrame, epsilon: float = 1e-12):
    """
    输入：Z (n x m) 的贫困向归一化矩阵，范围[0,1]
    输出：weights（m,），EPI（n,）
    """
    # 用于熵计算的矩阵：NaN 转 0（不建议大量缺失；若缺失较多应先插补）
    X = Z.fillna(0.0).to_numpy(dtype=float)

    n, m = X.shape
    if n < 2:
        raise ValueError("样本数n太小，无法进行熵权法（至少需要2行）。")

    # p_ij = x_ij / sum_i x_ij
    col_sums = X.sum(axis=0)
    # 避免某列全0导致除0：全0列给均匀分布（其差异度很低，权重也会接近0）
    P = np.zeros_like(X)
    for j in range(m):
        if np.isclose(col_sums[j], 0.0):
            P[:, j] = 1.0 / n
        else:
            P[:, j] = X[:, j] / col_sums[j]

    # 熵值 e_j = -k * sum_i p_ij * ln(p_ij)
    k = 1.0 / np.log(n)
    P_safe = np.clip(P, epsilon, 1.0)  # 防止ln(0)
    E = -k * np.sum(P_safe * np.log(P_safe), axis=0)

    # 差异系数 d_j = 1 - e_j；权重 w_j = d_j / sum d_j
    D = 1.0 - E
    if np.isclose(D.sum(), 0.0):
        W = np.ones(m) / m
    else:
        W = D / D.sum()

    weights = pd.Series(W, index=Z.columns, name="weight")

    # EPI = sum_j w_j * z_ij
    EPI = (Z.fillna(0.0) * weights).sum(axis=1)
    EPI.name = "EPI_entropy"

    return weights, EPI


# ========= 4) 主函数：读取 -> 归一化 -> 权重 -> 指数 -> 导出（不改顺序） =========
def compute_epi_entropy(df: pd.DataFrame, export_path: str = "EPI_entropy_output.xlsx"):
    # 检查列
    required_cols = [AREA_COL, YEAR_COL] + INDICATORS
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"数据缺少以下列：{missing}")

    # 保留原始顺序：绝不 sort / reset / groupby 改顺序
    base = df[[AREA_COL, YEAR_COL]].copy()
    X = df[INDICATORS].copy()

    # 1) Min-Max归一化（统一为“贫困向”）
    Z = minmax_normalize_poverty_oriented(df, INDICATORS, DIRECTIONS)

    # 2) 熵权
    weights, epi = entropy_weight_method(Z)

    # 3) 输出（保持原始行顺序）
    score_df = base.copy()
    score_df["EPI_entropy"] = epi.values

    weights_df = weights.reset_index().rename(columns={"index": "indicator"})

    normalized_df = pd.concat([base, Z], axis=1)
    normalized_df["EPI_entropy"] = epi.values

    # 4) 导出
    with pd.ExcelWriter(export_path, engine="openpyxl") as writer:
        score_df.to_excel(writer, sheet_name="EPI_score", index=False)
        weights_df.to_excel(writer, sheet_name="Weights", index=False)
        normalized_df.to_excel(writer, sheet_name="Normalized_Z", index=False)

    return score_df, weights_df, normalized_df


# ========= 5) 运行入口 =========
if __name__ == "__main__":
    data_path = "EP数据(2).xlsx"  # 改成你的文件名/路径

    if data_path.lower().endswith(".csv"):
        df0 = pd.read_csv(data_path, encoding="utf-8-sig")
    else:
        df0 = pd.read_excel(data_path)

    score_df, weights_df, norm_df = compute_epi_entropy(
        df0,
        export_path="EPI_entropy_output.xlsx"
    )

    print("Done. Saved to EPI_entropy_output.xlsx")
    print(score_df.head(10))
    print("\nTop weights:")
    print(weights_df.sort_values("weight", ascending=False).head(5))

# -*- coding: utf-8 -*-
"""
Compute Energy Poverty Index (EPI) using:
1) CRITIC
2) CV (Coefficient of Variation)
3) Deviation Maximization (DM)

Steps:
- Poverty-oriented Min-Max normalization to Z in [0,1] using your +/- directions
- Compute weights using each method on Z
- EPI_method = sum_j w_j * Z_ij
- Keep original row order (NO sorting)

Output Excel:
- EPI_score: AREA, YEAR, EPI_CRITIC, EPI_CV, EPI_DM
- Normalized_Z
- CRITIC_Weights / CV_Weights / DM_Weights
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

# ========= 2) 贫困向归一化 Z =========
def minmax_normalize_poverty_oriented(df: pd.DataFrame, cols: list, directions: dict) -> pd.DataFrame:
    """
    Poverty-oriented min-max normalize to [0,1]:
      '-' : Z=(max-x)/(max-min)
      '+' : Z=(x-min)/(max-min)
    """
    Z = pd.DataFrame(index=df.index)
    for c in cols:
        x = pd.to_numeric(df[c], errors="coerce")
        xmin = np.nanmin(x.values)
        xmax = np.nanmax(x.values)

        if np.isnan(xmin) or np.isnan(xmax) or np.isclose(xmax, xmin):
            Z[c] = 0.0
            continue

        if directions[c] == "+":
            Z[c] = (x - xmin) / (xmax - xmin)
        elif directions[c] == "-":
            Z[c] = (xmax - x) / (xmax - xmin)
        else:
            raise ValueError(f"Unknown direction for {c}: {directions[c]}")
    return Z

def safe_weights(raw_w: np.ndarray) -> np.ndarray:
    raw_w = np.asarray(raw_w, dtype=float)
    s = np.sum(raw_w)
    if np.isnan(s) or np.isclose(s, 0.0):
        return np.ones_like(raw_w) / len(raw_w)
    return raw_w / s

# ========= 3) CRITIC =========
def critic_weights(Z: pd.DataFrame):
    """
    CRITIC:
      std_j = std(Z_j)
      conflict_j = sum_k (1 - corr_jk)
      C_j = std_j * conflict_j
      w_j = C_j / sum C_j
    """
    Zf = Z.fillna(0.0).astype(float)
    X = Zf.to_numpy()
    m = X.shape[1]

    std = Zf.std(axis=0, ddof=1).to_numpy()
    corr = np.corrcoef(X, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)

    conflict = np.sum(1.0 - corr, axis=1)
    C = std * conflict
    w = safe_weights(C)

    weights = pd.Series(w, index=Z.columns, name="weight")
    detail = pd.DataFrame({
        "indicator": Z.columns,
        "std": std,
        "conflict": conflict,
        "C": C,
        "weight": w
    })
    epi = (Zf * weights).sum(axis=1).rename("EPI_CRITIC")
    return weights, epi, detail

# ========= 4) CV（变异系数法） =========
def cv_weights(Z: pd.DataFrame, eps: float = 1e-12):
    """
    CV_j = std(Z_j) / mean(Z_j)
    w_j = CV_j / sum(CV_j)
    """
    Zf = Z.fillna(0.0).astype(float)
    mean = Zf.mean(axis=0).to_numpy()
    std = Zf.std(axis=0, ddof=1).to_numpy()

    cv = std / (mean + eps)
    w = safe_weights(cv)

    weights = pd.Series(w, index=Z.columns, name="weight")
    detail = pd.DataFrame({
        "indicator": Z.columns,
        "mean": mean,
        "std": std,
        "CV": cv,
        "weight": w
    })
    epi = (Zf * weights).sum(axis=1).rename("EPI_CV")
    return weights, epi, detail

# ========= 5) Deviation Maximization（最大离差法） =========
def deviation_maximization_weights(Z: pd.DataFrame):
    """
    DM（常用实现之一）：
      D_j = sum_{i=1..n} |z_ij - mean(z_j)|
      w_j = D_j / sum(D_j)

    你也可以改成 sum_{i,k} |z_ij - z_kj|（更重，但结果同向）
    """
    Zf = Z.fillna(0.0).astype(float)
    mean = Zf.mean(axis=0)
    D = (Zf.sub(mean, axis=1).abs()).sum(axis=0).to_numpy()

    w = safe_weights(D)

    detail = pd.DataFrame({
        "indicator": Z.columns,
        "mean": mean.to_numpy(),
        "D": D,
        "weight": w
    })
    weights = pd.Series(w, index=Z.columns, name="weight")
    epi = (Zf * weights).sum(axis=1).rename("EPI_DM")
    return weights, epi, detail

# ========= 6) 主函数：计算三种方法并导出 =========
def compute_epi_critic_cv_dm(df: pd.DataFrame, export_path: str = "EPI_CRITIC_CV_DM_output.xlsx"):
    required_cols = [AREA_COL, YEAR_COL] + INDICATORS
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"数据缺少以下列：{missing}")

    base = df[[AREA_COL, YEAR_COL]].copy()  # 保持原始顺序

    # 1) Z
    Z = minmax_normalize_poverty_oriented(df, INDICATORS, DIRECTIONS)

    # 2) CRITIC
    w_c, epi_c, det_c = critic_weights(Z)

    # 3) CV
    w_v, epi_v, det_v = cv_weights(Z)

    # 4) DM
    w_d, epi_d, det_d = deviation_maximization_weights(Z)

    # 5) 输出（保持原始行顺序）
    score_df = base.copy()
    score_df["EPI_CRITIC"] = epi_c.values
    score_df["EPI_CV"] = epi_v.values
    score_df["EPI_DM"] = epi_d.values

    normalized_df = pd.concat([base, Z], axis=1)

    # 6) 导出
    with pd.ExcelWriter(export_path, engine="openpyxl") as writer:
        score_df.to_excel(writer, sheet_name="EPI_score", index=False)
        normalized_df.to_excel(writer, sheet_name="Normalized_Z", index=False)

        det_c.to_excel(writer, sheet_name="CRITIC_Weights", index=False)
        det_v.to_excel(writer, sheet_name="CV_Weights", index=False)
        det_d.to_excel(writer, sheet_name="DM_Weights", index=False)

    return score_df, det_c, det_v, det_d

# ========= 7) 运行入口 =========
if __name__ == "__main__":
    data_path = "EP数据(2).xlsx"  # 改成你的文件名/路径

    if data_path.lower().endswith(".csv"):
        df0 = pd.read_csv(data_path, encoding="utf-8-sig")
    else:
        df0 = pd.read_excel(data_path)

    score_df, det_c, det_v, det_d = compute_epi_critic_cv_dm(
        df0,
        export_path="EPI_CRITIC_CV_DM_output.xlsx"
    )

    print("Done. Saved to EPI_CRITIC_CV_DM_output.xlsx")
    print(score_df.head(10))

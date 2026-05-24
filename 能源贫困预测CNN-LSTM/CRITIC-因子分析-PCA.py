# -*- coding: utf-8 -*-
"""
Compute Energy Poverty Index (EPI) using:
1) CRITIC
2) Factor Analysis (FA)
3) PCA

Steps:
- Poverty-oriented Min-Max normalization to Z in [0,1] using your +/- directions
- CRITIC weights computed on Z
- PCA/FA computed on standardized Z (z-score), then composite score oriented & scaled to [0,1]
- Keep original row order (NO sorting)

Output Excel:
- EPI_score: AREA, YEAR, EPI_CRITIC, EPI_PCA, EPI_FA (+ raw scores)
- Normalized_Z: poverty-oriented normalized indicators
- CRITIC_Weights: CRITIC weights and components
- PCA_Info, PCA_Loadings
- FA_Info, FA_Loadings
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FactorAnalysis

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

SEED = 42

# ========= 2) 工具函数 =========
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

def minmax01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    xmin, xmax = np.nanmin(x), np.nanmax(x)
    if np.isnan(xmin) or np.isnan(xmax) or np.isclose(xmax, xmin):
        return np.zeros_like(x, dtype=float)
    return (x - xmin) / (xmax - xmin)

def orient_by_reference(score_raw: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """If corr(score, ref) < 0 then flip sign so that higher = more poverty."""
    s = np.asarray(score_raw, dtype=float).reshape(-1)
    r = np.asarray(ref, dtype=float).reshape(-1)
    if np.std(s) < 1e-12 or np.std(r) < 1e-12:
        return s
    corr = np.corrcoef(s, r)[0, 1]
    if np.isnan(corr):
        return s
    return -s if corr < 0 else s

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

    std = Zf.std(axis=0, ddof=1).to_numpy()  # (m,)
    corr = np.corrcoef(X, rowvar=False)      # (m,m)
    corr = np.nan_to_num(corr, nan=0.0)

    conflict = np.sum(1.0 - corr, axis=1)    # includes self (1-1=0), ok
    C = std * conflict

    if np.isclose(C.sum(), 0.0):
        w = np.ones(m) / m
    else:
        w = C / C.sum()

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

# ========= 4) PCA =========
def pca_index(Z: pd.DataFrame, var_threshold: float = 0.80):
    """
    PCA on standardized Z.
    Composite score = sum_{k=1..K} (var_ratio_k / sum var_ratio_1..K) * PC_score_k
    K chosen by cumulative explained variance >= var_threshold.
    Then orient and min-max to [0,1].
    """
    Zf = Z.fillna(0.0).astype(float)
    ref = Zf.mean(axis=1).to_numpy()

    scaler = StandardScaler()
    Xs = scaler.fit_transform(Zf.to_numpy())

    pca = PCA(random_state=SEED)
    pcs = pca.fit_transform(Xs)  # (n, m)

    var_ratio = pca.explained_variance_ratio_
    cum = np.cumsum(var_ratio)
    K = int(np.searchsorted(cum, var_threshold) + 1)
    K = max(1, min(K, pcs.shape[1]))

    w = var_ratio[:K]
    w = w / w.sum()

    score_raw = pcs[:, :K] @ w
    score_raw = orient_by_reference(score_raw, ref)
    score_01 = minmax01(score_raw)

    # loadings: (features x components)
    loadings = (pca.components_.T)  # columns are PCs
    load_df = pd.DataFrame(loadings[:, :K], index=Z.columns, columns=[f"PC{k+1}" for k in range(K)]).reset_index()
    load_df = load_df.rename(columns={"index": "indicator"})

    info_df = pd.DataFrame({
        "component": [f"PC{k+1}" for k in range(len(var_ratio))],
        "explained_variance_ratio": var_ratio,
        "cumulative_ratio": cum
    })
    meta = {"K": K, "var_threshold": var_threshold}

    return score_raw, score_01, info_df, load_df, meta

# ========= 5) Factor Analysis (FA) =========
def fa_index(Z: pd.DataFrame):
    """
    FA on standardized Z.
    Choose number of factors m by Kaiser criterion (eigenvalues of corr > 1), at least 1.
    Weight factor scores by corresponding eigenvalues proportion (first m eigenvalues).
    Then orient and min-max to [0,1].
    """
    Zf = Z.fillna(0.0).astype(float)
    ref = Zf.mean(axis=1).to_numpy()

    scaler = StandardScaler()
    Xs = scaler.fit_transform(Zf.to_numpy())

    # eigenvalues of correlation matrix for choosing number of factors
    R = np.corrcoef(Xs, rowvar=False)
    R = np.nan_to_num(R, nan=0.0)
    eigvals = np.linalg.eigvalsh(R)[::-1]  # descending
    m = int(np.sum(eigvals > 1.0))
    m = max(1, min(m, Xs.shape[1] - 1))  # keep safe

    fa = FactorAnalysis(n_components=m, random_state=SEED)
    factor_scores = fa.fit_transform(Xs)  # (n, m)

    # factor weights from eigenvalues
    w = eigvals[:m]
    if np.isclose(w.sum(), 0.0):
        w = np.ones(m) / m
    else:
        w = w / w.sum()

    score_raw = factor_scores @ w
    score_raw = orient_by_reference(score_raw, ref)
    score_01 = minmax01(score_raw)

    # loadings: sklearn FactorAnalysis components_ shape (m, features)
    loadings = fa.components_.T  # (features, m)
    load_df = pd.DataFrame(loadings, index=Z.columns, columns=[f"F{k+1}" for k in range(m)]).reset_index()
    load_df = load_df.rename(columns={"index": "indicator"})

    info_df = pd.DataFrame({
        "eigenvalue": eigvals,
        "kaiser_gt_1": eigvals > 1.0
    })
    meta = {"m": m}

    return score_raw, score_01, info_df, load_df, meta

# ========= 6) 主函数：计算三种方法并导出 =========
def compute_epi_critic_fa_pca(df: pd.DataFrame, export_path: str = "EPI_CRITIC_FA_PCA_output.xlsx"):
    # 检查列
    required_cols = [AREA_COL, YEAR_COL] + INDICATORS
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"数据缺少以下列：{missing}")

    base = df[[AREA_COL, YEAR_COL]].copy()  # 保持原始顺序

    # 1) Poverty-oriented min-max Z
    Z = minmax_normalize_poverty_oriented(df, INDICATORS, DIRECTIONS)

    # 2) CRITIC
    w_critic, epi_critic, critic_detail = critic_weights(Z)

    # 3) PCA
    pca_raw, pca_01, pca_info, pca_load, pca_meta = pca_index(Z, var_threshold=0.80)

    # 4) Factor Analysis
    fa_raw, fa_01, fa_info, fa_load, fa_meta = fa_index(Z)

    # 5) 输出表（保持原顺序）
    score_df = base.copy()
    score_df["EPI_CRITIC"] = epi_critic.values
    score_df["EPI_PCA"] = pca_01
    score_df["EPI_FA"] = fa_01
    score_df["EPI_PCA_raw"] = pca_raw
    score_df["EPI_FA_raw"] = fa_raw

    normalized_df = pd.concat([base, Z], axis=1)

    # 6) 导出Excel
    with pd.ExcelWriter(export_path, engine="openpyxl") as writer:
        score_df.to_excel(writer, sheet_name="EPI_score", index=False)
        normalized_df.to_excel(writer, sheet_name="Normalized_Z", index=False)

        critic_detail.to_excel(writer, sheet_name="CRITIC_Weights", index=False)

        pca_info.to_excel(writer, sheet_name="PCA_Info", index=False)
        pca_load.to_excel(writer, sheet_name="PCA_Loadings", index=False)
        pd.DataFrame([pca_meta]).to_excel(writer, sheet_name="PCA_Meta", index=False)

        fa_info.to_excel(writer, sheet_name="FA_Info", index=False)
        fa_load.to_excel(writer, sheet_name="FA_Loadings", index=False)
        pd.DataFrame([fa_meta]).to_excel(writer, sheet_name="FA_Meta", index=False)

    return score_df, critic_detail, pca_info, fa_info

# ========= 7) 运行入口 =========
if __name__ == "__main__":
    data_path = "EP数据(2).xlsx"  # 改成你的文件名/路径

    if data_path.lower().endswith(".csv"):
        df0 = pd.read_csv(data_path, encoding="utf-8-sig")
    else:
        df0 = pd.read_excel(data_path)

    score_df, critic_detail, pca_info, fa_info = compute_epi_critic_fa_pca(
        df0,
        export_path="EPI_CRITIC_FA_PCA_output.xlsx"
    )

    print("Done. Saved to EPI_CRITIC_FA_PCA_output.xlsx")
    print(score_df.head(10))

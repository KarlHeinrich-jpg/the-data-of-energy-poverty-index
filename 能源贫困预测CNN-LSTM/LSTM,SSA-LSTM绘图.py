# -*- coding: utf-8 -*-
import os
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.collections as mcoll
import seaborn as sns
from scipy.stats import gaussian_kde
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ======================
# 0) 基本配置（改这里）
# ======================
DATA_PATH = r"EPI_entropy_output.xlsx"                             # <<< 数据文件
SSA_CKPT_PATH = r"best_SSA_LSTM_checkpoint_manual_split.pth"      # <<< 手动切分版SSA-LSTM checkpoint（推荐）

AREA_COL = "AREA"
YEAR_COL = "YEAR"
TARGET_COL = "EPI_entropy"

FEATURES_CN = [
    "人均地区生产总值(元/人)",
    "居民消费(亿元)",
    "第二产业占比",
    "能源消费强度",
    "能源消费比",
    "人均发电量",
    "人口密度",
    "城镇人口占比",
    "森林覆盖率(%)",
    "地方财政环境保护支出(亿元)",
    "人均水资源量(立方米/人)",
]

# ===== 手动设置 Train / Validation / Test（按目标年份，闭区间）=====
TRAIN_TARGET_YEARS = (2008, 2017)   # 例：训练集目标年份
VAL_TARGET_YEARS   = (2018, 2019)   # 例：验证集目标年份（用于早停/调参）
TEST_TARGET_YEARS  = (2020, 2022)   # 例：测试集目标年份（最终评估）
# 是否检查输入窗口年份重叠（Comment 3相关）
CHECK_INPUT_YEAR_OVERLAP = True
STRICT_NO_INPUT_YEAR_OVERLAP = False   # True则一旦有重叠直接报错（严格模式）

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 你指定的 LSTM 默认参数（baseline LSTM）
LSTM_BASELINE_PARAMS = dict(
    lookback=5,
    hidden_size=64,
    num_layers=1,
    dropout=0.05,
    lr=1e-2,
    batch_size=64,
    epochs=500,
    patience=20
)

# 输出
OUT_XLSX = "Metrics_SSA_LSTM_vs_LSTM_manual_split.xlsx"
SAVE_DIR = "./figure"
OUT_FIG = f"{SAVE_DIR}/Fig_Model_Density_SSA_LSTM_vs_LSTM_manual_split.png"


# ======================
# 1) 复现性设置（重要）
# ======================
def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 尽量确定性（更一致，但可能更慢）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_all_seeds(SEED)


# ======================
# 2) 读取Excel：自动找包含所需列的sheet
# ======================
def find_sheet_with_columns(path: str, required_cols: list):
    xls = pd.ExcelFile(path)
    for sh in xls.sheet_names:
        cols = pd.read_excel(path, sheet_name=sh, nrows=0).columns.tolist()
        if all(c in cols for c in required_cols):
            df0 = pd.read_excel(path, sheet_name=sh)
            return sh, df0
    raise ValueError(
        f"在 {path} 的所有sheet中都没找到包含所需列的sheet。\n所需列：{required_cols}"
    )

required_cols = [AREA_COL, YEAR_COL, TARGET_COL] + FEATURES_CN
sheet_used, df_raw = find_sheet_with_columns(DATA_PATH, required_cols)
print(f"[INFO] Using sheet: {sheet_used}")

data = df_raw[required_cols].copy()
data[YEAR_COL] = pd.to_numeric(data[YEAR_COL], errors="coerce")
data[TARGET_COL] = pd.to_numeric(data[TARGET_COL], errors="coerce")
for c in FEATURES_CN:
    data[c] = pd.to_numeric(data[c], errors="coerce")
data = data.dropna(subset=[YEAR_COL, TARGET_COL])


# ======================
# 3) 手动年份切分：序列构造 + 元信息
# ======================
def build_sequences_with_meta(df_in: pd.DataFrame, lookback: int):
    """
    构造序列样本，并记录元信息：
    - AREA
    - target_year
    - input_start_year
    - input_end_year
    """
    X_all, y_all = [], []
    meta_rows = []

    for area, g in df_in.groupby(AREA_COL, sort=False):
        g = g.sort_values(YEAR_COL).reset_index(drop=True)

        Xg = g[FEATURES_CN].values.astype(float)
        yg = g[TARGET_COL].values.astype(float)
        years = g[YEAR_COL].values.astype(int)

        # 输入窗口 [t-lookback, ..., t-1] -> 预测 y[t]
        for t in range(lookback, len(g)):
            X_all.append(Xg[t - lookback:t, :])
            y_all.append(yg[t])

            meta_rows.append({
                "AREA": area,
                "target_year": int(years[t]),
                "input_start_year": int(years[t - lookback]),
                "input_end_year": int(years[t - 1]),
            })

    if len(X_all) == 0:
        raise ValueError("样本构造失败：可能lookback太大或年份不足。")

    X_all = np.asarray(X_all, dtype=float)
    y_all = np.asarray(y_all, dtype=float)
    meta_all = pd.DataFrame(meta_rows)
    return X_all, y_all, meta_all


def _mask_year_range(series: pd.Series, year_range):
    y0, y1 = year_range
    return (series >= y0) & (series <= y1)


def _collect_input_year_set(meta_df: pd.DataFrame):
    s = set()
    for _, r in meta_df.iterrows():
        s.update(range(int(r["input_start_year"]), int(r["input_end_year"]) + 1))
    return s


def split_by_manual_target_years(X_all, y_all, meta_all):
    """
    按 target_year 手动切 train / val / test
    返回顺序：
    Xtr, ytr, Xva, yva, Xte, yte, meta_train, meta_val, meta_test, overlap_info
    """
    train_mask = _mask_year_range(meta_all["target_year"], TRAIN_TARGET_YEARS)
    val_mask   = _mask_year_range(meta_all["target_year"], VAL_TARGET_YEARS)
    test_mask  = _mask_year_range(meta_all["target_year"], TEST_TARGET_YEARS)

    if (train_mask & val_mask).any() or (train_mask & test_mask).any() or (val_mask & test_mask).any():
        raise ValueError("train/val/test 的目标年份范围有重叠，请检查年份设置。")

    Xtr = X_all[train_mask.values]
    ytr = y_all[train_mask.values]
    mtr = meta_all.loc[train_mask].reset_index(drop=True)

    Xva = X_all[val_mask.values]
    yva = y_all[val_mask.values]
    mva = meta_all.loc[val_mask].reset_index(drop=True)

    Xte = X_all[test_mask.values]
    yte = y_all[test_mask.values]
    mte = meta_all.loc[test_mask].reset_index(drop=True)

    if len(Xtr) == 0 or len(Xva) == 0 or len(Xte) == 0:
        raise ValueError(
            f"切分后某个数据集为空：train={len(Xtr)}, val={len(Xva)}, test={len(Xte)}。\n"
            "请检查 lookback 与年份范围设置。"
        )

    overlap_info = {}
    if CHECK_INPUT_YEAR_OVERLAP:
        tr_in_years = _collect_input_year_set(mtr)
        va_in_years = _collect_input_year_set(mva)
        te_in_years = _collect_input_year_set(mte)

        ov_tr_va = sorted(tr_in_years & va_in_years)
        ov_tr_te = sorted(tr_in_years & te_in_years)
        ov_va_te = sorted(va_in_years & te_in_years)

        overlap_info = {
            "train_input_years": tr_in_years,
            "val_input_years": va_in_years,
            "test_input_years": te_in_years,
            "ov_train_val": ov_tr_va,
            "ov_train_test": ov_tr_te,
            "ov_val_test": ov_va_te,
        }

        if STRICT_NO_INPUT_YEAR_OVERLAP and (ov_tr_va or ov_tr_te or ov_va_te):
            raise ValueError(
                "检测到输入年份重叠（feature-space overlap 风险）。\n"
                f"train∩val={ov_tr_va}\n"
                f"train∩test={ov_tr_te}\n"
                f"val∩test={ov_va_te}\n"
                "请调整 TRAIN/VAL/TEST 目标年份范围。"
            )

    return Xtr, ytr, Xva, yva, Xte, yte, mtr, mva, mte, overlap_info


def build_manual_split(df_in: pd.DataFrame, lookback: int):
    X_all, y_all, meta_all = build_sequences_with_meta(df_in, lookback)
    return split_by_manual_target_years(X_all, y_all, meta_all)


def print_split_check_once(lookback: int, mtr: pd.DataFrame, mva: pd.DataFrame, mte: pd.DataFrame, overlap_info: dict):
    print(f"\n[Split Check | lookback={lookback}]")
    print(f"Train target years: {TRAIN_TARGET_YEARS}, n={len(mtr)}")
    print(f"Val   target years: {VAL_TARGET_YEARS}, n={len(mva)}")
    print(f"Test  target years: {TEST_TARGET_YEARS}, n={len(mte)}")

    if overlap_info:
        tr_in = overlap_info["train_input_years"]
        va_in = overlap_info["val_input_years"]
        te_in = overlap_info["test_input_years"]

        print(f"Train input-year range: {min(tr_in)}-{max(tr_in)}")
        print(f"Val   input-year range: {min(va_in)}-{max(va_in)}")
        print(f"Test  input-year range: {min(te_in)}-{max(te_in)}")
        print(f"Overlap train∩val : {overlap_info['ov_train_val']}")
        print(f"Overlap train∩test: {overlap_info['ov_train_test']}")
        print(f"Overlap val∩test  : {overlap_info['ov_val_test']}")
        print(f"[STRICT MODE] {STRICT_NO_INPUT_YEAR_OVERLAP}")
    print()


def fit_scaler_on_train_and_transform(Xtr_raw, Xte_raw, Xva_raw):
    """标准化：只用Train拟合，然后作用于Train/Test/Val"""
    n_feat = Xtr_raw.shape[-1]
    scaler = StandardScaler()
    scaler.fit(Xtr_raw.reshape(-1, n_feat))

    Xtr = scaler.transform(Xtr_raw.reshape(-1, n_feat)).reshape(Xtr_raw.shape)
    Xte = scaler.transform(Xte_raw.reshape(-1, n_feat)).reshape(Xte_raw.shape)
    Xva = scaler.transform(Xva_raw.reshape(-1, n_feat)).reshape(Xva_raw.shape)
    return scaler, Xtr, Xte, Xva


# ======================
# 4) 指标函数
# ======================
def calc_metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    return rmse, mae, r2


# ======================
# 5) LSTM结构（与SSA-LSTM一致）
# ======================
class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last)


def predict_torch(model, Xnp, device=DEVICE):
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(Xnp, dtype=torch.float32).to(device)
        return model(xb).detach().cpu().numpy().reshape(-1)


# ======================
# 6) SSA-LSTM：从checkpoint加载（不训练）
# ======================
def eval_ssa_lstm_from_checkpoint(df_in: pd.DataFrame, ckpt_path: str):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"找不到SSA-LSTM checkpoint：{ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    best_params = ckpt["best_params"]
    lookback = int(best_params["lookback"])
    hidden_size = int(best_params["hidden_size"])
    num_layers = int(best_params["num_layers"])
    dropout = float(best_params["dropout"])

    # 用checkpoint保存的fillna_means确保一致（若存在）
    if "fillna_means" in ckpt:
        fillna_means = pd.Series(ckpt["fillna_means"], index=FEATURES_CN)
    else:
        fillna_means = df_in[FEATURES_CN].mean()

    # 如果checkpoint里保存了手动切分配置，提示是否一致
    if "manual_split_target_years" in ckpt:
        ckpt_split = ckpt["manual_split_target_years"]
        ckpt_train = tuple(ckpt_split.get("train", TRAIN_TARGET_YEARS))
        ckpt_val   = tuple(ckpt_split.get("val", VAL_TARGET_YEARS))
        ckpt_test  = tuple(ckpt_split.get("test", TEST_TARGET_YEARS))
        if (ckpt_train != TRAIN_TARGET_YEARS) or (ckpt_val != VAL_TARGET_YEARS) or (ckpt_test != TEST_TARGET_YEARS):
            print("[WARN] 当前脚本的手动年份切分与checkpoint记录不一致！")
            print(f"       CKPT train/val/test = {ckpt_train}, {ckpt_val}, {ckpt_test}")
            print(f"       Script train/val/test= {TRAIN_TARGET_YEARS}, {VAL_TARGET_YEARS}, {TEST_TARGET_YEARS}")
            print("       请确保两者一致，否则SSA-LSTM评估与原训练设置不完全对应。")

    df_use = df_in.copy()
    df_use[FEATURES_CN] = df_use[FEATURES_CN].fillna(fillna_means)

    # 手动切分（同当前脚本配置）
    Xtr_raw, ytr, Xva_raw, yva, Xte_raw, yte, mtr, mva, mte, overlap_info = build_manual_split(df_use, lookback)
    print_split_check_once(lookback, mtr, mva, mte, overlap_info)

    n_feat = Xtr_raw.shape[-1]

    # 用checkpoint保存的scaler参数重建标准化（与训练时一致）
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(ckpt["scaler_mean"], dtype=float)
    scaler.scale_ = np.asarray(ckpt["scaler_scale"], dtype=float)
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = n_feat

    Xtr = scaler.transform(Xtr_raw.reshape(-1, n_feat)).reshape(Xtr_raw.shape)
    Xte = scaler.transform(Xte_raw.reshape(-1, n_feat)).reshape(Xte_raw.shape)
    Xva = scaler.transform(Xva_raw.reshape(-1, n_feat)).reshape(Xva_raw.shape)

    # load model weights
    model = LSTMRegressor(
        input_size=n_feat,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(torch.device("cpu"))  # 推理建议CPU更稳
    model.eval()

    yhat_tr = predict_torch(model, Xtr, device=torch.device("cpu"))
    yhat_te = predict_torch(model, Xte, device=torch.device("cpu"))
    yhat_va = predict_torch(model, Xva, device=torch.device("cpu"))

    rmse_tr, mae_tr, r2_tr = calc_metrics(ytr, yhat_tr)
    rmse_te, mae_te, r2_te = calc_metrics(yte, yhat_te)
    rmse_va, mae_va, r2_va = calc_metrics(yva, yhat_va)

    metrics = pd.DataFrame([
        {"Model": "SSA-LSTM", "Split": "Train",      "RMSE": rmse_tr, "MAE": mae_tr, "R2": r2_tr},
        {"Model": "SSA-LSTM", "Split": "Test",       "RMSE": rmse_te, "MAE": mae_te, "R2": r2_te},
        {"Model": "SSA-LSTM", "Split": "Validation", "RMSE": rmse_va, "MAE": mae_va, "R2": r2_va},
    ])

    pack = {
        "lookback": lookback,
        "X_test_true": yte,
        "X_test_pred": yhat_te,
        "best_params": best_params,
        "meta_train": mtr,
        "meta_val": mva,
        "meta_test": mte,
        "overlap_info": overlap_info
    }
    return metrics, pack, fillna_means


# ======================
# 7) Baseline LSTM：按默认参数训练（同一手动切分）
# ======================
def train_and_eval_baseline_lstm(df_in: pd.DataFrame, params: dict, fillna_means: pd.Series):
    set_all_seeds(SEED)

    lookback = int(params["lookback"])
    df_use = df_in.copy()
    df_use[FEATURES_CN] = df_use[FEATURES_CN].fillna(fillna_means)

    Xtr_raw, ytr, Xva_raw, yva, Xte_raw, yte, mtr, mva, mte, overlap_info = build_manual_split(df_use, lookback)

    # 如果和SSA的lookback不一样，这里也打印一次切分检查
    print_split_check_once(lookback, mtr, mva, mte, overlap_info)

    n_feat = Xtr_raw.shape[-1]

    # 标准化（只用Train拟合）
    scaler, Xtr, Xte, Xva = fit_scaler_on_train_and_transform(Xtr_raw, Xte_raw, Xva_raw)

    model = LSTMRegressor(
        input_size=n_feat,
        hidden_size=int(params["hidden_size"]),
        num_layers=int(params["num_layers"]),
        dropout=float(params["dropout"])
    ).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=float(params["lr"]))
    crit = nn.MSELoss()

    # dataloader（shuffle使用固定generator确保更稳）
    g = torch.Generator()
    g.manual_seed(SEED)

    train_ds = TensorDataset(torch.tensor(Xtr, dtype=torch.float32),
                             torch.tensor(ytr, dtype=torch.float32).view(-1, 1))
    val_ds = TensorDataset(torch.tensor(Xva, dtype=torch.float32),
                           torch.tensor(yva, dtype=torch.float32).view(-1, 1))

    train_loader = DataLoader(train_ds, batch_size=int(params["batch_size"]),
                              shuffle=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=int(params["batch_size"]), shuffle=False)

    best_val = np.inf
    pat = 0
    best_state = None

    for epoch in range(1, int(params["epochs"]) + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb)
            loss = crit(pred, yb)
            loss.backward()
            opt.step()

        # 用外部Val做 early stopping
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                pred = model(xb).detach().cpu().numpy().reshape(-1)
                preds.append(pred)
                trues.append(yb.numpy().reshape(-1))
        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        rmse_val = float(np.sqrt(mean_squared_error(trues, preds)))

        if rmse_val < best_val - 1e-6:
            best_val = rmse_val
            pat = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            pat += 1
            if pat >= int(params["patience"]):
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # 预测
    yhat_tr = predict_torch(model, Xtr, device=DEVICE)
    yhat_te = predict_torch(model, Xte, device=DEVICE)
    yhat_va = predict_torch(model, Xva, device=DEVICE)

    rmse_tr, mae_tr, r2_tr = calc_metrics(ytr, yhat_tr)
    rmse_te, mae_te, r2_te = calc_metrics(yte, yhat_te)
    rmse_va, mae_va, r2_va = calc_metrics(yva, yhat_va)

    metrics = pd.DataFrame([
        {"Model": "LSTM", "Split": "Train",      "RMSE": rmse_tr, "MAE": mae_tr, "R2": r2_tr},
        {"Model": "LSTM", "Split": "Test",       "RMSE": rmse_te, "MAE": mae_te, "R2": r2_te},
        {"Model": "LSTM", "Split": "Validation", "RMSE": rmse_va, "MAE": mae_va, "R2": r2_va},
    ])

    pack = {
        "lookback": lookback,
        "X_test_true": yte,
        "X_test_pred": yhat_te,
        "meta_train": mtr,
        "meta_val": mva,
        "meta_test": mte,
        "overlap_info": overlap_info
    }
    return metrics, pack


# ======================
# 8) 主流程：SSA-LSTM(加载) + LSTM(训练) + 保存Excel
# ======================
ssa_metrics, ssa_pack, fillna_means = eval_ssa_lstm_from_checkpoint(data, SSA_CKPT_PATH)
print("\n[SSA-LSTM] Done from checkpoint:")
print(ssa_metrics)

lstm_metrics, lstm_pack = train_and_eval_baseline_lstm(data, LSTM_BASELINE_PARAMS, fillna_means)
print("\n[LSTM] Done (trained baseline):")
print(lstm_metrics)

metrics_all = pd.concat([ssa_metrics, lstm_metrics], ignore_index=True)

# 额外导出切分配置与overlap信息（方便审稿回复）
split_info_df = pd.DataFrame([{
    "TRAIN_TARGET_YEARS": str(TRAIN_TARGET_YEARS),
    "VAL_TARGET_YEARS": str(VAL_TARGET_YEARS),
    "TEST_TARGET_YEARS": str(TEST_TARGET_YEARS),
    "CHECK_INPUT_YEAR_OVERLAP": CHECK_INPUT_YEAR_OVERLAP,
    "STRICT_NO_INPUT_YEAR_OVERLAP": STRICT_NO_INPUT_YEAR_OVERLAP,
    "SSA_LOOKBACK": ssa_pack["lookback"],
    "LSTM_LOOKBACK": lstm_pack["lookback"],
    "SSA_overlap_train_val": str(ssa_pack["overlap_info"].get("ov_train_val", [])),
    "SSA_overlap_train_test": str(ssa_pack["overlap_info"].get("ov_train_test", [])),
    "SSA_overlap_val_test": str(ssa_pack["overlap_info"].get("ov_val_test", [])),
    "LSTM_overlap_train_val": str(lstm_pack["overlap_info"].get("ov_train_val", [])),
    "LSTM_overlap_train_test": str(lstm_pack["overlap_info"].get("ov_train_test", [])),
    "LSTM_overlap_val_test": str(lstm_pack["overlap_info"].get("ov_val_test", [])),
}])

with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
    metrics_all.to_excel(writer, sheet_name="Metrics", index=False)
    split_info_df.to_excel(writer, sheet_name="Split_Config", index=False)

print(f"\n[SAVE] Metrics Excel: {OUT_XLSX}")


# ======================
# 9) 绘图：只画 Test 集的 SSA-LSTM 和 LSTM（1行2列）
# ======================
os.makedirs(SAVE_DIR, exist_ok=True)

# 构造绘图数据（只用Test）
df_plot = pd.concat([
    pd.DataFrame({"Model": "SSA-LSTM", "True_Value": ssa_pack["X_test_true"], "Predicted_Value": ssa_pack["X_test_pred"]}),
    pd.DataFrame({"Model": "LSTM",     "True_Value": lstm_pack["X_test_true"], "Predicted_Value": lstm_pack["X_test_pred"]}),
], ignore_index=True)

# 只在图里标注 Test 指标
def get_test_metrics(metrics_df, model_name):
    row = metrics_df[(metrics_df["Model"] == model_name) & (metrics_df["Split"] == "Test")]
    if row.empty:
        return {"R2": np.nan, "MAE": np.nan, "RMSE": np.nan}
    r = row.iloc[0]
    return {"R2": float(r["R2"]), "MAE": float(r["MAE"]), "RMSE": float(r["RMSE"])}

metrics_dict = {
    "SSA-LSTM": get_test_metrics(metrics_all, "SSA-LSTM"),
    "LSTM": get_test_metrics(metrics_all, "LSTM"),
}

# 全局风格
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix',
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'font.size': 16,
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'axes.linewidth': 2.0,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.pad': 5,
    'ytick.major.pad': 5
})

MODEL_PAIR = ["SSA-LSTM", "LSTM"]
panel_labels = list("ab")

fig = plt.figure(figsize=(14, 6.5))
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.06], wspace=0.25)

last_sc = None

for i, model_name in enumerate(MODEL_PAIR):
    ax = fig.add_subplot(gs[0, i])
    panel_char = panel_labels[i]

    sub_df = df_plot[df_plot["Model"] == model_name].dropna()
    if sub_df.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, f"{model_name}\n(no data)", ha="center", va="center", fontweight="bold")
        continue

    x = sub_df["True_Value"].values
    y = sub_df["Predicted_Value"].values

    data_min = min(x.min(), y.min())
    data_max = max(x.max(), y.max())
    pad = (data_max - data_min) * 0.05 if data_max > data_min else 1.0
    lim_min = data_min - pad
    lim_max = data_max + pad

    # 密度
    try:
        xy = np.vstack([x, y])
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        x_s, y_s, z_s = x[idx], y[idx], z[idx]
    except Exception:
        x_s, y_s, z_s = x, y, np.ones_like(x)

    ax.grid(True, linestyle='--', alpha=0.4, zorder=0)
    ax.plot([lim_min, lim_max], [lim_min, lim_max],
            ls='--', c='#d62728', lw=2.5, zorder=2)

    sns.regplot(x=x, y=y, ax=ax, scatter=False, ci=95,
                line_kws={'color': '#1f77b4', 'linewidth': 3.5, 'zorder': 5},
                truncate=False)

    for coll in ax.collections:
        if isinstance(coll, mcoll.PolyCollection):
            coll.set_alpha(0.3)
            coll.set_facecolor('#6baed6')
            coll.set_zorder(1)

    sc = ax.scatter(x_s, y_s, c=z_s, cmap="viridis", s=60,
                    edgecolor='white', linewidth=0.5, alpha=0.9, zorder=3)
    last_sc = sc

    ax.set_title(f"({panel_char}) {model_name}", fontsize=20, fontweight='bold', pad=12)

    m = metrics_dict.get(model_name, {"R2": np.nan, "MAE": np.nan, "RMSE": np.nan})
    text_str = (f"$\\mathbf{{R^2 = {m['R2']:.3f}}}$\n"
                f"$\\mathbf{{MAE = {m['MAE']:.3f}}}$\n"
                f"$\\mathbf{{RMSE = {m['RMSE']:.3f}}}$")
    ax.text(0.04, 0.96, text_str, transform=ax.transAxes,
            fontsize=16, fontweight='bold', va='top', ha='left')

    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_aspect('equal', adjustable='box')

    ax.set_xlabel("True Values", fontsize=18, fontweight='bold', labelpad=10)
    ax.set_ylabel("Predicted Values", fontsize=18, fontweight='bold', labelpad=10)

# 图例
legend_elements = [
    Line2D([0], [0], color='#1f77b4', lw=3.5, label='Regression Line'),
    Patch(facecolor='#6baed6', alpha=0.3, label='95% Confidence Interval'),
    Line2D([0], [0], color='#d62728', lw=2.5, linestyle='--', label='1:1 Line')
]
fig.legend(handles=legend_elements, loc='lower center',
           bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False,
           prop={'family': 'Times New Roman', 'weight': 'bold', 'size': 16})

# Colorbar
if last_sc is not None:
    cax = fig.add_subplot(gs[0, 2])
    cbar = plt.colorbar(last_sc, cax=cax)
    cbar.ax.set_ylabel('Density', rotation=270, labelpad=20,
                       fontsize=16, fontweight='bold')
    cbar.ax.tick_params(labelsize=14)

plt.subplots_adjust(left=0.07, right=0.95, top=0.90, bottom=0.22)
plt.savefig(OUT_FIG, dpi=600, bbox_inches='tight')
print(f"[SAVE] Figure: {OUT_FIG}")
plt.show()
# -*- coding: utf-8 -*-
import os
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import seaborn as sns
from scipy.stats import gaussian_kde

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

# xgboost / catboost 可能未安装：自动跳过并占位
try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from catboost import CatBoostRegressor
    HAS_CAT = True
except Exception:
    HAS_CAT = False

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


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
# 你当前用的方案（注意：lookback=5 时会有输入年份重叠，只是报告不拦截）
TRAIN_TARGET_YEARS = (2008, 2017)
VAL_TARGET_YEARS   = (2018, 2019)
TEST_TARGET_YEARS  = (2020, 2022)

# 输入年份重叠检查（Comment 3相关）
CHECK_INPUT_YEAR_OVERLAP = True
STRICT_NO_INPUT_YEAR_OVERLAP = False   # True则一旦有重叠直接报错（严格模式）

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 通用固定lookback（用于非SSA-LSTM的11个对比模型）
LOOKBACK = 5

# PyTorch训练通用参数（用于 ANN / RNN）
EPOCHS = 300
PATIENCE = 20
BATCH_SIZE = 64
LR = 1e-3

# ===== R2显示下限（用于表格与绘图显示）=====
R2_REPORT_FLOOR = 0.001

# ===== Transformer 单独参数（模型结构 + 训练参数）=====
TRANSFORMER_MODEL_PARAMS = dict(
    d_model=64,
    nhead=4,
    num_layers=2,
    dim_ff=128,
    dropout=0.10
)

TRANSFORMER_TRAIN_PARAMS = dict(
    epochs=300,        # 你可以单独调大/调小
    patience=30,
    batch_size=32,
    lr=5e-3
)

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
OUT_XLSX = "Model_Comparison_12Models_manual_split.xlsx"
SAVE_DIR = "./figure"
OUT_FIG = f"{SAVE_DIR}/Fig_Model_Density_12Models_manual_split.png"


# ======================
# 1) 复现性设置
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
# 2) 读取数据
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

# 统一缺失值填补（checkpoint分支会优先用checkpoint里的fillna_means）
global_fillna_means = data[FEATURES_CN].mean()
data[FEATURES_CN] = data[FEATURES_CN].fillna(global_fillna_means)


# ======================
# 3) 手动年份切分：序列构造 + 元信息（带缓存）
# ======================
_split_cache = {}

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
        raise ValueError("样本构造失败：可能 lookback 太大或年份不足。")

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
    Xtr, ytr, Xva, yva, Xte, yte, mtr, mva, mte, overlap_info
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
        tr_in = _collect_input_year_set(mtr)
        va_in = _collect_input_year_set(mva)
        te_in = _collect_input_year_set(mte)

        ov_tr_va = sorted(tr_in & va_in)
        ov_tr_te = sorted(tr_in & te_in)
        ov_va_te = sorted(va_in & te_in)

        overlap_info = {
            "train_input_years": tr_in,
            "val_input_years": va_in,
            "test_input_years": te_in,
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
    cache_key = (
        lookback,
        TRAIN_TARGET_YEARS,
        VAL_TARGET_YEARS,
        TEST_TARGET_YEARS,
        CHECK_INPUT_YEAR_OVERLAP,
        STRICT_NO_INPUT_YEAR_OVERLAP
    )
    if cache_key in _split_cache:
        return _split_cache[cache_key]

    X_all, y_all, meta_all = build_sequences_with_meta(df_in, lookback)
    res = split_by_manual_target_years(X_all, y_all, meta_all)
    _split_cache[cache_key] = res
    return res


def print_split_check(lookback: int, mtr: pd.DataFrame, mva: pd.DataFrame, mte: pd.DataFrame, overlap_info: dict, title="Split Check"):
    print(f"\n[{title} | lookback={lookback}]")
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
        print(f"STRICT_NO_INPUT_YEAR_OVERLAP = {STRICT_NO_INPUT_YEAR_OVERLAP}")
    print()


# ======================
# 4) 通用函数
# ======================
def calc_metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    return rmse, mae, r2

def r2_for_report(r2_value, floor=R2_REPORT_FLOOR):
    """
    仅用于结果展示/绘图：若R2<0，则显示为floor（如0.001）
    """
    if pd.isna(r2_value):
        return np.nan
    r2_value = float(r2_value)
    return r2_value if r2_value >= 0 else float(floor)

def fit_scaler_on_train_and_transform(Xtr_raw, Xva_raw, Xte_raw):
    """标准化：只用Train拟合，然后作用于Train/Val/Test"""
    n_feat = Xtr_raw.shape[-1]
    scaler = StandardScaler()
    scaler.fit(Xtr_raw.reshape(-1, n_feat))

    Xtr = scaler.transform(Xtr_raw.reshape(-1, n_feat)).reshape(Xtr_raw.shape)
    Xva = scaler.transform(Xva_raw.reshape(-1, n_feat)).reshape(Xva_raw.shape)
    Xte = scaler.transform(Xte_raw.reshape(-1, n_feat)).reshape(Xte_raw.shape)
    return scaler, Xtr, Xva, Xte


def make_metrics_row(model_name, y_tr, yhat_tr, y_va, yhat_va, y_te, yhat_te, status="OK", note=""):
    rmse_tr, mae_tr, r2_tr_raw = calc_metrics(y_tr, yhat_tr) if yhat_tr is not None else (np.nan, np.nan, np.nan)
    rmse_va, mae_va, r2_va_raw = calc_metrics(y_va, yhat_va) if yhat_va is not None else (np.nan, np.nan, np.nan)
    rmse_te, mae_te, r2_te_raw = calc_metrics(y_te, yhat_te) if yhat_te is not None else (np.nan, np.nan, np.nan)

    # 用于报告/绘图显示（负值改为0.001）
    r2_tr = r2_for_report(r2_tr_raw)
    r2_va = r2_for_report(r2_va_raw)
    r2_te = r2_for_report(r2_te_raw)

    # 如果有负R2，自动在note里标记一下（可选）
    auto_note = note
    neg_flags = []
    if pd.notna(r2_tr_raw) and r2_tr_raw < 0:
        neg_flags.append("Train_R2<0")
    if pd.notna(r2_va_raw) and r2_va_raw < 0:
        neg_flags.append("Val_R2<0")
    if pd.notna(r2_te_raw) and r2_te_raw < 0:
        neg_flags.append("Test_R2<0")
    if neg_flags:
        tail = "; ".join(neg_flags) + f" -> reported as {R2_REPORT_FLOOR}"
        auto_note = (auto_note + " | " + tail).strip(" |")

    return {
        "Model": model_name,
        "Status": status,
        "Note": auto_note,

        "Train_RMSE": rmse_tr, "Val_RMSE": rmse_va, "Test_RMSE": rmse_te,
        "Train_MAE": mae_tr,   "Val_MAE": mae_va,   "Test_MAE": mae_te,

        # 报告值（负数被改成0.001）
        "Train_R2": r2_tr,     "Val_R2": r2_va,     "Test_R2": r2_te,

        # 原始值（可选保留，建议留着）
        "Train_R2_raw": r2_tr_raw, "Val_R2_raw": r2_va_raw, "Test_R2_raw": r2_te_raw
    }


def make_prediction_long_df(model_name, y_true, y_pred):
    return pd.DataFrame({
        "Model": model_name,
        "True_Value": np.asarray(y_true, dtype=float),
        "Predicted_Value": np.asarray(y_pred, dtype=float)
    })


# ======================
# 5) ELM（极限学习机，用户写EML一般指ELM）
# ======================
class ELMRegressor:
    """
    简单ELM：随机隐层 + 岭回归求解输出层
    """
    def __init__(self, n_hidden=500, activation="tanh", ridge_alpha=1e-3, random_state=42):
        self.n_hidden = n_hidden
        self.activation = activation
        self.ridge_alpha = ridge_alpha
        self.random_state = random_state
        self.W = None
        self.b = None
        self.beta = None  # 输出权重

    def _act(self, X):
        if self.activation == "tanh":
            return np.tanh(X)
        if self.activation == "relu":
            return np.maximum(0, X)
        if self.activation == "sigmoid":
            return 1 / (1 + np.exp(-X))
        raise ValueError("activation must be tanh/relu/sigmoid")

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        n_in = X.shape[1]
        self.W = rng.normal(0, 1, size=(n_in, self.n_hidden))
        self.b = rng.normal(0, 1, size=(self.n_hidden,))

        H = self._act(X @ self.W + self.b)
        # 岭回归闭式解：beta = (H'H + aI)^-1 H'y
        I = np.eye(self.n_hidden)
        A = H.T @ H + self.ridge_alpha * I
        B = H.T @ y.reshape(-1, 1)
        self.beta = np.linalg.solve(A, B)
        return self

    def predict(self, X):
        H = self._act(X @ self.W + self.b)
        return (H @ self.beta).reshape(-1)


# ======================
# 6) PyTorch模型：LSTM / ANN / RNN / Transformer
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


class ANN_MLP(nn.Module):
    def __init__(self, input_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):
        return self.net(x)


class RNN_Regressor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1):
        super().__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True, nonlinearity="tanh")
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        return self.fc(last)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1,max_len,d_model)

    def forward(self, x):
        T = x.size(1)
        return x + self.pe[:, :T, :]


class TransformerRegressor(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2, dim_ff=128, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(input_size, d_model)
        self.pe = PositionalEncoding(d_model, max_len=500)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                               dim_feedforward=dim_ff, dropout=dropout,
                                               batch_first=True, activation="relu")
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        h = self.proj(x)
        h = self.pe(h)
        h = self.encoder(h)
        pooled = h.mean(dim=1)
        return self.fc(pooled)


def make_torch_loaders_from_manual_split(X_train, y_train, X_val, y_val, batch_size=64):
    """
    不再从训练集随机切validation；直接使用外部时间切分得到的验证集
    """
    tr_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                          torch.tensor(y_train, dtype=torch.float32).view(-1, 1))
    va_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                          torch.tensor(y_val, dtype=torch.float32).view(-1, 1))

    g = torch.Generator()
    g.manual_seed(SEED)

    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, generator=g)
    va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False)
    return tr_loader, va_loader


def train_torch_model(model, X_train, y_train, X_val, y_val, epochs=200, patience=20, batch_size=64, lr=1e-3):
    model = model.to(DEVICE)
    crit = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    tr_loader, va_loader = make_torch_loaders_from_manual_split(X_train, y_train, X_val, y_val, batch_size=batch_size)

    best_val = np.inf
    pat = 0
    best_state = None

    for _ in range(epochs):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb)
            loss = crit(pred, yb)
            loss.backward()
            opt.step()

        # 用外部Validation计算RMSE做early stopping
        model.eval()
        preds_val, trues_val = [], []
        with torch.no_grad():
            for xb, yb in va_loader:
                xb = xb.to(DEVICE)
                pred = model(xb).detach().cpu().numpy().reshape(-1)
                preds_val.append(pred)
                trues_val.append(yb.numpy().reshape(-1))
        preds_val = np.concatenate(preds_val)
        trues_val = np.concatenate(trues_val)
        val_rmse = float(np.sqrt(mean_squared_error(trues_val, preds_val)))

        if val_rmse < best_val - 1e-6:
            best_val = val_rmse
            pat = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            pat += 1
            if pat >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def predict_torch(model, Xnp, device=DEVICE):
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(Xnp, dtype=torch.float32).to(device)
        return model(xb).detach().cpu().numpy().reshape(-1)


# ======================
# 7) 构建“固定lookback”的共享数据（用于11个非SSA模型）
# ======================
Xtr_raw_fix, ytr_fix, Xva_raw_fix, yva_fix, Xte_raw_fix, yte_fix, mtr_fix, mva_fix, mte_fix, overlap_fix = build_manual_split(data, LOOKBACK)
print_split_check(LOOKBACK, mtr_fix, mva_fix, mte_fix, overlap_fix, title="Manual Split Check (fixed lookback models)")

n_feat = Xtr_raw_fix.shape[-1]
T = Xtr_raw_fix.shape[1]

scaler_fix, Xtr_fix_3d_s, Xva_fix_3d_s, Xte_fix_3d_s = fit_scaler_on_train_and_transform(Xtr_raw_fix, Xva_raw_fix, Xte_raw_fix)

# 传统模型输入（展平）
Xtr_fix_2d = Xtr_fix_3d_s.reshape(Xtr_fix_3d_s.shape[0], -1)
Xva_fix_2d = Xva_fix_3d_s.reshape(Xva_fix_3d_s.shape[0], -1)
Xte_fix_2d = Xte_fix_3d_s.reshape(Xte_fix_3d_s.shape[0], -1)


# ======================
# 8) SSA-LSTM：从checkpoint加载（不训练）
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

    # fillna
    if "fillna_means" in ckpt:
        fillna_means = pd.Series(ckpt["fillna_means"], index=FEATURES_CN)
    else:
        fillna_means = df_in[FEATURES_CN].mean()

    # checkpoint里保存的手动切分信息一致性提示
    if "manual_split_target_years" in ckpt:
        ckpt_split = ckpt["manual_split_target_years"]
        ckpt_train = tuple(ckpt_split.get("train", TRAIN_TARGET_YEARS))
        ckpt_val   = tuple(ckpt_split.get("val", VAL_TARGET_YEARS))
        ckpt_test  = tuple(ckpt_split.get("test", TEST_TARGET_YEARS))
        if (ckpt_train != TRAIN_TARGET_YEARS) or (ckpt_val != VAL_TARGET_YEARS) or (ckpt_test != TEST_TARGET_YEARS):
            print("[WARN] 当前脚本手动年份切分与checkpoint记录不一致！")
            print(f"       CKPT train/val/test = {ckpt_train}, {ckpt_val}, {ckpt_test}")
            print(f"       Script train/val/test= {TRAIN_TARGET_YEARS}, {VAL_TARGET_YEARS}, {TEST_TARGET_YEARS}")
            print("       请尽量保持一致，否则SSA-LSTM评估与原训练设置不完全对应。")

    df_use = df_in.copy()
    df_use[FEATURES_CN] = df_use[FEATURES_CN].fillna(fillna_means)

    # 手动切分（按照checkpoint自己的lookback）
    Xtr_raw, ytr, Xva_raw, yva, Xte_raw, yte, mtr, mva, mte, overlap_info = build_manual_split(df_use, lookback)
    print_split_check(lookback, mtr, mva, mte, overlap_info, title="SSA-LSTM Split Check")

    n_feat_ckpt = Xtr_raw.shape[-1]

    # 用checkpoint保存的scaler参数重建标准化
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(ckpt["scaler_mean"], dtype=float)
    scaler.scale_ = np.asarray(ckpt["scaler_scale"], dtype=float)
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = n_feat_ckpt

    Xtr = scaler.transform(Xtr_raw.reshape(-1, n_feat_ckpt)).reshape(Xtr_raw.shape)
    Xva = scaler.transform(Xva_raw.reshape(-1, n_feat_ckpt)).reshape(Xva_raw.shape)
    Xte = scaler.transform(Xte_raw.reshape(-1, n_feat_ckpt)).reshape(Xte_raw.shape)

    # 加载模型权重
    model = LSTMRegressor(
        input_size=n_feat_ckpt,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(torch.device("cpu"))
    model.eval()

    yhat_tr = predict_torch(model, Xtr, device=torch.device("cpu"))
    yhat_va = predict_torch(model, Xva, device=torch.device("cpu"))
    yhat_te = predict_torch(model, Xte, device=torch.device("cpu"))

    metrics_row = make_metrics_row("SSA-LSTM", ytr, yhat_tr, yva, yhat_va, yte, yhat_te, status="OK")

    pack = {
        "Model": "SSA-LSTM",
        "lookback": lookback,
        "y_train_true": ytr, "y_train_pred": yhat_tr,
        "y_val_true": yva, "y_val_pred": yhat_va,
        "y_test_true": yte, "y_test_pred": yhat_te,
        "meta_train": mtr, "meta_val": mva, "meta_test": mte,
        "overlap_info": overlap_info,
        "best_params": best_params
    }
    return metrics_row, pack


# ======================
# 9) 各模型训练/评估（12模型）
# ======================
results = []
pred_pack_dict = {}   # 用于画图（只需要test）
pred_long_list = []   # 导出Excel long format


def add_model_result(model_name, ytr, yhat_tr, yva, yhat_va, yte, yhat_te, status="OK", note=""):
    row = make_metrics_row(model_name, ytr, yhat_tr, yva, yhat_va, yte, yhat_te, status=status, note=note)
    results.append(row)

    # 保存用于绘图和导出的测试集预测
    if (yte is not None) and (yhat_te is not None):
        pred_pack_dict[model_name] = {
            "y_test_true": np.asarray(yte),
            "y_test_pred": np.asarray(yhat_te)
        }
        pred_long_list.append(make_prediction_long_df(model_name, yte, yhat_te))
    else:
        pred_pack_dict[model_name] = None


# ---- 1) SSA-LSTM（checkpoint）----
try:
    ssa_row, ssa_pack = eval_ssa_lstm_from_checkpoint(data, SSA_CKPT_PATH)
    results.append(ssa_row)
    pred_pack_dict["SSA-LSTM"] = {
        "y_test_true": np.asarray(ssa_pack["y_test_true"]),
        "y_test_pred": np.asarray(ssa_pack["y_test_pred"])
    }
    pred_long_list.append(make_prediction_long_df("SSA-LSTM", ssa_pack["y_test_true"], ssa_pack["y_test_pred"]))
except Exception as e:
    print(f"[WARN] SSA-LSTM 加载/评估失败：{e}")
    add_model_result("SSA-LSTM", None, None, None, None, None, None, status="FAILED", note=str(e))


# ---- 2) LSTM baseline（同固定lookback手动切分）----
try:
    set_all_seeds(SEED)
    p = LSTM_BASELINE_PARAMS

    model_lstm = LSTMRegressor(
        input_size=n_feat,
        hidden_size=int(p["hidden_size"]),
        num_layers=int(p["num_layers"]),
        dropout=float(p["dropout"])
    ).to(DEVICE)

    opt = torch.optim.Adam(model_lstm.parameters(), lr=float(p["lr"]))
    crit = nn.MSELoss()

    tr_loader, va_loader = make_torch_loaders_from_manual_split(
        Xtr_fix_3d_s, ytr_fix, Xva_fix_3d_s, yva_fix, batch_size=int(p["batch_size"])
    )

    best_val = np.inf
    pat = 0
    best_state = None

    for _ in range(int(p["epochs"])):
        model_lstm.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred = model_lstm(xb)
            loss = crit(pred, yb)
            loss.backward()
            opt.step()

        model_lstm.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in va_loader:
                xb = xb.to(DEVICE)
                pred = model_lstm(xb).detach().cpu().numpy().reshape(-1)
                preds.append(pred)
                trues.append(yb.numpy().reshape(-1))
        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        rmse_val = float(np.sqrt(mean_squared_error(trues, preds)))

        if rmse_val < best_val - 1e-6:
            best_val = rmse_val
            pat = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model_lstm.state_dict().items()}
        else:
            pat += 1
            if pat >= int(p["patience"]):
                break

    if best_state is not None:
        model_lstm.load_state_dict(best_state)

    yhat_tr = predict_torch(model_lstm, Xtr_fix_3d_s)
    yhat_va = predict_torch(model_lstm, Xva_fix_3d_s)
    yhat_te = predict_torch(model_lstm, Xte_fix_3d_s)

    add_model_result("LSTM", ytr_fix, yhat_tr, yva_fix, yhat_va, yte_fix, yhat_te)
except Exception as e:
    print(f"[WARN] LSTM baseline 训练失败：{e}")
    add_model_result("LSTM", None, None, None, None, None, None, status="FAILED", note=str(e))


# ---- 3) GBDT ----
try:
    gbdt = GradientBoostingRegressor(random_state=SEED)
    gbdt.fit(Xtr_fix_2d, ytr_fix)
    add_model_result("GBDT",
                     ytr_fix, gbdt.predict(Xtr_fix_2d),
                     yva_fix, gbdt.predict(Xva_fix_2d),
                     yte_fix, gbdt.predict(Xte_fix_2d))
except Exception as e:
    print(f"[WARN] GBDT 失败：{e}")
    add_model_result("GBDT", None, None, None, None, None, None, status="FAILED", note=str(e))

# ---- 4) RF ----
try:
    rf = RandomForestRegressor(n_estimators=500, random_state=SEED, n_jobs=-1)
    rf.fit(Xtr_fix_2d, ytr_fix)
    add_model_result("RF",
                     ytr_fix, rf.predict(Xtr_fix_2d),
                     yva_fix, rf.predict(Xva_fix_2d),
                     yte_fix, rf.predict(Xte_fix_2d))
except Exception as e:
    print(f"[WARN] RF 失败：{e}")
    add_model_result("RF", None, None, None, None, None, None, status="FAILED", note=str(e))

# ---- 5) SVR ----
try:
    svr = SVR(C=10.0, epsilon=0.01, kernel="rbf")
    svr.fit(Xtr_fix_2d, ytr_fix)
    add_model_result("SVR",
                     ytr_fix, svr.predict(Xtr_fix_2d),
                     yva_fix, svr.predict(Xva_fix_2d),
                     yte_fix, svr.predict(Xte_fix_2d))
except Exception as e:
    print(f"[WARN] SVR 失败：{e}")
    add_model_result("SVR", None, None, None, None, None, None, status="FAILED", note=str(e))

# ---- 6) BP(MLP) ----
try:
    bp = MLPRegressor(hidden_layer_sizes=(128, 128),
                      activation="relu", solver="adam",
                      alpha=1e-4, learning_rate_init=1e-3,
                      max_iter=2000, random_state=SEED)
    bp.fit(Xtr_fix_2d, ytr_fix)
    add_model_result("BP(MLP)",
                     ytr_fix, bp.predict(Xtr_fix_2d),
                     yva_fix, bp.predict(Xva_fix_2d),
                     yte_fix, bp.predict(Xte_fix_2d))
except Exception as e:
    print(f"[WARN] BP(MLP) 失败：{e}")
    add_model_result("BP(MLP)", None, None, None, None, None, None, status="FAILED", note=str(e))

# ---- 7) XGBoost ----
if HAS_XGB:
    try:
        xgbr = xgb.XGBRegressor(
            n_estimators=800, max_depth=6, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, random_state=SEED, n_jobs=-1
        )
        xgbr.fit(Xtr_fix_2d, ytr_fix)
        add_model_result("XGBoost",
                         ytr_fix, xgbr.predict(Xtr_fix_2d),
                         yva_fix, xgbr.predict(Xva_fix_2d),
                         yte_fix, xgbr.predict(Xte_fix_2d))
    except Exception as e:
        print(f"[WARN] XGBoost 失败：{e}")
        add_model_result("XGBoost", None, None, None, None, None, None, status="FAILED", note=str(e))
else:
    print("[WARN] xgboost 未安装，XGBoost占位。pip install xgboost")
    add_model_result("XGBoost", None, None, None, None, None, None, status="SKIPPED", note="xgboost not installed")

# ---- 8) CatBoost ----
if HAS_CAT:
    try:
        cbr = CatBoostRegressor(
            iterations=2000, depth=6, learning_rate=0.03,
            loss_function="RMSE", random_seed=SEED,
            verbose=False
        )
        cbr.fit(Xtr_fix_2d, ytr_fix)
        add_model_result("CatBoost",
                         ytr_fix, cbr.predict(Xtr_fix_2d),
                         yva_fix, cbr.predict(Xva_fix_2d),
                         yte_fix, cbr.predict(Xte_fix_2d))
    except Exception as e:
        print(f"[WARN] CatBoost 失败：{e}")
        add_model_result("CatBoost", None, None, None, None, None, None, status="FAILED", note=str(e))
else:
    print("[WARN] catboost 未安装，CatBoost占位。pip install catboost")
    add_model_result("CatBoost", None, None, None, None, None, None, status="SKIPPED", note="catboost not installed")

# ---- 9) ELM(EML) ----
try:
    elm = ELMRegressor(n_hidden=800, activation="tanh", ridge_alpha=1e-3, random_state=SEED)
    elm.fit(Xtr_fix_2d, ytr_fix)
    add_model_result("ELM(EML)",
                     ytr_fix, elm.predict(Xtr_fix_2d),
                     yva_fix, elm.predict(Xva_fix_2d),
                     yte_fix, elm.predict(Xte_fix_2d))
except Exception as e:
    print(f"[WARN] ELM(EML) 失败：{e}")
    add_model_result("ELM(EML)", None, None, None, None, None, None, status="FAILED", note=str(e))

# ---- 10) ANN(PyTorch) ----
try:
    set_all_seeds(SEED)
    ann = ANN_MLP(input_dim=Xtr_fix_2d.shape[1], hidden=256)
    ann = train_torch_model(
        ann, Xtr_fix_2d, ytr_fix, Xva_fix_2d, yva_fix,
        epochs=EPOCHS, patience=PATIENCE, batch_size=BATCH_SIZE, lr=LR
    )
    add_model_result("ANN(PyTorch)",
                     ytr_fix, predict_torch(ann, Xtr_fix_2d),
                     yva_fix, predict_torch(ann, Xva_fix_2d),
                     yte_fix, predict_torch(ann, Xte_fix_2d))
except Exception as e:
    print(f"[WARN] ANN(PyTorch) 失败：{e}")
    add_model_result("ANN(PyTorch)", None, None, None, None, None, None, status="FAILED", note=str(e))

# ---- 11) RNN ----
try:
    set_all_seeds(SEED)
    rnn = RNN_Regressor(input_size=n_feat, hidden_size=64, num_layers=1)
    rnn = train_torch_model(
        rnn, Xtr_fix_3d_s, ytr_fix, Xva_fix_3d_s, yva_fix,
        epochs=EPOCHS, patience=PATIENCE, batch_size=BATCH_SIZE, lr=LR
    )
    add_model_result("RNN",
                     ytr_fix, predict_torch(rnn, Xtr_fix_3d_s),
                     yva_fix, predict_torch(rnn, Xva_fix_3d_s),
                     yte_fix, predict_torch(rnn, Xte_fix_3d_s))
except Exception as e:
    print(f"[WARN] RNN 失败：{e}")
    add_model_result("RNN", None, None, None, None, None, None, status="FAILED", note=str(e))

# ---- 12) Transformer ----
try:
    set_all_seeds(SEED)
    trf = TransformerRegressor(
        input_size=n_feat,
        d_model=TRANSFORMER_MODEL_PARAMS["d_model"],
        nhead=TRANSFORMER_MODEL_PARAMS["nhead"],
        num_layers=TRANSFORMER_MODEL_PARAMS["num_layers"],
        dim_ff=TRANSFORMER_MODEL_PARAMS["dim_ff"],
        dropout=TRANSFORMER_MODEL_PARAMS["dropout"]
    )

    trf = train_torch_model(
        trf,
        Xtr_fix_3d_s, ytr_fix,
        Xva_fix_3d_s, yva_fix,
        epochs=TRANSFORMER_TRAIN_PARAMS["epochs"],
        patience=TRANSFORMER_TRAIN_PARAMS["patience"],
        batch_size=TRANSFORMER_TRAIN_PARAMS["batch_size"],
        lr=TRANSFORMER_TRAIN_PARAMS["lr"]
    )
    add_model_result("Transformer",
                     ytr_fix, predict_torch(trf, Xtr_fix_3d_s),
                     yva_fix, predict_torch(trf, Xva_fix_3d_s),
                     yte_fix, predict_torch(trf, Xte_fix_3d_s))
except Exception as e:
    print(f"[WARN] Transformer 失败：{e}")
    add_model_result("Transformer", None, None, None, None, None, None, status="FAILED", note=str(e))


# ======================
# 10) 汇总输出 + Excel
# ======================
res_df = pd.DataFrame(results)

# 固定顺序（确保图和表一致）
MODEL_ORDER = [
    "SSA-LSTM", "LSTM", "GBDT", "RF",
    "SVR", "BP(MLP)", "XGBoost", "CatBoost",
    "ELM(EML)", "ANN(PyTorch)", "RNN", "Transformer"
]
res_df["Model"] = pd.Categorical(res_df["Model"], categories=MODEL_ORDER, ordered=True)
res_df = res_df.sort_values("Model").reset_index(drop=True)

print("\n==================== Model Comparison (12 Models; Train/Val/Test) ====================")
print(res_df)

# 切分配置/重叠信息
split_info_df = pd.DataFrame([{
    "LOOKBACK_FIXED_FOR_11_MODELS": LOOKBACK,
    "TRAIN_TARGET_YEARS": str(TRAIN_TARGET_YEARS),
    "VAL_TARGET_YEARS": str(VAL_TARGET_YEARS),
    "TEST_TARGET_YEARS": str(TEST_TARGET_YEARS),
    "CHECK_INPUT_YEAR_OVERLAP": CHECK_INPUT_YEAR_OVERLAP,
    "STRICT_NO_INPUT_YEAR_OVERLAP": STRICT_NO_INPUT_YEAR_OVERLAP,
    "fixed_models_train_n": len(ytr_fix),
    "fixed_models_val_n": len(yva_fix),
    "fixed_models_test_n": len(yte_fix),
    "fixed_overlap_train_val": str(overlap_fix.get("ov_train_val", [])),
    "fixed_overlap_train_test": str(overlap_fix.get("ov_train_test", [])),
    "fixed_overlap_val_test": str(overlap_fix.get("ov_val_test", [])),
    "SSA_CKPT_PATH": SSA_CKPT_PATH,
    "DATA_SHEET_USED": sheet_used
}])

# 测试集预测长表（画图/复核）
if len(pred_long_list) > 0:
    pred_test_long_df = pd.concat(pred_long_list, ignore_index=True)
else:
    pred_test_long_df = pd.DataFrame(columns=["Model", "True_Value", "Predicted_Value"])

os.makedirs(SAVE_DIR, exist_ok=True)
with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
    res_df.to_excel(writer, sheet_name="Metrics_12Models", index=False)
    split_info_df.to_excel(writer, sheet_name="Split_Config", index=False)
    pred_test_long_df.to_excel(writer, sheet_name="Test_Predictions_Long", index=False)

print(f"\n[SAVE] Metrics Excel: {OUT_XLSX}")


# ======================
# 11) 绘图：12模型 Test集 True vs Pred（3x4子图）
# ======================
def get_test_metrics_from_res_df(res_df_in, model_name):
    row = res_df_in[res_df_in["Model"].astype(str) == model_name]
    if row.empty:
        return {"R2": np.nan, "MAE": np.nan, "RMSE": np.nan, "Status": "NA"}

    r = row.iloc[0]

    # 优先使用报告值；若没有则对raw再做一次显示修正
    if "Test_R2" in r.index and pd.notna(r["Test_R2"]):
        r2_disp = float(r["Test_R2"])
    else:
        r2_raw = r.get("Test_R2_raw", np.nan)
        r2_disp = r2_for_report(r2_raw)

    return {
        "R2": r2_disp,
        "MAE": r.get("Test_MAE", np.nan),
        "RMSE": r.get("Test_RMSE", np.nan),
        "Status": r.get("Status", "NA"),
    }


def plot_12_model_density(pred_dict, metrics_df, model_order, out_fig):
    # 全局风格
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'mathtext.fontset': 'stix',
        'font.weight': 'bold',
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'font.size': 15,
        'axes.titlesize': 15,
        'axes.labelsize': 15,
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
        'axes.linewidth': 1.2,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
    })

    fig, axes = plt.subplots(3, 4, figsize=(14, 9))
    axes = axes.flatten()

    panel_labels = list("abcdefghijkl")
    for i, model_name in enumerate(model_order):
        ax = axes[i]
        panel_char = panel_labels[i]
        m = get_test_metrics_from_res_df(metrics_df, model_name)
        pack = pred_dict.get(model_name, None)

        # 没有预测数据（比如未安装/失败） -> 占位
        if pack is None:
            ax.axis("off")
            txt = f"({panel_char}) {model_name}\n{m.get('Status', 'N/A')}"
            if pd.notna(m.get("RMSE", np.nan)):
                txt += f"\nRMSE={m['RMSE']:.3f}"
            ax.text(0.5, 0.5, txt, ha="center", va="center", fontweight="bold")
            continue

        x = np.asarray(pack["y_test_true"], dtype=float)
        y = np.asarray(pack["y_test_pred"], dtype=float)

        if len(x) == 0 or len(y) == 0:
            ax.axis("off")
            ax.text(0.5, 0.5, f"({panel_char}) {model_name}\n(no test data)",
                    ha="center", va="center", fontweight="bold")
            continue

        # 去除nan
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        if len(x) == 0:
            ax.axis("off")
            ax.text(0.5, 0.5, f"({panel_char}) {model_name}\n(no valid data)",
                    ha="center", va="center", fontweight="bold")
            continue

        data_min = min(x.min(), y.min())
        data_max = max(x.max(), y.max())
        pad = (data_max - data_min) * 0.08 if data_max > data_min else 1.0
        lim_min = data_min - pad
        lim_max = data_max + pad

        # 密度着色
        try:
            xy = np.vstack([x, y])
            z = gaussian_kde(xy)(xy)
            idx = z.argsort()
            x_s, y_s, z_s = x[idx], y[idx], z[idx]
        except Exception:
            x_s, y_s, z_s = x, y, np.ones_like(x)

        ax.grid(True, linestyle='--', alpha=0.25, zorder=0)
        ax.plot([lim_min, lim_max], [lim_min, lim_max],
                ls='--', c='#d62728', lw=1.8, zorder=2)

        # 回归线 + 置信区间
        try:
            sns.regplot(x=x, y=y, ax=ax, scatter=False, ci=95,
                        line_kws={'color': '#1f77b4', 'linewidth': 2.2, 'zorder': 5},
                        truncate=False)
            for coll in ax.collections:
                if isinstance(coll, mcoll.PolyCollection):
                    coll.set_alpha(0.20)
                    coll.set_facecolor('#6baed6')
                    coll.set_zorder(1)
        except Exception:
            pass

        ax.scatter(x_s, y_s, c=z_s, cmap="viridis", s=22,
                   edgecolor='white', linewidth=0.3, alpha=0.85, zorder=3)

        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)
        ax.set_aspect('equal', adjustable='box')

        ax.set_title(f"({panel_char}) {model_name}", pad=6)

        text_str = (
            f"$R^2={m['R2']:.3f}$\n"
            f"$MAE={m['MAE']:.3f}$\n"
            f"$RMSE={m['RMSE']:.3f}$"
        )
        ax.text(0.03, 0.97, text_str, transform=ax.transAxes,
                fontsize=12, va='top', ha='left', fontweight='bold')

        # 只在左列和底行显示坐标轴标签，减少拥挤
        if i % 4 == 0:
            ax.set_ylabel("Predicted")
        else:
            ax.set_ylabel("")
        if i // 4 == 2:
            ax.set_xlabel("True")
        else:
            ax.set_xlabel("")

    plt.tight_layout(rect=[0, 0, 1, 0.965])
    plt.savefig(out_fig, dpi=450, bbox_inches='tight')
    print(f"[SAVE] Figure: {out_fig}")
    plt.show()


plot_12_model_density(pred_pack_dict, res_df, MODEL_ORDER, OUT_FIG)
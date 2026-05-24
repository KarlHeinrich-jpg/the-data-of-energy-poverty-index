# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")

import random
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

# xgboost / catboost 可能未安装：自动跳过并提示
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
# 0) 基本配置
# ======================
DATA_PATH = r"EPI_entropy_output.xlsx"   # <<< 改成你的文件路径
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

LOOKBACK = 5

# ===== 手动设置 Train / Validation / Test（按目标年份，闭区间）=====
TRAIN_TARGET_YEARS = (2008, 2017)   # 例：训练集目标年份
VAL_TARGET_YEARS   = (2018, 2019)   # 例：验证集目标年份（用于早停/调参）
TEST_TARGET_YEARS  = (2020, 2022)   # 例：测试集目标年份（最终评估）

# 输入年份重叠检查（Comment 3 相关）
CHECK_INPUT_YEAR_OVERLAP = True
STRICT_NO_INPUT_YEAR_OVERLAP = False  # True则发现输入年份重叠直接报错

# PyTorch训练通用参数（用于 ANN/RNN/Transformer）
EPOCHS = 300
PATIENCE = 20
BATCH_SIZE = 64
LR = 1e-3
SEED = 42

np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================
# 1) 读取数据（优先 EPI_score sheet）
# ======================
def read_main_sheet(path: str) -> pd.DataFrame:
    xls = pd.ExcelFile(path)
    if "EPI_score" in xls.sheet_names:
        return pd.read_excel(path, sheet_name="EPI_score")
    return pd.read_excel(path, sheet_name=xls.sheet_names[0])

df = read_main_sheet(DATA_PATH)

need_cols = [AREA_COL, YEAR_COL, TARGET_COL] + FEATURES_CN
missing = [c for c in need_cols if c not in df.columns]
if missing:
    raise ValueError(f"数据缺少以下列：{missing}\n请检查表头是否一致（括号全角/半角、空格等）。")

data = df[need_cols].copy()

# 数值化
data[YEAR_COL] = pd.to_numeric(data[YEAR_COL], errors="coerce")
data[TARGET_COL] = pd.to_numeric(data[TARGET_COL], errors="coerce")
for c in FEATURES_CN:
    data[c] = pd.to_numeric(data[c], errors="coerce")

data = data.dropna(subset=[YEAR_COL, TARGET_COL])
data[FEATURES_CN] = data[FEATURES_CN].fillna(data[FEATURES_CN].mean())


# ======================
# 2) 手动年份切分：构造序列样本 + 元信息
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
            X_all.append(Xg[t-lookback:t, :])
            y_all.append(yg[t])

            meta_rows.append({
                "AREA": area,
                "target_year": int(years[t]),
                "input_start_year": int(years[t-lookback]),
                "input_end_year": int(years[t-1]),
            })

    if len(X_all) == 0:
        raise ValueError("构造序列样本失败：可能 LOOKBACK 太大或年份不足。")

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
    返回：
    X_train_3d, y_train, X_val_3d, y_val, X_test_3d, y_test, meta_train, meta_val, meta_test, overlap_info
    """
    train_mask = _mask_year_range(meta_all["target_year"], TRAIN_TARGET_YEARS)
    val_mask   = _mask_year_range(meta_all["target_year"], VAL_TARGET_YEARS)
    test_mask  = _mask_year_range(meta_all["target_year"], TEST_TARGET_YEARS)

    # 检查目标年份范围重叠
    if (train_mask & val_mask).any() or (train_mask & test_mask).any() or (val_mask & test_mask).any():
        raise ValueError("train/val/test 的目标年份范围有重叠，请检查年份设置。")

    X_train_3d = X_all[train_mask.values]
    y_train = y_all[train_mask.values]
    meta_train = meta_all.loc[train_mask].reset_index(drop=True)

    X_val_3d = X_all[val_mask.values]
    y_val = y_all[val_mask.values]
    meta_val = meta_all.loc[val_mask].reset_index(drop=True)

    X_test_3d = X_all[test_mask.values]
    y_test = y_all[test_mask.values]
    meta_test = meta_all.loc[test_mask].reset_index(drop=True)

    if len(X_train_3d) == 0 or len(X_val_3d) == 0 or len(X_test_3d) == 0:
        raise ValueError(
            f"切分后某个数据集为空：train={len(X_train_3d)}, val={len(X_val_3d)}, test={len(X_test_3d)}。\n"
            "请检查 LOOKBACK 与年份范围设置。"
        )

    overlap_info = {}
    if CHECK_INPUT_YEAR_OVERLAP:
        tr_in = _collect_input_year_set(meta_train)
        va_in = _collect_input_year_set(meta_val)
        te_in = _collect_input_year_set(meta_test)

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

    return X_train_3d, y_train, X_val_3d, y_val, X_test_3d, y_test, meta_train, meta_val, meta_test, overlap_info


X_all_3d, y_all, meta_all = build_sequences_with_meta(data, LOOKBACK)
X_train_3d, y_train, X_val_3d, y_val, X_test_3d, y_test, meta_train, meta_val, meta_test, overlap_info = split_by_manual_target_years(
    X_all_3d, y_all, meta_all
)

# 打印切分检查（只打印一次）
print("\n==================== Manual Split Check ====================")
print(f"Train target years: {TRAIN_TARGET_YEARS}, n={len(meta_train)}")
print(f"Val   target years: {VAL_TARGET_YEARS}, n={len(meta_val)}")
print(f"Test  target years: {TEST_TARGET_YEARS}, n={len(meta_test)}")
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
print("===========================================================\n")

n_feat = X_train_3d.shape[-1]
T = X_train_3d.shape[1]


# ======================
# 3) 标准化（对所有模型统一：按训练集拟合）
# ======================
scaler = StandardScaler()
X_train_2d_flat = X_train_3d.reshape(-1, n_feat)
X_val_2d_flat   = X_val_3d.reshape(-1, n_feat)
X_test_2d_flat  = X_test_3d.reshape(-1, n_feat)

scaler.fit(X_train_2d_flat)

X_train_3d_s = scaler.transform(X_train_2d_flat).reshape(X_train_3d.shape)
X_val_3d_s   = scaler.transform(X_val_2d_flat).reshape(X_val_3d.shape)
X_test_3d_s  = scaler.transform(X_test_2d_flat).reshape(X_test_3d.shape)

# 给“传统机器学习模型”的输入：把序列展平为二维
X_train_2d = X_train_3d_s.reshape(X_train_3d_s.shape[0], -1)  # (N, T*n_feat)
X_val_2d   = X_val_3d_s.reshape(X_val_3d_s.shape[0], -1)
X_test_2d  = X_test_3d_s.reshape(X_test_3d_s.shape[0], -1)


# ======================
# 4) 指标函数
# ======================
def calc_metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return rmse, mae, r2


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
        yhat = (H @ self.beta).reshape(-1)
        return yhat


# ======================
# 6) PyTorch模型：ANN / RNN / Transformer
# ======================
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

        # val
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


def predict_torch(model, X):
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        yhat = model(xb).detach().cpu().numpy().reshape(-1)
    return yhat


# --- ANN（MLP）---
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

# --- RNN（vanilla）---
class RNN_Regressor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1):
        super().__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True, nonlinearity="tanh")
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):  # x: (B,T,F)
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        return self.fc(last)

# --- Transformer ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

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

    def forward(self, x):  # (B,T,F)
        h = self.proj(x)
        h = self.pe(h)
        h = self.encoder(h)
        pooled = h.mean(dim=1)   # mean pooling
        return self.fc(pooled)


# ======================
# 7) 训练并评估：多模型（统一输出Train/Val/Test）
# ======================
results = []

def add_result(name, yhat_tr, yhat_va, yhat_te):
    rmse_tr, mae_tr, r2_tr = calc_metrics(y_train, yhat_tr)
    rmse_va, mae_va, r2_va = calc_metrics(y_val, yhat_va)
    rmse_te, mae_te, r2_te = calc_metrics(y_test, yhat_te)
    results.append({
        "Model": name,
        "Train_RMSE": rmse_tr, "Val_RMSE": rmse_va, "Test_RMSE": rmse_te,
        "Train_MAE": mae_tr,   "Val_MAE": mae_va,   "Test_MAE": mae_te,
        "Train_R2": r2_tr,     "Val_R2": r2_va,     "Test_R2": r2_te
    })


# ---- 传统模型（2D输入）----
# GBDT
gbdt = GradientBoostingRegressor(random_state=SEED)
gbdt.fit(X_train_2d, y_train)
add_result("GBDT", gbdt.predict(X_train_2d), gbdt.predict(X_val_2d), gbdt.predict(X_test_2d))

# RF
rf = RandomForestRegressor(n_estimators=500, random_state=SEED, n_jobs=-1)
rf.fit(X_train_2d, y_train)
add_result("RF", rf.predict(X_train_2d), rf.predict(X_val_2d), rf.predict(X_test_2d))

# SVR（可能慢）
svr = SVR(C=10.0, epsilon=0.01, kernel="rbf")
svr.fit(X_train_2d, y_train)
add_result("SVR", svr.predict(X_train_2d), svr.predict(X_val_2d), svr.predict(X_test_2d))

# BP（sklearn MLPRegressor）
bp = MLPRegressor(hidden_layer_sizes=(128, 128),
                  activation="relu", solver="adam",
                  alpha=1e-4, learning_rate_init=1e-3,
                  max_iter=2000, random_state=SEED)
bp.fit(X_train_2d, y_train)
add_result("BP(MLP)", bp.predict(X_train_2d), bp.predict(X_val_2d), bp.predict(X_test_2d))

# XGBoost
if HAS_XGB:
    xgbr = xgb.XGBRegressor(
        n_estimators=800, max_depth=6, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, random_state=SEED, n_jobs=-1
    )
    xgbr.fit(X_train_2d, y_train)
    add_result("XGBoost", xgbr.predict(X_train_2d), xgbr.predict(X_val_2d), xgbr.predict(X_test_2d))
else:
    print("[WARN] xgboost 未安装，已跳过 XGBoost。pip install xgboost")

# CatBoost
if HAS_CAT:
    cbr = CatBoostRegressor(
        iterations=2000, depth=6, learning_rate=0.03,
        loss_function="RMSE", random_seed=SEED,
        verbose=False
    )
    cbr.fit(X_train_2d, y_train)
    add_result("CatBoost", cbr.predict(X_train_2d), cbr.predict(X_val_2d), cbr.predict(X_test_2d))
else:
    print("[WARN] catboost 未安装，已跳过 CatBoost。pip install catboost")

# ELM（你写的EML）
elm = ELMRegressor(n_hidden=800, activation="tanh", ridge_alpha=1e-3, random_state=SEED)
elm.fit(X_train_2d, y_train)
add_result("ELM(EML)", elm.predict(X_train_2d), elm.predict(X_val_2d), elm.predict(X_test_2d))


# ---- 深度模型（ANN用2D；RNN/Transformer用3D）----
# ANN (PyTorch MLP)
ann = ANN_MLP(input_dim=X_train_2d.shape[1], hidden=256)
ann = train_torch_model(
    ann, X_train_2d, y_train, X_val_2d, y_val,
    epochs=EPOCHS, patience=PATIENCE, batch_size=BATCH_SIZE, lr=LR
)
add_result("ANN(PyTorch)", predict_torch(ann, X_train_2d), predict_torch(ann, X_val_2d), predict_torch(ann, X_test_2d))

# RNN
rnn = RNN_Regressor(input_size=n_feat, hidden_size=64, num_layers=1)
rnn = train_torch_model(
    rnn, X_train_3d_s, y_train, X_val_3d_s, y_val,
    epochs=EPOCHS, patience=PATIENCE, batch_size=BATCH_SIZE, lr=LR
)
add_result("RNN", predict_torch(rnn, X_train_3d_s), predict_torch(rnn, X_val_3d_s), predict_torch(rnn, X_test_3d_s))

# Transformer
trf = TransformerRegressor(input_size=n_feat, d_model=64, nhead=4, num_layers=2, dim_ff=128, dropout=0.1)
trf = train_torch_model(
    trf, X_train_3d_s, y_train, X_val_3d_s, y_val,
    epochs=EPOCHS, patience=PATIENCE, batch_size=BATCH_SIZE, lr=LR
)
add_result("Transformer", predict_torch(trf, X_train_3d_s), predict_torch(trf, X_val_3d_s), predict_torch(trf, X_test_3d_s))


# ======================
# 8) 汇总输出 + 另存Excel
# ======================
res_df = pd.DataFrame(results)

# 排序：按 Test_RMSE 从小到大（不想排序就注释掉）
res_df = res_df.sort_values("Test_RMSE", ascending=True).reset_index(drop=True)

print("\n==================== Model Comparison (Train/Val/Test) ====================")
print(res_df)

split_info_df = pd.DataFrame([{
    "LOOKBACK": LOOKBACK,
    "TRAIN_TARGET_YEARS": str(TRAIN_TARGET_YEARS),
    "VAL_TARGET_YEARS": str(VAL_TARGET_YEARS),
    "TEST_TARGET_YEARS": str(TEST_TARGET_YEARS),
    "CHECK_INPUT_YEAR_OVERLAP": CHECK_INPUT_YEAR_OVERLAP,
    "STRICT_NO_INPUT_YEAR_OVERLAP": STRICT_NO_INPUT_YEAR_OVERLAP,
    "train_n": len(y_train),
    "val_n": len(y_val),
    "test_n": len(y_test),
    "overlap_train_val": str(overlap_info.get("ov_train_val", [])),
    "overlap_train_test": str(overlap_info.get("ov_train_test", [])),
    "overlap_val_test": str(overlap_info.get("ov_val_test", [])),
}])

out_xlsx = "Model_Comparison_manual_split.xlsx"
with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
    res_df.to_excel(writer, sheet_name="Metrics", index=False)
    split_info_df.to_excel(writer, sheet_name="Split_Config", index=False)

print(f"\nSaved: {out_xlsx}")
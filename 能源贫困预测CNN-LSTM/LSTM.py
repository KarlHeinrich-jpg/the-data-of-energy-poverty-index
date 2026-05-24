# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ======================
# 0) 基本配置
# ======================
DATA_PATH = r"EPI_entropy_output.xlsx"   # 改成你的文件路径
AREA_COL = "AREA"
YEAR_COL = "YEAR"
TARGET_COL = "EPI_entropy"

# 11个外生变量（中文列名）
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

# LSTM设置（后续你可用SSA优化这些超参）
LOOKBACK = 3
HIDDEN_SIZE = 64
NUM_LAYERS = 1
DROPOUT = 0.0  # NUM_LAYERS=1时dropout不生效
LR = 1e-3
BATCH_SIZE = 64
EPOCHS = 500
PATIENCE = 50

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# 0.1) 手动设置三个数据集（按目标年份切分）
# ======================
# ⚠️ 请按你的实际设计修改年份范围（闭区间）
TRAIN_TARGET_YEARS = (2008, 2017)   # 例：训练集目标年份
VAL_TARGET_YEARS   = (2018, 2019)   # 例：验证集目标年份（用于早停/调参）
TEST_TARGET_YEARS  = (2020, 2022)   # 例：测试集目标年份（最终评估）

# 是否打印输入窗口年份重叠检查（便于排查审稿人提到的feature-space overlap）
CHECK_OVERLAP = True


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

# 转数值 + 清洗
data[YEAR_COL] = pd.to_numeric(data[YEAR_COL], errors="coerce")
data[TARGET_COL] = pd.to_numeric(data[TARGET_COL], errors="coerce")
for c in FEATURES_CN:
    data[c] = pd.to_numeric(data[c], errors="coerce")

data = data.dropna(subset=[YEAR_COL, TARGET_COL])
# 简单填补缺失（也可以改成“按省份插补”）
data[FEATURES_CN] = data[FEATURES_CN].fillna(data[FEATURES_CN].mean())


# ======================
# 2) 构造序列样本（按省份时间顺序）+ 保存元信息
# ======================
def build_sequences_with_meta(df_in: pd.DataFrame, lookback: int):
    X_list, y_list = [], []
    meta_rows = []

    for area, g in df_in.groupby(AREA_COL, sort=False):
        g = g.sort_values(YEAR_COL).reset_index(drop=True)

        Xg = g[FEATURES_CN].values.astype(float)
        yg = g[TARGET_COL].values.astype(float)
        years = g[YEAR_COL].values.astype(int)

        # t 表示目标时点索引，输入窗口是 [t-lookback, ..., t-1]，预测目标 y[t]
        for t in range(lookback, len(g)):
            seqX = Xg[t - lookback:t, :]
            seqy = yg[t]

            target_year = int(years[t])
            input_start_year = int(years[t - lookback])
            input_end_year = int(years[t - 1])

            X_list.append(seqX)
            y_list.append(seqy)
            meta_rows.append({
                "AREA": area,
                "target_year": target_year,
                "input_start_year": input_start_year,
                "input_end_year": input_end_year,
            })

    if len(X_list) == 0:
        raise ValueError("构造序列样本失败：请检查LOOKBACK或数据长度。")

    X = np.asarray(X_list, dtype=float)
    y = np.asarray(y_list, dtype=float)
    meta = pd.DataFrame(meta_rows)

    return X, y, meta


X_all, y_all, meta_all = build_sequences_with_meta(data, LOOKBACK)
n_feat = X_all.shape[-1]

# ======================
# 2.1) 按“目标年份”手动切 train/val/test（不随机）
# ======================
def mask_year_range(series: pd.Series, year_range):
    y0, y1 = year_range
    return (series >= y0) & (series <= y1)

train_mask = mask_year_range(meta_all["target_year"], TRAIN_TARGET_YEARS)
val_mask   = mask_year_range(meta_all["target_year"], VAL_TARGET_YEARS)
test_mask  = mask_year_range(meta_all["target_year"], TEST_TARGET_YEARS)

# 检查是否有重叠
overlap_tv = (train_mask & val_mask).any()
overlap_tt = (train_mask & test_mask).any()
overlap_vt = (val_mask & test_mask).any()
if overlap_tv or overlap_tt or overlap_vt:
    raise ValueError("train/val/test 的目标年份范围有重叠，请重新设置。")

# 切分
X_train, y_train, meta_train = X_all[train_mask.values], y_all[train_mask.values], meta_all.loc[train_mask].reset_index(drop=True)
X_val,   y_val,   meta_val   = X_all[val_mask.values],   y_all[val_mask.values],   meta_all.loc[val_mask].reset_index(drop=True)
X_test,  y_test,  meta_test  = X_all[test_mask.values],  y_all[test_mask.values],  meta_all.loc[test_mask].reset_index(drop=True)

if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
    raise ValueError(
        f"切分后某个数据集为空：train={len(X_train)}, val={len(X_val)}, test={len(X_test)}。\n"
        f"请检查 LOOKBACK={LOOKBACK} 与年份范围设置。"
    )

# ======================
# 2.2) （可选）检查输入窗口年份是否重叠（用于审稿问题排查）
# ======================
def collect_input_year_set(meta_df: pd.DataFrame):
    year_set = set()
    for _, r in meta_df.iterrows():
        year_set.update(range(int(r["input_start_year"]), int(r["input_end_year"]) + 1))
    return year_set

if CHECK_OVERLAP:
    tr_input_years = collect_input_year_set(meta_train)
    va_input_years = collect_input_year_set(meta_val)
    te_input_years = collect_input_year_set(meta_test)

    print("\n==================== Split Summary ====================")
    print(f"Train target years: {TRAIN_TARGET_YEARS}, n={len(meta_train)}")
    print(f"Val   target years: {VAL_TARGET_YEARS}, n={len(meta_val)}")
    print(f"Test  target years: {TEST_TARGET_YEARS}, n={len(meta_test)}")
    print(f"Train input years range: {min(tr_input_years)}-{max(tr_input_years)}")
    print(f"Val   input years range: {min(va_input_years)}-{max(va_input_years)}")
    print(f"Test  input years range: {min(te_input_years)}-{max(te_input_years)}")

    print(f"Input-year overlap (train ∩ val): {sorted(tr_input_years & va_input_years)}")
    print(f"Input-year overlap (train ∩ test): {sorted(tr_input_years & te_input_years)}")
    print(f"Input-year overlap (val ∩ test): {sorted(va_input_years & te_input_years)}")
    print("=======================================================\n")


# ======================
# 3) 标准化（仅用训练集拟合）
# ======================
scaler = StandardScaler()
X_train_2d = X_train.reshape(-1, n_feat)
X_val_2d   = X_val.reshape(-1, n_feat)
X_test_2d  = X_test.reshape(-1, n_feat)

scaler.fit(X_train_2d)
X_train_s = scaler.transform(X_train_2d).reshape(X_train.shape)
X_val_s   = scaler.transform(X_val_2d).reshape(X_val.shape)
X_test_s  = scaler.transform(X_test_2d).reshape(X_test.shape)


# ======================
# 4) PyTorch LSTM 模型
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
        out, _ = self.lstm(x)      # (B,T,H)
        last = out[:, -1, :]       # (B,H)
        yhat = self.fc(last)       # (B,1)
        return yhat

model = LSTMRegressor(n_feat, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# 用“时间切分”的 train / val，不再 random_split
tr_ds = TensorDataset(
    torch.tensor(X_train_s, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
)
val_ds = TensorDataset(
    torch.tensor(X_val_s, dtype=torch.float32),
    torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
)

tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

best_val = np.inf
pat = 0
best_state = None

for epoch in range(1, EPOCHS + 1):
    model.train()
    tr_loss = 0.0
    for xb, yb in tr_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        tr_loss += loss.item() * xb.size(0)
    tr_loss /= len(tr_loader.dataset)

    model.eval()
    va_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb)
            loss = criterion(pred, yb)
            va_loss += loss.item() * xb.size(0)
    va_loss /= len(val_loader.dataset)

    print(f"Epoch {epoch:03d} | train_loss={tr_loss:.6f} | val_loss={va_loss:.6f}")

    if va_loss < best_val - 1e-6:
        best_val = va_loss
        pat = 0
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    else:
        pat += 1
        if pat >= PATIENCE:
            print("Early stopping.")
            break

if best_state is not None:
    model.load_state_dict(best_state)


# ======================
# 5) 预测 + 评价指标（RMSE/MAE/R2）
# ======================
def predict_numpy(model, X_np):
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(X_np, dtype=torch.float32).to(DEVICE)
        pred = model(xb).detach().cpu().numpy().reshape(-1)
    return pred

def eval_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2

yhat_train = predict_numpy(model, X_train_s)
yhat_val   = predict_numpy(model, X_val_s)
yhat_test  = predict_numpy(model, X_test_s)

rmse_train, mae_train, r2_train = eval_metrics(y_train, yhat_train)
rmse_val,   mae_val,   r2_val   = eval_metrics(y_val, yhat_val)
rmse_test,  mae_test,  r2_test  = eval_metrics(y_test, yhat_test)

print("\n==================== LSTM Evaluation (Manual Train/Val/Test Split) ====================")
print(f"Train | RMSE: {rmse_train:.6f} | MAE: {mae_train:.6f} | R2: {r2_train:.6f}")
print(f"Val   | RMSE: {rmse_val:.6f} | MAE: {mae_val:.6f} | R2: {r2_val:.6f}")
print(f"Test  | RMSE: {rmse_test:.6f} | MAE: {mae_test:.6f} | R2: {r2_test:.6f}")
print("========================================================================================")


# ======================
# 6) 可选：导出预测结果（含年份信息，便于画图/做SHAP筛样本）
# ======================
pred_train_df = meta_train.copy()
pred_train_df["y_true"] = y_train
pred_train_df["y_pred"] = yhat_train
pred_train_df["split"] = "train"

pred_val_df = meta_val.copy()
pred_val_df["y_true"] = y_val
pred_val_df["y_pred"] = yhat_val
pred_val_df["split"] = "val"

pred_test_df = meta_test.copy()
pred_test_df["y_true"] = y_test
pred_test_df["y_pred"] = yhat_test
pred_test_df["split"] = "test"

pred_all_df = pd.concat([pred_train_df, pred_val_df, pred_test_df], axis=0, ignore_index=True)

# 如果你想导出，取消注释
# pred_all_df.to_excel("LSTM_predictions_manual_split.xlsx", index=False)
# print("Saved predictions: LSTM_predictions_manual_split.xlsx")
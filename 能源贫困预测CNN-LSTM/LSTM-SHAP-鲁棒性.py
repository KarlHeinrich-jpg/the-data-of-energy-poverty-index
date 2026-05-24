# -*- coding: utf-8 -*-
"""
PyTorch版：分别以 EPI_CRITIC / EPI_FA / EPI_PCA 为因变量训练 LSTM，并输出对应 SHAP 蜂群图 + 叠加条形图
- 不再分东/中/西部
- 自变量：11个外生变量（中文列名）
- 因变量：EPI_CRITIC, EPI_FA, EPI_PCA（你的数据中必须已存在这三列）
- 输出：SHAP_LSTM_EPI_CRITIC.png / SHAP_LSTM_EPI_FA.png / SHAP_LSTM_EPI_PCA.png

注意：
1) 该脚本假设你的 Excel 同时包含：AREA, YEAR, 11个外生变量列，以及 EPI_CRITIC/EPI_FA/EPI_PCA 三列。
2) 数据集划分：按“每省内部按YEAR排序”，末尾30%为测试（不打乱，避免时间泄露）。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.preprocessing import StandardScaler
import shap

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ======================
# 全局字体与风格设置（与你之前一致）
# ======================
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.weight': 'bold',
    'font.size': 16,
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'axes.labelsize': 18,
    'axes.titlesize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'axes.linewidth': 2.0,
    'axes.edgecolor': 'black',
    'axes.unicode_minus': False,
    'xtick.major.width': 2.0,
    'ytick.major.width': 2.0,
    'xtick.major.size': 8,
    'ytick.major.size': 8,
})

# ======================
# 0) 配置：文件路径 + 列名
# ======================
DATA_PATH = r"EPI_entropy_output - 副本.xlsx"  # <<< 改成你的文件（需包含三种EPI列+外生变量列）
AREA_COL = "AREA"
YEAR_COL = "YEAR"

# 三个因变量（你要求的）
TARGETS = ["EWM","CRITIC", "FA", "PCA","CV","DM"]

# 中文列名 -> 简称（用于SHAP图的特征名称显示）
CN2SYM = {
    "人均地区生产总值(元/人)": "GDP",
    "居民消费(亿元)": "HC",
    "第二产业占比": "SSI",
    "人口密度": "PD",
    "城镇人口占比": "UR",
    "能源消费强度": "EI",
    "能源消费比": "ECS",
    "人均发电量": "PGPC",
    "森林覆盖率(%)": "FCR",
    "地方财政环境保护支出(亿元)": "LFEEP",
    "人均水资源量(立方米/人)": "WRPC",
}
FEATURE_CN = list(CN2SYM.keys())
FEATURE_SYM = [CN2SYM[c] for c in FEATURE_CN]

# ======================
# 1) LSTM & SHAP 参数（可按你论文统一口径）
# ======================
LOOKBACK = 5
TEST_RATIO = 0.30

EPOCHS = 200
BATCH_SIZE = 64
HIDDEN_SIZE = 64
NUM_LAYERS = 1
DROPOUT = 0.0
LR = 1e-2
PATIENCE = 20

BG_SIZE = 100
TEST_SHAP_SIZE = 300

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# 2) 读sheet：自动找到包含所需列的sheet（更稳）
# ======================
def find_sheet_with_columns(path: str, required_cols: list) -> pd.DataFrame:
    xls = pd.ExcelFile(path)
    for sh in xls.sheet_names:
        cols = pd.read_excel(path, sheet_name=sh, nrows=0).columns.tolist()
        if all(c in cols for c in required_cols):
            print(f"[INFO] Using sheet: {sh}")
            return pd.read_excel(path, sheet_name=sh)
    raise ValueError(f"在 {path} 的所有sheet中都没找到包含所需列的sheet。\n所需列：{required_cols}")

# ======================
# 3) 构造序列样本（按省内YEAR排序；每省末尾30%做测试）
# ======================
def split_panel_train_test(df_in: pd.DataFrame, lookback: int, test_ratio: float, target_col: str):
    Xtr, ytr, Xte, yte = [], [], [], []
    for area, g in df_in.groupby(AREA_COL, sort=False):
        g = g.sort_values(YEAR_COL)

        Xg = g[FEATURE_CN].values.astype(float)
        yg = g[target_col].values.astype(float)

        seqX, seqy = [], []
        for t in range(lookback, len(g)):
            seqX.append(Xg[t-lookback:t, :])
            seqy.append(yg[t])
        seqX = np.asarray(seqX, dtype=float)
        seqy = np.asarray(seqy, dtype=float)

        if len(seqX) < 5:
            continue

        n = len(seqX)
        n_test = max(1, int(np.floor(n * test_ratio)))
        n_train = n - n_test

        Xtr.append(seqX[:n_train]); ytr.append(seqy[:n_train])
        Xte.append(seqX[n_train:]); yte.append(seqy[n_train:])

    if len(Xtr) == 0 or len(Xte) == 0:
        raise ValueError(f"目标列 {target_col} 下可用序列样本为空（可能某些省年份不足或数据缺失）。")

    X_train = np.concatenate(Xtr, axis=0)
    y_train = np.concatenate(ytr, axis=0)
    X_test = np.concatenate(Xte, axis=0)
    y_test = np.concatenate(yte, axis=0)
    return X_train, X_test, y_train, y_test

# ======================
# 4) LSTM 模型 + 训练
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
        out, _ = self.lstm(x)     # (B,T,H)
        last = out[:, -1, :]      # (B,H)
        yhat = self.fc(last)      # (B,1)
        return yhat

def train_lstm(X_train_s, y_train):
    n_feat = X_train_s.shape[-1]
    model = LSTMRegressor(n_feat, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    ds = TensorDataset(
        torch.tensor(X_train_s, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    )

    # 训练/验证（用于early stopping）
    val_ratio = 0.2
    n_total = len(ds)
    n_val = int(n_total * val_ratio)
    n_tr = n_total - n_val
    tr_ds, val_ds = torch.utils.data.random_split(
        ds, [n_tr, n_val], generator=torch.Generator().manual_seed(SEED)
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

        print(f"Epoch {epoch:03d} | train={tr_loss:.6f} | val={va_loss:.6f}")

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
    model.eval()
    return model

# ======================
# 5) SHAP 计算（与之前一致）
# ======================
def compute_shap(model, X_train_s, X_test_s, lookback, n_feat):
    rng = np.random.default_rng(SEED)
    bg_idx = rng.choice(len(X_train_s), size=min(BG_SIZE, len(X_train_s)), replace=False)
    te_idx = rng.choice(len(X_test_s), size=min(TEST_SHAP_SIZE, len(X_test_s)), replace=False)

    X_bg = torch.tensor(X_train_s[bg_idx], dtype=torch.float32).to(DEVICE)
    X_te = torch.tensor(X_test_s[te_idx], dtype=torch.float32).to(DEVICE)

    shap_values = None
    try:
        explainer = shap.DeepExplainer(model, X_bg)
        shap_values = explainer.shap_values(X_te)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
    except Exception:
        try:
            explainer = shap.GradientExplainer(model, X_bg)
            shap_values = explainer.shap_values(X_te)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
        except Exception:
            # KernelExplainer（慢但通用）
            X_bg_np = X_bg.detach().cpu().numpy().reshape(X_bg.shape[0], -1)
            X_te_np = X_te.detach().cpu().numpy().reshape(X_te.shape[0], -1)

            def f_predict(x_flat):
                x_seq = torch.tensor(x_flat, dtype=torch.float32).to(DEVICE).reshape(-1, lookback, n_feat)
                with torch.no_grad():
                    yhat = model(x_seq).detach().cpu().numpy()
                return yhat

            explainer = shap.KernelExplainer(f_predict, X_bg_np)
            sv = explainer.shap_values(X_te_np, nsamples=200)
            if isinstance(sv, list):
                sv = sv[0]
            shap_values = sv.reshape(X_te.shape[0], lookback, n_feat)

    if isinstance(shap_values, torch.Tensor):
        shap_values = shap_values.detach().cpu().numpy()

    if shap_values.ndim != 3:
        raise ValueError(f"SHAP值维度异常：{shap_values.shape}，期望 (n_samples, lookback, n_features)")

    # 对时间维取均值，得到特征级SHAP（与你之前一致）
    shap_feat = shap_values.mean(axis=1)  # (n_samples, n_feat)
    X_te_feat = X_te.detach().cpu().numpy().mean(axis=1)

    X_te_df = pd.DataFrame(X_te_feat, columns=FEATURE_SYM)
    shap_df = pd.DataFrame(shap_feat, columns=FEATURE_SYM)
    return shap_feat, X_te_df, shap_df

# ======================
# 6) 绘图并保存（与你之前一致）
# ======================
def plot_and_save(shap_feat, X_te_df, shap_df, out_png, title_text):
    plt.figure(figsize=(16, 10), dpi=300)

    shap.summary_plot(shap_feat, X_te_df, plot_type="dot", cmap="viridis", show=False)

    ax1 = plt.gca()
    ax1.set_position([0.3, 0.1, 0.65, 0.85])

    for collection in ax1.collections:
        try:
            collection.set_sizes([40])
        except Exception:
            pass

    ax2 = ax1.twiny()
    feature_order = [t.get_text() for t in ax1.get_yticklabels()]
    ordered_importance = [np.abs(shap_df[f]).mean() for f in feature_order]

    viridis = cm.get_cmap('viridis', len(feature_order))
    colors = viridis(np.linspace(0, 1, len(feature_order)))

    ax2.barh(
        range(len(feature_order)),
        ordered_importance,
        height=0.7,
        color=colors,
        alpha=0.3,
        edgecolor='black',
        linewidth=2.0
    )

    ax1.set_xlabel('SHAP Value (Impact on LSTM Model Output)', fontweight='bold', fontsize=16, color='black')
    ax2.set_xlabel('Mean |SHAP Value| (Feature Importance)', fontweight='bold', fontsize=16, color='black')
    ax2.xaxis.set_label_position('top')
    ax2.xaxis.tick_top()
    ax1.set_ylabel('Features', fontweight='bold', fontsize=16, color='black')

    ax1.set_title(title_text, fontweight='bold', fontsize=18, color='black', pad=12)

    ax1.grid(True, axis='x', linestyle='--', alpha=0.3, linewidth=1.5)
    ax2.grid(True, axis='x', linestyle='--', alpha=0.3, linewidth=1.5)

    for tick in ax1.get_xticklabels() + ax1.get_yticklabels() + ax2.get_xticklabels():
        tick.set_color('black')
        tick.set_fontweight('bold')

    ax1.axvline(x=0, color='black', linestyle='-', linewidth=2.0, alpha=0.7)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight", format="png")
    plt.close()
    print(f"Saved: {out_png}")

# ======================
# 7) 主流程：对三个Y分别训练 + SHAP + 保存
# ======================
required_cols = [AREA_COL, YEAR_COL] + FEATURE_CN + TARGETS
df_all = find_sheet_with_columns(DATA_PATH, required_cols)

# 清洗：转数值
df_all[YEAR_COL] = pd.to_numeric(df_all[YEAR_COL], errors="coerce")
for c in FEATURE_CN:
    df_all[c] = pd.to_numeric(df_all[c], errors="coerce")
for ycol in TARGETS:
    df_all[ycol] = pd.to_numeric(df_all[ycol], errors="coerce")

# 去掉YEAR或任一目标缺失行（按需）
df_all = df_all.dropna(subset=[YEAR_COL])
df_all[FEATURE_CN] = df_all[FEATURE_CN].fillna(df_all[FEATURE_CN].mean())

for target_col in TARGETS:
    print("\n" + "=" * 80)
    print(f"Target Y = {target_col}")

    data_use = df_all.dropna(subset=[target_col]).copy()
    if data_use.empty:
        print(f"[Skip] {target_col} 全为空或不存在有效数据。")
        continue

    # 构造序列（按省内时间顺序分训练/测试）
    X_train, X_test, y_train, y_test = split_panel_train_test(data_use, LOOKBACK, TEST_RATIO, target_col)
    n_feat = X_train.shape[-1]

    # 标准化（仅用训练集拟合）
    scaler = StandardScaler()
    X_train_2d = X_train.reshape(-1, n_feat)
    X_test_2d = X_test.reshape(-1, n_feat)
    scaler.fit(X_train_2d)

    X_train_s = scaler.transform(X_train_2d).reshape(X_train.shape)
    X_test_s = scaler.transform(X_test_2d).reshape(X_test.shape)

    # 训练LSTM
    model = train_lstm(X_train_s, y_train)

    # SHAP
    shap_feat, X_te_df, shap_df = compute_shap(model, X_train_s, X_test_s, LOOKBACK, n_feat)

    # 保存图
    out_png = f"SHAP_LSTM_{target_col}.png"
    title = f"SHAP Summary ({target_col})"
    plot_and_save(shap_feat, X_te_df, shap_df, out_png, title)

print("\nAll done.")

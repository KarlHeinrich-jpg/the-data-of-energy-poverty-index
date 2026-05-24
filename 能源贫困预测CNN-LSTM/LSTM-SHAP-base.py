# -*- coding: utf-8 -*-
"""
PyTorch 版：LSTM + SHAP（保持你给的绘图风格：Times New Roman + bold + viridis + 蜂群图叠加条形图）
- 数据列：AREA, YEAR, EPI_entropy + 11个外生变量（中文列名）
- 特征显示：用简称（GDP, HC, SSI, PD, UR, EI, ECS, PGPC, FCR, LFEEP, WRPC）
- 模型：PyTorch LSTM（回归）
- SHAP：优先 DeepExplainer / GradientExplainer；不行自动回退 KernelExplainer（慢但通用）
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
# 全局字体与风格设置（与你原代码一致）
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
DATA_PATH = r"EPI_entropy_output.xlsx"  # <<< 改成你的路径
AREA_COL = "AREA"
YEAR_COL = "YEAR"
TARGET_COL = "EPI_entropy"

# 中文列名 -> 简称（你给的Table2）
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
# 1) LSTM 训练/SHAP 参数（你可后续让SSA优化）
# ======================
LOOKBACK = 5          # 用过去5期预测下一期
TEST_RATIO = 0.30     # 每省最后30%序列做测试（不打乱）
EPOCHS = 200
BATCH_SIZE = 64
HIDDEN_SIZE = 64
NUM_LAYERS = 1
DROPOUT = 0.0         # NUM_LAYERS=1时dropout不会生效（PyTorch设定）
LR = 1e-2
PATIENCE = 20

# SHAP采样（避免太慢）
BG_SIZE = 100
TEST_SHAP_SIZE = 300

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================
# 2) 读取 & 清洗（兼容中文导入；剔除非数据尾部内容）
# ======================
df = pd.read_excel(DATA_PATH)

need_cols = [AREA_COL, YEAR_COL, TARGET_COL] + FEATURE_CN
missing = [c for c in need_cols if c not in df.columns]
if missing:
    raise ValueError(f"数据缺少以下列：{missing}\n请检查Excel表头是否完全一致（括号全角/半角、空格等）。")

data = df[need_cols].copy()

data[YEAR_COL] = pd.to_numeric(data[YEAR_COL], errors="coerce")
data[TARGET_COL] = pd.to_numeric(data[TARGET_COL], errors="coerce")
for c in FEATURE_CN:
    data[c] = pd.to_numeric(data[c], errors="coerce")

# YEAR或目标为空的行剔除（通常能去掉表尾的“Table 2”文本）
data = data.dropna(subset=[YEAR_COL, TARGET_COL])

# 特征缺失：简单均值填补（也可换更严谨的分省插补）
data[FEATURE_CN] = data[FEATURE_CN].fillna(data[FEATURE_CN].mean())


# ======================
# 3) 构造“省内时间序列”样本：X:(N, lookback, n_feat), y:(N,)
# ======================
def split_panel_train_test(df_in: pd.DataFrame, lookback: int, test_ratio: float):
    Xtr, ytr, Xte, yte = [], [], [], []
    for area, g in df_in.groupby(AREA_COL, sort=False):
        g = g.sort_values(YEAR_COL)
        Xg = g[FEATURE_CN].values.astype(float)
        yg = g[TARGET_COL].values.astype(float)

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

        Xtr.append(seqX[:n_train])
        ytr.append(seqy[:n_train])
        Xte.append(seqX[n_train:])
        yte.append(seqy[n_train:])

    X_train = np.concatenate(Xtr, axis=0)
    y_train = np.concatenate(ytr, axis=0)
    X_test = np.concatenate(Xte, axis=0)
    y_test = np.concatenate(yte, axis=0)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = split_panel_train_test(data, LOOKBACK, TEST_RATIO)

n_feat = X_train.shape[-1]


# ======================
# 4) 标准化（按训练集拟合；对三维序列展平处理）
# ======================
scaler = StandardScaler()
X_train_2d = X_train.reshape(-1, n_feat)
X_test_2d = X_test.reshape(-1, n_feat)

scaler.fit(X_train_2d)
X_train_s = scaler.transform(X_train_2d).reshape(X_train.shape)
X_test_s = scaler.transform(X_test_2d).reshape(X_test.shape)


# ======================
# 5) PyTorch LSTM 模型
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
        # x: (B, T, F)
        out, _ = self.lstm(x)          # out: (B, T, H)
        last = out[:, -1, :]           # last: (B, H)
        yhat = self.fc(last)           # (B, 1)
        return yhat


model = LSTMRegressor(
    input_size=n_feat,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT
).to(DEVICE)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

train_ds = TensorDataset(
    torch.tensor(X_train_s, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

# 简单early stopping（用训练集切分出val）
val_ratio = 0.2
n_total = len(train_ds)
n_val = int(n_total * val_ratio)
n_tr = n_total - n_val
tr_ds, val_ds = torch.utils.data.random_split(train_ds, [n_tr, n_val], generator=torch.Generator().manual_seed(SEED))

tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

best_val = np.inf
pat = 0
best_state = None

for epoch in range(1, EPOCHS + 1):
    model.train()
    tr_loss = 0.0
    for xb, yb in tr_loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)
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
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
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

# ======================
# 6) 计算 SHAP（优先 Deep/Gradient，失败则 Kernel）
# ======================
model.eval()

rng = np.random.default_rng(SEED)
bg_idx = rng.choice(len(X_train_s), size=min(BG_SIZE, len(X_train_s)), replace=False)
te_idx = rng.choice(len(X_test_s), size=min(TEST_SHAP_SIZE, len(X_test_s)), replace=False)

X_bg = torch.tensor(X_train_s[bg_idx], dtype=torch.float32).to(DEVICE)
X_te = torch.tensor(X_test_s[te_idx], dtype=torch.float32).to(DEVICE)

shap_values = None

# DeepExplainer / GradientExplainer（PyTorch）
try:
    explainer = shap.DeepExplainer(model, X_bg)
    shap_values = explainer.shap_values(X_te)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
except Exception as e1:
    try:
        explainer = shap.GradientExplainer(model, X_bg)
        shap_values = explainer.shap_values(X_te)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
    except Exception as e2:
        # KernelExplainer：需要numpy二维输入（把序列展平）
        X_bg_np = X_bg.detach().cpu().numpy().reshape(X_bg.shape[0], -1)
        X_te_np = X_te.detach().cpu().numpy().reshape(X_te.shape[0], -1)

        def f_predict(x_flat):
            x_seq = torch.tensor(x_flat, dtype=torch.float32).to(DEVICE).reshape(-1, LOOKBACK, n_feat)
            with torch.no_grad():
                yhat = model(x_seq).detach().cpu().numpy()
            return yhat

        explainer = shap.KernelExplainer(f_predict, X_bg_np)
        sv = explainer.shap_values(X_te_np, nsamples=200)
        if isinstance(sv, list):
            sv = sv[0]
        shap_values = sv.reshape(X_te.shape[0], LOOKBACK, n_feat)

# shap_values 期望形状：(n_samples, lookback, n_feat)
if isinstance(shap_values, torch.Tensor):
    shap_values = shap_values.detach().cpu().numpy()

if shap_values.ndim != 3:
    raise ValueError(f"SHAP值维度异常：{shap_values.shape}，期望 (n_samples, lookback, n_features)")

# ======================
# 7) 汇总到特征维度：对时间维取平均（你也可以改成sum或abs-sum）
# ======================
shap_feat = shap_values.mean(axis=1)  # (n_samples, n_feat)

X_te_np = X_te.detach().cpu().numpy()
X_te_feat = X_te_np.mean(axis=1)      # (n_samples, n_feat)

X_te_df = pd.DataFrame(X_te_feat, columns=FEATURE_SYM)
shap_df = pd.DataFrame(shap_feat, columns=FEATURE_SYM)

# ======================
# 8) 绘图：蜂群图 + 右侧叠加条形图（格式与原代码一致）
# ======================
plt.figure(figsize=(16, 10), dpi=300)

shap.summary_plot(
    shap_feat,
    X_te_df,
    plot_type="dot",
    cmap="viridis",
    show=False
)

ax1 = plt.gca()
ax1.set_position([0.3, 0.1, 0.65, 0.85])

# 放大蜂群图的点
for collection in ax1.collections:
    try:
        collection.set_sizes([40])
    except Exception:
        pass

# 右侧叠加条形图
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

ax1.set_xlabel('SHAP Value', fontweight='bold', fontsize=16, color='black')
ax2.set_xlabel('Mean |SHAP Value| (Feature Importance)', fontweight='bold', fontsize=16, color='black')
ax2.xaxis.set_label_position('top')
ax2.xaxis.tick_top()
ax1.set_ylabel('Features', fontweight='bold', fontsize=16, color='black')

ax1.grid(True, axis='x', linestyle='--', alpha=0.3, linewidth=1.5)
ax2.grid(True, axis='x', linestyle='--', alpha=0.3, linewidth=1.5)

for tick in ax1.get_xticklabels() + ax1.get_yticklabels() + ax2.get_xticklabels():
    tick.set_color('black')
    tick.set_fontweight('bold')

ax1.axvline(x=0, color='black', linestyle='-', linewidth=2.0, alpha=0.7)

OUT_PNG = r"SHAP_LSTM_beeswarm_bar_with ur.png"  # <<< 改成你要保存的位置/文件名
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight", format="png")
plt.show()

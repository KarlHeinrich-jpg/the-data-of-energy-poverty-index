# -*- coding: utf-8 -*-
"""
SSA-LSTM (PyTorch) + Save Best Weights (Checkpoint) [Manual Year Split Version]

主要修改：
1) 不再使用 7/2/1 比例切分
2) 改为手动设置 Train / Validation / Test 的目标年份范围（按 target_year 切）
3) 支持检查输入窗口年份重叠（feature-space overlap 风险）
4) 其余SSA + LSTM + checkpoint逻辑尽量保持不变
"""

import os
import random
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
DATA_PATH = r"EPI_entropy_output.xlsx"  # <<< 改成你的文件
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
# 你可以按需要修改这三个范围（例如：train=(2008,2015), val=(2016,2018), test=(2019,2022)）
TRAIN_TARGET_YEARS = (2008, 2017)   # 例：训练集目标年份
VAL_TARGET_YEARS   = (2018, 2019)   # 例：验证集目标年份（用于早停/调参）
TEST_TARGET_YEARS  = (2020, 2022)   # 例：测试集目标年份（最终评估）
# 是否检查输入窗口年份重叠（Comment 3 相关）
CHECK_INPUT_YEAR_OVERLAP = True
# 若为 True，一旦发现 train/val/test 输入年份集合重叠就直接报错
# 先调试时建议 False；等你确定年份方案后再改成 True
STRICT_NO_INPUT_YEAR_OVERLAP = False

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_all_seeds(seed=42):
    """训练与推理尽量可复现（GPU仍可能有极小差异）"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 尽量确定性（可能变慢）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_all_seeds(SEED)


# ======================
# 1) SSA 参数
# ======================
SSA_POP = 50
SSA_ITER = 200

# ======================
# 2) LSTM训练设置
# ======================
EPOCHS_SSA = 100
PATIENCE_SSA = 30

EPOCHS_FINAL = 300
PATIENCE_FINAL = 30

# ======================
# 3) SSA 搜索空间
# ======================
BATCH_CHOICES = [32, 64, 128]

BOUNDS = {
    "lookback": (2, 10),                # int
    "hidden": (16, 128),                # int
    "num_layers": (1, 2),               # int
    "dropout": (0.0, 0.5),              # float
    "log10_lr": (-4.0, -1.0),           # float -> lr = 10**log10_lr
    "batch_idx": (0, len(BATCH_CHOICES)-1),  # int
}


def decode_position(x: np.ndarray) -> dict:
    x = x.copy()
    lb = np.array([
        BOUNDS["lookback"][0], BOUNDS["hidden"][0], BOUNDS["num_layers"][0],
        BOUNDS["dropout"][0], BOUNDS["log10_lr"][0], BOUNDS["batch_idx"][0]
    ], dtype=float)
    ub = np.array([
        BOUNDS["lookback"][1], BOUNDS["hidden"][1], BOUNDS["num_layers"][1],
        BOUNDS["dropout"][1], BOUNDS["log10_lr"][1], BOUNDS["batch_idx"][1]
    ], dtype=float)
    x = np.clip(x, lb, ub)

    lookback = int(np.round(x[0]))
    hidden = int(np.round(x[1]))
    num_layers = int(np.round(x[2]))
    dropout = float(x[3])
    lr = float(10 ** x[4])
    batch_idx = int(np.round(x[5]))
    batch_idx = int(np.clip(batch_idx, 0, len(BATCH_CHOICES)-1))
    batch_size = int(BATCH_CHOICES[batch_idx])

    return {
        "lookback": lookback,
        "hidden_size": hidden,
        "num_layers": num_layers,
        "dropout": dropout,
        "lr": lr,
        "batch_size": batch_size
    }


def init_population(pop_size: int, dim: int) -> np.ndarray:
    lb = np.array([
        BOUNDS["lookback"][0], BOUNDS["hidden"][0], BOUNDS["num_layers"][0],
        BOUNDS["dropout"][0], BOUNDS["log10_lr"][0], BOUNDS["batch_idx"][0]
    ], dtype=float)
    ub = np.array([
        BOUNDS["lookback"][1], BOUNDS["hidden"][1], BOUNDS["num_layers"][1],
        BOUNDS["dropout"][1], BOUNDS["log10_lr"][1], BOUNDS["batch_idx"][1]
    ], dtype=float)
    return lb + (ub - lb) * np.random.rand(pop_size, dim)


def clip_pop(P: np.ndarray) -> np.ndarray:
    lb = np.array([
        BOUNDS["lookback"][0], BOUNDS["hidden"][0], BOUNDS["num_layers"][0],
        BOUNDS["dropout"][0], BOUNDS["log10_lr"][0], BOUNDS["batch_idx"][0]
    ], dtype=float)
    ub = np.array([
        BOUNDS["lookback"][1], BOUNDS["hidden"][1], BOUNDS["num_layers"][1],
        BOUNDS["dropout"][1], BOUNDS["log10_lr"][1], BOUNDS["batch_idx"][1]
    ], dtype=float)
    return np.clip(P, lb, ub)


# ======================
# 4) 读取Excel：自动找sheet
# ======================
def find_sheet_with_columns(path: str, required_cols: list) -> tuple[str, pd.DataFrame]:
    xls = pd.ExcelFile(path)
    for sh in xls.sheet_names:
        cols = pd.read_excel(path, sheet_name=sh, nrows=0).columns.tolist()
        if all(c in cols for c in required_cols):
            df0 = pd.read_excel(path, sheet_name=sh)
            return sh, df0
    raise ValueError(f"在 {path} 的所有sheet中都没找到包含所需列的sheet。\n所需列：{required_cols}")


required_cols = [AREA_COL, YEAR_COL, TARGET_COL] + FEATURES_CN
sheet_name, df = find_sheet_with_columns(DATA_PATH, required_cols)
print(f"[INFO] Using sheet: {sheet_name}")

data = df[required_cols].copy()
data[YEAR_COL] = pd.to_numeric(data[YEAR_COL], errors="coerce")
data[TARGET_COL] = pd.to_numeric(data[TARGET_COL], errors="coerce")
for c in FEATURES_CN:
    data[c] = pd.to_numeric(data[c], errors="coerce")
data = data.dropna(subset=[YEAR_COL, TARGET_COL])

# 记录 fillna 用的均值（下次加载权重也用它填补，确保一致）
fillna_means = data[FEATURES_CN].mean()
data[FEATURES_CN] = data[FEATURES_CN].fillna(fillna_means)


# ======================
# 5) 序列构造 + 手动年份切分（cache）
# ======================
_seq_cache_manual = {}
_split_check_printed = set()  # 避免SSA里重复打印同一个lookback的切分信息


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

        for t in range(lookback, len(g)):
            # 输入窗口: [t-lookback, ..., t-1], 目标: y[t]
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
    year_set = set()
    for _, r in meta_df.iterrows():
        year_set.update(range(int(r["input_start_year"]), int(r["input_end_year"]) + 1))
    return year_set


def split_by_manual_target_years(X_all, y_all, meta_all):
    """
    按 target_year 手动切 train / val / test
    返回顺序：
    Xtr, ytr, Xva, yva, Xte, yte, meta_train, meta_val, meta_test, overlap_info
    """
    train_mask = _mask_year_range(meta_all["target_year"], TRAIN_TARGET_YEARS)
    val_mask   = _mask_year_range(meta_all["target_year"], VAL_TARGET_YEARS)
    test_mask  = _mask_year_range(meta_all["target_year"], TEST_TARGET_YEARS)

    # 检查目标年份范围是否重叠
    if (train_mask & val_mask).any() or (train_mask & test_mask).any() or (val_mask & test_mask).any():
        raise ValueError("train/val/test 的目标年份范围有重叠，请检查 TRAIN_TARGET_YEARS / VAL_TARGET_YEARS / TEST_TARGET_YEARS。")

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
            f"请检查 lookback 与年份范围设置。"
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

        if STRICT_NO_INPUT_YEAR_OVERLAP and (len(ov_tr_va) > 0 or len(ov_tr_te) > 0 or len(ov_va_te) > 0):
            raise ValueError(
                "检测到输入年份重叠（feature-space overlap 风险）。\n"
                f"train∩val={ov_tr_va}\n"
                f"train∩test={ov_tr_te}\n"
                f"val∩test={ov_va_te}\n"
                "请重新设置目标年份范围（可增加时间间隔/purge gap）。"
            )

    return Xtr, ytr, Xva, yva, Xte, yte, mtr, mva, mte, overlap_info


def get_data_manual(lookback: int):
    """
    缓存：同一lookback只构造/切分一次
    返回：
    Xtr, ytr, Xva, yva, Xte, yte, meta_train, meta_val, meta_test, overlap_info
    """
    cache_key = (
        lookback,
        TRAIN_TARGET_YEARS,
        VAL_TARGET_YEARS,
        TEST_TARGET_YEARS,
        CHECK_INPUT_YEAR_OVERLAP,
        STRICT_NO_INPUT_YEAR_OVERLAP,
    )

    if cache_key in _seq_cache_manual:
        return _seq_cache_manual[cache_key]

    X_all, y_all, meta_all = build_sequences_with_meta(data, lookback)
    res = split_by_manual_target_years(X_all, y_all, meta_all)
    _seq_cache_manual[cache_key] = res

    # 只打印一次，避免SSA迭代刷屏
    if CHECK_INPUT_YEAR_OVERLAP and (lookback, TRAIN_TARGET_YEARS, VAL_TARGET_YEARS, TEST_TARGET_YEARS) not in _split_check_printed:
        _, _, _, _, _, _, mtr, mva, mte, overlap_info = res

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
            if STRICT_NO_INPUT_YEAR_OVERLAP:
                print("[STRICT MODE] No overlap allowed.")
            else:
                print("[STRICT MODE OFF] Overlap is reported but not blocked.")
        print()

        _split_check_printed.add((lookback, TRAIN_TARGET_YEARS, VAL_TARGET_YEARS, TEST_TARGET_YEARS))

    return res


# ======================
# 6) LSTM 模型
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


# ======================
# 7) fitness：Validation RMSE（基于手动时间切分）
# ======================
def fit_lstm_and_eval_val_rmse(params: dict) -> float:
    # 为了让同一参数每次评估更稳定（尤其SSA内部多次调用）
    set_all_seeds(SEED)

    lookback = params["lookback"]
    Xtr_raw, ytr, Xva_raw, yva, _, _, _, _, _, _ = get_data_manual(lookback)
    n_feat = Xtr_raw.shape[-1]

    # 标准化（只用train拟合）
    scaler = StandardScaler()
    Xtr_2d = Xtr_raw.reshape(-1, n_feat)
    Xva_2d = Xva_raw.reshape(-1, n_feat)
    scaler.fit(Xtr_2d)
    Xtr = scaler.transform(Xtr_2d).reshape(Xtr_raw.shape)
    Xva = scaler.transform(Xva_2d).reshape(Xva_raw.shape)

    model = LSTMRegressor(
        input_size=n_feat,
        hidden_size=params["hidden_size"],
        num_layers=params["num_layers"],
        dropout=params["dropout"]
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    criterion = nn.MSELoss()

    train_ds = TensorDataset(
        torch.tensor(Xtr, dtype=torch.float32),
        torch.tensor(ytr, dtype=torch.float32).view(-1, 1)
    )
    val_ds = TensorDataset(
        torch.tensor(Xva, dtype=torch.float32),
        torch.tensor(yva, dtype=torch.float32).view(-1, 1)
    )

    g = torch.Generator()
    g.manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=params["batch_size"], shuffle=False)

    best_val = np.inf
    pat = 0
    best_state = None

    for _ in range(EPOCHS_SSA):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

        # 用外部validation做early stopping
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
            if pat >= PATIENCE_SSA:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # fitness 就是 validation RMSE
    # （这里再算一次保持逻辑清晰）
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

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return rmse_val


# ======================
# 8) SSA 主过程（与原逻辑基本一致）
# ======================
def ssa_optimize():
    dim = 6
    pop = init_population(SSA_POP, dim)

    PD = 0.2
    SD = 0.1
    ST = 0.8

    fitness = np.zeros(SSA_POP, dtype=float)
    for i in range(SSA_POP):
        params = decode_position(pop[i])
        fitness[i] = fit_lstm_and_eval_val_rmse(params)

    iter_logs = []
    pop_logs = []

    for it in range(1, SSA_ITER + 1):
        order = np.argsort(fitness)
        pop = pop[order]
        fitness = fitness[order]

        best_x = pop[0].copy()
        worst_x = pop[-1].copy()

        best_fit = float(fitness[0])
        mean_fit = float(np.mean(fitness))
        worst_fit = float(fitness[-1])

        best_params = decode_position(best_x)

        iter_logs.append({
            "iteration": it,
            "best_fitness_val_RMSE": best_fit,
            "mean_fitness_val_RMSE": mean_fit,
            "worst_fitness_val_RMSE": worst_fit,
            **best_params
        })

        for rank in range(SSA_POP):
            p = decode_position(pop[rank])
            pop_logs.append({
                "iteration": it,
                "rank": rank + 1,
                "fitness_val_RMSE": float(fitness[rank]),
                **p
            })

        print(f"[SSA] Iter {it:03d}/{SSA_ITER} | best(val_RMSE)={best_fit:.6f} | mean={mean_fit:.6f}")

        n_producers = int(np.ceil(PD * SSA_POP))
        n_aware = int(np.ceil(SD * SSA_POP))

        # Producer update
        for i in range(n_producers):
            r2 = np.random.rand()
            if r2 < ST:
                pop[i] = pop[i] * np.exp(-(i + 1) / (np.random.rand() * SSA_ITER + 1e-12))
            else:
                pop[i] = pop[i] + np.random.normal(0, 1, size=dim)

        # Scrounger update
        for i in range(n_producers, SSA_POP):
            A = np.random.choice([-1, 1], size=dim)
            if i > SSA_POP / 2:
                pop[i] = np.random.normal(0, 1, size=dim) * np.exp((worst_x - pop[i]) / ((i + 1) ** 2))
            else:
                pop[i] = best_x + np.abs(pop[i] - best_x) * A

        # Aware sparrows update
        aware_idx = np.random.choice(SSA_POP, size=n_aware, replace=False)
        for j in aware_idx:
            if fitness[j] > best_fit:
                beta = np.random.normal(0, 1, size=dim)
                pop[j] = best_x + beta * np.abs(pop[j] - best_x)
            else:
                K = np.random.uniform(-1, 1)
                pop[j] = pop[j] + K * (np.abs(pop[j] - worst_x) / (fitness[j] - worst_fit + 1e-12))

        pop = clip_pop(pop)

        # Re-evaluate
        for i in range(SSA_POP):
            params = decode_position(pop[i])
            fitness[i] = fit_lstm_and_eval_val_rmse(params)

    # Final best
    order = np.argsort(fitness)
    pop = pop[order]
    fitness = fitness[order]
    best_params = decode_position(pop[0])
    best_fitness = float(fitness[0])

    return best_params, best_fitness, pd.DataFrame(iter_logs), pd.DataFrame(pop_logs)


# ======================
# 9) 最终训练 + 保存 checkpoint（权重+scaler+meta）
# ======================
def train_final_and_eval_and_save(best_params: dict,
                                  checkpoint_path="best_SSA_LSTM_checkpoint_manual_split.pth"):
    set_all_seeds(SEED)

    lookback = best_params["lookback"]
    Xtr_raw, ytr, Xva_raw, yva, Xte_raw, yte, meta_tr, meta_va, meta_te, overlap_info = get_data_manual(lookback)
    n_feat = Xtr_raw.shape[-1]

    # 标准化：只用Train拟合（保存scaler参数）
    scaler = StandardScaler()
    Xtr_2d = Xtr_raw.reshape(-1, n_feat)
    Xva_2d = Xva_raw.reshape(-1, n_feat)
    Xte_2d = Xte_raw.reshape(-1, n_feat)

    scaler.fit(Xtr_2d)
    Xtr = scaler.transform(Xtr_2d).reshape(Xtr_raw.shape)
    Xva = scaler.transform(Xva_2d).reshape(Xva_raw.shape)
    Xte = scaler.transform(Xte_2d).reshape(Xte_raw.shape)

    model = LSTMRegressor(
        input_size=n_feat,
        hidden_size=best_params["hidden_size"],
        num_layers=best_params["num_layers"],
        dropout=best_params["dropout"]
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"])
    criterion = nn.MSELoss()

    train_ds = TensorDataset(
        torch.tensor(Xtr, dtype=torch.float32),
        torch.tensor(ytr, dtype=torch.float32).view(-1, 1)
    )
    val_ds = TensorDataset(
        torch.tensor(Xva, dtype=torch.float32),
        torch.tensor(yva, dtype=torch.float32).view(-1, 1)
    )

    g = torch.Generator()
    g.manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=best_params["batch_size"], shuffle=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=best_params["batch_size"], shuffle=False)

    best_val = np.inf
    pat = 0
    best_state = None

    for _ in range(EPOCHS_FINAL):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

        # validation RMSE for early stopping
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
            if pat >= PATIENCE_FINAL:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    def predict(Xnp):
        model.eval()
        with torch.no_grad():
            xb = torch.tensor(Xnp, dtype=torch.float32).to(DEVICE)
            return model(xb).detach().cpu().numpy().reshape(-1)

    yhat_tr = predict(Xtr)
    yhat_va = predict(Xva)
    yhat_te = predict(Xte)

    def metrics(y_true, y_pred):
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))
        return rmse, mae, r2

    rmse_tr, mae_tr, r2_tr = metrics(ytr, yhat_tr)
    rmse_va, mae_va, r2_va = metrics(yva, yhat_va)
    rmse_te, mae_te, r2_te = metrics(yte, yhat_te)

    metrics_df = pd.DataFrame([
        {"Split": "Train", "RMSE": rmse_tr, "MAE": mae_tr, "R2": r2_tr},
        {"Split": "Validation", "RMSE": rmse_va, "MAE": mae_va, "R2": r2_va},
        {"Split": "Test", "RMSE": rmse_te, "MAE": mae_te, "R2": r2_te},
    ])

    # ===== 保存 checkpoint：权重 + scaler + 预处理信息 =====
    ckpt = {
        "state_dict": model.state_dict(),
        "best_params": best_params,
        "feature_names": FEATURES_CN,
        "target_col": TARGET_COL,
        "area_col": AREA_COL,
        "year_col": YEAR_COL,
        "sheet_used": sheet_name,
        "seed": SEED,
        "manual_split_target_years": {
            "train": TRAIN_TARGET_YEARS,
            "val": VAL_TARGET_YEARS,
            "test": TEST_TARGET_YEARS
        },
        "overlap_check_config": {
            "check_input_year_overlap": CHECK_INPUT_YEAR_OVERLAP,
            "strict_no_input_year_overlap": STRICT_NO_INPUT_YEAR_OVERLAP
        },
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "fillna_means": fillna_means.values,  # 与 FEATURES_CN 对齐
    }
    torch.save(ckpt, checkpoint_path)
    print(f"[SAVE] Best model checkpoint saved: {checkpoint_path}")

    # 小型权重摘要（避免写全部权重到Excel过大）
    weights_summary = []
    total_params = 0
    for name, p in model.state_dict().items():
        numel = int(p.numel())
        total_params += numel
        weights_summary.append({
            "param_name": name,
            "shape": str(tuple(p.shape)),
            "num_params": numel
        })
    weights_summary_df = pd.DataFrame(weights_summary)
    total_params_df = pd.DataFrame([{"total_params": total_params}])

    # 可选输出切分元信息（便于审稿回复/自查）
    split_meta_df = pd.concat([
        meta_tr.assign(split="train"),
        meta_va.assign(split="val"),
        meta_te.assign(split="test")
    ], axis=0, ignore_index=True)

    overlap_rows = []
    if overlap_info:
        overlap_rows.append({
            "lookback": lookback,
            "train_target_years": str(TRAIN_TARGET_YEARS),
            "val_target_years": str(VAL_TARGET_YEARS),
            "test_target_years": str(TEST_TARGET_YEARS),
            "overlap_train_val": str(overlap_info.get("ov_train_val", [])),
            "overlap_train_test": str(overlap_info.get("ov_train_test", [])),
            "overlap_val_test": str(overlap_info.get("ov_val_test", [])),
        })
    overlap_df = pd.DataFrame(overlap_rows)

    return metrics_df, weights_summary_df, total_params_df, split_meta_df, overlap_df


# ======================
# 10) 下次：直接加载权重评估（不训练 -> 结果一致）
# ======================
def evaluate_from_checkpoint(checkpoint_path: str, run_on_cpu: bool = True):
    """
    只做推理/评估，不训练。
    只要数据、切分、预处理一致，输出会与保存时一致（建议CPU获得最稳定一致性）。
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    best_params = ckpt["best_params"]
    lookback = best_params["lookback"]

    # 使用当前脚本中的手动年份配置重建数据（需与你保存checkpoint时一致）
    Xtr_raw, ytr, Xva_raw, yva, Xte_raw, yte, _, _, _, _ = get_data_manual(lookback)
    n_feat = Xtr_raw.shape[-1]

    # 用保存的scaler参数重建标准化
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(ckpt["scaler_mean"], dtype=float)
    scaler.scale_ = np.asarray(ckpt["scaler_scale"], dtype=float)
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = n_feat

    Xtr = scaler.transform(Xtr_raw.reshape(-1, n_feat)).reshape(Xtr_raw.shape)
    Xva = scaler.transform(Xva_raw.reshape(-1, n_feat)).reshape(Xva_raw.shape)
    Xte = scaler.transform(Xte_raw.reshape(-1, n_feat)).reshape(Xte_raw.shape)

    # 重建模型并加载权重
    model = LSTMRegressor(
        input_size=n_feat,
        hidden_size=best_params["hidden_size"],
        num_layers=best_params["num_layers"],
        dropout=best_params["dropout"]
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    device = torch.device("cpu") if run_on_cpu else DEVICE
    model.to(device)

    def predict(Xnp):
        with torch.no_grad():
            xb = torch.tensor(Xnp, dtype=torch.float32).to(device)
            return model(xb).detach().cpu().numpy().reshape(-1)

    yhat_tr = predict(Xtr)
    yhat_va = predict(Xva)
    yhat_te = predict(Xte)

    def metrics(y_true, y_pred):
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))
        return rmse, mae, r2

    rmse_tr, mae_tr, r2_tr = metrics(ytr, yhat_tr)
    rmse_va, mae_va, r2_va = metrics(yva, yhat_va)
    rmse_te, mae_te, r2_te = metrics(yte, yhat_te)

    metrics_df = pd.DataFrame([
        {"Split": "Train", "RMSE": rmse_tr, "MAE": mae_tr, "R2": r2_tr},
        {"Split": "Validation", "RMSE": rmse_va, "MAE": mae_va, "R2": r2_va},
        {"Split": "Test", "RMSE": rmse_te, "MAE": mae_te, "R2": r2_te},
    ])
    return metrics_df


# ======================
# 11) 主程序
# ======================
if __name__ == "__main__":
    print("Start SSA optimization: fitness = Validation RMSE (Manual target-year split)...")
    print(f"Train target years: {TRAIN_TARGET_YEARS}")
    print(f"Val   target years: {VAL_TARGET_YEARS}")
    print(f"Test  target years: {TEST_TARGET_YEARS}")
    print(f"Overlap check: CHECK={CHECK_INPUT_YEAR_OVERLAP}, STRICT={STRICT_NO_INPUT_YEAR_OVERLAP}")

    best_params, best_fit, iter_df, pop_df = ssa_optimize()

    print("\n==================== Best SSA Params ====================")
    print(best_params)
    print(f"Best Fitness (Validation RMSE): {best_fit:.6f}")
    print("=========================================================\n")

    # 最终训练 + 保存权重
    ckpt_path = "best_SSA_LSTM_checkpoint_manual_split.pth"
    metrics_df, weights_summary_df, total_params_df, split_meta_df, overlap_df = train_final_and_eval_and_save(best_params, ckpt_path)
    print(metrics_df)

    # 输出到Excel
    out_xlsx = "SSA_LSTM_Results_manual_split.xlsx"
    best_params_df = pd.DataFrame([{
        **best_params,
        "best_fitness_val_RMSE": best_fit,
        "SSA_POP": SSA_POP,
        "SSA_ITER": SSA_ITER,
        "TRAIN_TARGET_YEARS": str(TRAIN_TARGET_YEARS),
        "VAL_TARGET_YEARS": str(VAL_TARGET_YEARS),
        "TEST_TARGET_YEARS": str(TEST_TARGET_YEARS),
        "CHECK_INPUT_YEAR_OVERLAP": CHECK_INPUT_YEAR_OVERLAP,
        "STRICT_NO_INPUT_YEAR_OVERLAP": STRICT_NO_INPUT_YEAR_OVERLAP,
        "EPOCHS_SSA": EPOCHS_SSA,
        "EPOCHS_FINAL": EPOCHS_FINAL,
        "SHEET_USED": sheet_name
    }])

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        metrics_df.to_excel(writer, sheet_name="Metrics", index=False)
        best_params_df.to_excel(writer, sheet_name="Best_Params", index=False)
        iter_df.to_excel(writer, sheet_name="SSA_Iteration", index=False)
        pop_df.to_excel(writer, sheet_name="SSA_Population", index=False)
        weights_summary_df.to_excel(writer, sheet_name="Weights_Summary", index=False)
        total_params_df.to_excel(writer, sheet_name="Weights_TotalParams", index=False)
        split_meta_df.to_excel(writer, sheet_name="Split_Meta", index=False)
        if len(overlap_df) > 0:
            overlap_df.to_excel(writer, sheet_name="Split_Overlap_Check", index=False)

    print(f"\nSaved Excel: {out_xlsx}")

    # ===== 演示：下次不训练，直接加载权重评估（应与上面Metrics一致）=====
    print("\n[CHECK] Evaluate from checkpoint (no training):")
    metrics_ckpt = evaluate_from_checkpoint(ckpt_path, run_on_cpu=True)
    print(metrics_ckpt)
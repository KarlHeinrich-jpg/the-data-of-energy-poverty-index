import pandas as pd
import numpy as np
import xgboost
import shap
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib
from matplotlib.lines import Line2D

# 设置全局字体与风格
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

# 配置：文件路径 + 列名
DATA_PATH = r"EPI_entropy_output - 副本 (2).xlsx"  # 数据文件路径
AREA_COL = "AREA"
YEAR_COL = "YEAR"
TARGET_COL = "EPI_entropy"

# 你给出的6个特征
FEATURE_SYM = ["GDP", "EI", "PGPC", "FCR", "UR", "SSI"]

# ======================
# 读取数据
# ======================
df = pd.read_excel(DATA_PATH)  # 读取数据

# 确保必要列存在
need_cols = [AREA_COL, YEAR_COL, TARGET_COL] + FEATURE_SYM
missing = [c for c in need_cols if c not in df.columns]
if missing:
    raise ValueError(f"数据缺少以下列：{missing}\n请检查表头是否一致（括号全角/半角、空格等）。")

data = df[need_cols].copy()
data[YEAR_COL] = pd.to_numeric(data[YEAR_COL], errors="coerce")
data[TARGET_COL] = pd.to_numeric(data[TARGET_COL], errors="coerce")
for c in FEATURE_SYM:
    data[c] = pd.to_numeric(data[c], errors="coerce")

data = data.dropna(subset=[YEAR_COL, TARGET_COL])
data[FEATURE_SYM] = data[FEATURE_SYM].fillna(data[FEATURE_SYM].mean())

# ======================
# 划分数据集
# ======================
X = data[FEATURE_SYM]  # 特征
y = data[TARGET_COL]  # 目标变量

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ======================
# 模型训练与超参数搜索
# ======================
param_grid = {'n_estimators': [100, 200, 300]}  # 超参数网格
xgb_model = xgboost.XGBRegressor(objective='reg:squarederror', random_state=42)  # 初始化XGBoost

grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=3, verbose=1, n_jobs=-1
)

grid_search.fit(X_train, y_train)  # 拟合模型
model = grid_search.best_estimator_  # 获取最佳模型实例

# ======================
# 计算SHAP值
# ======================
explainer = shap.TreeExplainer(model)  # 创建解释器
shap_values = explainer(X_test)  # 计算测试集的SHAP值

def simple_beeswarm(x_values, nbins=40, width=0.1):
    hist_range = (np.min(x_values), np.max(x_values))
    if hist_range[0] == hist_range[1]:
        hist_range = (hist_range[0] - 0.1, hist_range[1] + 0.1)
    counts, edges = np.histogram(x_values, bins=nbins, range=hist_range)
    bin_indices = np.digitize(x_values, edges) - 1
    bin_indices = np.clip(bin_indices, 0, nbins - 1)
    y_values = np.zeros_like(x_values)
    max_count = counts.max()
    if max_count == 0:
        return np.random.uniform(-0.1, 0.1, len(x_values))
    for i in range(len(counts)):
        idxs = np.where(bin_indices == i)[0]
        if len(idxs) == 0:
            continue
        current_width = (counts[i] / max_count) * width
        ys = np.linspace(-current_width, current_width, len(idxs))
        np.random.shuffle(ys)
        y_values[idxs] = ys
    return y_values

# ======================
# 配色方案：选择黄绿色（Viridis）
# ======================
color_schemes = {
    1: {'cmap': 'viridis', 'bar_color': '#d62728'},  # Viridis配色方案
}

selected_scheme = 1  # 配色方案为viridis

# ======================
# 计算阈值点（拟合线穿过 SHAP=0）
# ======================
def find_knee_point(x_data, y_data):
    z = np.polyfit(x_data, y_data, 3)  # 对数据进行3次多项式拟合，获取拟合系数
    p = np.poly1d(z)  # 根据系数生成多项式函数
    roots = p.roots  # 计算多项式方程的所有根（即拟合线穿过0线的x值）
    real_roots = roots[np.isreal(roots)].real  # 提取实数根并转为实数类型
    x_min, x_max = np.min(x_data), np.max(x_data)  # 获取x数据的最小值和最大值
    valid_roots = [r for r in real_roots if x_min <= r <= x_max]  # 筛选出在 x 数据范围内的根
    if len(valid_roots) > 0:
        return min(valid_roots)  # 返回最小的那个根（最左边的交叉点）
    else:
        return np.median(x_data)  # 若没有找到有效根，返回中位数

# ======================
# SHAP分析绘图函数（仅绘制依赖图）
# ======================
def plot_shap_analysis(shap_values_obj, X_data, y_data, target_name, scheme_idx):
    scheme = color_schemes[scheme_idx]
    current_cmap = plt.get_cmap(scheme['cmap'])
    bar_color = scheme['bar_color']

    # 创建画布和网格
    fig = plt.figure(figsize=(14, 9))
    gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.45, hspace=0.4)  # 修改为2行3列

    # 依赖图部分：右侧
    top_6_features = FEATURE_SYM  # 获取你指定的6个特征
    axes_scatter = []  # 初始化依赖图子轴列表
    for i in range(2):  # 遍历行
        for j in range(3):  # 遍历列
            axes_scatter.append(fig.add_subplot(gs[i, j]))  # 添加子图到网格右侧区域

    # 绘制每个依赖图
    for i, feature in enumerate(top_6_features):  # 遍历指定的6个特征
        ax = axes_scatter[i]  # 获取当前子图对象
        feature_idx = X_data.columns.get_loc(feature)  # 获取特征索引
        x_col_data = X_data[feature]  # 获取特征数据
        y_col_data = shap_values_obj.values[:, feature_idx]  # 获取 SHAP 值
        scatter = ax.scatter(x_col_data,
                             y_col_data,
                             c=y_data,
                             cmap=current_cmap,
                             s=25,
                             alpha=0.8)

        # SHAP=0 的水平参考线
        ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.7)

        # 计算并绘制 3 阶多项式拟合曲线
        z = np.polyfit(x_col_data, y_col_data, 3)
        p = np.poly1d(z)  # 生成多项式函数
        x_range = np.linspace(x_col_data.min(), x_col_data.max(), 100)
        ax.plot(x_range, p(x_range), color='black', alpha=0.4, linewidth=2)

        # 添加中位数与阈值线
        median_val = x_col_data.median()  # 计算中位数
        threshold_val = find_knee_point(x_col_data, y_col_data)  # 计算拟合线穿过 SHAP=0 的阈值点

        ax.axvline(median_val, color='black', linestyle='--', linewidth=1)  # 中位数线
        ax.axvline(threshold_val, color='red', linestyle=':', linewidth=1.2)  # 阈值线

        # 添加图例
        line_handles = [
            Line2D([0], [0], color='black', lw=1, linestyle='--', label=f'Median: {median_val:.2f}'),
            Line2D([0], [0], color='red', lw=1, linestyle=':', label=f'Threshold: {threshold_val:.2f}'),
            Line2D([0], [0], color='black', lw=2, linestyle='-', alpha=0.4, label='Trend Fit')
        ]
        ax.legend(handles=line_handles, loc='best', fontsize=14)

        ax.set_xlabel(feature, fontsize=16)
        ax.set_ylabel("SHAP", fontsize=14, labelpad=-8)

        # 设置刻度字体大小
        ax.tick_params(axis='both', which='major', labelsize=14)

    # 保存图像
    plt.tight_layout(rect=[0, 0.18, 1, 0.98])
    plt.savefig('shap_analysis_plot_with_polynomial_2rows_3cols_yellowgreen.png', dpi=300, bbox_inches='tight')
    plt.show()

# ======================
# 执行绘图
# ======================
plot_shap_analysis(shap_values_obj=shap_values, X_data=X_test, y_data=y_test, target_name=TARGET_COL,
                   scheme_idx=selected_scheme)

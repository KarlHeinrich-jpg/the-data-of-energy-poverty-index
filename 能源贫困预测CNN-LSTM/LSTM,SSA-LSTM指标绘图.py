# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import matplotlib.patheffects as path_effects

# ======================
# 1) 数据（直接用你的结果）
# ======================
models = ['SSA-LSTM', 'LSTM']

rmse_train = [0.021, 0.026]
rmse_test  = [0.027, 0.042]

mae_train  = [0.016, 0.019]
mae_test   = [0.022, 0.031]

r2_train   = [0.919, 0.916]
r2_test    = [0.722, 0.327]

# ======================
# 2) 风格设置
# ======================
plt.rcParams.update({
    "font.family": 'Times New Roman',
    "font.size": 24,
    "font.weight": 'bold',
    "mathtext.fontset": 'stix',
    "axes.linewidth": 3.5,
    "axes.axisbelow": True,
    "axes.labelweight": 'bold',
    "xtick.major.width": 3.0,
    "ytick.major.width": 3.0,
})

# ======================
# 3) 配色（换一组）
# ======================
viridis = cm.get_cmap('viridis')
train_color = viridis(0.25)
test_color  = viridis(0.85)

# ======================
# 4) 创建图形
# ======================
fig, axes = plt.subplots(1, 3, figsize=(24, 8), dpi=200)
fig.patch.set_facecolor('white')

bar_width = 0.30
x = np.arange(len(models))

for ax in axes:
    ax.set_facecolor('#FCFCFC')
    for spine in ax.spines.values():
        spine.set_linewidth(3.5)
        spine.set_color('#333333')

# ======================
# 5) RMSE
# ======================
axes[0].bar(x - bar_width/2, rmse_train, bar_width, label='Train',
            color=train_color, edgecolor='black', linewidth=2.0)
axes[0].bar(x + bar_width/2, rmse_test, bar_width, label='Test',
            color=test_color, edgecolor='black', linewidth=2.0)

axes[0].set_ylim(0, max(max(rmse_train), max(rmse_test)) * 1.25)
axes[0].set_title('RMSE Comparison', fontsize=30, pad=18)
axes[0].set_ylabel('RMSE', fontsize=28, labelpad=12)
axes[0].set_xticks(x)
axes[0].set_xticklabels(models, fontsize=26)
axes[0].tick_params(axis='y', labelsize=24, length=12, width=3.0, pad=10)
axes[0].grid(True, linestyle='--', alpha=0.4, color='#888888', linewidth=1.8)

rmse_offset = max(rmse_train + rmse_test) * 0.03
for i, v in enumerate(rmse_train):
    txt = axes[0].text(i - bar_width/2, v + rmse_offset, f'{v:.3f}',
                       ha='center', va='bottom', fontsize=22, fontweight='bold', color='black')
    txt.set_path_effects([path_effects.withStroke(linewidth=3, foreground='white')])
for i, v in enumerate(rmse_test):
    txt = axes[0].text(i + bar_width/2, v + rmse_offset, f'{v:.3f}',
                       ha='center', va='bottom', fontsize=22, fontweight='bold', color='black')
    txt.set_path_effects([path_effects.withStroke(linewidth=3, foreground='white')])

# ======================
# 6) MAE
# ======================
axes[1].bar(x - bar_width/2, mae_train, bar_width, label='Train',
            color=train_color, edgecolor='black', linewidth=2.0)
axes[1].bar(x + bar_width/2, mae_test, bar_width, label='Test',
            color=test_color, edgecolor='black', linewidth=2.0)

axes[1].set_ylim(0, max(max(mae_train), max(mae_test)) * 1.25)
axes[1].set_title('MAE Comparison', fontsize=30, pad=18)
axes[1].set_ylabel('MAE', fontsize=28, labelpad=12)
axes[1].set_xticks(x)
axes[1].set_xticklabels(models, fontsize=26)
axes[1].tick_params(axis='y', labelsize=24, length=12, width=3.0, pad=10)
axes[1].grid(True, linestyle='--', alpha=0.4, color='#888888', linewidth=1.8)

mae_offset = max(mae_train + mae_test) * 0.03
for i, v in enumerate(mae_train):
    txt = axes[1].text(i - bar_width/2, v + mae_offset, f'{v:.3f}',
                       ha='center', va='bottom', fontsize=22, fontweight='bold', color='black')
    txt.set_path_effects([path_effects.withStroke(linewidth=3, foreground='white')])
for i, v in enumerate(mae_test):
    txt = axes[1].text(i + bar_width/2, v + mae_offset, f'{v:.3f}',
                       ha='center', va='bottom', fontsize=22, fontweight='bold', color='black')
    txt.set_path_effects([path_effects.withStroke(linewidth=3, foreground='white')])

# ======================
# 7) R²
# ======================
axes[2].bar(x - bar_width/2, r2_train, bar_width, label='Train',
            color=train_color, edgecolor='black', linewidth=2.0)
axes[2].bar(x + bar_width/2, r2_test, bar_width, label='Test',
            color=test_color, edgecolor='black', linewidth=2.0)

axes[2].set_ylim(0, 1.05)
axes[2].set_title('$R^2$ Comparison', fontsize=30, pad=18)
axes[2].set_ylabel('$R^2$', fontsize=28, labelpad=12)
axes[2].set_xticks(x)
axes[2].set_xticklabels(models, fontsize=26)
axes[2].tick_params(axis='y', labelsize=24, length=12, width=3.0, pad=10)
axes[2].grid(True, linestyle='--', alpha=0.4, color='#888888', linewidth=1.8)

for i, v in enumerate(r2_train):
    txt = axes[2].text(i - bar_width/2, v + 0.02, f'{v:.3f}',
                       ha='center', va='bottom', fontsize=22, fontweight='bold', color='black')
    txt.set_path_effects([path_effects.withStroke(linewidth=3, foreground='white')])
for i, v in enumerate(r2_test):
    txt = axes[2].text(i + bar_width/2, v + 0.02, f'{v:.3f}',
                       ha='center', va='bottom', fontsize=22, fontweight='bold', color='black')
    txt.set_path_effects([path_effects.withStroke(linewidth=3, foreground='white')])

# ======================
# 8) 图例（关键：放到图外底部，不重叠）
# ======================
handles, labels = axes[0].get_legend_handles_labels()
leg = fig.legend(handles, labels,
                 loc='lower center',
                 ncol=2,
                 frameon=True,
                 bbox_to_anchor=(0.5, -0.00),  # ✅ 往下放
                 fontsize=20,
                 handlelength=4,
                 handleheight=2.5,
                 framealpha=0.95,
                 edgecolor='black',
                 facecolor='white',
                 borderpad=1.2,
                 columnspacing=3.5)
leg.get_frame().set_linewidth(2.5)

# ======================
# 9) 布局与保存（关键：给底部图例预留空间）
# ======================
plt.tight_layout(rect=[0, 0.18, 1, 0.98])  # ✅ 底部预留更大空间
plt.savefig("SSA_LSTM_vs_LSTM_Metrics.png", dpi=400, bbox_inches="tight")
plt.show()

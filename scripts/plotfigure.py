import matplotlib.pyplot as plt
import numpy as np

# 1. 准备横坐标 (样本数)
xtr = np.array([1, 2, 5, 10, 20, 50, 100])

# --- 你的实验数据 (MEAN) ---
baseline_mean = np.array([0.3508, 0.5080, 0.7438, 0.8323, 0.8065, 0.9016, 0.9087])
proposed_lp_mean = np.array([0.7194, 0.7793, 0.8084, 0.8234, 0.8364, 0.8453, 0.8502])
proposed_ft_mean = np.array([0.7458, 0.8297, 0.8844, 0.8941, 0.9123, 0.9228, 0.9277])

# --- 你的实验数据 (STD) ---
baseline_std = np.array([0.1387, 0.1279, 0.0754, 0.0333, 0.1661, 0.0047, 0.0055])
proposed_lp_std = np.array([0.0969, 0.0710, 0.0636, 0.0554, 0.0504, 0.0483, 0.0456])
proposed_ft_std = np.array([0.1105, 0.0650, 0.0165, 0.0133, 0.0046, 0.0044, 0.0035])

# --- 原论文对比数据 (根据截图右侧表格录入) ---
paper_baseline = np.array([0.535, 0.673, 0.792, 0.831, 0.906, 0.928, 0.931])
paper_lp = np.array([0.833, 0.857, 0.870, 0.877, 0.886, 0.893, 0.896])
paper_ft = np.array([0.847, 0.882, 0.902, 0.912, 0.925, 0.936, 0.941])

# 2. 开始绘图
plt.figure(figsize=(11, 7), dpi=120)

def plot_group(x, mean, std, p_data, label, color, line_marker, dot_marker):
    # A. 画你的结果：折线 + 阴影
    plt.plot(x, mean, label=f'Ours: {label}', color=color, marker=line_marker, linewidth=2, markersize=6)
    plt.fill_between(x, mean - std, mean + std, color=color, alpha=0.12)
    
    # B. 画论文的结果：散点 (使用对应颜色的不同标记)
    plt.scatter(x, p_data, color=color, marker=dot_marker, s=80, 
                label=f'Paper: {label}', edgecolors='black', linewidths=0.5, zorder=5)

# 绘制三组对比
plot_group(xtr, baseline_mean, baseline_std, paper_baseline, 'Baseline', '#7f8c8d', 'o', 'X')  # 灰色
plot_group(xtr, proposed_lp_mean, proposed_lp_std, paper_lp, 'Linear Probing', '#2980b9', 's', 'D') # 蓝色
plot_group(xtr, proposed_ft_mean, proposed_ft_std, paper_ft, 'Fine-tuning', '#e74c3c', '^', '*')    # 红色

# 3. 图表修饰
plt.xscale('log')
plt.xticks(xtr, xtr)
plt.ylim(0.2, 1.0)
plt.xlabel('Number of Training Samples ($X_{tr}$)', fontsize=12, fontweight='bold')
plt.ylabel('Dice Score', fontsize=12, fontweight='bold')
plt.title('Performance Comparison: Our Implementation vs. Original Paper', fontsize=14, pad=15)

# 添加网格和图例
plt.grid(True, which="both", ls="--", alpha=0.4)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, frameon=True)

plt.tight_layout()

plt.savefig('dice_comparison_results.png', dpi=300, bbox_inches='tight')
plt.show()

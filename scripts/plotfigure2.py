import matplotlib.pyplot as plt
import numpy as np

# 1. 准备数据 (根据上传的图片数据录入)
xtr = np.array([1, 2, 5, 10, 20, 50, 100])

# --- Attention Gate (AG) 数据: Mean 和 STD ---
ag_mean_ft = np.array([0.8123, 0.8536, 0.8940, 0.8956, 0.9153, 0.9247, 0.9298])
ag_std_ft  = np.array([0.0376, 0.0338, 0.0188, 0.0173, 0.0069, 0.0039, 0.0042])

ag_mean_lp = np.array([0.7791, 0.8059, 0.8341, 0.8389, 0.8528, 0.8626, 0.8642])
ag_std_lp  = np.array([0.0859, 0.0798, 0.0705, 0.0615, 0.0602, 0.0556, 0.0545])

# --- CBAM 数据: 单次 Run 结果 ---
cbam_ft = np.array([0.8437, 0.8843, 0.9033, 0.9150, 0.9195, 0.9296, 0.9311])
cbam_lp = np.array([0.8612, 0.8705, 0.8797, 0.8876, 0.8926, 0.8978, 0.8992])

# 2. 开始绘图
plt.figure(figsize=(11, 7), dpi=120)

def plot_with_range(x, ag_mean, ag_std, cbam_val, label, color_base):
    # 画 AG 的均值折线 (实线)
    plt.plot(x, ag_mean, label=f'AG {label} (Mean)', color=color_base, linewidth=2, marker='o')
    # 画 AG 的 STD 阴影区间
    plt.fill_between(x, ag_mean - ag_std, ag_mean + ag_std, color=color_base, alpha=0.15, label=f'AG {label} Range (±STD)')
    # 画 CBAM 的单次折线 (虚线)
    plt.plot(x, cbam_val, label=f'CBAM {label} (Single Run)', color=color_base, linestyle='--', marker='x', alpha=0.8)

# 绘制 Fine-tuning (FT) 组 - 红色系
plot_with_range(xtr, ag_mean_ft, ag_std_ft, cbam_ft, 'Fine-tuning', '#e74c3c')

# 绘制 Linear Probing (LP) 组 - 蓝色系
plot_with_range(xtr, ag_mean_lp, ag_std_lp, cbam_lp, 'Linear Probing', '#2980b9')

# 3. 图表修饰
plt.xscale('log')
plt.xticks(xtr, xtr)
plt.ylim(0.65, 1.0) # 根据数据范围调整起始点
plt.xlabel('Number of Training Samples ($X_{tr}$)', fontsize=12, fontweight='bold')
plt.ylabel('Dice Score', fontsize=12, fontweight='bold')
plt.title('Comparison: CBAM Performance within Attention Gate Statistical Range', fontsize=14, pad=15)

plt.grid(True, which="both", ls="--", alpha=0.3)
plt.legend(loc='lower right', fontsize=9, frameon=True, ncol=2)

plt.tight_layout()
plt.savefig('ag_vs_cbam_comparison.png', dpi=300)
plt.show()
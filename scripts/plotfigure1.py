import matplotlib.pyplot as plt
import numpy as np

# 1. 数据录入
xtr = np.array([1, 2, 5, 10, 20, 50, 100])
labels = ['1', '2', '5', '10', '20', '50', '100']

# --- No Attention (底端数据) ---
no_att_ft = np.array([0.7550, 0.8201, 0.8902, 0.8927, 0.9150, 0.9234, 0.9294])
no_att_lp = np.array([0.7314, 0.7885, 0.8269, 0.8369, 0.8498, 0.8582, 0.8611])
no_att_bs = np.array([0.2926, 0.5769, 0.7815, 0.8305, 0.8093, 0.8389, 0.9115])

# --- With Attention (顶端数据) ---
with_att_ft = np.array([0.8123, 0.8536, 0.8940, 0.8956, 0.9153, 0.9247, 0.9298])
with_att_lp = np.array([0.7791, 0.8059, 0.8341, 0.8389, 0.8528, 0.8626, 0.8642])
with_att_bs = np.array([0.3283, 0.5420, 0.7801, 0.8188, 0.8785, 0.9092, 0.9112])

# 2. 绘图设置
plt.figure(figsize=(12, 8), dpi=120)
colors = {'FT': '#e74c3c', 'LP': '#2980b9', 'BS': '#7f8c8d'}

def draw_diff_segments(x_indices, low, high, color, label, offset, group_key):
    x_coords = x_indices + offset
    # 画底端 (No Attention) 和顶端 (With Attention)
    plt.scatter(x_coords, low, color=color, marker='o', s=40, alpha=0.3)
    plt.scatter(x_coords, high, color=color, marker='^', s=80, label=label)
    
    # 画垂直增益线
    for i in range(len(x_indices)):
        plt.vlines(x_coords[i], low[i], high[i], colors=color, linewidth=2.5, alpha=0.7)
        
        # 在 Xtr=1 和 Xtr=50 的 Baseline 处标注 *
        if group_key == 'BS' and xtr[i] in [1, 50]:
            plt.text(x_coords[i], high[i] + 0.015, '*', ha='center', va='bottom', 
                     color='black', fontsize=16, fontweight='bold')

# 3. 绘制 Xtr 刻度之间的垂直隔离虚线
x_idx = np.arange(len(xtr))
for i in range(len(x_idx) - 1):
    # 在两个索引的中间位置画线
    mid_point = (x_idx[i] + x_idx[i+1]) / 2
    plt.axvline(x=mid_point, color='gray', linestyle=':', alpha=0.2, linewidth=1.5)

# 4. 绘制三组核心数据
draw_diff_segments(x_idx, no_att_bs, with_att_bs, colors['BS'], 'Baseline Gap', -0.2, 'BS')
draw_diff_segments(x_idx, no_att_lp, with_att_lp, colors['LP'], 'Linear Probing Gap', 0.0, 'LP')
draw_diff_segments(x_idx, no_att_ft, with_att_ft, colors['FT'], 'Proposed (Fine-tuning) Gap', 0.2, 'FT')

# 5. 图表完善
plt.xticks(x_idx, labels)
plt.xlabel('Number of Training Samples ($X_{tr}$)', fontsize=12, labelpad=10)
plt.ylabel('Dice Score', fontsize=12)
plt.title('Performance Gap: Attention Impact across $X_{tr}$ Samples', fontsize=14, pad=20)
plt.ylim(0.2, 1.05)
plt.grid(axis='y', ls='--', alpha=0.15) # 保留水平辅助线
plt.legend(loc='lower right', frameon=True, fontsize=10)

# 标注图表信息
plt.text(0, 0.22, '* supported by Wilcoxon Signed Rank test $p < 0.05$ ', 
         fontsize=9, style='italic', bbox=dict(facecolor='white', alpha=0.5))

plt.tight_layout()
# 自动保存
plt.savefig('attention_gap_segmented.png', dpi=300)
plt.show()
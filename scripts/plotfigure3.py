import matplotlib.pyplot as plt
import numpy as np

# 1. 准备横坐标 (训练样本数)
xtr = np.array([1, 2, 5, 10, 20, 50, 100])

# 2. 录入数据 (严格对应 image_6a0e1b.png 中的五列)
baseline = np.array([0.204787, 0.625860, 0.705163, 0.783891, 0.864405, 0.905403, 0.908972])
prop_lp  = np.array([0.773207, 0.789962, 0.824008, 0.836641, 0.846135, 0.858308, 0.860078])
prop_ft  = np.array([0.812115, 0.842301, 0.879014, 0.899784, 0.908765, 0.921762, 0.926952])
rand_lp  = np.array([0.764679, 0.784530, 0.813589, 0.832655, 0.844301, 0.847094, 0.853709])
rand_ft  = np.array([0.812254, 0.829822, 0.876437, 0.897298, 0.911452, 0.923532, 0.926414])
rand_fe  = np.array([0.802691, 0.830659, 0.874530, 0.894828, 0.907998, 0.916102, 0.921648])

# 3. 开始绘图
plt.figure(figsize=(10, 6.5), dpi=120)

# 绘制 Baseline
plt.plot(xtr, baseline, label='Baseline (Scratch)', color='#95a5a6', marker='o', linestyle=':', linewidth=2)

# 绘制 Proposed (使用实线)
plt.plot(xtr, prop_lp, label='Proposed Linear Probing', color='#2980b9', marker='s', linewidth=2)
plt.plot(xtr, prop_ft, label='Proposed (Fine-tuning)', color='#c0392b', marker='^', linewidth=2)

# 绘制 Randomized Attention (使用同色系的虚线，方便视觉归类)
plt.plot(xtr, rand_lp, label='Randomized Attn (LP)', color='#3498db', marker='s', linestyle='--', alpha=0.7)
plt.plot(xtr, rand_ft, label='Randomized Attn (FT)', color='#e74c3c', marker='^', linestyle='--', alpha=0.7)
plt.plot(xtr, rand_fe,  label='Randomized Attn (Frozen Encoder)', color="#de3ce7", marker='^', linestyle='--', alpha=0.7)

# 4. 图表细节优化
plt.xscale('log') # 使用对数坐标更清晰地展示小样本区间的巨大差异
plt.xticks(xtr, labels=['1', '2', '5', '10', '20', '50', '100'])
plt.xlabel('Number of Training Samples ($X_{tr}$)', fontsize=11)
plt.ylabel('Dice Score', fontsize=11)
plt.title('Ablation Study: Impact of Pre-trained vs. Randomized Attention', fontsize=13, pad=15)
plt.ylim(0.15, 1.0) # 考虑到 Baseline 的起始点

plt.grid(True, which="both", ls="--", alpha=0.3)
plt.legend(loc='lower right', fontsize=9)

plt.tight_layout()
# 保存并显示
plt.savefig('ablation_comparison_5_lines.png', dpi=300)
plt.show()
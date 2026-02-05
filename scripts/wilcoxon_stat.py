import pandas as pd
from scipy.stats import wilcoxon

# 1. 数据准备
df = pd.read_excel('results3.xlsx')
df[['Xtr', 'Method']] = df[['Xtr', 'Method']].ffill()

# 2. 全局配对分析 (忽略 Xtr，只分 Attention vs Non-attention)
# 确保每一行都是有效的配对
valid_pairs = df.dropna(subset=['Non-attention', 'Attention']).copy()

# 计算全局差值
valid_pairs['Improvement'] = valid_pairs['Attention'] - valid_pairs['Non-attention']

# 执行 Wilcoxon 符号秩检验
# 这将告诉我们：在所有实验条件下，Attention 是否显著改变了结果
stat, p_global = wilcoxon(valid_pairs['Attention'], valid_pairs['Non-attention'], alternative='greater')

print("=== 全局 Wilcoxon 配对检验结果 (所有 Xtr 汇总) ===")
print(f"总配对样本数 (N): {len(valid_pairs)}")
print(f"Non-Attention 总体均值: {valid_pairs['Non-attention'].mean():.4f}")
print(f"Attention 总体均值:     {valid_pairs['Attention'].mean():.4f}")
print(f"平均提升 (Mean Diff):   {valid_pairs['Improvement'].mean():.4f}")
print(f"P-Value:                {p_global:.6f}")
print(f"显著性结果:             {'显著 (Significant)' if p_global < 0.05 else '不显著'}")

# 3. 种子级分析 (分析哪些 Seed 表现最稳)
# 我们可以看每个 Seed 在不同 Xtr 下的表现均值
seed_summary = valid_pairs.groupby('Seed').agg({
    'Improvement': ['mean', 'std', 'count']
}).reset_index()

print("\n=== 种子级表现分析 (部分展示) ===")
print(seed_summary.head(10))
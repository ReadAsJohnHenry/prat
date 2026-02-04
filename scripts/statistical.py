import pandas as pd
from scipy.stats import mannwhitneyu

df = pd.read_excel('results2.xlsx')
df[['Xtr', 'Method']] = df[['Xtr', 'Method']].fillna(method='ffill')

summary_aggregate = []

# 只根据 Xtr 分组，不再区分 Method
for xtr, group in df.groupby('Xtr'):
    non_attn = group['Non-attention'].dropna().values
    attn = group['Attention'].dropna().values
    
    if len(non_attn) > 1:
        stat, p = mannwhitneyu(attn, non_attn, alternative='greater')
        summary_aggregate.append({
            'Xtr': xtr,
            'N_Total': len(non_attn) + len(attn),
            'Mean_Non': non_attn.mean(),
            'Mean_Attn': attn.mean(),
            'P-Value': p,
            'Significant': 'Yes' if p < 0.05 else 'No'
        })

print(pd.DataFrame(summary_aggregate).to_string())
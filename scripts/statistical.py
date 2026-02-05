import pandas as pd
from scipy.stats import wilcoxon
import numpy as np

# ---------------------------------------------------------
# 1. Setup and Data Cleaning
# ---------------------------------------------------------
file_path = 'results2.xlsx'
df = pd.read_excel(file_path)

# Forward fill to handle merged cells in Excel
df['Seed'] = df['Seed'].ffill()
df['Xtr'] = df['Xtr'].ffill()

# Calculate paired differences
valid_df = df.dropna(subset=['Non-attention', 'Attention']).copy()
valid_df['Improvement'] = valid_df['Attention'] - valid_df['Non-attention']

# ---------------------------------------------------------
# 2. Define the Wilcoxon Helper Function
# ---------------------------------------------------------
def perform_wilcoxon(data_sub):
    """
    Safely performs Wilcoxon signed-rank test.
    Null Hypothesis (H0): No difference between Attention and Non-attention.
    Alternative Hypothesis (H1): Attention > Non-attention.
    """
    if len(data_sub) < 2:
        return np.nan
    # If all values are identical, wilcoxon cannot compute ranks
    if (data_sub['Improvement'] == 0).all():
        return 1.0
    try:
        # One-sided test ('greater') as per senior's hypothesis
        _, p = wilcoxon(data_sub['Attention'], data_sub['Non-attention'], alternative='greater')
        return p
    except:
        return np.nan

# ---------------------------------------------------------
# 3. Multi-Level Statistical Analysis
# ---------------------------------------------------------

# --- A. Analysis PER METHOD (Combining all Xtr and Seeds) ---
method_audit = valid_df.groupby('Method').apply(lambda x: pd.Series({
    'N_Pairs': len(x),
    'Mean_Imp': x['Improvement'].mean(),
    'P_Value': perform_wilcoxon(x),
}), include_groups=False).reset_index()

# --- B. Analysis PER XTR (Combining all Methods and Seeds) ---
xtr_audit = valid_df.groupby('Xtr').apply(lambda x: pd.Series({
    'N_Pairs': len(x),
    'Mean_Imp': x['Improvement'].mean(),
    'P_Value': perform_wilcoxon(x),
}), include_groups=False).reset_index()

# --- C. Analysis PER (METHOD + XTR) CROSS-GROUP ---
# This addresses the senior's specific question about isolating conditions
detailed_cross_audit = []
for (method, xtr), group in valid_df.groupby(['Method', 'Xtr']):
    p_val = perform_wilcoxon(group)
    detailed_cross_audit.append({
        'Method': method,
        'Xtr': xtr,
        'N_Seeds': len(group),
        'Avg_Imp': group['Improvement'].mean(),
        'P_Value': p_val,
        'Significant': 'Yes' if p_val < 0.05 else 'No'
    })
df_cross = pd.DataFrame(detailed_cross_audit)

# --- D: SINGLE SEED ANALYSIS (Highest Number of Runs) --
per_seed_results = []
for seed, group in valid_df.groupby('Seed'):
    p_val = perform_wilcoxon(group)
    per_seed_results.append({
        'Seed': seed,
        'N_Pairs': len(group),
        'Avg_Non_Attn': group['Non-attention'].mean(),
        'Avg_Attn': group['Attention'].mean(),
        'Avg_Improvement': group['Improvement'].mean(),
        'Win_Rate': (group['Improvement'] > 0).mean(),
        'P_Value': p_val,
        'Significant': 'Yes' if p_val < 0.05 else 'No'
    })
df_per_seed = pd.DataFrame(per_seed_results)

# ---------------------------------------------------------
# 4. Save and Export
# ---------------------------------------------------------
with pd.ExcelWriter('refined_statistical_audit.xlsx') as writer:
    method_audit.to_excel(writer, sheet_name='By_Method', index=False)
    xtr_audit.to_excel(writer, sheet_name='By_Xtr', index=False)
    df_cross.to_excel(writer, sheet_name='Detailed_Cross_Analysis', index=False)
    df_per_seed.to_excel(writer, sheet_name='Single_Seed_Analysis', index=False)

print("=== STATS PER METHOD ===")
print(method_audit.to_string(index=False))
print("\n=== STATS PER XTR ===")
print(xtr_audit.to_string(index=False))
print("\n=== DETAILED CROSS ANALYSIS ===")
print(df_cross.to_string(index=False))
print("\n=== SINGLE SEED ANALYSIS ===")
print(df_per_seed.to_string(index=False))

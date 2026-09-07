import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

# 1. LOAD DATA
features_df = pd.read_csv('z_agg1.csv')
nfl_df = pd.read_excel('early_slope.xlsx') 
clinical_df = pd.read_excel('clinical_covariates.xlsx')

# 2. STANDARDIZE KEYS
def clean_code(code):
    if pd.isna(code): return code
    return code.replace('.', '_').split('_with_emission')[0]

for df in [features_df, nfl_df, clinical_df]:
    df['code_clean'] = df['code'].apply(clean_code)

# 3. MERGE
merged_df = pd.merge(nfl_df, clinical_df, on='code_clean', how='inner', suffixes=('', '_clinical'))
merged_df = pd.merge(merged_df, features_df, on='code_clean', how='inner', suffixes=('', '_features'))

target_col = 'Nfl Concentration Pg Per Ml'
merged_df[target_col] = pd.to_numeric(merged_df[target_col], errors='coerce')
merged_df['Sex_num'] = merged_df['Sex'].map({'Male': 0, 'Female': 1})
merged_df['class_num'] = merged_df['class'].map({'CTR': 0, 'ALS': 1})
merged_df = merged_df.dropna(subset=[target_col, 'Age Years', 'Sex_num', 'class_num'])

print(f"Final sample size: {len(merged_df)}")

# 4. FEATURE SELECTION
spectral_cols = [c for c in merged_df.columns if c.startswith('dim')]
correlations = merged_df[spectral_cols].corrwith(merged_df[target_col]).sort_values(ascending=False)

# --- KEY CHANGE: SELECT ABSOLUTE BEST FEATURE ---
best_feat = correlations.abs().idxmax()
best_corr = correlations[best_feat]

print(f"Strongest Predictor: {best_feat}")
print(f"Correlation: {best_corr:.4f}")

# 5. STATISTICAL MODELING
covariates = ['Age Years', 'Sex_num', 'class_num']
X_combined = sm.add_constant(merged_df[[best_feat] + covariates])
model_combined = sm.OLS(merged_df[target_col], X_combined).fit()

print(f"\n--- COMBINED MODEL ({best_feat} + Clinical) ---")
print(model_combined.summary().tables[1])

# 6. PLOT
plt.figure(figsize=(8, 6))
sns.regplot(data=merged_df, x=best_feat, y=target_col, color='red')
plt.title(f'NfL vs. {best_feat} (r={best_corr:.2f})')
plt.ylabel('NfL Concentration (pg/mL)')
plt.show()
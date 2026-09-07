import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

# 1. LOAD DATA
chirality_df = pd.read_csv('chirality_interp_descriptors.csv')
nfl_df = pd.read_excel('early_slope.xlsx') 
clinical_df = pd.read_excel('clinical_covariates.xlsx')

# 2. STANDARDIZE KEYS FOR MERGING
def clean_code(code):
    if pd.isna(code): return code
    # Standardize '1.3d' to '1_3d' and remove suffix
    cleaned = code.replace('.', '_').split('_with_emission')[0]
    return cleaned

for df in [chirality_df, nfl_df, clinical_df]:
    df['code_clean'] = df['code'].apply(clean_code)

# 3. MERGE DATASETS
merged_df = pd.merge(nfl_df, clinical_df, on='code_clean', how='inner', suffixes=('', '_clinical'))
merged_df = pd.merge(merged_df, chirality_df, on='code_clean', how='inner', suffixes=('', '_chirality'))

# --- KEY CHANGE: TARGET = TOTAL SCORE ---
target_col = 'Total Alsfrsr Score'
# Covariates: Age and Sex only. 
# (We don't use 'Early Slope' to predict current score, usually).
covariates = ['Age Years', 'Sex_num']

# --- FILTER FOR ALS PATIENTS ONLY ---
# We assess severity within the disease group. 
merged_df = merged_df[merged_df['class'] == 'ALS']
print(f"Filtered to ALS patients only. N = {len(merged_df)}")

# Clean Data
merged_df['Sex_num'] = merged_df['Sex'].map({'Male': 0, 'Female': 1})
merged_df[target_col] = pd.to_numeric(merged_df[target_col], errors='coerce')

# Drop samples with missing target or covariates
merged_df = merged_df.dropna(subset=[target_col] + covariates)
print(f"Final sample size for modeling: N = {len(merged_df)}")

# 4. FEATURE SELECTION (CORRELATION)
spectral_cols = [c for c in merged_df.columns if c.startswith('ch')]
correlations = merged_df[spectral_cols].corrwith(merged_df[target_col]).sort_values(ascending=False)

# Identify strongest feature (Absolute correlation)
top_feature = correlations.abs().idxmax()
top_corr_val = correlations[top_feature]

print(f"\nTop Spectral Predictor: {top_feature}")
print(f"Correlation (r): {top_corr_val:.3f}")

# 5. GENERATE PLOTS
plt.figure(figsize=(8, 6))
sns.regplot(data=merged_df, x=top_feature, y=target_col, color='green')
plt.title(f'ALSFRS-R Score vs. {top_feature}\n(ALS Patients Only, r={top_corr_val:.2f})')
plt.xlabel(top_feature)
plt.ylabel('Disability Score (ALSFRS-R)')
plt.show()

# 6. STATISTICAL MODELING (Statsmodels)

# Model 1: Clinical Demographics Only
X_clin = sm.add_constant(merged_df[covariates])
model_clin = sm.OLS(merged_df[target_col], X_clin).fit()

# Model 2: Clinical + Top Spectral Feature
X_combined = sm.add_constant(merged_df[[top_feature] + covariates])
model_combined = sm.OLS(merged_df[target_col], X_combined).fit()

print("\n--- MODEL 1: CLINICAL DEMOGRAPHICS ONLY ---")
print(model_clin.summary().tables[1])

print(f"\n--- MODEL 2: COMBINED ({top_feature} + Clinical) ---")
print(model_combined.summary().tables[1])

# Save results
merged_df.to_csv('processed_score_data.csv', index=False)
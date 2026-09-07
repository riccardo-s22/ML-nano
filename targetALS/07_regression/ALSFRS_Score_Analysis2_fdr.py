import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# 1. LOAD DATA
chirality_df = pd.read_csv('chirality_interp_descriptors.csv')
nfl_df = pd.read_excel('early_slope.xlsx') 
clinical_df = pd.read_excel('clinical_covariates.xlsx')

# 2. STANDARDIZE KEYS FOR MERGING
def clean_code(code):
    if pd.isna(code): return code
    cleaned = code.replace('.', '_').split('_with_emission')[0]
    return cleaned

for df in [chirality_df, nfl_df, clinical_df]:
    df['code_clean'] = df['code'].apply(clean_code)

# 3. MERGE DATASETS
merged_df = pd.merge(nfl_df, clinical_df, on='code_clean', how='inner', suffixes=('', '_clinical'))
merged_df = pd.merge(merged_df, chirality_df, on='code_clean', how='inner', suffixes=('', '_chirality'))

# --- DEFINE TARGET & COVARIATES ---
target_col = 'Total Alsfrsr Score'
covariates = ['Age Years', 'Sex_num']

# --- FILTER FOR ALS PATIENTS ONLY ---
merged_df = merged_df[merged_df['class'] == 'ALS']
merged_df['Sex_num'] = merged_df['Sex'].map({'Male': 0, 'Female': 1})
merged_df[target_col] = pd.to_numeric(merged_df[target_col], errors='coerce')
merged_df = merged_df.dropna(subset=[target_col] + covariates)

# 4. FEATURE SELECTION
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests

print("\n--- FEATURE SELECTION WITH FDR CORRECTION ---")

# 1. Filter valid spectral columns
# Changed 'ch' to 'dim' to match the columns in z_agg1.csv
spectral_cols = [c for c in merged_df.columns if c.startswith('dim') and merged_df[c].nunique() > 1]
correlations = []
p_values = []
valid_features = []

# 2. Calculate raw Pearson r and p-value for each feature
for col in spectral_cols:
    # Drop NaNs for the calculation to prevent errors
    valid_data = merged_df[[col, target_col]].dropna()
    
    if len(valid_data) > 2:  # Need at least 3 points for a correlation
        r, p = stats.pearsonr(valid_data[col], valid_data[target_col])
        correlations.append(r)
        p_values.append(p)
        valid_features.append(col)

# 3. Create a results DataFrame
fdr_df = pd.DataFrame({
    'Feature': valid_features,
    'Pearson_r': correlations,
    'Raw_p': p_values
})

# 4. Apply Benjamini-Hochberg FDR Correction
# multipletests returns a tuple; index 1 contains the corrected p-values
fdr_df['FDR_Corrected_p'] = multipletests(fdr_df['Raw_p'], alpha=0.05, method='fdr_bh')[1]

# 5. Sort by most significant (lowest FDR corrected p-value)
fdr_df = fdr_df.sort_values('FDR_Corrected_p')

print(fdr_df.head(10))

# Redefine top predictor for the rest of your script based on FDR significance
top_feature = fdr_df.iloc[0]['Feature']
top_corr_val = fdr_df.iloc[0]['Pearson_r']

print(f"\nTop Spectral Predictor (Post-FDR): {top_feature}")
print(f"Correlation (r): {top_corr_val:.3f}")
print(f"FDR Corrected p-value: {fdr_df.iloc[0]['FDR_Corrected_p']:.4e}")
# =========================================================
# --- NEW: HEATMAP OF TOP 10 FEATURES + CLINICAL DATA ---
# =========================================================
top_10_features = fdr_df['Feature'].head(10).tolist()
plot_cols = [target_col] + covariates + top_10_features
plt.figure(figsize=(12, 10))
corr_matrix = merged_df[plot_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", center=0)
plt.title(f'Correlation Heatmap: ALSFRS Score vs. Clinical & Top Spectral Features')
plt.tight_layout()
plt.show()
# =========================================================

# 5. REGRESSION PLOT
plt.figure(figsize=(8, 6))
sns.regplot(data=merged_df, x=top_feature, y=target_col, color='green')
# Use the top_corr_val variable that was already calculated and saved earlier
plt.title(f'ALSFRS Score vs. {top_feature}\n(r={top_corr_val:.2f})')
plt.show()

# 6. STATISTICAL MODELING
X_combined = sm.add_constant(merged_df[[top_feature] + covariates])
model_combined = sm.OLS(merged_df[target_col], X_combined).fit()
print(model_combined.summary().tables[1])
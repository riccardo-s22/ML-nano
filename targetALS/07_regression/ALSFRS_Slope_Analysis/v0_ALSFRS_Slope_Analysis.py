import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests

# 1. LOAD DATA
# Ensure this is pointed to your desired input file ('chirality_interp_descriptors.csv' or 'z_agg3.csv')
chirality_df = pd.read_csv('chirality_interp_descriptors.csv')
slope_df = pd.read_excel('early_slope.xlsx') 
clinical_df = pd.read_excel('clinical_covariates.xlsx')

# 2. STANDARDIZE KEYS FOR MERGING
def clean_code(code):
    if pd.isna(code): return code
    cleaned = str(code).replace('.', '_').split('_with_emission')[0]
    return cleaned

for df in [chirality_df, slope_df, clinical_df]:
    if 'code' in df.columns:
        df['code_clean'] = df['code'].apply(clean_code)

# 3. MERGE DATASETS
merged_df = pd.merge(slope_df, clinical_df, on='code_clean', how='inner', suffixes=('', '_clinical'))
merged_df = pd.merge(merged_df, chirality_df, on='code_clean', how='inner', suffixes=('', '_chirality'))

# --- DEFINE TARGET & COVARIATES ---
# IMPORTANT: Update this to match the exact column header for the slope in early_slope.xlsx
target_col = 'Total Alsfrsr Score'
covariates = ['Age Years', 'Sex_num', 'Early Slope (pts/mo)']

# --- FILTER FOR ALS PATIENTS ONLY ---
if 'class' in merged_df.columns:
    merged_df = merged_df[merged_df['class'] == 'ALS']
    
if 'Sex' in merged_df.columns:
    merged_df['Sex_num'] = merged_df['Sex'].map({'Male': 0, 'Female': 1})
    
merged_df[target_col] = pd.to_numeric(merged_df[target_col], errors='coerce')
merged_df = merged_df.dropna(subset=[target_col] + covariates)

print(f"Final sample size for modeling: N = {len(merged_df)}")

# 4. FEATURE SELECTION WITH FDR CORRECTION
print("\n--- FEATURE SELECTION WITH FDR CORRECTION ---")

# Automatically detect either 'ch' or 'dim' features and filter zero-variance columns
spectral_cols = [c for c in merged_df.columns if (c.startswith('ch') or c.startswith('dim')) and merged_df[c].nunique() > 1]

if len(spectral_cols) == 0:
    print("ERROR: No valid spectral features found. Check your CSV column headers.")
    exit()

correlations = []
p_values = []
valid_features = []

# Calculate raw Pearson r and p-value for each feature
for col in spectral_cols:
    valid_data = merged_df[[col, target_col]].dropna()
    
    if len(valid_data) > 2:  # Need at least 3 valid points to calculate a correlation
        r, p = stats.pearsonr(valid_data[col], valid_data[target_col])
        correlations.append(r)
        p_values.append(p)
        valid_features.append(col)

# Create the results DataFrame (Fixes the NameError)
fdr_df = pd.DataFrame({
    'Feature': valid_features,
    'Pearson_r': correlations,
    'Raw_p': p_values
})

# Apply Benjamini-Hochberg FDR Correction (Fixes the missing import error)
fdr_df['FDR_Corrected_p'] = multipletests(fdr_df['Raw_p'], alpha=0.05, method='fdr_bh')[1]

# Sort by most significant (lowest FDR corrected p-value)
fdr_df = fdr_df.sort_values('FDR_Corrected_p')
print(fdr_df.head(10))

# Define top predictor variables for plotting
top_feature = fdr_df.iloc[0]['Feature']
top_corr_val = fdr_df.iloc[0]['Pearson_r']
top_p_val = fdr_df.iloc[0]['FDR_Corrected_p']

print(f"\nTop Spectral Predictor: {top_feature}")
print(f"Correlation (r): {top_corr_val:.3f}")
print(f"FDR Corrected p-value: {top_p_val:.4e}")

# 5. HEATMAP OF TOP 10 FEATURES + CLINICAL DATA
top_10_features = fdr_df['Feature'].head(10).tolist()
plot_cols = [target_col] + covariates + top_10_features

plt.figure(figsize=(12, 10))
corr_matrix = merged_df[plot_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", center=0)
plt.title(f'Correlation Heatmap: {target_col} vs. Clinical & Top Spectral Features')
plt.tight_layout()
plt.show()

# 6. REGRESSION PLOT (Fixes the list string index error)
plt.figure(figsize=(8, 6))
sns.regplot(data=merged_df, x=top_feature, y=target_col, color='green')
plt.title(f'{target_col} vs. {top_feature}\n(r = {top_corr_val:.2f}, FDR p = {top_p_val:.3f})')
plt.xlabel(top_feature)
plt.ylabel(target_col)
plt.show()

# 7. STATISTICAL MODELING
print(f"\n--- OLS REGRESSION MODEL: {top_feature} + Clinical ---")
X_combined = sm.add_constant(merged_df[[top_feature] + covariates])
model_combined = sm.OLS(merged_df[target_col], X_combined).fit()
print(model_combined.summary().tables[1])
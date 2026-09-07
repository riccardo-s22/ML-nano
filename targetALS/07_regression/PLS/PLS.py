import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

# 1. LOAD DATA
chirality_df = pd.read_csv('chirality_interp_descriptors.csv')

# USE read_excel FOR XLSX FILES
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

# Clean missing values and types
target_col = 'Nfl Concentration Pg Per Ml'
merged_df[target_col] = pd.to_numeric(merged_df[target_col], errors='coerce')
merged_df = merged_df.dropna(subset=[target_col, 'Age Years', 'Sex', 'class'])

# Map categorical variables
merged_df['Sex_num'] = merged_df['Sex'].map({'Male': 0, 'Female': 1})
merged_df['class_num'] = merged_df['class'].map({'CTR': 0, 'ALS': 1})

import scipy.stats as stats
from statsmodels.stats.multitest import multipletests

print("\n--- FEATURE SELECTION WITH FDR CORRECTION ---")

# 1. Filter valid spectral columns
spectral_cols = [c for c in merged_df.columns if c.startswith('ch') and merged_df[c].nunique() > 1]

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
# 5. GENERATE PLOTS

# A. Correlation Heatmap (Top 10 Features + Clinical)
# Grab the top 10 features directly from the FDR-corrected dataframe
top_10_features = fdr_df['Feature'].head(10).tolist()
plot_cols = [target_col, 'Age Years', 'Sex_num', 'class_num'] + top_10_features
plt.figure(figsize=(12, 10))
sns.heatmap(merged_df[plot_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix: NfL vs. SWCNT Features')
plt.tight_layout()
plt.show()

# B. Regression: Top Spectral Feature vs NfL
plt.figure(figsize=(8, 6))
sns.regplot(data=merged_df, x=correlations.index[0], y=target_col, 
            scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
plt.title(f'NfL Prediction via {correlations.index[0]}')
plt.xlabel('Spectral Feature Intensity/Shift')
plt.ylabel('SIMOA NfL (pg/mL)')
plt.show()

# C. Boxplot: Feature by Clinical Class
plt.figure(figsize=(8, 6))
sns.boxplot(data=merged_df, x='class', y=correlations.index[0])
sns.stripplot(data=merged_df, x='class', y=correlations.index[0], color='black', alpha=0.3)
plt.title('Spectral Feature Distribution by Diagnosis')
plt.show()

# 6. STATISTICAL MODELING (Statsmodels)

# Model 1: Clinical Covariates Only
X_clin = sm.add_constant(merged_df[['Age Years', 'Sex_num', 'class_num']])
model_clin = sm.OLS(merged_df[target_col], X_clin).fit()

# Model 2: Clinical + Top Spectral Feature
X_combined = sm.add_constant(merged_df[[correlations.index[0], 'Age Years', 'Sex_num', 'class_num']])
model_combined = sm.OLS(merged_df[target_col], X_combined).fit()

print("\n--- CLINICAL ONLY MODEL ---")
print(model_clin.summary().tables[1])
print("\n--- COMBINED SPECTRAL + CLINICAL MODEL ---")
print(model_combined.summary().tables[1])

# 7. MULTIVARIATE PREDICTION (PLSR)
X_all = merged_df[spectral_cols + ['Age Years', 'Sex_num', 'class_num']]
y = merged_df[target_col]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)

# Leave-One-Out Cross-Validation
pls = PLSRegression(n_components=2)
y_pred = cross_val_predict(pls, X_scaled, y, cv=LeaveOneOut())

print(f"\nMultivariate PLSR R^2: {r2_score(y, y_pred):.3f}")

# Save the merged dataset for external use
merged_df.to_csv('processed_nfl_data.csv', index=False)
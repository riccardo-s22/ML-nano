import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# 1. LOAD DATA
# CHANGED: Load the new feature file
features_df = pd.read_csv('z_agg3.csv')
nfl_df = pd.read_excel('early_slope.xlsx') 
clinical_df = pd.read_excel('clinical_covariates.xlsx')

# 2. STANDARDIZE KEYS FOR MERGING
def clean_code(code):
    if pd.isna(code): return code
    # Standardize '1.3d' to '1_3d' and remove suffix
    cleaned = code.replace('.', '_').split('_with_emission')[0]
    return cleaned

for df in [features_df, nfl_df, clinical_df]:
    df['code_clean'] = df['code'].apply(clean_code)

# 3. MERGE DATASETS
# Merge target and clinical first
merged_df = pd.merge(nfl_df, clinical_df, on='code_clean', how='inner', suffixes=('', '_clinical'))
# Merge with new features
merged_df = pd.merge(merged_df, features_df, on='code_clean', how='inner', suffixes=('', '_features'))

# Define Target
target_col = 'Nfl Concentration Pg Per Ml'

# Clean Data
merged_df[target_col] = pd.to_numeric(merged_df[target_col], errors='coerce')
merged_df['Sex_num'] = merged_df['Sex'].map({'Male': 0, 'Female': 1})
merged_df['class_num'] = merged_df['class'].map({'CTR': 0, 'ALS': 1})

# Drop NaNs
merged_df = merged_df.dropna(subset=[target_col, 'Age Years', 'Sex_num', 'class_num'])
print(f"Final sample size: {len(merged_df)}")

# 4. FEATURE SELECTION (CORRELATION)
# CHANGED: Select columns starting with 'dim' (for new file) OR 'ch' (for old file)
spectral_cols = [c for c in merged_df.columns if c.startswith('dim') or c.startswith('ch')]

if len(spectral_cols) == 0:
    print("ERROR: No spectral features found! Check column names.")
else:
    correlations = merged_df[spectral_cols].corrwith(merged_df[target_col]).sort_values(ascending=False)

    print("Top Predictor:", correlations.index[0], "with r =", correlations.iloc[0])
    print("Top Negative Predictor:", correlations.index[-1], "with r =", correlations.iloc[-1])

    # 5. STATISTICAL MODELING (Top Feature + Clinical)
    top_feature = correlations.index[0]
    covariates = ['Age Years', 'Sex_num', 'class_num']
    
    X_combined = sm.add_constant(merged_df[[top_feature] + covariates])
    model_combined = sm.OLS(merged_df[target_col], X_combined).fit()
    
    print("\n--- COMBINED MODEL (Top Feature + Clinical) ---")
    print(model_combined.summary().tables[1])

    # 6. GENERATE PLOTS
    # Heatmap of Top 10 Features
    top_10 = correlations.abs().head(10).index.tolist()
    plot_cols = [target_col] + covariates + top_10
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(merged_df[plot_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap: New Features vs NfL')
    plt.show()
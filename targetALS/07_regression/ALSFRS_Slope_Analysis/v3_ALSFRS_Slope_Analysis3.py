import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# 1. LOAD DATA
chirality_df = pd.read_csv('z_agg3.csv')
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
target_col = 'Early Slope (pts/mo)'
covariates = ['Age Years', 'Sex_num', 'Total Alsfrsr Score']

# --- FILTER FOR ALS PATIENTS ONLY ---
merged_df = merged_df[merged_df['class'] == 'ALS']
merged_df['Sex_num'] = merged_df['Sex'].map({'Male': 0, 'Female': 1})
merged_df[target_col] = pd.to_numeric(merged_df[target_col], errors='coerce')
merged_df['Total Alsfrsr Score'] = pd.to_numeric(merged_df['Total Alsfrsr Score'], errors='coerce')
merged_df = merged_df.dropna(subset=[target_col] + covariates)

# 4. FEATURE SELECTION
# --- FIX: LOOK FOR 'dim' OR 'ch' ---
spectral_cols = [c for c in merged_df.columns if c.startswith('dim') or c.startswith('ch')]

if len(spectral_cols) == 0:
    print("ERROR: No spectral features found. Check your column headers.")
else:
    correlations = merged_df[spectral_cols].corrwith(merged_df[target_col]).sort_values(ascending=False)

    top_feature = correlations.abs().idxmax()
    print(f"Top Predictor: {top_feature} (r={correlations[top_feature]:.3f})")

    # =========================================================
    # --- HEATMAP OF TOP 10 FEATURES + CLINICAL DATA ---
    # =========================================================
    top_10_features = correlations.abs().head(10).index.tolist()
    # Define columns to plot: Target + Clinical Factors + Top 10 Spectral
    plot_cols = [target_col] + covariates + top_10_features

    plt.figure(figsize=(12, 10))
    corr_matrix = merged_df[plot_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", center=0)
    plt.title(f'Correlation Heatmap: ALSFRS Slope vs. Clinical & Top Spectral Features')
    plt.tight_layout()
    plt.show()
    # =========================================================

    # 5. REGRESSION PLOT (Top Feature Only)
    plt.figure(figsize=(8, 6))
    sns.regplot(data=merged_df, x=top_feature, y=target_col, color='purple')
    plt.title(f'ALSFRS Slope vs. {top_feature}\n(r={correlations[top_feature]:.2f})')
    plt.show()

    # 6. STATISTICAL MODELING
    X_combined = sm.add_constant(merged_df[[top_feature] + covariates])
    model_combined = sm.OLS(merged_df[target_col], X_combined).fit()
    
    print("\n--- COMBINED MODEL (Top Feature + Clinical) ---")
    print(model_combined.summary().tables[1])
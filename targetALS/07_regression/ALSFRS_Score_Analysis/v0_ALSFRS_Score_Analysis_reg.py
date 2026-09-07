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

from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ... [Keep your Steps 1-3 exactly as they are] ...

# 4. PREPARE DATA & FIX RUNTIME WARNING
print("\n--- PREPARING DATA ---")
raw_spectral_cols = [c for c in merged_df.columns if c.startswith('ch')]

# Filter out zero-variance columns to avoid divide-by-zero errors
spectral_cols = [c for c in raw_spectral_cols if merged_df[c].nunique() > 1]
print(f"Dropped {len(raw_spectral_cols) - len(spectral_cols)} zero-variance spectral features.")

# Define features (X) and target (y)
# (Assuming 'covariates' is defined earlier in your script, e.g., ['Age Years', 'Sex_num'])
X_raw = merged_df[covariates + spectral_cols]
y = merged_df[target_col]

# Standardize features (Mandatory for Regularization)
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_raw), columns=X_raw.columns, index=X_raw.index)

# 5. REGULARIZED FEATURE SELECTION (LassoCV)
print("\n--- FITTING LASSO MODEL ---")

# LassoCV automatically tests different penalty strengths (alphas) using cross-validation
cv_strategy = LeaveOneOut()

# Fit the model with L1 penalty to shrink uninformative SWCNT features to exactly zero
lasso = LassoCV(cv=cv_strategy, random_state=42, max_iter=10000)
lasso.fit(X_scaled, y)

print(f"Optimal Alpha (Regularization Strength): {lasso.alpha_:.4f}")

# Extract and rank the non-zero coefficients selected by the model
coef_series = pd.Series(lasso.coef_, index=X_scaled.columns)
selected_features = coef_series[coef_series != 0].sort_values(key=abs, ascending=False)

print(f"\nFeatures selected by Lasso (out of {len(X_scaled.columns)} input features): {len(selected_features)}")
print(selected_features)

# 6. LEAVE-ONE-OUT CROSS VALIDATION (LOOCV) PERFORMANCE
print("\n--- LOOCV EVALUATION ---")

# Generate cross-validated predictions: trains on N-1 patients, predicts the 1 left out
y_pred_cv = cross_val_predict(lasso, X_scaled, y, cv=cv_strategy)

# Calculate generalization metrics
mse_cv = mean_squared_error(y, y_pred_cv)
rmse_cv = np.sqrt(mse_cv)
r2_cv = r2_score(y, y_pred_cv) 

print(f"LOOCV RMSE (Error in ALSFRS-R points): {rmse_cv:.2f}")
print(f"LOOCV R-squared: {r2_cv:.2f}")

# 7. VISUALIZE PREDICTIONS
plt.figure(figsize=(8, 6))
sns.regplot(x=y, y=y_pred_cv, scatter_kws={'color': 'blue'}, line_kws={'color': 'red'})
plt.title(f'LOOCV Predictions vs Actual ALSFRS-R\n(Lasso Regression, N={len(y)})')
plt.xlabel('Actual ALSFRS-R Score')
plt.ylabel('Predicted ALSFRS-R Score (Cross-Validated)')

# Add a 1:1 Identity line to show perfect prediction
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=1, label='Perfect Prediction (1:1)')
plt.legend()
plt.show()
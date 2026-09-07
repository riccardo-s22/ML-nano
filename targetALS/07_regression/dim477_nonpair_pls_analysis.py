import os
import re
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score

base = r"C:\Users\riccardo-s\Documents\CNT\targetALS\PLS"

z = pd.read_csv(os.path.join(base, "z_agg3.csv"))
chir = pd.read_csv(os.path.join(base, "chirality_interp_descriptors.csv"))
nfl = pd.read_excel(os.path.join(base, "early_slope.xlsx"))

def norm_code(x):
    return str(x).replace(".", "_")

def count_chiralities(feature_name: str):
    return len(set(re.findall(r"ch(\d+_\d+)", feature_name)))

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def compute_vip(pls_model):
    T = pls_model.x_scores_
    W = pls_model.x_weights_
    Q = pls_model.y_loadings_
    p, h = W.shape
    s = np.diag(T.T @ T @ Q.T @ Q).reshape(h, -1)
    total_s = np.sum(s)
    vip = np.zeros((p,))
    for i in range(p):
        weight = np.array([(W[i, j] ** 2) / np.sum(W[:, j] ** 2) for j in range(h)])
        vip[i] = np.sqrt(p * np.sum(s.flatten() * weight) / total_s)
    return vip

z["code_norm"] = z["code"].map(norm_code)
nfl["code_norm"] = nfl["code"].map(norm_code)
chir["code_norm"] = chir["code"].astype(str)

merged = (
    z[["code_norm", "dim477"]]
    .merge(nfl[["code_norm", "Nfl Concentration Pg Per Ml"]], on="code_norm", how="inner")
    .merge(chir, on="code_norm", how="inner")
)

all_desc = [c for c in chir.columns if c not in {"code", "code_norm"}]
nonpair = [c for c in all_desc if count_chiralities(c) == 1]

df = merged[["code_norm", "dim477", "Nfl Concentration Pg Per Ml"] + nonpair].dropna(subset=["dim477", "Nfl Concentration Pg Per Ml"]).reset_index(drop=True)

usable = []
for c in nonpair:
    s = df[c]
    if s.notna().sum() >= 6 and s.dropna().nunique() > 1:
        usable.append(c)

X_df = df[usable].apply(lambda s: s.fillna(s.median()), axis=0)
y = df["Nfl Concentration Pg Per Ml"].to_numpy(dtype=float)
dim = df["dim477"].to_numpy(dtype=float)

loo = LeaveOneOut()
max_comp = min(10, X_df.shape[0] - 2, X_df.shape[1])
rows = []
for n_comp in range(1, max_comp + 1):
    pipe = Pipeline([("scaler", StandardScaler()), ("pls", PLSRegression(n_components=n_comp))])
    y_cv = cross_val_predict(pipe, X_df, y, cv=loo).ravel()
    rows.append((n_comp, r2_score(y, y_cv), rmse(y, y_cv), pearsonr(y, y_cv)[0], spearmanr(y, y_cv)[0]))

cv = pd.DataFrame(rows, columns=["n_components", "cv_r2", "cv_rmse", "cv_pearson_r", "cv_spearman_rho"])
best_n = int(cv.sort_values(["cv_r2", "cv_pearson_r"], ascending=False).iloc[0]["n_components"])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_df)
pls = PLSRegression(n_components=best_n)
pls.fit(X_scaled, y)

y_cv_best = cross_val_predict(Pipeline([("scaler", StandardScaler()), ("pls", PLSRegression(n_components=best_n))]), X_df, y, cv=loo).ravel()
T = pls.x_scores_

print("Best n_components:", best_n)
print("LOOCV R2:", cv.loc[cv["n_components"] == best_n, "cv_r2"].iloc[0])
print("dim477 vs PLS t1 Pearson r:", pearsonr(dim, T[:, 0])[0])
print("dim477 vs LOOCV-predicted NFL Pearson r:", pearsonr(dim, y_cv_best)[0])

vip = compute_vip(pls)
vip_df = pd.DataFrame({"feature": X_df.columns, "vip": vip, "coef": pls.coef_.ravel()}).sort_values("vip", ascending=False)
print(vip_df.head(10).to_string(index=False))

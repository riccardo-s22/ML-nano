import numpy as np
import pandas as pd

def common_mode_subtract_timepoint(X):  # X: (n_samples, n_chiralities)
    cm = X.mean(axis=1, keepdims=True)
    return X - cm

for t in ["0h","6h","24h"]:
    df = pd.read_csv(f"features_sum5x5_{t}.csv")
    chir_cols = [c for c in df.columns if c != "sample_id"]
    X = df[chir_cols].astype(float).to_numpy()

    R_raw = pd.DataFrame(X, columns=chir_cols).corr(method="pearson")
    X_res = common_mode_subtract_timepoint(X)
    R_res = pd.DataFrame(X_res, columns=chir_cols).corr(method="pearson")

    R_raw.to_csv(f"corr_raw_{t}.csv")
    R_res.to_csv(f"corr_residual_{t}.csv")

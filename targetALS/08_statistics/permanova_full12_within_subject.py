import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# ----------------------------
# SETTINGS
# ----------------------------
DATA_DIR = r"C:\Users\riccardo-s\Documents\CNT\targetALS\exp5\multiplexing_results"
FILES = {
    "0h":  "features_max5x5_0h.csv",
    "6h":  "features_max5x5_6h.csv",
    "24h": "features_max5x5_24h.csv",
}
TIMEPOINTS = ["0h", "6h", "24h"]
N_PERM = 9999
SEED_GLOBAL = 42
SEED_PAIRWISE = 123
OUT_DIR = os.path.join(DATA_DIR, "permanova_full12_out")
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------
# HELPERS
# ----------------------------
def load_features(path):
    df = pd.read_csv(path)
    if "sample_id" not in df.columns:
        raise ValueError(f"Missing sample_id in {path}")
    chir_cols = [c for c in df.columns if c != "sample_id"]
    return df, chir_cols

def holm_adjust(pvals):
    pvals = np.array(pvals, float)
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    prev = 0.0
    for i, idx in enumerate(order):
        val = min(1.0, (m - i) * pvals[idx])
        prev = max(prev, val)
        adj[idx] = prev
    return adj

def permanova_F_R2_repeated(Tz):
    """
    Tz shape: (N, k, p) where
      N = number of subjects/samples
      k = number of repeated conditions (timepoints = 3)
      p = features (12)
    Groups are defined by axis=1 (timepoints). Uses Euclidean sums-of-squares.
    """
    N, k, p = Tz.shape
    grand = Tz.reshape(-1, p).mean(axis=0)
    ss_total = ((Tz.reshape(-1, p) - grand) ** 2).sum()

    cent = Tz.mean(axis=0)  # (k, p)
    ss_within = ((Tz - cent[None, :, :]) ** 2).sum()
    ss_between = ss_total - ss_within

    df_between = k - 1
    df_within = N * k - k

    F = (ss_between / df_between) / (ss_within / df_within)
    R2 = ss_between / ss_total if ss_total > 0 else np.nan
    return float(F), float(R2), float(ss_between), float(ss_within), float(ss_total), int(df_between), int(df_within)

def within_subject_permutation_F(Tz, n_perm=9999, seed=0):
    """
    Permute labels within each subject by permuting the timepoint axis (k=3).
    Returns p-value based on pseudo-F.
    """
    rng = np.random.default_rng(seed)
    N = Tz.shape[0]

    perms3 = np.array([
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ], dtype=int)

    F_obs, R2_obs, ss_b, ss_w, ss_t, df_b, df_w = permanova_F_R2_repeated(Tz)

    F_perm = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        choice = rng.integers(0, 6, size=N)
        idx = perms3[choice]                    # (N,3)
        Tp = Tz[np.arange(N)[:, None], idx, :]  # permute time axis per subject
        F_perm[i], *_ = permanova_F_R2_repeated(Tp)

    p = (1 + np.sum(F_perm >= F_obs)) / (1 + n_perm)

    return {
        "F": F_obs,
        "R2": R2_obs,
        "p": float(p),
        "n_perm": int(n_perm),
        "df_between": df_b,
        "df_within": df_w,
        "F_perm": F_perm
    }

def pairwise_permanova_repeated(Tz, a_idx, b_idx, n_perm=9999, seed=0):
    """
    Pairwise repeated-measures PERMANOVA between two timepoints.
    Permutation is swapping the two conditions within each subject.
    """
    rng = np.random.default_rng(seed)
    Tp = Tz[:, [a_idx, b_idx], :]  # (N,2,p)
    N, k, p = Tp.shape

    def F_R2_2(Tp2):
        grand = Tp2.reshape(-1, p).mean(axis=0)
        ss_total = ((Tp2.reshape(-1, p) - grand) ** 2).sum()
        cent = Tp2.mean(axis=0)  # (2,p)
        ss_within = ((Tp2 - cent[None, :, :]) ** 2).sum()
        ss_between = ss_total - ss_within

        df_between = 1
        df_within = N * 2 - 2
        F = (ss_between / df_between) / (ss_within / df_within)
        R2 = ss_between / ss_total if ss_total > 0 else np.nan
        return float(F), float(R2)

    F_obs, R2_obs = F_R2_2(Tp)

    F_perm = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        swap = rng.integers(0, 2, size=N).astype(bool)
        Tpp = Tp.copy()
        Tpp[swap, 0, :], Tpp[swap, 1, :] = Tp[swap, 1, :], Tp[swap, 0, :]
        F_perm[i], _ = F_R2_2(Tpp)

    p = (1 + np.sum(F_perm >= F_obs)) / (1 + n_perm)
    return F_obs, R2_obs, float(p)

# ----------------------------
# LOAD + ALIGN + STANDARDIZE
# ----------------------------
dfs = {}
chir_cols = None
for tp in TIMEPOINTS:
    df, cols = load_features(os.path.join(DATA_DIR, FILES[tp]))
    dfs[tp] = df
    chir_cols = cols if chir_cols is None else chir_cols

common_ids = sorted(set(dfs["0h"]["sample_id"])
                    .intersection(dfs["6h"]["sample_id"])
                    .intersection(dfs["24h"]["sample_id"]))

for tp in TIMEPOINTS:
    dfs[tp] = dfs[tp].set_index("sample_id").loc[common_ids].reset_index()

X0 = dfs["0h"][chir_cols].astype(float).to_numpy()
X6 = dfs["6h"][chir_cols].astype(float).to_numpy()
X24 = dfs["24h"][chir_cols].astype(float).to_numpy()

T = np.stack([X0, X6, X24], axis=1)  # (N,3,12)

# Global standardization across all observations
scaler = StandardScaler()
Tz = scaler.fit_transform(T.reshape(-1, T.shape[-1])).reshape(T.shape)

# ----------------------------
# GLOBAL PERMANOVA
# ----------------------------
res = within_subject_permutation_F(Tz, n_perm=N_PERM, seed=SEED_GLOBAL)

global_df = pd.DataFrame([{
    "n_samples": Tz.shape[0],
    "n_observations": int(Tz.shape[0] * Tz.shape[1]),
    "n_features": int(Tz.shape[2]),
    "F": res["F"],
    "R2": res["R2"],
    "p": res["p"],
    "n_perm": res["n_perm"],
    "df_between": res["df_between"],
    "df_within": res["df_within"],
}])
global_df.to_csv(os.path.join(OUT_DIR, "permanova_global_full12.csv"), index=False)

# Null histogram
fig, ax = plt.subplots(figsize=(7.5, 5.5))
ax.hist(res["F_perm"], bins=40, alpha=0.8)
ax.axvline(res["F"], linewidth=2)
ax.set_title("PERMANOVA (within-subject) null distribution of F\nFull 12-chirality standardized features")
ax.set_xlabel("Pseudo-F under permutation")
ax.set_ylabel("Count")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "permanova_null_hist_full12.png"), dpi=300)
plt.close(fig)

# ----------------------------
# PAIRWISE PERMANOVA
# ----------------------------
tp_idx = {"0h": 0, "6h": 1, "24h": 2}
pairs = [("0h", "6h"), ("6h", "24h"), ("0h", "24h")]

rows = []
for a, b in pairs:
    Fp, R2p, pp = pairwise_permanova_repeated(
        Tz, tp_idx[a], tp_idx[b], n_perm=N_PERM, seed=SEED_PAIRWISE
    )
    rows.append({"comparison": f"{a} vs {b}", "F": Fp, "R2": R2p, "p": pp, "n_samples": Tz.shape[0]})

pair_df = pd.DataFrame(rows)
pair_df["p_holm"] = holm_adjust(pair_df["p"].values)
pair_df.to_csv(os.path.join(OUT_DIR, "permanova_pairwise_full12.csv"), index=False)

print("Done. Results saved in:", OUT_DIR)
print(global_df)
print(pair_df)

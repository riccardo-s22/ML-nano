import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.formula.api as smf
from patsy import build_design_matrices
import matplotlib.pyplot as plt

# -------------------------
# Settings
# -------------------------
FEATURE_FILES = {
    "0h":  "features_max5x5_0h.csv",
    "6h":  "features_max5x5_6h.csv",
    "24h": "features_max5x5_24h.csv",
}
LABELS_FILE = "sample_labels.csv"   # biological samples only (others = technical controls)

# Outputs (suffix is added automatically if COMMONMODE_REMOVE=True)
OUT_EMM_BASE   = "emm_by_chirality_time"
OUT_CONTR_BASE = "contrasts_by_chirality_fdr"
OUT_FIG_BASE   = "chirality_trajectories_emm_ci"

# Preprocessing choices
USE_LOG1P = True
WITHIN_CHIRALITY_ZSCORE = True      # keep consistent with your prior model
COMMONMODE_REMOVE = True            # <-- NEW: subtract within-(sample_id,time) mean across chiralities
ALPHA = 0.05

# -------------------------
# Helpers
# -------------------------
def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    adj = ranked * n / (np.arange(1, n + 1))
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out = np.empty_like(adj)
    out[order] = adj
    return out

def melt_features(df, time_label):
    chir_cols = [c for c in df.columns if c != "sample_id"]
    x = df.melt(
        id_vars=["sample_id"],
        value_vars=chir_cols,
        var_name="chirality",
        value_name="y"
    )
    x["time"] = time_label
    return x

def build_long_df(feature_files, keep_samples=None):
    dfs = []
    for t, path in feature_files.items():
        df = pd.read_csv(path)
        if keep_samples is not None:
            df = df[df["sample_id"].isin(keep_samples)].copy()
        dfs.append(melt_features(df, t))
    long = pd.concat(dfs, ignore_index=True)
    return long

def make_grid_over_samples(samples, chirality, time_levels):
    """Create grid with all samples for a fixed chirality, across time_levels."""
    return pd.DataFrame({
        "sample_id": np.repeat(samples, len(time_levels)),
        "chirality": chirality,
        "time": np.tile(time_levels, len(samples))
    })

def avg_design_row(ols_model, new_df):
    """
    Compute the average (across rows) design vector for new_df using Patsy design_info.
    Returns mean_x (p,), and optionally individual X for debugging.
    """
    design_info = ols_model.model.data.design_info
    X = build_design_matrices([design_info], new_df, return_type="dataframe")[0]
    mean_x = X.mean(axis=0).to_numpy()
    return mean_x, X

def estimate_from_x(mean_x, beta, cov, df_denom):
    """Delta-method mean + SE + 95% CI from linear predictor mean_x."""
    est = float(mean_x @ beta)
    var = float(mean_x @ cov @ mean_x)
    se = np.sqrt(max(var, 0.0))
    tcrit = st.t.ppf(0.975, df_denom)
    lo = est - tcrit * se
    hi = est + tcrit * se
    return est, se, lo, hi

# -------------------------
# Output names (avoid overwrite)
# -------------------------
suffix = "_commonmode" if COMMONMODE_REMOVE else ""
OUT_EMM   = f"{OUT_EMM_BASE}{suffix}.csv"
OUT_CONTR = f"{OUT_CONTR_BASE}{suffix}.csv"
OUT_FIG   = f"{OUT_FIG_BASE}{suffix}.png"

# -------------------------
# Load sample list (exclude technical controls)
# -------------------------
labels = pd.read_csv(LABELS_FILE)
keep_samples = labels["sample"].tolist() if "sample" in labels.columns else labels["sample_id"].tolist()

# -------------------------
# Build long df
# -------------------------
long = build_long_df(FEATURE_FILES, keep_samples=keep_samples)

# Ensure categorical with fixed ordering early (used by grouping too)
time_levels = ["0h", "6h", "24h"]
long["time"] = pd.Categorical(long["time"], categories=time_levels, ordered=True)
long["chirality"] = long["chirality"].astype("category")

# -------------------------
# Preprocess
# -------------------------
if USE_LOG1P:
    long["y"] = np.log1p(long["y"])

if COMMONMODE_REMOVE:
    # Remove within-(sample_id, time) mean across chiralities:
    # y*_{i,t,c} = y_{i,t,c} - (1/12) * sum_c y_{i,t,c}
    long["y"] = long["y"] - long.groupby(["sample_id", "time"], observed=False)["y"].transform("mean")

if WITHIN_CHIRALITY_ZSCORE:
    # z-score within chirality across all samples+times (as in your prior approach)
    long["y"] = long.groupby("chirality", observed=False)["y"].transform(
        lambda x: (x - x.mean()) / x.std(ddof=0)
    )

# -------------------------
# Fit within-subject model with cluster-robust SE by sample
# (sample fixed effects emulate a repeated-measures model)
# -------------------------
ols = smf.ols("y ~ time * chirality + C(sample_id)", data=long).fit()
ols_cr = ols.get_robustcov_results(cov_type="cluster", groups=long["sample_id"])

beta = np.asarray(ols_cr.params, dtype=float)
cov = np.asarray(ols_cr.cov_params(), dtype=float)

# Cluster df ~ (#clusters - 1)
n_clusters = long["sample_id"].nunique()
df_denom = n_clusters - 1
print(f"Using {n_clusters} samples (clusters); df_denom={df_denom}")

# -------------------------
# EMMs: predicted mean per chirality at each time, averaged over samples
# -------------------------
samples = sorted(long["sample_id"].unique())
chiralities = list(long["chirality"].cat.categories)

emm_rows = []
mean_x_cache = {}  # cache mean design vectors for contrasts

for chir in chiralities:
    for t in time_levels:
        grid = make_grid_over_samples(samples, chir, [t])
        grid["time"] = pd.Categorical(grid["time"], categories=time_levels, ordered=True)
        grid["chirality"] = pd.Categorical(grid["chirality"], categories=chiralities)
        mean_x, _ = avg_design_row(ols, grid)

        est, se, lo, hi = estimate_from_x(mean_x, beta, cov, df_denom)
        emm_rows.append({
            "chirality": chir,
            "time": t,
            "emm": est,
            "se": se,
            "ci95_lo": lo,
            "ci95_hi": hi,
        })
        mean_x_cache[(chir, t)] = mean_x

emm = pd.DataFrame(emm_rows)
emm.to_csv(OUT_EMM, index=False)
print(f"Wrote EMMs to {OUT_EMM}")

# -------------------------
# Planned contrasts per chirality, with BH-FDR
# -------------------------
contrasts = [
    ("6h-0h", "6h", "0h"),
    ("24h-6h", "24h", "6h"),
    ("24h-0h", "24h", "0h"),
]

contr_rows = []
pvals = []

for chir in chiralities:
    for name, t1, t0 in contrasts:
        x1 = mean_x_cache[(chir, t1)]
        x0 = mean_x_cache[(chir, t0)]
        d = x1 - x0

        est = float(d @ beta)
        var = float(d @ cov @ d)
        se = np.sqrt(max(var, 0.0))
        tstat = est / se if se > 0 else np.nan
        p = 2 * (1 - st.t.cdf(abs(tstat), df_denom)) if np.isfinite(tstat) else np.nan

        contr_rows.append({
            "chirality": chir,
            "contrast": name,
            "estimate": est,
            "se": se,
            "t": tstat,
            "p": p,
        })
        pvals.append(p)

contr_df = pd.DataFrame(contr_rows)
contr_df["p_fdr"] = bh_fdr(np.array(pvals, dtype=float))
contr_df["signif_fdr_0.05"] = contr_df["p_fdr"] < ALPHA
contr_df.to_csv(OUT_CONTR, index=False)
print(f"Wrote contrasts to {OUT_CONTR}")

# -------------------------
# Plot: per-chirality trajectories with 95% CI
# -------------------------
emm_plot = emm.copy()
emm_plot["time_num"] = emm_plot["time"].map({"0h": 0, "6h": 6, "24h": 24})

plt.figure(figsize=(10, 6))
for chir in chiralities:
    sub = emm_plot[emm_plot["chirality"] == chir].sort_values("time_num")
    x = sub["time_num"].to_numpy()
    y = sub["emm"].to_numpy()
    lo = sub["ci95_lo"].to_numpy()
    hi = sub["ci95_hi"].to_numpy()

    plt.plot(x, y, marker="o", linewidth=1)
    plt.fill_between(x, lo, hi, alpha=0.15)

plt.xticks([0, 6, 24], ["0h", "6h", "24h"])
plt.xlabel("Time")
plt.ylabel("EMM (model scale)")
title = "Chirality-specific temporal trajectories (EMMs ± 95% CI)"
if COMMONMODE_REMOVE:
    title += " [within-(sample,time) common-mode removed]"
if WITHIN_CHIRALITY_ZSCORE:
    title += " [within-chirality z-scored]"
elif USE_LOG1P:
    title += " [log1p scale]"
plt.title(title)
plt.tight_layout()
plt.savefig(OUT_FIG, dpi=300)
print(f"Wrote figure to {OUT_FIG}")

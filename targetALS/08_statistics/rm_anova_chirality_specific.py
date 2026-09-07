import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from statsmodels.stats.anova import AnovaRM

# -------- SETTINGS --------
DATA_DIR = r"C:\Users\riccardo-s\Documents\CNT\targetALS\exp5\multiplexing_results"
FILES = {
    "0h":  "features_max5x5_0h.csv",
    "6h":  "features_max5x5_6h.csv",
    "24h": "features_max5x5_24h.csv",
}
TIME_ORDER = ["0h", "6h", "24h"]
OUT_DIR = os.path.join(DATA_DIR, "rm_anova_chirality_out")
os.makedirs(OUT_DIR, exist_ok=True)

# -------- HELPERS --------
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

def bh_fdr(pvals):
    p = np.array(pvals, float)
    m = len(p)
    order = np.argsort(p)
    ranks = np.arange(1, m + 1)
    q = np.empty(m)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        idx = order[i]
        val = min(prev, p[idx] * m / ranks[i])
        prev = val
        q[idx] = val
    return q

def cohen_dz(diff):
    diff = np.asarray(diff, float)
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else np.nan

# -------- LOAD + ALIGN --------
f0 = pd.read_csv(os.path.join(DATA_DIR, FILES["0h"]))
f6 = pd.read_csv(os.path.join(DATA_DIR, FILES["6h"]))
f24 = pd.read_csv(os.path.join(DATA_DIR, FILES["24h"]))

chir_cols = [c for c in f0.columns if c != "sample_id"]

common_ids = sorted(set(f0["sample_id"]).intersection(f6["sample_id"]).intersection(f24["sample_id"]))
f0 = f0.set_index("sample_id").loc[common_ids].reset_index()
f6 = f6.set_index("sample_id").loc[common_ids].reset_index()
f24 = f24.set_index("sample_id").loc[common_ids].reset_index()

def make_long(df, tp):
    out = df[["sample_id"] + chir_cols].copy()
    out["timepoint"] = tp
    return out

long = pd.concat([make_long(f0, "0h"), make_long(f6, "6h"), make_long(f24, "24h")], ignore_index=True)
long["sample_id"] = long["sample_id"].astype(str)
long["timepoint"] = pd.Categorical(long["timepoint"], categories=TIME_ORDER, ordered=True)

wide = long.pivot(index="sample_id", columns="timepoint", values=chir_cols)

# -------- RM-ANOVA + POSTHOC --------
anova_rows = []
posthoc_rows = []

for c in chir_cols:
    dat = long[["sample_id", "timepoint", c]].rename(columns={c: "y"}).dropna()

    # RM-ANOVA
    aov = AnovaRM(dat, depvar="y", subject="sample_id", within=["timepoint"]).fit()
    tab = aov.anova_table
    F = float(tab.loc["timepoint", "F Value"])
    pval = float(tab.loc["timepoint", "Pr > F"])
    df_num = float(tab.loc["timepoint", "Num DF"])
    df_den = float(tab.loc["timepoint", "Den DF"])
    partial_eta2 = (F * df_num) / (F * df_num + df_den)

    # Friedman robustness check
    w = wide[c].loc[:, TIME_ORDER].dropna()
    fr = stats.friedmanchisquare(w["0h"].values, w["6h"].values, w["24h"].values)

    anova_rows.append({
        "chirality": c,
        "n": int(w.shape[0]),
        "rm_anova_F": F,
        "rm_anova_p": pval,
        "partial_eta2": float(partial_eta2),
        "friedman_chi2": float(fr.statistic),
        "friedman_p": float(fr.pvalue),
        "mean_0h": float(w["0h"].mean()),
        "mean_6h": float(w["6h"].mean()),
        "mean_24h": float(w["24h"].mean()),
    })

    # Post-hoc paired t-tests (with Holm correction within chirality)
    pairs = [("0h", "6h"), ("6h", "24h"), ("0h", "24h")]
    pvals_pair = []
    tmp = []

    for a, b in pairs:
        diff = (w[b] - w[a]).values
        t = stats.ttest_rel(w[b].values, w[a].values)

        try:
            wx = stats.wilcoxon(w[b].values, w[a].values, zero_method="wilcox", alternative="two-sided")
            w_p = float(wx.pvalue)
            w_stat = float(wx.statistic)
        except Exception:
            w_p, w_stat = np.nan, np.nan

        tmp.append({
            "chirality": c,
            "comparison": f"{a} vs {b}",
            "n": int(len(diff)),
            "mean_diff_(b-a)": float(diff.mean()),
            "median_diff_(b-a)": float(np.median(diff)),
            "paired_t_stat": float(t.statistic),
            "paired_t_p": float(t.pvalue),
            "cohen_dz": cohen_dz(diff),
            "wilcoxon_stat": w_stat,
            "wilcoxon_p": w_p
        })
        pvals_pair.append(float(t.pvalue))

    adj = holm_adjust(pvals_pair)
    for i in range(len(tmp)):
        tmp[i]["paired_t_p_holm_within_chirality"] = float(adj[i])
        posthoc_rows.append(tmp[i])

anova_df = pd.DataFrame(anova_rows)
anova_df["rm_anova_p_fdr_bh_12ch"] = bh_fdr(anova_df["rm_anova_p"].values)
posthoc_df = pd.DataFrame(posthoc_rows)

anova_df.to_csv(os.path.join(OUT_DIR, "rm_anova_by_chirality.csv"), index=False)
posthoc_df.to_csv(os.path.join(OUT_DIR, "posthoc_paired_tests_by_chirality.csv"), index=False)

# Simple plot of -log10(p) per chirality
fig, ax = plt.subplots(figsize=(9, 4.8))
p = anova_df["rm_anova_p"].values
ax.bar(anova_df["chirality"].astype(str).values, -np.log10(np.maximum(p, 1e-300)))
ax.axhline(-np.log10(0.05), linewidth=1)
ax.set_title("Repeated-measures ANOVA per chirality (time effect)\nBar height = -log10(p)")
ax.set_ylabel("-log10(p)")
ax.set_xlabel("Chirality")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "rm_anova_pvalues_barplot.png"), dpi=300)
plt.close(fig)

print("Done. Outputs in:", OUT_DIR)
print(anova_df.sort_values("rm_anova_p"))

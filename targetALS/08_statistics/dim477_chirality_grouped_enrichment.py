
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, mannwhitneyu

base = r"C:\Users\riccardo-s\Documents\CNT\targetALS\PLS"

z = pd.read_csv(os.path.join(base, "z_agg3.csv"))
chir = pd.read_csv(os.path.join(base, "chirality_interp_descriptors.csv"))
nfl = pd.read_excel(os.path.join(base, "early_slope.xlsx"))

def norm_code(x: str) -> str:
    return str(x).replace(".", "_")

z["code_norm"] = z["code"].map(norm_code)
nfl["code_norm"] = nfl["code"].map(norm_code)
chir["code_norm"] = chir["code"].astype(str)

merged = (
    z[["code_norm", "dim477"]]
    .merge(nfl[["code_norm", "Nfl Concentration Pg Per Ml"]], on="code_norm", how="inner")
    .merge(chir, on="code_norm", how="inner")
)

desc_cols = [c for c in chir.columns if c not in {"code", "code_norm"}]

def extract_chiralities(feature_name: str):
    return re.findall(r"ch(\d+_\d+)", feature_name)

rows = []
for col in desc_cols:
    sub = merged[["dim477", "Nfl Concentration Pg Per Ml", col]].dropna()
    if len(sub) < 6:
        continue
    r_dim, p_dim = spearmanr(sub["dim477"], sub[col])
    r_nfl, p_nfl = spearmanr(sub["Nfl Concentration Pg Per Ml"], sub[col])
    if np.isnan(r_dim) or np.isnan(r_nfl):
        continue
    rows.append(
        {
            "feature": col,
            "n": len(sub),
            "rho_dim477": r_dim,
            "p_dim477": p_dim,
            "rho_nfl": r_nfl,
            "p_nfl": p_nfl,
            "chiralities": extract_chiralities(col),
        }
    )

feature_df = pd.DataFrame(rows)
exploded = feature_df.explode("chiralities").rename(columns={"chiralities": "chirality"})

group_df = (
    exploded.groupby("chirality")
    .agg(
        mean_abs_rho_dim477=("rho_dim477", lambda s: float(np.mean(np.abs(s)))),
        mean_abs_rho_nfl=("rho_nfl", lambda s: float(np.mean(np.abs(s)))),
        median_abs_rho_dim477=("rho_dim477", lambda s: float(np.median(np.abs(s)))),
        median_abs_rho_nfl=("rho_nfl", lambda s: float(np.median(np.abs(s)))),
        n_features=("feature", "count"),
    )
    .reset_index()
    .sort_values("mean_abs_rho_nfl", ascending=False)
    .reset_index(drop=True)
)

top_k = 4
top_chirs = set(group_df.head(top_k)["chirality"])

top_vals = group_df.loc[group_df["chirality"].isin(top_chirs), "mean_abs_rho_dim477"]
other_vals = group_df.loc[~group_df["chirality"].isin(top_chirs), "mean_abs_rho_dim477"]

u_stat, p_val = mannwhitneyu(top_vals, other_vals, alternative="greater")
group_df["nfl_top_group"] = np.where(group_df["chirality"].isin(top_chirs), "NFL-top chiralities", "Other chiralities")

group_df.to_csv(os.path.join(base, "dim477_chirality_grouped_enrichment.csv"), index=False)

fig = plt.figure(figsize=(11, 5.5))

ax1 = fig.add_subplot(1, 2, 1)
for label in ["Other chiralities", "NFL-top chiralities"]:
    sub = group_df[group_df["nfl_top_group"] == label]
    ax1.scatter(sub["mean_abs_rho_nfl"], sub["mean_abs_rho_dim477"], s=80, label=label)

for _, row in group_df.iterrows():
    ax1.text(
        row["mean_abs_rho_nfl"] + 0.001,
        row["mean_abs_rho_dim477"] + 0.001,
        row["chirality"],
        fontsize=8,
    )

ax1.set_xlabel("Mean |Spearman rho| with NFL")
ax1.set_ylabel("Mean |Spearman rho| with dim477")
ax1.set_title("Chirality-grouped correlations")
ax1.legend(frameon=False)

ax2 = fig.add_subplot(1, 2, 2)
plot_df = group_df.sort_values("mean_abs_rho_dim477", ascending=True)
bar_colors = ["tab:orange" if c in top_chirs else "tab:blue" for c in plot_df["chirality"]]
ax2.barh(plot_df["chirality"], plot_df["mean_abs_rho_dim477"], color=bar_colors)
ax2.set_xlabel("Mean |Spearman rho| with dim477")
ax2.set_title(f"Enrichment test: U={u_stat:.1f}, p={p_val:.4g}")

plt.tight_layout()
plt.savefig(os.path.join(base, "dim477_chirality_grouped_enrichment.png"), dpi=200, bbox_inches="tight")

print("Top 4 NFL-linked chiralities:", ", ".join(group_df.head(top_k)["chirality"]))
print(f"Mann–Whitney U = {u_stat:.3f}, p = {p_val:.6g}")

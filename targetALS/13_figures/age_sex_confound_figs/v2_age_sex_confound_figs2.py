"""Two SEPARATE confounding figures:
  fig_dim477_vs_age.png  -- mechanism: dim477 flat vs age within group
  fig_auc_forest.png     -- robustness: AUC unchanged under matching/adjustment,
                             age/sex alone far below the AE
"""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
rng = np.random.default_rng(42)

OUT = "age_sex_confound_outputs"
df = pd.read_csv(os.path.join(OUT, "merged_dim477_age_sex.csv"))
ALS, CTRL = "#c0392b", "#2980b9"
pal = {"ALS": ALS, "CTRL": CTRL}

# ---- 1:1 nearest-age matched subset (caliper 5y) ----
als = df[df.y == 1].sort_values("age"); ctl = df[df.y == 0]; used = set(); pr = []
for _, a in als.iterrows():
    cand = ctl[~ctl.code.isin(used)].copy(); cand["d"] = (cand.age - a.age).abs()
    cand = cand[cand.d <= 5]
    if len(cand):
        c = cand.sort_values("d").iloc[0]; used.add(c.code); pr += [a.code, c.code]
matched = df[df.code.isin(pr)]
band = df[(df.age >= max(df.age[df.y==1].min(), df.age[df.y==0].min())) &
          (df.age <= min(df.age[df.y==1].max(), df.age[df.y==0].max()))]

def auc_or(y, s): a = roc_auc_score(y, s); return max(a, 1 - a)
def boot_auc(sub, col, n=2000):
    y = sub.y.values; s = sub[col].values; idx = np.arange(len(y)); out = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if y[b].sum() in (0, len(b)): continue
        out.append(auc_or(y[b], s[b]))
    return auc_or(y, s), np.percentile(out, 2.5), np.percentile(out, 97.5)
def loo(cols, n=2000):
    X = df[cols].values.astype(float); y = df.y.values; p = np.zeros(len(df))
    for tr, te in LeaveOneOut().split(X):
        sc = StandardScaler().fit(X[tr])
        p[te] = LogisticRegression().fit(sc.transform(X[tr]), y[tr]).predict_proba(sc.transform(X[te]))[:,1]
    base = roc_auc_score(y, p); out = []
    for _ in range(n):
        b = rng.choice(np.arange(len(y)), len(y), replace=True)
        if y[b].sum() in (0, len(b)): continue
        out.append(roc_auc_score(y[b], p[b]))
    return base, np.percentile(out, 2.5), np.percentile(out, 97.5)

# =====================================================================
# FIGURE 1 : dim477 vs age, within-group flat fits (+ age hist inset)
# =====================================================================
fig, ax = plt.subplots(figsize=(7.2, 5.6))
# overall (ignoring group) trend for contrast
bo = np.polyfit(df.age, df.dim477, 1); xs = np.array([df.age.min(), df.age.max()])
ax.plot(xs, np.polyval(bo, xs), color="grey", lw=1.4, ls="-", alpha=.7,
        label=f"pooled (ignores group): rho={stats.spearmanr(df.age,df.dim477)[0]:+.2f}")
for g in ["CTRL", "ALS"]:
    m = df.group == g
    ax.scatter(df.age[m], df.dim477[m], c=pal[g], s=70, edgecolor="k", lw=.5, zorder=3, label=g)
    b = np.polyfit(df.age[m], df.dim477[m], 1); gx = np.array([df.age[m].min(), df.age[m].max()])
    r, p = stats.spearmanr(df.age[m], df.dim477[m])
    ax.plot(gx, np.polyval(b, gx), color=pal[g], lw=2.2, ls="--", zorder=2,
            label=f"{g} within-group: rho={r:+.2f} (p={p:.2f})")
ax.set_xlabel("Age (years)"); ax.set_ylabel("dim477 (aggregated latent)")
ax.set_title("dim477 separates the groups at every age\n"
             "within-group slopes are flat -> age is not the driver")
ax.legend(fontsize=8, loc="upper left", framealpha=.95)
# age-distribution inset
ins = ax.inset_axes([0.62, 0.06, 0.35, 0.30])
for g in ["CTRL", "ALS"]:
    ins.hist(df.age[df.group==g], bins=np.arange(15,80,7), alpha=.6, color=pal[g])
ins.set_title("age imbalance (p=0.026)", fontsize=7); ins.tick_params(labelsize=6)
ins.set_xlabel("age", fontsize=6)
fig.tight_layout(); f1 = os.path.join(OUT, "fig_dim477_vs_age.png")
fig.savefig(f1, dpi=160); print("saved", f1)

# =====================================================================
# FIGURE 2 : AUC forest
# =====================================================================
rows = []  # (label, auc, lo, hi, kind)
for lab, sub in [("dim477  -  full cohort (n=39)", df),
                 ("dim477  -  age-overlap band (n=%d)" % len(band), band),
                 ("dim477  -  1:1 age-matched (n=%d, p=0.91)" % len(matched), matched)]:
    a, lo, hi = boot_auc(sub, "dim477"); rows.append((lab, a, lo, hi, "sig"))
a, lo, hi = boot_auc(df, "P_ALS"); rows.append(("AE classifier score P_ALS (n=39)", a, lo, hi, "sig"))
for lab, cols in [("dim477 + age + sex  (LOO)", ["dim477","age","male"]),
                  ("AGE alone  (LOO)", ["age"]),
                  ("AGE + SEX alone  (LOO)", ["age","male"])]:
    a, lo, hi = loo(cols); rows.append((lab, a, lo, hi, "base" if "alone" not in lab else "conf"))
rows = rows[::-1]

fig, ax = plt.subplots(figsize=(8.4, 5.2))
for i, (lab, a, lo, hi, kind) in enumerate(rows):
    col = {"sig": ALS, "base": "#27ae60", "conf": "#7f8c8d"}[kind]
    ax.plot([lo, hi], [i, i], color=col, lw=2.4, zorder=2)
    ax.scatter(a, i, color=col, s=95, edgecolor="k", zorder=3)
    ax.text(hi + 0.008, i, f"{a:.2f}", va="center", fontsize=9)
ax.axvspan(0.4, 0.5, color="grey", alpha=.12)  # chance shading edge
ax.axvline(0.5, color="k", ls=":", lw=1, label="chance")
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=9)
ax.set_xlim(0.45, 1.03); ax.set_xlabel("AUC (95% CI, bootstrap)")
ax.set_title("AE signal is unchanged by age matching/adjustment;\n"
             "age & sex alone reach only ~0.6")
import matplotlib.patches as mp
ax.legend(handles=[mp.Patch(color=ALS, label="AE signal (dim477 / P_ALS)"),
                   mp.Patch(color="#7f8c8d", label="covariate-only baseline")],
          loc="lower right", fontsize=8)
fig.tight_layout(); f2 = os.path.join(OUT, "fig_auc_forest.png")
fig.savefig(f2, dpi=160); print("saved", f2)

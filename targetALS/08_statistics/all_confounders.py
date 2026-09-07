"""
Systematic confounder screen for the ALS-vs-CTRL corona signal.

A covariate can only manufacture a false case/control signal if it BOTH
 (1) differs between ALS and CTRL, AND (2) associates with the AE signal.
Signals tested: dim477 (aggregated latent) and P_ALS (OOF classifier score).

Covariate classes in this cohort:
 - cross-group (present in both arms): age, sex, ethnicity, race, trial coenrollment
       -> test group-association (Fisher/MWU) AND signal-association
 - ALS-only (collinear with disease): riluzole, edaravone, C9orf72, El Escorial,
       ALSFRS-R, symptom site, disease duration, SVC
       -> cannot bias controls, but test WITHIN-ALS whether the signal is really
          "on-drug"/"severity"/"subtype" rather than disease
"""
import os, numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
pd.options.mode.chained_assignment = None

ROOT = "/mnt/c/Users/riccardo-s/Documents/CNT/targetALS/exp5"
OUT = os.path.join(ROOT, "all_confounders_outputs"); os.makedirs(OUT, exist_ok=True)
rep = open(os.path.join(OUT, "confounder_report.txt"), "w")
def p(*a):
    m = " ".join(str(x) for x in a); print(m); rep.write(m + "\n")

# ---- signals + demographics already merged earlier ----
sig = pd.read_csv(os.path.join(ROOT, "age_sex_confound_outputs", "merged_dim477_age_sex.csv"))
sig = sig[["code", "group", "y", "age", "male", "dim477", "P_ALS"]]  # code = int id

# ---- full clinical table (bounded read to dodge phantom rows) ----
v = pd.read_excel(os.path.join(ROOT, "visit1.xlsx"), sheet_name="Foglio1", nrows=40)
v = v[v["code"].notna()].copy(); v["code"] = v["code"].astype(int)
def col(name):  # fuzzy column getter
    for c in v.columns:
        if c.strip().lower().startswith(name.strip().lower()): return c
    raise KeyError(name)

df = sig.merge(v, on="code", how="left")
ALS = df[df.y == 1]; N = len(df)
p(f"n={N}  ALS={df.y.sum()}  CTRL={(1-df.y).sum()}\n")

def auc_or(s, y): a = roc_auc_score(y, s); return max(a, 1 - a)

def cross_group_cat(label, series):
    """2-level covariate available in both arms: does it differ by group AND
    associate with the signals (within group)?"""
    s = series.astype(str).str.strip()
    lv = sorted(s[df.y == 1].unique().tolist() + s[df.y == 0].unique().tolist())
    tab = pd.crosstab(df.group, s)
    both = all((s[df.y == g].nunique() > 1) for g in [0, 1])
    fp = stats.fisher_exact(tab.values)[1] if tab.shape == (2, 2) else np.nan
    p(f"--- {label}  (cross-group)")
    p("   " + tab.to_string().replace("\n", "\n   "))
    p(f"   assoc with GROUP: Fisher p={fp:.3g}   both-arms-vary={both}")
    for sg in ["dim477", "P_ALS"]:
        parts = []
        for g, gn in [(1, "ALS"), (0, "CTRL")]:
            m = df.y == g; vals = s[m]
            u = vals.unique()
            if len(u) == 2:
                a = df.loc[m & (s == u[0]), sg]; b = df.loc[m & (s == u[1]), sg]
                if len(a) >= 2 and len(b) >= 2:
                    pv = stats.mannwhitneyu(a, b).pvalue
                    parts.append(f"{gn}:{u[0]}={a.median():.3f}/{u[1]}={b.median():.3f} p={pv:.2f}")
        p(f"   {sg} within-group by {label}: " + ("; ".join(parts) if parts else "n/a (no variation)"))
    p("")

def within_als_cat(label, series):
    s = ALS[series.name if hasattr(series, 'name') else series].astype(str).str.strip() if isinstance(series, str) else series
    s = ALS[series].astype(str).str.strip()
    u = [x for x in s.unique() if x not in ("nan", "Not Applicable", "None")]
    p(f"--- {label}  (ALS-only; within-ALS test)")
    cnt = s.value_counts().to_dict(); p(f"   counts: {cnt}")
    if len(u) < 2:
        p("   no within-ALS variation -> cannot confound within ALS\n"); return
    for sg in ["dim477", "P_ALS"]:
        groups = [ALS.loc[s == x, sg] for x in u if (s == x).sum() >= 2]
        labs = [x for x in u if (s == x).sum() >= 2]
        if len(groups) >= 2:
            if len(groups) == 2:
                pv = stats.mannwhitneyu(groups[0], groups[1]).pvalue
            else:
                pv = stats.kruskal(*groups).pvalue
            meds = "; ".join(f"{l}={g.median():.3f}(n{len(g)})" for l, g in zip(labs, groups))
            p(f"   {sg}: {meds}  p={pv:.2f}")
    p("")

def within_als_cont(label, series, transform=None):
    x = pd.to_numeric(ALS[series], errors="coerce")
    if transform: x = transform(x)
    p(f"--- {label}  (ALS-only continuous; within-ALS Spearman)")
    for sg in ["dim477", "P_ALS"]:
        m = x.notna()
        if m.sum() >= 5:
            r, pv = stats.spearmanr(x[m], ALS.loc[m, sg])
            p(f"   {sg} vs {label}: rho={r:+.2f} p={pv:.2f} (n={m.sum()})")
    p("")

p("="*70); p("A. CROSS-GROUP COVARIATES (can bias case/control if imbalanced)"); p("="*70)
cross_group_cat("Ethnicity", df[col("Ethnicity")])
cross_group_cat("Race", df[col("Race")])
cross_group_cat("TrialCoenroll", df[col("Coenrollment")])

p("="*70); p("B. ALS-ONLY COVARIATES (collinear w/ disease; within-ALS 'what is the signal')"); p("="*70)
within_als_cat("Riluzole", col("Taking Rilutek"))
within_als_cat("Edaravone", col("Taking Radicava"))
within_als_cat("C9orf72", col("C 9 Orf 72"))
within_als_cat("ElEscorial", col("Als Diagnosis By Revised"))
within_als_cat("OnsetSite", col("Site Of Symptom"))
within_als_cont("ALSFRS_R_total", col("Total Alsfrsr"))
within_als_cont("SVC_pred_pct", col("Trial 1 Predicted"))
# disease duration = sample year (2024 assumed) - onset year
dur = 2024 - pd.to_numeric(ALS[col("Year Of Symptom Onset")], errors="coerce")
ALS["_dur"] = dur
within_als_cont("DiseaseDuration_yr", "_dur")

# ---- reference: strength of the actual disease signal for scale ----
p("="*70); p("C. REFERENCE: disease signal magnitude")
p(f"   dim477 ALS/CTRL AUC={auc_or(df.dim477, df.y):.3f}   P_ALS OOF AUC={auc_or(df.P_ALS, df.y):.3f}")
p("\nsaved ->", OUT); rep.close()

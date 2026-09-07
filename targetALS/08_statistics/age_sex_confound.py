"""
Assess age & sex confounding on (a) the AE latent representation broadly and
(b) the canonical dim477 latent, plus the AE OOF classifier score.

Data sources
------------
- demographics : visit1.xlsx  (code=int subject id, Age Years, Sex, Subject Group)
- dim477 + full 512-d latent : PLS/z_agg3.csv  (per-subject aggregated fold3 latent)
- AE OOF classifier score    : model_metrics_eval/existing_models_oof_predictions.csv (P_ALS)

Confounding logic: a covariate C confounds the AE<->group signal only if
  (1) C is imbalanced between groups (association C~group), AND
  (2) C drives the AE signal independent of group (association C~signal WITHIN group).
We then test whether the AE signal survives adjustment for age+sex, and whether
age+sex alone can reproduce the AE's class separation.
"""
import os, re, warnings
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

ROOT = "/mnt/c/Users/riccardo-s/Documents/CNT/targetALS"
EXP5 = os.path.join(ROOT, "exp5")
OUT  = os.path.join(EXP5, "age_sex_confound_outputs")
os.makedirs(OUT, exist_ok=True)

def sep(t): print("\n" + "="*74 + f"\n{t}\n" + "="*74)

# ---------------- load demographics ----------------
import openpyxl
wb = openpyxl.load_workbook(os.path.join(EXP5, "visit1.xlsx"), data_only=True)
ws = wb["Foglio1"]; rows = list(ws.iter_rows(values_only=True)); hdr = rows[0]
ix = {n: i for i, n in enumerate(hdr)}
dem = []
for r in rows[1:]:
    if r[0] is None: continue
    dem.append(dict(code=int(r[ix["code"]]),
                    group=("ALS" if "ALS" in str(r[ix["Subject Group"]]) else "CTRL"),
                    age=float(r[ix["Age Years"]]),
                    sex=str(r[ix["Sex"]]).strip()))
dem = pd.DataFrame(dem)
dem["male"] = (dem["sex"] == "Male").astype(int)

# ---------------- load latent + OOF ----------------
z = pd.read_csv(os.path.join(ROOT, "PLS", "z_agg3.csv")).drop(columns=["group"])
z["code"] = z["code"].str.extract(r"^(\d+)").astype(int)
oof = pd.read_csv(os.path.join(EXP5, "model_metrics_eval", "existing_models_oof_predictions.csv"))
oof["code"] = oof["code"].str.extract(r"^(\d+)").astype(int)
oof = oof[["code", "P_ALS"]]

df = dem.merge(z, on="code").merge(oof, on="code")
df["y"] = (df["group"] == "ALS").astype(int)           # 1 = ALS
dim_cols = [c for c in z.columns if c.startswith("dim")]
print(f"merged n={len(df)}  ALS={df.y.sum()}  CTRL={(1-df.y).sum()}  latent dims={len(dim_cols)}")

# ================================================================
sep("1. IMBALANCE  (is age/sex actually associated with group?)")
for g in ["ALS", "CTRL"]:
    s = df.loc[df.group == g, "age"]
    print(f"  {g}: age mean={s.mean():.1f}  sd={s.std():.1f}  median={s.median():.0f}  range={s.min():.0f}-{s.max():.0f}  n={len(s)}")
a_als, a_ctl = df.age[df.y == 1], df.age[df.y == 0]
tt = stats.ttest_ind(a_als, a_ctl, equal_var=False)
mw = stats.mannwhitneyu(a_als, a_ctl)
d = (a_als.mean() - a_ctl.mean()) / np.sqrt((a_als.var(ddof=1) + a_ctl.var(ddof=1)) / 2)
print(f"  AGE  Welch t p={tt.pvalue:.3g}  MWU p={mw.pvalue:.3g}  Cohen d={d:+.2f}")
ct = pd.crosstab(df.group, df.sex)
print("  SEX table:\n", ct.to_string())
fish = stats.fisher_exact(ct.values)
print(f"  SEX Fisher exact p={fish[1]:.3g}  (ALS %male={df.male[df.y==1].mean()*100:.0f}, CTRL %male={df.male[df.y==0].mean()*100:.0f})")

# ================================================================
sep("2. DOES AGE/SEX DRIVE THE AE SIGNAL *WITHIN* GROUP? (condition 2)")
# within-group correlation strips the group-mediated path; only a real
# age->signal effect survives.
def within_group_corr(sig):
    out = {}
    for g in ["ALS", "CTRL"]:
        m = df.group == g
        r, p = stats.spearmanr(df.loc[m, "age"], df.loc[m, sig])
        out[g] = (r, p)
    # pooled within-group partial (age vs sig | group) via residualization
    ag = df.age - df.groupby("y").age.transform("mean")
    sg = df[sig] - df.groupby("y")[sig].transform("mean")
    rp, pp = stats.spearmanr(ag, sg)
    return out, (rp, pp)

for sig in ["dim477", "P_ALS"]:
    wg, pooled = within_group_corr(sig)
    print(f"\n  {sig} vs AGE within-group (Spearman):")
    for g in ["ALS", "CTRL"]:
        print(f"     {g}: rho={wg[g][0]:+.2f}  p={wg[g][1]:.3g}")
    print(f"     pooled within-group partial: rho={pooled[0]:+.2f}  p={pooled[1]:.3g}")
    # sex within group
    print(f"  {sig} vs SEX within-group (MWU male vs female):")
    for g in ["ALS", "CTRL"]:
        m = df.group == g
        mm, ff = df.loc[m & (df.male == 1), sig], df.loc[m & (df.male == 0), sig]
        if len(mm) > 1 and len(ff) > 1:
            p = stats.mannwhitneyu(mm, ff).pvalue
            print(f"     {g}: male med={mm.median():+.3f} female med={ff.median():+.3f}  p={p:.3g}")

# ================================================================
sep("3. DOES THE AE SIGNAL SURVIVE ADJUSTMENT FOR AGE + SEX?")
# Nested logistic regressions predicting ALS status (in-sample LR fit; small n,
# so this is an association test, not held-out performance).
def fit_report(cols):
    X = StandardScaler().fit_transform(df[cols].values)
    lr = LogisticRegression().fit(X, df.y.values)
    p = lr.predict_proba(X)[:, 1]
    auc = roc_auc_score(df.y, p)
    return auc, dict(zip(cols, lr.coef_[0]))

for cols in (["age"], ["male"], ["age", "male"], ["dim477"],
             ["dim477", "age"], ["dim477", "age", "male"]):
    auc, coef = fit_report(cols)
    cstr = "  ".join(f"{k}:{v:+.2f}" for k, v in coef.items())
    print(f"  in-sample AUC={auc:.3f}  [{'+'.join(cols):22s}]  coef {cstr}")

# proper held-out: LOO AUC for age+sex alone vs dim477 alone vs combined
def loo_auc(cols):
    X = df[cols].values.astype(float); yv = df.y.values
    pr = np.zeros(len(df))
    for tr, te in LeaveOneOut().split(X):
        sc = StandardScaler().fit(X[tr])
        lr = LogisticRegression().fit(sc.transform(X[tr]), yv[tr])
        pr[te] = lr.predict_proba(sc.transform(X[te]))[:, 1]
    return roc_auc_score(yv, pr)
print("\n  Leave-one-out AUC (honest generalization):")
for cols in (["age"], ["male"], ["age", "male"], ["dim477"],
             ["dim477", "age", "male"], ["P_ALS"]):
    print(f"     {'+'.join(cols):20s}  LOO-AUC={loo_auc(cols):.3f}")

# ================================================================
sep("4. AGE-MATCHED SUBSET  (remove the imbalance, re-test AE signal)")
# controls skew young; restrict both arms to the overlapping age band.
lo = max(df.age[df.y == 1].min(), df.age[df.y == 0].min())
hi = min(df.age[df.y == 1].max(), df.age[df.y == 0].max())
sub = df[(df.age >= lo) & (df.age <= hi)]
print(f"  overlap age band = [{lo:.0f}, {hi:.0f}]  -> n={len(sub)}  ALS={sub.y.sum()} CTRL={(1-sub.y).sum()}")
p_age = stats.mannwhitneyu(sub.age[sub.y == 1], sub.age[sub.y == 0]).pvalue
print(f"  age MWU within band p={p_age:.3g}  (ALS med={sub.age[sub.y==1].median():.0f}, CTRL med={sub.age[sub.y==0].median():.0f})")
for sig in ["dim477", "P_ALS"]:
    a = roc_auc_score(sub.y, sub[sig]); a = max(a, 1 - a)
    p = stats.mannwhitneyu(sub[sig][sub.y == 1], sub[sig][sub.y == 0]).pvalue
    print(f"  {sig}: age-matched AUC={a:.3f}  MWU p={p:.3g}   (full-cohort AUC="
          f"{max(roc_auc_score(df.y, df[sig]), 1-roc_auc_score(df.y, df[sig])):.3f})")

# ================================================================
sep("5. AE LATENT SPACE WIDE:  age/sex vs group structure across 512 dims")
# For every latent dim: |correlation with group| vs |within-group corr with age|.
res = []
ag = df.age - df.groupby("y").age.transform("mean")   # within-group age
for c in dim_cols:
    v = df[c].values
    r_grp = abs(stats.pointbiserialr(df.y, v)[0])
    r_age = abs(stats.spearmanr(ag, v - df.groupby("y")[c].transform("mean"))[0])
    # sex within group
    sxg = df.male - df.groupby("y").male.transform("mean")
    r_sex = abs(stats.spearmanr(sxg, v - df.groupby("y")[c].transform("mean"))[0])
    res.append((c, r_grp, r_age, r_sex))
R = pd.DataFrame(res, columns=["dim", "r_group", "r_age_wg", "r_sex_wg"])
R.to_csv(os.path.join(OUT, "per_dim_group_vs_age_sex.csv"), index=False)
n_grp = (R.r_group > 0.4).sum()
n_age = (R.r_age_wg > 0.4).sum()
n_sex = (R.r_sex_wg > 0.4).sum()
print(f"  latent dims with |r|>0.4 :  group-linked={n_grp}   age-linked(within-grp)={n_age}   sex-linked(within-grp)={n_sex}")
print(f"  mean |r| across 512 dims :  group={R.r_group.mean():.3f}  age_wg={R.r_age_wg.mean():.3f}  sex_wg={R.r_sex_wg.mean():.3f}")
d4 = R[R.dim == "dim477"].iloc[0]
print(f"  dim477 specifically      :  r_group={d4.r_group:.3f}  r_age_wg={d4.r_age_wg:.3f}  r_sex_wg={d4.r_sex_wg:.3f}")
print(f"  corr(|r_group|,|r_age_wg|) across dims = {stats.spearmanr(R.r_group, R.r_age_wg)[0]:+.3f}"
      "  (if ~0, group-linked dims are NOT the age-linked ones)")

df.to_csv(os.path.join(OUT, "merged_dim477_age_sex.csv"), index=False)
print("\nsaved outputs ->", OUT)

#!/usr/bin/env python3
"""exp4 SERUM (STMN_22) -- BLIND FITTED per-condition analysis (mean gauss_max over clean chiralities).
Cross-check of the model-free result. Same geometry (missing grid pos 91). (7,5) excluded (frac r2>.95=0.24)."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr, friedmanchisquare
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

BASE="/mnt/c/Users/riccardo-s/Documents/CNT/cryptic_mRNA/exp4/serum"
GOOD=["ch8_3","ch6_5","ch10_2","ch9_4","ch8_4","ch7_6","ch8_6","ch8_7","ch9_5","ch10_3","ch10_5"]
fit=pd.read_csv(f"{BASE}/chirality_gaussian_descriptors_exp4_serum.csv")
fit["fit_mean"]=fit[[f"{ch}_gauss_max" for ch in GOOD]].mean(axis=1)
df=fit.reset_index(drop=True)
CONDS=sorted(df["cond"].unique())

# gradients (raw fitted mean)
rho_r,p_r=spearmanr(df["row"],df["fit_mean"])
df["res"]=df["fit_mean"]-df.groupby("row")["fit_mean"].transform("mean")
rho_c,p_c=spearmanr(df["col"],df["res"])
print(f"FITTED gradients: row(top-down) rho={rho_r:+.2f} p={p_r:.2g}   within-row col(left-right) rho={rho_c:+.2f} p={p_c:.2g}")

df["fit_rn"]=df["fit_mean"]/df.groupby("row")["fit_mean"].transform("mean")
F=anova_lm(smf.ols("fit_rn ~ 1",data=df).fit(),smf.ols("fit_rn ~ C(cond)",data=df).fit())["Pr(>F)"].iloc[1]
piv=df.pivot_table(index="row",columns="cond",values="fit_rn").apply(lambda s:s.fillna(s.median()))
fr=friedmanchisquare(*[piv[c].values for c in CONDS])
print(f"FITTED C(cond) F-test p={F:.2g}  Friedman chi2={fr.statistic:.1f} p={fr.pvalue:.2g}")
print("\nfitted row-norm per BLIND condition (sorted brightest first):")
cm=df.groupby("cond")["fit_rn"].agg(["mean","sem"])
for c in cm["mean"].sort_values(ascending=False).index:
    print(f"   {c}:  {cm.loc[c,'mean']:.3f} +/- {cm.loc[c,'sem']:.3f}")

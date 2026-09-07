#!/usr/bin/env python3
"""
Per-chirality dose analysis for exp2 (chirality_gaussian_descriptors_exp2.csv).
Drift removed by row-normalization within (plate,row). Pool non-water wells across plates.
Q1: is the dose-brightening UNIFORM across chiralities (amplitude scaling) or selective?
Q2: (7,5) Delta-lambda (paper readout): does fitted peak center shift with dose?
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import statsmodels.formula.api as smf

BASE="/mnt/c/Users/riccardo-s/Documents/CNT/cryptic_mRNA/exp2/STMN_2/STMN_2"
df=pd.read_csv(f"{BASE}/chirality_gaussian_descriptors_exp2.csv")
GOOD=["ch8_3","ch6_5","ch7_5","ch10_2","ch9_4","ch8_4","ch7_6"]   # frac r2>0.95 ~1.0
ALL=["ch8_3","ch6_5","ch7_5","ch10_2","ch9_4","ch8_4","ch7_6","ch8_6","ch8_7","ch9_5","ch10_3","ch10_5"]
RANK={0.0:0,0.1:1,1.0:2,10.0:3,100.0:4,1000.0:5}
df["rank"]=df["dose"].map(RANK)
nw=df[~df.is_water].copy()

def rownorm(col):
    g=df.groupby(["plate","row"])[col].transform("mean")
    return df[col]/g

print("="*74); print("Q1: per-chirality DOSE response on gauss_max (drift removed, non-water pooled)"); print("="*74)
print(f"{'chir':7s} {'dose_rho':>8s} {'p':>9s} {'frac/dec':>9s}  (brightening per 10x dose)")
res={}
for ch in ALL:
    col=f"{ch}_gauss_max"
    df["rn"]=rownorm(col)
    s=df[~df.is_water]
    rho,p=spearmanr(s["rank"],s["rn"])
    # fractional slope per decade: log10(gauss_max) ~ log10(dose) + C(plate,row block via rn)
    sl=smf.ols("rn ~ rank",data=s).fit().params["rank"]
    res[ch]=(rho,p,sl)
    flag=" *" if ch in GOOD else "  (dim)"
    print(f"{ch:7s} {rho:+8.2f} {p:9.1g} {sl:+9.3f}{flag}")

print("\n--- UNIFORM vs SELECTIVE (good chiralities only) ---")
# stack good chiralities' row-normalized gauss_max; test dose:chirality interaction
recs=[]
for ch in GOOD:
    df["rn"]=rownorm(f"{ch}_gauss_max")
    for _,r in df[~df.is_water].iterrows():
        recs.append(dict(rn=r["rn"],rank=r["rank"],ch=ch,plate=r["plate"]))
L=pd.DataFrame(recs)
m0=smf.ols("rn ~ rank + C(ch) + C(plate)",data=L).fit()
m1=smf.ols("rn ~ rank * C(ch) + C(plate)",data=L).fit()
from statsmodels.stats.anova import anova_lm
pint=anova_lm(m0,m1)["Pr(>F)"].iloc[1]
print(f"  common dose slope (rank): {m0.params['rank']:+.4f}  p={m0.pvalues['rank']:.1g}")
print(f"  dose x chirality INTERACTION p = {pint:.2g}  "
      f"({'uniform amplitude scaling' if pint>0.05 else 'chirality-selective'})")
print("  per-chirality dose slopes (fraction per rank step):")
for ch in GOOD:
    print(f"    {ch:7s} {res[ch][2]:+.4f}")

print("\n"+"="*74); print("Q2: (7,5) Delta-lambda  [paper readout]  + other good chiralities"); print("="*74)
# additive drift removal on center: subtract (plate,row) mean center; Dlam = lam0 - lam (blue +)
for ch in ["ch7_5"]+[c for c in GOOD if c!="ch7_5"]:
    cc=f"{ch}_gauss_em_center"
    df["cc_dt"]=df[cc]-df.groupby(["plate","row"])[cc].transform("mean")  # drift-removed center residual
    s=df[~df.is_water & df[f"{ch}_gauss_r2"].gt(0.95)]
    rho,p=spearmanr(s["rank"],s["cc_dt"])
    # Dlam vs dose: mean center residual per dose, relative to dose=0
    base=s[s["dose"]==0]["cc_dt"].mean()
    by=s.groupby("dose")["cc_dt"].mean()
    dl_top=-(by.get(s["dose"].max(),np.nan)-base)   # blue-positive Dlam at top dose
    # brightness coupling: does center move with its own gauss_max independent of dose?
    s2=s.copy(); s2["amp"]=s2[f"{ch}_gauss_max"]
    mc=smf.ols("cc_dt ~ rank + amp",data=s2).fit()
    print(f"{ch:7s}: center-shift rho(dose)={rho:+.2f} p={p:.2g}  "
          f"Dlam(top-0)={dl_top:+.2f}nm  |  dose-coef p={mc.pvalues['rank']:.2g} amp-coef p={mc.pvalues['amp']:.2g}")

# ---- figure ----
fig,ax=plt.subplots(1,2,figsize=(13,5))
ax[0].axhline(0,color="k",lw=0.6)
xs=range(len(ALL)); slopes=[res[c][2] for c in ALL]; cols=["#2c7fb8" if c in GOOD else "#bbb" for c in ALL]
ax[0].bar(xs,slopes,color=cols)
ax[0].set_xticks(list(xs)); ax[0].set_xticklabels(ALL,rotation=45,ha="right",fontsize=8)
ax[0].set_ylabel("dose slope (Δ row-norm intensity per rank step)")
ax[0].set_title("Q1: per-chirality brightening (blue=clean fit, gray=dim)")
# Q1 detail: good chiralities dose curves
for ch in GOOD:
    df["rn"]=rownorm(f"{ch}_gauss_max"); s=df[~df.is_water]
    g=s.groupby("rank")["rn"].mean()
    ax[1].plot(g.index,g.values,marker="o",label=ch)
ax[1].set_xticks(range(6)); ax[1].set_xticklabels(["0","0.1","1","10","100","1000"])
ax[1].axhline(1,color="k",lw=0.6,ls=":"); ax[1].set_xlabel("STMN2-CE concentration")
ax[1].set_ylabel("row-normalized gauss_max"); ax[1].legend(fontsize=7,ncol=2); ax[1].grid(alpha=0.3)
ax[1].set_title("Good chiralities all brighten ~together")
fig.suptitle("exp2 per-chirality fitted analysis: uniform amplitude dose-response",y=1.02)
fig.tight_layout(); fig.savefig(f"{BASE}/chirality_dose_exp2.png",dpi=130,bbox_inches="tight")
print("\nSaved chirality_dose_exp2.png")

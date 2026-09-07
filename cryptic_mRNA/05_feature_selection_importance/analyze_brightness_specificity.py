#!/usr/bin/env python3
"""
Is the global brightening SPECIFIC to STMN2-CE on GT15-STMN2 vs the GT15 control?
The control (bare (GT)15 anchor, no STMN2 capture domain) should NOT respond if
the brightening is hybridization-specific.

Test on WATER, brightness = clean-tube total (ch6_5+ch7_5+ch8_3) and fit-free integ:
 1. fold-change & dose Spearman, sensor vs control
 2. FORMAL interaction: log10(brightness) ~ logc * is_sensor -> sensor extra slope
 3. PERMUTATION test (robust to small control n): shuffle sensor/control labels,
    recompute the sensor-minus-control dose-slope difference 10000x -> empirical p.
"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import statsmodels.formula.api as smf

HERE="/mnt/c/Users/riccardo-s/Documents/CNT/cryptic_mRNA/exp1"
TUB=["ch6_5","ch7_5","ch8_3"]
g=pd.read_csv(f"{HERE}/chirality_gaussian_descriptors.csv")
f=pd.read_csv(f"{HERE}/fit_free_descriptors.csv")
g["total3"]=g[[f"{c}_gauss_max" for c in TUB]].clip(lower=0).sum(axis=1)
g=g.merge(f[["well"]+[f"{c}_ff_integ" for c in TUB]],on="well")
g["ffint3"]=g[[f"{c}_ff_integ" for c in TUB]].sum(axis=1)
g["logc"]=np.log10(g["copies"]+1); g["is_sensor"]=(g.sensor=="GT15-STMN2").astype(int)
W=g[g.matrix=="water"].copy()

def slope(d,col):  # OLS slope of log10(brightness) on logc
    dd=d[[col,"logc"]].copy(); dd["y"]=np.log10(dd[col].clip(lower=1e-30))
    return smf.ols("y ~ logc",data=dd).fit().params["logc"]

for col,name in [("total3","clean-tube total (gauss_max)"),
                 ("ffint3","clean-tube integrated (fit-free)")]:
    print("="*72); print(f"BRIGHTNESS = {name}"); print("="*72)
    s=W[W.is_sensor==1]; c=W[W.is_sensor==0]
    rs,ps=spearmanr(s["logc"],s[col]); rc,pc=spearmanr(c["logc"],c[col])
    fcs=s[s.copies==s.copies.max()][col].mean()/s[s.copies==0][col].mean()
    fcc=c[c.copies==c.copies.max()][col].mean()/c[c.copies==0][col].mean()
    print(f"  sensor : rho={rs:+.2f} p={ps:.1e}  fold(max/0)={fcs:.2f}x  (n={len(s)})")
    print(f"  control: rho={rc:+.2f} p={pc:.2f}  fold(max/0)={fcc:.2f}x  (n={len(c)})")

    W["y"]=np.log10(W[col].clip(lower=1e-30))
    m=smf.ols("y ~ logc * is_sensor",data=W).fit()
    bi=m.params["logc:is_sensor"]; pi=m.pvalues["logc:is_sensor"]
    print(f"  FORMAL interaction (sensor extra slope): {bi:+.3f}/dec  p={pi:.3f}")

    # permutation test on the slope difference
    obs=slope(s,col)-slope(c,col)
    rng=np.random.default_rng(11); lab=W["is_sensor"].to_numpy(); n=10000; cnt=0
    for _ in range(n):
        perm=rng.permutation(lab); Wp=W.assign(_s=perm)
        d=slope(Wp[Wp._s==1],col)-slope(Wp[Wp._s==0],col)
        if abs(d)>=abs(obs): cnt+=1
    print(f"  PERMUTATION: observed sensor-control slope diff={obs:+.3f}/dec  "
          f"empirical p={cnt/n:.3f}\n")

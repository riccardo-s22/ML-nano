#!/usr/bin/env python3
"""
Rank every (chirality, feature) by SPECIFIC STMN2-CE dose response:
 strong correlation in GT15-STMN2, weak/absent in the GT15 aspecific control.

For each (tube,feat) in the WATER arm:
  rho_s,p_s  = Spearman(feature, logc) for GT15-STMN2
  rho_c,p_c  = Spearman(feature, logc) for GT15 control
  interaction: OLS  y ~ logc * is_sensor  -> logc:is_sensor coef & p  (the formal
               sensor-specific extra slope; small p = response differs from control)
  spec_score = |rho_s| - |rho_c|   (want high sensor, low control)

Gaussian SHAPE features (em_center, fwhm) are R2>0.95 gated per well; amplitude/
area and all fit-free features are ungated (valid even when dim).
"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import statsmodels.formula.api as smf

HERE="/mnt/c/Users/riccardo-s/Documents/CNT/cryptic_mRNA/exp1"
CHIR=["ch8_3","ch6_5","ch7_5","ch10_2","ch9_4","ch8_4",
      "ch7_6","ch8_6","ch8_7","ch9_5","ch10_3","ch10_5"]
g=pd.read_csv(f"{HERE}/chirality_gaussian_descriptors.csv")
f=pd.read_csv(f"{HERE}/fit_free_descriptors.csv")
df=g.merge(f[["well"]+[c for c in f.columns if c.startswith("ch")]], on="well")
df["logc"]=np.log10(df["copies"]+1); df["is_sensor"]=(df.sensor=="GT15-STMN2").astype(int)

GAUSS=["gauss_max","gauss_auc","gauss_em_center","gauss_fwhm"]
FREE =["ff_integ","ff_peak","ff_centroid","ff_width"]
SHAPE_GATED={"gauss_em_center","gauss_fwhm"}

rows=[]
w=df[df.matrix=="water"].copy()
for ch in CHIR:
    for feat in GAUSS+FREE:
        col=f"{ch}_{feat}"
        y=w[col].copy()
        if feat in SHAPE_GATED:
            y=y.where(w[f"{ch}_gauss_r2"]>0.95)
        sub=pd.DataFrame({"y":y,"logc":w["logc"],"is_sensor":w["is_sensor"]}).dropna()
        s=sub[sub.is_sensor==1]; c=sub[sub.is_sensor==0]
        if len(s)<8 or len(c)<6 or s.y.nunique()<3: continue
        rs,ps=spearmanr(s.logc,s.y); rc,pc=spearmanr(c.logc,c.y)
        try:
            m=smf.ols("y ~ logc * is_sensor",data=sub).fit()
            bi=m.params.get("logc:is_sensor",np.nan); pi=m.pvalues.get("logc:is_sensor",np.nan)
        except Exception:
            bi=pi=np.nan
        rows.append(dict(chirality=ch,feature=feat,
                         rho_sensor=rs,p_sensor=ps,rho_ctrl=rc,p_ctrl=pc,
                         spec_score=abs(rs)-abs(rc),inter_p=pi,n_s=len(s),n_c=len(c)))
R=pd.DataFrame(rows)
R.to_csv(f"{HERE}/specificity_ranking.csv",index=False)

# headline: significant sensor (p_s<0.05), non-sig control, ranked by spec_score
cand=R[(R.p_sensor<0.05)&(R.p_ctrl>0.10)].sort_values("spec_score",ascending=False)
pd.set_option("display.width",200)
print("="*92)
print("MOST SPECIFIC dose responders: sensor p<0.05, control p>0.10, ranked by |rho_s|-|rho_c|")
print("="*92)
print(f"{'chir':7} {'feature':14} {'rho_s':>6} {'p_s':>8} {'rho_c':>6} {'p_c':>6} "
      f"{'spec':>6} {'inter_p':>8}")
for _,r in cand.head(15).iterrows():
    print(f"{r.chirality:7} {r.feature:14} {r.rho_sensor:+6.2f} {r.p_sensor:8.1e} "
          f"{r.rho_ctrl:+6.2f} {r.p_ctrl:6.2f} {r.spec_score:+6.2f} {r.inter_p:8.2f}")

print("\nTop 10 by formal interaction (sensor-specific extra slope, smallest inter_p):")
byint=R.dropna(subset=["inter_p"]).sort_values("inter_p")
print(f"{'chir':7} {'feature':14} {'rho_s':>6} {'rho_c':>6} {'inter_p':>8}")
for _,r in byint.head(10).iterrows():
    print(f"{r.chirality:7} {r.feature:14} {r.rho_sensor:+6.2f} {r.rho_ctrl:+6.2f} {r.inter_p:8.3f}")
print("\nSaved specificity_ranking.csv")

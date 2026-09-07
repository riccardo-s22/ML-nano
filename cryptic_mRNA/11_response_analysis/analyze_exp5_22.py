#!/usr/bin/env python3
"""exp5 / 22nt GT15-STMN2-CE sensor in DEFINED BACKGROUND (SDS 1% + BSA 0.5% + random RNA 30 ng/mL).
Randomized split layout: TOP rows 0-3 and BOTTOM rows 4-7 have DIFFERENT column->concentration maps,
so each concentration is sampled at two different column positions -> dose is decorrelated from column.

File format CHANGED: 512 emission pts x 95 columns, ALL 95 columns are wells (no wavelength col, no ref
ramp). Reuse the instrument emission axis from exp4 (same 512-pt calibration 852.6-1675.9 nm).
Geometry: 95 wells, 8x12 row-major, missing = col11/row7 = LAST grid cell -> row=w//12, col=w%12 (no shift).
NOTE: TOP col0 (=1e5) is the WRONG sensor (STMN-CE 20 loaded into a 22 plate) -> EXCLUDE from 22 analysis.
Two kinds of 0: 'blank_noRNA' (0, no RNA at all) vs 'zero_bg' (0 target, random-RNA background present)."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr, mannwhitneyu
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

BASE="/mnt/c/Users/riccardo-s/Documents/CNT/cryptic_mRNA/exp5/22"; PREFIX="STMN_22_WATER_2H_"
EXC=[570,640,670,750,780]; NCOL=12
AX=np.loadtxt("/mnt/c/Users/riccardo-s/Documents/CNT/cryptic_mRNA/exp4/water/STMN_22_WATER__570.txt")[:,0]

# 0-indexed col -> (copies, label).  label 'b'=blank_noRNA, 'z'=zero_bg, 'd'=dose
TOP={0:(1e5,'wrong'),1:(0,'b'),2:(5000,'d'),3:(1,'d'),4:(1000,'d'),5:(10,'d'),
     6:(0,'z'),7:(50,'d'),8:(500,'d'),9:(0.1,'d'),10:(100,'d'),11:(1e4,'d')}
BOT={0:(0.1,'d'),1:(1e4,'d'),2:(50,'d'),3:(500,'d'),4:(0,'b'),5:(1000,'d'),
     6:(5000,'d'),7:(100,'d'),8:(10,'d'),9:(1e5,'d'),10:(1,'d'),11:(0,'z')}

# ---- model-free integrated intensity, summed over 5 excitations ----
pk=(AX>=900)&(AX<=1350); bw=(AX>=1400)&(AX<=1600)
inten={}
for e in EXC:
    W=np.loadtxt(f"{BASE}/{PREFIX}{e}.txt")          # 512 x 95, all wells
    inten[e]=(W[pk,:]-W[bw,:].mean(axis=0)).sum(axis=0)
int_all=sum(inten[e] for e in EXC)

rows=[]
for w in range(len(int_all)):
    r=w//NCOL; c=w%NCOL
    cp,lab=(TOP if r<=3 else BOT)[c]
    rows.append(dict(w=w,row=r,col=c,half=("top" if r<=3 else "bot"),copies=cp,label=lab,int_all=int_all[w]))
df=pd.DataFrame(rows)
df=df[df.int_all>0].reset_index(drop=True)
df["int_rn"]=df["int_all"]/df.groupby("row")["int_all"].transform("mean")

wrong=df[df.label=="wrong"].copy()          # top c0: 20nt sensor at 1e5 (bonus)
dd=df[df.label!="wrong"].reset_index(drop=True)   # the actual 22nt sensor wells

# ================= GRADIENTS =================
print("="*78); print("GRADIENTS (raw ∫I, 22nt sensor wells only)"); print("="*78)
rr,pr=spearmanr(dd["row"],dd["int_all"]); print(f"ROW top-down heating: rho={rr:+.2f} p={pr:.2g}")
rm=dd.groupby("row")["int_all"].mean(); print("  row means:", " ".join(f"r{r}={rm[r]:.3g}" for r in sorted(dd.row.unique())))
dd["res"]=dd["int_all"]-dd.groupby("row")["int_all"].transform("mean")
rc,pc=spearmanr(dd["col"],dd["res"]); print(f"LEFT-RIGHT within-row residual: rho={rc:+.2f} p={pc:.2g}")

# ================= SPECIFICITY: two kinds of zero =================
print("\n"+"="*78); print("SPECIFICITY controls (row-normalized)"); print("="*78)
for lab,name in [('b','blank_noRNA (0, no RNA)'),('z','zero_bg (0 target + random RNA)')]:
    s=dd[dd.label==lab]["int_rn"]; print(f"  {name:38s} n={len(s)}  mean={s.mean():.3f} +/- {s.sem():.3f}")
b=dd[dd.label=='b']["int_rn"]; z=dd[dd.label=='z']["int_rn"]
print(f"  random-RNA background effect (zero_bg vs blank_noRNA): MWU p={mannwhitneyu(z,b).pvalue:.2g}")
zero_ref=dd[dd.label.isin(['b','z'])]["int_rn"]   # pooled 0-target reference

# ================= DOSE-RESPONSE (target > 0) =================
print("\n"+"="*78); print("STMN2-CE DOSE-RESPONSE in background (0 target pooled as baseline)"); print("="*78)
dd["lc"]=np.log10(dd["copies"]+1)
order=[0,0.1,1,10,50,100,500,1000,5000,1e4,1e5]
print(f"{'copies':>8} {'topcol':>6} {'botcol':>6} {'top_rn':>7} {'bot_rn':>7} {'all_rn':>7} {'n':>3}")
for cp in order:
    s=dd[dd.copies==cp]; t=s[s.half=='top']; bo=s[s.half=='bot']
    tc=int(t.col.iloc[0]) if len(t) else -1; bc=int(bo.col.iloc[0]) if len(bo) else -1
    print(f"{cp:>8g} {tc:>6} {bc:>6} {t.int_rn.mean() if len(t) else np.nan:>7.3f} "
          f"{bo.int_rn.mean() if len(bo) else np.nan:>7.3f} {s.int_rn.mean():>7.3f} {len(s):>3}")
rho,p=spearmanr(dd["lc"],dd["int_rn"]); print(f"\nSpearman(copies, int_rn) ALL wells: rho={rho:+.2f} p={p:.2g}")
pos=dd[dd.copies>0]
rho2,p2=spearmanr(pos["lc"],pos["int_rn"]); print(f"Spearman among target>0 only:      rho={rho2:+.2f} p={p2:.2g}")

# ================= KEY: dose controlling for COLUMN POSITION (decorrelated by design) =================
print("\n"+"="*78); print("DOSE vs POSITION  (randomized layout decorrelates them)"); print("="*78)
print(f"corr(log_copies, col) = {np.corrcoef(dd.lc,dd.col)[0,1]:+.2f}  (near 0 => separable)")
m0=smf.ols("int_rn ~ col",data=dd).fit()
m1=smf.ols("int_rn ~ col + lc",data=dd).fit()
Fp=anova_lm(m0,m1)["Pr(>F)"].iloc[1]
print(f"OLS int_rn ~ col + log_copies : log_copies coef={m1.params['lc']:+.4f} "
      f"p={m1.pvalues['lc']:.2g}; adds over col-only F-test p={Fp:.2g}")

# ================= TOP-vs-BOTTOM agreement (same conc, different columns) =================
print("\n"+"="*78); print("TOP vs BOTTOM agreement per concentration (same dose, DIFFERENT column)"); print("="*78)
prof=dd[dd.copies>=0].groupby(["copies","half"])["int_rn"].mean().unstack()
prof=prof.dropna()
rho_tb,p_tb=spearmanr(prof["top"],prof["bot"])
print(prof.round(3).to_string())
print(f"\nSpearman(top-profile, bottom-profile) across concentrations: rho={rho_tb:+.2f} p={p_tb:.2g}")
print("  (high rho => brightness tracks DOSE, since column position differs between the two halves)")

# ================= high vs zero =================
hi=dd[dd.copies>=1000]["int_rn"]; midlo=dd[(dd.copies>0)&(dd.copies<1000)]["int_rn"]
print("\n"+"="*78); print("THRESHOLD check"); print("="*78)
print(f"high(>=1e3)={hi.mean():.3f}  vs zero-pooled={zero_ref.mean():.3f}  MWU p={mannwhitneyu(hi,zero_ref).pvalue:.2g}")
print(f"high(>=1e3)={hi.mean():.3f}  vs low(0.1-500)={midlo.mean():.3f}  MWU p={mannwhitneyu(hi,midlo).pvalue:.2g}")
print(f"low(0.1-500)={midlo.mean():.3f} vs zero-pooled={zero_ref.mean():.3f} MWU p={mannwhitneyu(midlo,zero_ref).pvalue:.2g}")

# ================= bonus: wrong sensor (20nt @1e5, top c0) vs correct 22nt @1e5 (bot c9) =================
print("\n"+"="*78); print("BONUS: wrong-sensor well (20nt @1e5, top c0)"); print("="*78)
c22_1e5=dd[(dd.copies==1e5)&(dd.half=='bot')]["int_rn"]
print(f"  20nt@1e5 (top c0, row-norm): mean={wrong['int_all'].div(df.groupby('row')['int_all'].transform('mean')[wrong.index]).mean():.3f} "
      f"(n={len(wrong)}); 22nt@1e5 (bot c9): mean={c22_1e5.mean():.3f} n={len(c22_1e5)}")

# ================= figure =================
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(15,6))
xa={cp:i for i,cp in enumerate(order)}
for half,cl,mk in [("top","#1f77b4","o"),("bot","#d62728","s")]:
    s=dd[dd.half==half].groupby("copies")["int_rn"].agg(["mean","sem"])
    xs=[cp for cp in order if cp in s.index]
    ax1.errorbar([xa[cp] for cp in xs],[s.loc[cp,"mean"] for cp in xs],[s.loc[cp,"sem"] for cp in xs],
                 marker=mk,ms=6,capsize=3,color=cl,lw=1.6,label=f"{half} rows")
g=dd.groupby("copies")["int_rn"].agg(["mean","sem"])
ax1.errorbar([xa[cp] for cp in order],[g.loc[cp,"mean"] for cp in order],[g.loc[cp,"sem"] for cp in order],
             marker="D",ms=8,color="k",lw=2.4,capsize=3,label="pooled",zorder=10)
ax1.axhline(zero_ref.mean(),color="grey",lw=0.8,ls=":",label="0-target level")
ax1.set_xticks(range(len(order))); ax1.set_xticklabels([f"{c:g}" for c in order],rotation=45,fontsize=8)
ax1.set_xlabel("STMN2-CE (copies/µL) in SDS/BSA/randomRNA background"); ax1.set_ylabel("within-row-normalized ∫I")
ax1.set_title(f"exp5 22nt in background — dose-response (ρ={rho:+.2f}, p={p:.1g})"); ax1.grid(alpha=0.3); ax1.legend(fontsize=8)
hm=dd.pivot_table(index="row",columns="col",values="int_all")
im=ax2.imshow(hm.values,aspect="auto",cmap="viridis")
ax2.set_xticks(range(NCOL)); ax2.set_yticks(range(8)); ax2.set_yticklabels([f"r{r}" for r in range(8)])
ax2.set_xlabel("plate column (layout differs top vs bottom)"); ax2.set_ylabel("row")
ax2.axhline(3.5,color="w",lw=1.5,ls="--")
ax2.set_title(f"raw ∫I heat-map (row ρ={rr:+.2f}, col-resid ρ={rc:+.2f})")
fig.colorbar(im,ax=ax2,fraction=0.046)
fig.suptitle("exp5 / 22nt STMN2-CE sensor in defined background — randomized split layout",y=1.02,fontsize=12)
fig.tight_layout(); fig.savefig(f"{BASE}/exp5_22_analysis.png",dpi=130,bbox_inches="tight")
print(f"\nSaved {BASE}/exp5_22_analysis.png")

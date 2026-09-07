#!/usr/bin/env python3
"""exp4 WATER (STMN_22 = 22nt GT15-STMN2-CE sensor) full pipeline: blind ranking + gradients
+ unblind dose-response + left-right spatial-artifact test (mirrors exp4 serum).
Grid 8x12 row-major; missing = col11 last row = LAST grid cell -> NO shift; row=w//12, col=w%12.
Layout (0-idx col -> copies/uL): c0=1e6,c1=10,c2=1e5,c3=100,c4=20,c5=200,c6=0,c7=1e3,c8=1,c9=1e4,c10=50,c11=0.1."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr, mannwhitneyu, friedmanchisquare
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

BASE="/mnt/c/Users/riccardo-s/Documents/CNT/cryptic_mRNA/exp4/water"; PREFIX="STMN_22_WATER__"
EXC=[570,640,670,750,780]; NCOL=12
GOOD=["ch6_5","ch7_5","ch10_2","ch9_4","ch8_4","ch7_6"]   # frac r2>0.95 == 1.0 in water
COPIES={0:1e6,1:10.,2:1e5,3:100.,4:20.,5:200.,6:0.,7:1e3,8:1.,9:1e4,10:50.,11:0.1}
LOGC={c:np.log10(v+1) for c,v in COPIES.items()}

# ---- model-free ∫I ----
em=None; inten={}
for e in EXC:
    a=np.loadtxt(f"{BASE}/{PREFIX}{e}.txt")
    if em is None: em=a[:,0]; pk=(em>=900)&(em<=1350); bw=(em>=1400)&(em<=1600)
    W=a[:,2:]; inten[e]=(W[pk,:]-W[bw,:].mean(axis=0)).sum(axis=0)
int_all=sum(inten[e] for e in EXC)
fit=pd.read_csv(f"{BASE}/chirality_gaussian_descriptors_exp4_water.csv")
fit["int_all"]=[int_all[int(r.w)] for _,r in fit.iterrows()]
fit["fit_mean"]=fit[[f"{ch}_gauss_max" for ch in GOOD]].mean(axis=1)
fit["copies"]=fit["col"].map(COPIES)
df=fit[fit.int_all>0].reset_index(drop=True)
for e in EXC: df[f"int_{e}"]=[inten[e][int(w)] for w in df.w]
df["int_rn"]=df["int_all"]/df.groupby("row")["int_all"].transform("mean")
df["fit_rn"]=df["fit_mean"]/df.groupby("row")["fit_mean"].transform("mean")

# ---- gradients (raw) ----
print("="*74); print("GRADIENTS (raw combined ∫I)"); print("="*74)
rho_r,p_r=spearmanr(df["row"],df["int_all"]); print(f"ROW top-down: rho={rho_r:+.2f} p={p_r:.2g}")
rm=df.groupby("row")["int_all"].mean()
print("   row means:", " ".join(f"r{r}={rm[r]:.3g}" for r in sorted(df.row.unique())))
df["res"]=df["int_all"]-df.groupby("row")["int_all"].transform("mean")
rho_c,p_c=spearmanr(df["col"],df["res"]); print(f"COLUMN within-row residual (left-right): rho={rho_c:+.2f} p={p_c:.2g}")

# ---- BLIND per-condition ----
CONDS=sorted(df["cond"].unique()) if "cond" in df else [f"C{c:02d}" for c in range(NCOL)]
df["cond"]=df["col"].map(lambda c:f"C{c:02d}")
mF=anova_lm(smf.ols("int_rn ~ 1",data=df).fit(),smf.ols("int_rn ~ C(cond)",data=df).fit())["Pr(>F)"].iloc[1]
print("\n"+"="*74); print(f"BLIND per-condition (row-normalized): C(cond) F-test p={mF:.2g}"); print("="*74)
cm=df.groupby("cond")["int_rn"].agg(["mean","sem"])
for c in cm["mean"].sort_values(ascending=False).index:
    print(f"   {c}:  {cm.loc[c,'mean']:.3f} +/- {cm.loc[c,'sem']:.3f}")

# ---- UNBLIND dose-response ----
print("\n"+"="*74); print("UNBLIND STMN2-CE dose-response (row-normalized)"); print("="*74)
order=[0,0.1,1,10,20,50,100,200,1e3,1e4,1e5,1e6]
print(f"{'copies':>8} {'col':>3} {'fit_rn':>7} {'mf_rn':>7} {'n':>3}")
for cp in order:
    s=df[df.copies==cp]; c=int(s['col'].iloc[0])
    print(f"{cp:>8g} {c:>3} {s.fit_rn.mean():>7.3f} {s.int_rn.mean():>7.3f} {len(s):>3}")
for lab,c in [("fitted","fit_rn"),("model-free","int_rn")]:
    rho,p=spearmanr(df["copies"],df[c]); print(f"  Spearman(copies) {lab:11s} rho={rho:+.2f} p={p:.2g}")

# ---- left-right spatial artifact test (baseline from below-threshold cols) ----
print("\n"+"="*74); print("SPATIAL left-right test + position detrend (baseline from <=200cp cols)"); print("="*74)
low=df[df.copies<=200]
rho_pos,p_pos=spearmanr(low["col"],low["int_rn"]); print(f"below-threshold cols: int_rn vs column position rho={rho_pos:+.2f} p={p_pos:.2g}")
low_cm=low.groupby("col").agg(m=("int_rn","mean"),nbr=("col",lambda s:np.mean([LOGC[cc] for cc in (s.iloc[0]-1,s.iloc[0]+1) if 0<=cc<NCOL])))
rho_nbr,p_nbr=spearmanr(low_cm["m"],low_cm["nbr"]); print(f"below-threshold cols: colmean vs neighbor-conc score rho={rho_nbr:+.2f} p={p_nbr:.2g}")
b1,b0=np.polyfit(low["col"],low["int_rn"],1)
print(f"spatial baseline: int_rn ~= {b0:.4f} {b1:+.4f}*col")
df["pos_base"]=b0+b1*df["col"]; df["int_detr"]=df["int_rn"]-df["pos_base"]+1.0
print(f"\n{'copies':>8} {'col':>3} {'raw_rn':>7} {'posbase':>7} {'detrended':>9}")
for cp in order:
    s=df[df.copies==cp]; c=int(s['col'].iloc[0])
    print(f"{cp:>8g} {c:>3} {s.int_rn.mean():>7.3f} {s.pos_base.mean():>7.3f} {s.int_detr.mean():>9.3f}")
hi=df[df.copies>=1e3]["int_detr"]; bl=df[df.copies==0]["int_detr"]; mid=df[(df.copies>=10)&(df.copies<=200)]["int_detr"]
print(f"\nafter detrend: high>=1e3 mean={hi.mean():.3f} vs blank={bl.mean():.3f} MWU p={mannwhitneyu(hi,bl).pvalue:.2g};"
      f" vs mid(10-200) p={mannwhitneyu(hi,mid).pvalue:.2g}")
rho,p=spearmanr(df["copies"],df["int_detr"]); print(f"Spearman(copies, detrended) rho={rho:+.2f} p={p:.2g}")

# ---- figure ----
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(15,6))
xa={cp:i for i,cp in enumerate(order)}
for col,mk,lc,lab in [("int_rn","s","#ff7f0e","raw row-norm ∫I"),("int_detr","D","#2ca02c","position-detrended")]:
    g=df.groupby("copies")[col].agg(["mean","sem"])
    ax1.errorbar([xa[cp] for cp in order],[g.loc[cp,"mean"] for cp in order],[g.loc[cp,"sem"] for cp in order],
                 marker=mk,ms=6,capsize=3,color=lc,lw=1.6,label=lab)
ax1.axhline(1,color="k",lw=0.6,ls=":"); ax1.set_xticks(range(len(order)))
ax1.set_xticklabels(["0"]+[f"{c:g}" for c in order[1:]],rotation=45,fontsize=8)
ax1.set_xlabel("STMN2-CE (copies/µL) in water"); ax1.set_ylabel("normalized intensity")
ax1.set_title(f"exp4 WATER 22nt: dose-response (Spearman raw ρ={spearmanr(df.copies,df.int_rn)[0]:+.2f})"); ax1.grid(alpha=0.3); ax1.legend()
hm=df.pivot_table(index="row",columns="col",values="int_all")
im=ax2.imshow(hm.values,aspect="auto",cmap="viridis")
ax2.set_xticks(range(NCOL)); ax2.set_xticklabels([f"{COPIES[c]:g}" for c in range(NCOL)],rotation=45,fontsize=7)
ax2.set_yticks(range(8)); ax2.set_yticklabels([f"r{r}" for r in range(8)])
ax2.set_title(f"raw ∫I heat-map (row ρ={rho_r:+.2f}, col-resid ρ={rho_c:+.2f})"); ax2.set_xlabel("copies/µL"); ax2.set_ylabel("row")
fig.colorbar(im,ax=ax2,fraction=0.046)
fig.suptitle("exp4 WATER (STMN_22 sensor) — dose-response + gradients",y=1.02,fontsize=12)
fig.tight_layout(); fig.savefig(f"{BASE}/exp4_water_analysis.png",dpi=130,bbox_inches="tight")
print("\nSaved exp4_water_analysis.png")

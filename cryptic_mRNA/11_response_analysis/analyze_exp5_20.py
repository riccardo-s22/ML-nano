#!/usr/bin/env python3
"""exp5 / 20nt GT15-STMN2-CE sensor in defined background (SDS1%+BSA0.5%+randomRNA 30ng/mL).
Same pipeline as the 22 plate. OLD format: col0=emission, col1=ref(IGNORE), cols2+=96 wells; FULL grid.
Randomized split layout (TOP rows0-3 vs BOT rows4-7 different maps) => dose decorrelated from column.
1e5 fully testable here (both halves, correct 20nt sensor). NO wrong-sensor exclusion."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr, mannwhitneyu, rankdata, chi2 as chi2d
from itertools import combinations
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests

BASE="/mnt/c/Users/riccardo-s/Documents/CNT/cryptic_mRNA/exp5/20"; PREFIX="STMN_20_WATER_2H__"; NCOL=12
EXC=[570,640,670,750,780]
GOOD=['ch6_5','ch10_2','ch9_4','ch8_4','ch7_6','ch8_6','ch8_7','ch9_5','ch10_3','ch10_5']
TOP={0:(1e5,'d'),1:(0,'b'),2:(5000,'d'),3:(1,'d'),4:(1000,'d'),5:(10,'d'),6:(0,'z'),7:(50,'d'),8:(500,'d'),9:(0.1,'d'),10:(100,'d'),11:(1e4,'d')}
BOT={0:(0.1,'d'),1:(1e4,'d'),2:(50,'d'),3:(500,'d'),4:(0,'b'),5:(1000,'d'),6:(5000,'d'),7:(100,'d'),8:(10,'d'),9:(1e5,'d'),10:(1,'d'),11:(0,'z')}

# ---- model-free ∫I (old format: cols2+) ----
em=None; inten={}
for e in EXC:
    a=np.loadtxt(f"{BASE}/{PREFIX}{e}.txt")
    if em is None: em=a[:,0]; pk=(em>=900)&(em<=1350); bw=(em>=1400)&(em<=1600)
    W=a[:,2:]; inten[e]=(W[pk,:]-W[bw,:].mean(axis=0)).sum(axis=0)
int_all=sum(inten[e] for e in EXC)
rows=[]
for w in range(len(int_all)):
    r=w//NCOL; c=w%NCOL; cp,lab=(TOP if r<=3 else BOT)[c]
    rows.append(dict(w=w,row=r,col=c,half=("top" if r<=3 else "bot"),copies=cp,label=lab,int_all=int_all[w]))
df=pd.DataFrame(rows); df=df[df.int_all>0].reset_index(drop=True)
df["int_rn"]=df["int_all"]/df.groupby("row")["int_all"].transform("mean")
df["lc"]=np.log10(df["copies"]+1)
# fitted readout
fit=pd.read_csv(f"{BASE}/chirality_gaussian_descriptors_exp5_20.csv")
df["fit"]=fit[[f"{ch}_gauss_max" for ch in GOOD]].mean(axis=1).values[df.w.values]
df["fit_rn"]=df["fit"]/df.groupby("row")["fit"].transform("mean")

print("="*78); print("exp5/20  GRADIENTS"); print("="*78)
rr,pr=spearmanr(df.row,df.int_all); df["res"]=df.int_all-df.groupby("row")["int_all"].transform("mean")
rc,pc=spearmanr(df.col,df.res); print(f"row heating rho={rr:+.2f} p={pr:.2g} | left-right residual rho={rc:+.2f} p={pc:.2g}")

print("\n"+"="*78); print("SPECIFICITY: blank_noRNA vs zero_bg (raw + left-right detrend)"); print("="*78)
b1,b0=np.polyfit(df.col,df.int_rn,1); df["detr"]=df.int_rn-(b0+b1*df.col)+1.0
b=df[df.label=='b']; z=df[df.label=='z']
print(f"blank_noRNA (cols{sorted(b.col.unique())}) rn={b.int_rn.mean():.3f} vs zero_bg (cols{sorted(z.col.unique())}) rn={z.int_rn.mean():.3f}"
      f"  raw MWU p={mannwhitneyu(b.int_rn,z.int_rn).pvalue:.2g}")
print(f"  after left-right detrend: {b.detr.mean():.3f} vs {z.detr.mean():.3f} MWU p={mannwhitneyu(b.detr,z.detr).pvalue:.2g}")

print("\n"+"="*78); print("DOSE-RESPONSE (all concs incl 1e5, which is valid on the 20 plate)"); print("="*78)
order=[0,0.1,1,10,50,100,500,1000,5000,1e4,1e5]
print(f"{'copies':>8} {'topcol':>6} {'botcol':>6} {'top_rn':>7} {'bot_rn':>7} {'all_rn':>7} {'n':>3}")
for cp in order:
    s=df[df.copies==cp]; t=s[s.half=='top']; bo=s[s.half=='bot']
    print(f"{cp:>8g} {int(t.col.iloc[0]) if len(t) else -1:>6} {int(bo.col.iloc[0]) if len(bo) else -1:>6} "
          f"{t.int_rn.mean() if len(t) else np.nan:>7.3f} {bo.int_rn.mean() if len(bo) else np.nan:>7.3f} {s.int_rn.mean():>7.3f} {len(s):>3}")
print(f"corr(log_copies,col)={np.corrcoef(df.lc,df.col)[0,1]:+.2f}")
for lab,col in [("model-free","int_rn"),("fitted","fit_rn")]:
    rho,p=spearmanr(df.lc,df[col]); m1=smf.ols(f"{col}~col+lc",df).fit()
    print(f"  {lab:11s}: Spearman(copies) rho={rho:+.2f} p={p:.2g} | OLS ~col+log_copies: coef={m1.params['lc']:+.4f} p={m1.pvalues['lc']:.2g}")
# cross-half discriminator
for lab,col in [("model-free","int_rn"),("fitted","fit_rn")]:
    prof=df.groupby(["copies","half"])[col].mean().unstack().dropna()
    rt,pt=spearmanr(prof["top"],prof["bot"]); print(f"  CROSS-HALF {lab:11s} top-vs-bottom profile: rho={rt:+.2f} p={pt:.2g} (n={len(prof)})")
# threshold
hi=df[df.copies>=1000]["int_rn"]; zero=df[df.label.isin(['b','z'])]["int_rn"]; low=df[(df.copies>0)&(df.copies<1000)]["int_rn"]
print(f"  high(>=1e3)={hi.mean():.3f} vs zero-pooled={zero.mean():.3f} MWU p={mannwhitneyu(hi,zero).pvalue:.2g} | vs low(0.1-500)={low.mean():.3f} p={mannwhitneyu(hi,low).pvalue:.2g}")

print("\n"+"="*78); print("WITHIN-ROW trend + concordance"); print("="*78)
for r in range(8):
    s=df[df.row==r]; rho,p=spearmanr(s.lc,s.int_rn); print(f"  row{r} {s.half.iloc[0]}: rho={rho:+.2f} p={p:.2g}")
def kw(m):
    a,n=m.shape; R=np.vstack([rankdata(m[i]) for i in range(a)]); Rj=R.sum(0)
    W=12*((Rj-Rj.mean())**2).sum()/(a**2*(n**3-n)); return W,chi2d.sf(a*(n-1)*W,n-1)
for half,rws in [('TOP',[0,1,2,3]),('BOT',[4,5,6,7])]:
    piv=df[df.row.isin(rws)].pivot_table(index='row',columns='copies',values='int_rn').dropna(axis=1)
    W,pW=kw(piv.values); print(f"  {half} within-half Kendall W={W:.2f} p={pW:.2g} (fixed map => spatial)")

print("\n"+"="*78); print("RATIOMETRIC / intensity-independent readouts (cross-half filter)"); print("="*78)
d=fit[(fit.copies!=1e5)].copy() if False else fit.copy()
d["lc"]=np.log10(d["copies"]+1)
readouts={}
for a_,b_ in combinations(GOOD,2): d[f"R_{a_}/{b_}"]=np.log(d[f"{a_}_gauss_max"]/d[f"{b_}_gauss_max"]); readouts[f"R_{a_}/{b_}"]="ratio"
for ch in GOOD: readouts[f"{ch}_gauss_em_center"]="wavelength"; readouts[f"{ch}_gauss_fwhm"]="fwhm"
res=[]
for col,fam in readouts.items():
    tp=d[d.half=='top']; bt=d[d.half=='bot']
    rt,pt=spearmanr(tp.lc,tp[col]); rb,pb=spearmanr(bt.lc,bt[col]); rp,pp=spearmanr(d.lc,d[col])
    res.append(dict(readout=col,family=fam,rho_top=rt,p_top=pt,rho_bot=rb,p_bot=pb,
                    both_ok=(np.sign(rt)==np.sign(rb) and pt<0.05 and pb<0.05),rho_pool=rp,p_pool=pp))
R=pd.DataFrame(res); R["fdr"]=multipletests(R.p_pool,method="fdr_bh")[1]
print(f"Scanned {len(R)} readouts. Pass stringent cross-half filter (signif same-sign both halves): {int(R.both_ok.sum())}")
hit=R[R.both_ok].sort_values("p_pool")
if len(hit): print(hit[["readout","family","rho_top","p_top","rho_bot","p_bot","rho_pool","fdr"]].to_string(index=False))
print("Any surviving FDR<0.05?:", "NONE" if (R.fdr<0.05).sum()==0 else "")
if (R.fdr<0.05).sum(): print(R[R.fdr<0.05][["readout","family","rho_pool","p_pool","fdr","both_ok"]].to_string(index=False))
print("Top-3 pooled for reference:")
print(R.reindex(R.p_pool.sort_values().index)[["readout","family","rho_pool","p_pool","fdr","both_ok"]].head(3).to_string(index=False))

# ---- figure ----
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(15,6)); xa={cp:i for i,cp in enumerate(order)}
for half,cl,mk in [("top","#1f77b4","o"),("bot","#d62728","s")]:
    s=df[df.half==half].groupby("copies")["int_rn"].agg(["mean","sem"]); xs=[cp for cp in order if cp in s.index]
    ax1.errorbar([xa[cp] for cp in xs],[s.loc[cp,"mean"] for cp in xs],[s.loc[cp,"sem"] for cp in xs],marker=mk,ms=6,capsize=3,color=cl,lw=1.6,label=f"{half} rows")
g=df.groupby("copies")["int_rn"].agg(["mean","sem"])
ax1.errorbar([xa[cp] for cp in order],[g.loc[cp,"mean"] for cp in order],[g.loc[cp,"sem"] for cp in order],marker="D",ms=8,color="k",lw=2.4,capsize=3,label="pooled",zorder=10)
ax1.axhline(zero.mean(),color="grey",lw=0.8,ls=":",label="0-target level")
ax1.set_xticks(range(len(order))); ax1.set_xticklabels([f"{c:g}" for c in order],rotation=45,fontsize=8)
ax1.set_xlabel("STMN2-CE (copies/µL) in SDS/BSA/randomRNA"); ax1.set_ylabel("within-row-normalized ∫I")
ax1.set_title(f"exp5 20nt in background — dose (ρ={spearmanr(df.lc,df.int_rn)[0]:+.2f})"); ax1.grid(alpha=0.3); ax1.legend(fontsize=8)
hm=df.pivot_table(index="row",columns="col",values="int_all"); im=ax2.imshow(hm.values,aspect="auto",cmap="viridis")
ax2.set_xticks(range(NCOL)); ax2.set_yticks(range(8)); ax2.set_yticklabels([f"r{r}" for r in range(8)]); ax2.axhline(3.5,color="w",lw=1.5,ls="--")
ax2.set_xlabel("plate column (layout differs top/bottom)"); ax2.set_ylabel("row")
ax2.set_title(f"raw ∫I heat-map (row ρ={rr:+.2f}, col-resid ρ={rc:+.2f})"); fig.colorbar(im,ax=ax2,fraction=0.046)
fig.suptitle("exp5 / 20nt STMN2-CE sensor in defined background — randomized split layout",y=1.02,fontsize=12)
fig.tight_layout(); fig.savefig(f"{BASE}/exp5_20_analysis.png",dpi=130,bbox_inches="tight"); print(f"\nSaved {BASE}/exp5_20_analysis.png")

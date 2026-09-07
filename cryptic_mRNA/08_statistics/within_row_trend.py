#!/usr/bin/env python3
"""Within-EACH-ROW concentration trend, and whether it is STABLE (reproducible) across the 8 rows.
Each row = one independent replicate of the full 12-point titration (row-major, one well per conc/row).
- Remove top-down heating by dividing each well by its own row mean (within-row normalization).
- Per-row Spearman(copies, int) : is the trend present in that row?
- Kendall's W concordance across rows: do all rows agree on the concentration RANK-ordering?
- Friedman test: is there any reproducible conc effect blocking on row?
Run for both SERUM and WATER."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr, friedmanchisquare, rankdata

PLATES = {
 "SERUM": dict(base="/mnt/c/Users/riccardo-s/Documents/CNT/cryptic_mRNA/exp4/serum",
               prefix="STMN_22_Serum__", MISS=90,
               COPIES={0:1e6,1:0.1,2:1e4,3:1.,4:1e3,5:10.,6:0.,7:20.,8:50.,9:100.,10:500.,11:1e5}),
 "WATER": dict(base="/mnt/c/Users/riccardo-s/Documents/CNT/cryptic_mRNA/exp4/water",
               prefix="STMN_22_WATER__", MISS=95,
               COPIES={0:1e6,1:10.,2:1e5,3:100.,4:20.,5:200.,6:0.,7:1e3,8:1.,9:1e4,10:50.,11:0.1}),
}
EXC=[570,640,670,750,780]; NCOL=12
ORDER=[0,0.1,1,10,20,50,100,200,500,1e3,1e4,1e5,1e6]

def load(cfg):
    em=None; inten={}
    for e in EXC:
        a=np.loadtxt(f"{cfg['base']}/{cfg['prefix']}{e}.txt")
        if em is None: em=a[:,0]; pk=(em>=900)&(em<=1350); bw=(em>=1400)&(em<=1600)
        W=a[:,2:]; inten[e]=(W[pk,:]-W[bw,:].mean(axis=0)).sum(axis=0)
    int_all=sum(inten[e] for e in EXC)
    rows=[]
    for w in range(len(int_all)):
        p=w if w<cfg["MISS"] else w+1; r=p//NCOL; c=p%NCOL
        rows.append(dict(w=w,row=r,col=c,copies=cfg["COPIES"][c],int_all=int_all[w]))
    df=pd.DataFrame(rows); df=df[df.int_all>0].reset_index(drop=True)
    df["int_rn"]=df["int_all"]/df.groupby("row")["int_all"].transform("mean")
    return df

def kendall_w(mat):
    """mat: rows=judges(plate rows), cols=items(concentrations). Rank items within each judge, sum ranks."""
    m,n=mat.shape                       # m judges, n items
    R=np.vstack([rankdata(mat[i]) for i in range(m)])
    Rj=R.sum(axis=0); S=((Rj-Rj.mean())**2).sum()
    W=12*S/(m**2*(n**3-n))
    chi2=m*(n-1)*W                      # ~ chi2 with n-1 df
    from scipy.stats import chi2 as chi2d
    return W, chi2, chi2d.sf(chi2,n-1)

figs={}
for name,cfg in PLATES.items():
    df=load(cfg)
    conc=sorted(df.copies.unique())
    print("="*76); print(f"{name}: within-row concentration trend + cross-row stability"); print("="*76)
    print("Per-row Spearman(log copies, within-row-normalized ∫I):")
    print(f"{'row':>3} {'mean∫I':>9} {'rho':>6} {'p':>8}")
    perrow=[]
    for r in sorted(df.row.unique()):
        s=df[df.row==r]; rho,p=spearmanr(np.log10(s.copies+1),s.int_rn)
        perrow.append((r,rho,p)); print(f"{r:>3} {s.int_all.mean():>9.3g} {rho:>+6.2f} {p:>8.2g}")
    rhos=np.array([x[1] for x in perrow])
    print(f"  -> per-row ρ: mean={rhos.mean():+.2f}, all positive={ (rhos>0).all() }, "
          f"range [{rhos.min():+.2f},{rhos.max():+.2f}], #signif(p<.05)={sum(x[2]<.05 for x in perrow)}/8")

    # row x concentration matrix (one well per conc per row; blank col has 7)
    piv=df.pivot_table(index="row",columns="copies",values="int_rn")
    # keep only concentrations present in every row for a clean concordance matrix
    full=piv.dropna(axis=1)
    W,chi2,pW=kendall_w(full.values)
    print(f"\nKendall's W (concordance of concentration ranking across rows, {full.shape[1]} concs x {full.shape[0]} rows):")
    print(f"  W={W:.3f}  chi2={chi2:.1f}  p={pW:.2g}   (W=1 => rows rank concentrations identically)")
    # Friedman on same matrix: subjects=rows, treatments=concentrations -> pass each conc column
    fr=friedmanchisquare(*[full.values[:,j] for j in range(full.shape[1])])
    print(f"Friedman (reproducible conc effect blocking on row): chi2={fr.statistic:.1f} p={fr.pvalue:.2g}")

    # mean profile across rows + how tightly rows track it (mean pairwise Spearman between rows)
    from itertools import combinations
    prs=[spearmanr(full.values[i],full.values[j]).correlation for i,j in combinations(range(full.shape[0]),2)]
    print(f"Mean pairwise between-row Spearman of the conc-profile: {np.nanmean(prs):+.2f} "
          f"(range {np.nanmin(prs):+.2f}..{np.nanmax(prs):+.2f})")
    figs[name]=(df,piv,perrow,W,pW)

# ---------- figure: per-row dose curves, one panel per plate ----------
fig,axes=plt.subplots(1,2,figsize=(16,6.5))
for ax,(name,(df,piv,perrow,W,pW)) in zip(axes,figs.items()):
    concs=[c for c in ORDER if c in piv.columns]
    x=range(len(concs)); cmap=plt.cm.viridis(np.linspace(0,1,8))
    for r in sorted(df.row.unique()):
        y=[piv.loc[r,c] if c in piv.columns else np.nan for c in concs]
        ax.plot(x,y,marker="o",ms=4,lw=1,color=cmap[r],alpha=0.8,label=f"row {r}")
    mean_prof=[piv[c].mean() if c in piv.columns else np.nan for c in concs]
    ax.plot(x,mean_prof,marker="s",ms=8,lw=2.6,color="k",label="row-mean",zorder=10)
    ax.axhline(1,color="grey",lw=0.6,ls=":")
    ax.set_xticks(list(x)); ax.set_xticklabels([f"{c:g}" for c in concs],rotation=45,fontsize=8)
    ax.set_xlabel("STMN2-CE (copies/µL)"); ax.set_ylabel("within-row-normalized ∫I")
    ax.set_title(f"exp4 {name} — per-row titration (Kendall W={W:.2f}, p={pW:.1g})")
    ax.grid(alpha=0.25); ax.legend(fontsize=7,ncol=2)
fig.suptitle("Within-each-row concentration trend and its stability across rows",y=1.02,fontsize=13)
fig.tight_layout()
out="/mnt/c/Users/riccardo-s/Documents/CNT/cryptic_mRNA/exp4/exp4_within_row_trend.png"
fig.savefig(out,dpi=130,bbox_inches="tight"); print(f"\nSaved {out}")

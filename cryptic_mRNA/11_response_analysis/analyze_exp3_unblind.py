#!/usr/bin/env python3
"""
exp3 UNBLINDED (v2 - no bogus position gradient; only legitimate row/heating drift removed).
One plate (water_2h, 8 rows x 11 cols), TWO experiments:
  A) STMN2-CE titration in constant 70 pg/uL neuron background (cols C0-C5):
       C0=20, C1=100, C4=10, C5=1 copies/uL; C2=0.1(top4)/0(bottom4); C3=0(top4)/0.1(bottom4)
  B) neuron-mRNA (non-target/matrix) titration, 0 STMN2-CE (cols C6-C10):
       C7=0, C10=10, C8=20, C9=70, C6=700 pg/uL
The whole-plate brightness is driven by NEURON mRNA (Exp A left half is uniformly 70pg -> uniformly
brighter than the mostly-low-neuron right half; that is the matrix effect, NOT a position artifact).
Metrics: fitted across-chir mean gauss_max + model-free integrated intensity. Drift removed by
row-normalization within plate only.
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr, mannwhitneyu

BASE="/mnt/c/Users/riccardo-s/Documents/CNT/cryptic_mRNA/exp3"
PREFIX="water_2h__"; EXC=[570,640,670,750,780]; NCOL=11
GOOD=["ch8_3","ch7_5","ch10_2","ch9_4","ch8_4","ch7_6","ch8_6"]

# ---- model-free integrated intensity per well ----
em=None; inten={}
for e in EXC:
    a=np.loadtxt(f"{BASE}/{PREFIX}{e}.txt")
    if em is None:
        em=a[:,0]; pk=(em>=900)&(em<=1350); bw=(em>=1400)&(em<=1600)
    W=a[:,2:]; inten[e]=(W[pk,:]-W[bw,:].mean(axis=0)).sum(axis=0)
int_all=sum(inten[e] for e in EXC)
fit=pd.read_csv(f"{BASE}/chirality_gaussian_descriptors_exp3.csv")

def label(r,c):
    if c in (0,1,4,5): return dict(expt="A", stmn={0:20.,1:100.,4:10.,5:1.}[c], neuron=70.)
    if c==2: return dict(expt="A", stmn=(0.1 if r<4 else 0.0), neuron=70.)
    if c==3: return dict(expt="A", stmn=(0.0 if r<4 else 0.1), neuron=70.)
    return dict(expt="B", stmn=0.0, neuron={6:700.,7:0.,8:20.,9:70.,10:10.}[c])

recs=[]
for _,row in fit.iterrows():
    r,c=int(row["row"]),int(row["col"]); w=r*NCOL+c
    d=dict(row=r,col=c,cond=row["cond"],int_all=int_all[w],
           fit_mean=np.mean([row[f"{ch}_gauss_max"] for ch in GOOD]))
    d.update(label(r,c)); recs.append(d)
df=pd.DataFrame(recs)
df=df[df["int_all"]>0].reset_index(drop=True)   # drop 1 dead well (C6 r5)
# legitimate drift removal ONLY: row (acquisition heating) normalization within plate
df["fit_rn"]=df["fit_mean"]/df.groupby("row")["fit_mean"].transform("mean")
df["int_rn"]=df["int_all"]/df.groupby("row")["int_all"].transform("mean")

# ===== Exp A: STMN2-CE dose (all at 70pg neuron) =====
A=df[df.expt=="A"].copy()
print("="*74); print("Exp A — STMN2-CE titration (0-100 copies/uL) at constant 70 pg/uL neuron"); print("="*74)
for lab,col in [("fitted","fit_rn"),("model-free","int_rn")]:
    rho,p=spearmanr(A["stmn"],A[col]); print(f"  {lab:11s} Spearman(copies) rho={rho:+.2f} p={p:.2g}")
z=A[A.stmn==0]["int_rn"]; lo=A[A.stmn==0.1]["int_rn"]; u,pu=mannwhitneyu(lo,z)
print(f"  balanced 0-vs-0.1 control (int): 0={z.mean():.3f} 0.1={lo.mean():.3f} MWU p={pu:.2g}")
print("  by copies (model-free):",{d:round(A[A.stmn==d]["int_rn"].mean(),3) for d in sorted(A.stmn.unique())})

# ===== Neuron-mRNA dose-response across the WHOLE plate, using ZERO-target wells =====
# 0-target wells: all Exp B + Exp A 0-copy wells (C2 bottom4 / C3 top4). All at their neuron level.
zt=df[df.stmn==0].copy()
print("\n"+"="*74); print("Neuron-mRNA (non-target) dose-response — ZERO-STMN2-CE wells, whole plate"); print("="*74)
for lab,col in [("fitted","fit_rn"),("model-free","int_rn")]:
    rho,p=spearmanr(zt["neuron"],zt[col]); print(f"  {lab:11s} Spearman(neuron pg) rho={rho:+.2f} p={p:.2g}")
    g=zt.groupby("neuron")[col].agg(["mean","sem","count"])
    for d in sorted(zt.neuron.unique()):
        print(f"     {d:6g} pg/uL : {g.loc[d,'mean']:.3f} +/- {g.loc[d,'sem']:.3f}  (n={int(g.loc[d,'count'])})")

# ---- figures ----
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,6))
Ax=[0,0.1,1,10,20,100]; xa={v:i for i,v in enumerate(Ax)}
for col,mk,lc,lab in [("fit_rn","o","#1f77b4","fitted gauss_max"),("int_rn","s","#ff7f0e","model-free ∫I")]:
    g=A.groupby("stmn")[col].agg(["mean","sem"])
    ax1.errorbar([xa[v] for v in Ax],[g.loc[v,"mean"] for v in Ax],[g.loc[v,"sem"] for v in Ax],
                 marker=mk,ms=7,capsize=3,color=lc,lw=1.8,label=lab)
ax1.axhline(1,color="k",lw=0.6,ls=":"); ax1.set_xticks(range(len(Ax))); ax1.set_xticklabels(map(str,Ax))
ax1.set_xlabel("STMN2-CE (copies/µL)  [70 pg/µL neuron background]")
ax1.set_ylabel("row-normalized intensity (heating drift removed)")
ax1.set_title("Exp A — STMN2-CE (target): FLAT, no response"); ax1.grid(alpha=0.3); ax1.legend()

Bx=[0,10,20,70,700]; xb={v:i for i,v in enumerate(Bx)}
for col,mk,lc,lab in [("fit_rn","o","#1f77b4","fitted gauss_max"),("int_rn","s","#ff7f0e","model-free ∫I")]:
    g=zt.groupby("neuron")[col].agg(["mean","sem"])
    ax2.errorbar([xb[v] for v in Bx],[g.loc[v,"mean"] for v in Bx],[g.loc[v,"sem"] for v in Bx],
                 marker=mk,ms=7,capsize=3,color=lc,lw=1.8,label=lab)
ax2.axhline(1,color="k",lw=0.6,ls=":"); ax2.set_xticks(range(len(Bx))); ax2.set_xticklabels(map(str,Bx))
ax2.set_xlabel("neuron mRNA (pg/µL)  [0 STMN2-CE]")
ax2.set_ylabel("row-normalized intensity (heating drift removed)")
ax2.set_title("Neuron mRNA (non-target/matrix): dose-dependent rise"); ax2.grid(alpha=0.3); ax2.legend()
fig.suptitle("exp3 water_2h UNBLINDED — no STMN2-CE target response; bulk neuron mRNA drives the signal",
             y=1.02,fontsize=13)
fig.tight_layout(); fig.savefig(f"{BASE}/exp3_unblinded_doseresponse.png",dpi=130,bbox_inches="tight")
print("\nSaved exp3_unblinded_doseresponse.png")

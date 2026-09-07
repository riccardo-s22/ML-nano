#!/usr/bin/env python3
"""exp3 SERUM UNBLINDED. 8 rows x 12 cols row-major (col11=7 reps). In serum matrix:
  STMN2-CE titration: C6=0, C9=0.1, C7=1, C5=10, C2=20, C3=50, C4=100, C8=1000, C1=10000 copies/uL;
                      C0 = STMN2-CE positive control 10 uM (synthetic, huge)
  neuron mRNA controls (0 STMN2-CE): C10=70 pg/uL, C11=700 pg/uL   (serum blank = C6, 0 copies)
Metrics: fitted mean gauss_max (10 clean chir) + model-free ∫I. Row(heating)-drift removed only."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr

BASE="/mnt/c/Users/riccardo-s/Documents/CNT/cryptic_mRNA/exp3/serum"; PREFIX="serum__"
EXC=[570,640,670,750,780]; NCOL=12
GOOD=["ch8_3","ch10_2","ch9_4","ch8_4","ch7_6","ch8_6","ch8_7","ch9_5","ch10_3","ch10_5"]
# column -> (kind, value)
CMAP={0:("posctrl",np.nan),1:("stmn",10000.),2:("stmn",20.),3:("stmn",50.),4:("stmn",100.),
      5:("stmn",10.),6:("stmn",0.),7:("stmn",1.),8:("stmn",1000.),9:("stmn",0.1),
      10:("neuron",70.),11:("neuron",700.)}

em=None; inten={}
for e in EXC:
    a=np.loadtxt(f"{BASE}/{PREFIX}{e}.txt")
    if em is None: em=a[:,0]; pk=(em>=900)&(em<=1350); bw=(em>=1400)&(em<=1600)
    W=a[:,2:]; inten[e]=(W[pk,:]-W[bw,:].mean(axis=0)).sum(axis=0)
int_all=sum(inten[e] for e in EXC)
fit=pd.read_csv(f"{BASE}/chirality_gaussian_descriptors_serum.csv")
fit["int_all"]=[int_all[int(r.w)] for _,r in fit.iterrows()]   # index model-free ∫I by file-well w
fit["fit_mean"]=fit[[f"{ch}_gauss_max" for ch in GOOD]].mean(axis=1)
fit["kind"]=fit["col"].map(lambda c:CMAP[c][0]); fit["val"]=fit["col"].map(lambda c:CMAP[c][1])
df=fit[fit.int_all>0].reset_index(drop=True)
df["fit_rn"]=df["fit_mean"]/df.groupby("row")["fit_mean"].transform("mean")
df["int_rn"]=df["int_all"]/df.groupby("row")["int_all"].transform("mean")

# ---- STMN2-CE dose-response (numeric copy series C1-C9,C6) ----
S=df[df.kind=="stmn"].copy()
print("="*76); print("STMN2-CE titration in SERUM (0 -> 10000 copies/uL)"); print("="*76)
for lab,col in [("fitted","fit_rn"),("model-free","int_rn")]:
    rho,p=spearmanr(S["val"],S[col]); print(f"  {lab:11s} Spearman(copies) rho={rho:+.2f} p={p:.2g}")
for lab,col in [("model-free","int_rn")]:
    g=S.groupby("val")[col].agg(["mean","sem"])
    print("   copies/uL   mean_rn   sem")
    for d in sorted(S["val"].unique()): print(f"     {d:9g}   {g.loc[d,'mean']:.3f}   {g.loc[d,'sem']:.3f}")
pc=df[df.kind=="posctrl"]["int_rn"]
print(f"  10 uM positive control (C0): {pc.mean():.3f} +/- {pc.sem():.3f}")

# ---- neuron mRNA in serum (0 = serum blank C6, 70, 700) ----
print("\n"+"="*76); print("neuron-mRNA controls in SERUM (0 STMN2-CE)"); print("="*76)
blank=df[(df.kind=='stmn')&(df.val==0)]["int_rn"]
n70=df[(df.kind=='neuron')&(df.val==70)]["int_rn"]; n700=df[(df.kind=='neuron')&(df.val==700)]["int_rn"]
for lab,v in [("0 (serum blank, C6)",blank),("70 pg/uL (C10)",n70),("700 pg/uL (C11)",n700)]:
    print(f"   neuron {lab:22s}: {v.mean():.3f} +/- {v.sem():.3f}")

# ---- figure ----
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(15,6))
Sx=[0,0.1,1,10,20,50,100,1000,10000]; xa={v:i for i,v in enumerate(Sx)}
for col,mk,lc,lab in [("fit_rn","o","#1f77b4","fitted mean"),("int_rn","s","#ff7f0e","model-free ∫I")]:
    g=S.groupby("val")[col].agg(["mean","sem"])
    ax1.errorbar([xa[v] for v in Sx],[g.loc[v,"mean"] for v in Sx],[g.loc[v,"sem"] for v in Sx],
                 marker=mk,ms=6,capsize=3,color=lc,lw=1.6,label=lab)
    # positive control at far right
    pcm=df[df.kind=='posctrl'][col]
    ax1.errorbar([len(Sx)+0.6],[pcm.mean()],[pcm.sem()],marker="*",ms=15,capsize=3,color=lc)
ax1.axhline(1,color="k",lw=0.6,ls=":")
ax1.set_xticks(list(range(len(Sx)))+[len(Sx)+0.6]); ax1.set_xticklabels([str(v) for v in Sx]+["10 µM\nPos.Ctrl"])
ax1.set_xlabel("STMN2-CE (copies/µL)  [in serum]"); ax1.set_ylabel("row-normalized intensity")
ax1.set_title("STMN2-CE titration (target): rising dose-response (ρ≈+0.62, p~1e-9); pos.ctrl brightest")
ax1.grid(alpha=0.3); ax1.legend()
Bx=[0,70,700]; xb={v:i for i,v in enumerate(Bx)}
for col,mk,lc,lab in [("fit_rn","o","#1f77b4","fitted mean"),("int_rn","s","#ff7f0e","model-free ∫I")]:
    vals={0:blank,70:n70,700:n700}
    ax2.errorbar([xb[v] for v in Bx],[df.loc[vals[v].index,col].mean() for v in Bx],
                 [df.loc[vals[v].index,col].sem() for v in Bx],marker=mk,ms=7,capsize=3,color=lc,lw=1.8,label=lab)
ax2.axhline(1,color="k",lw=0.6,ls=":"); ax2.set_xticks(range(len(Bx))); ax2.set_xticklabels(["0\n(serum blank)","70","700"])
ax2.set_xlabel("neuron mRNA (pg/µL)  [in serum, 0 STMN2-CE]"); ax2.set_ylabel("row-normalized intensity")
ax2.set_title("neuron mRNA (non-target): dim, slightly DECREASING — no matrix brightening"); ax2.grid(alpha=0.3); ax2.legend()
fig.suptitle("exp3 SERUM UNBLINDED (corrected read: first well missing) — STMN2-CE dose-response + bright 10 µM pos.ctrl; neuron mRNA does NOT brighten",
             y=1.02,fontsize=12)
fig.tight_layout(); fig.savefig(f"{BASE}/serum_unblinded_doseresponse.png",dpi=130,bbox_inches="tight")
print("\nSaved serum_unblinded_doseresponse.png")

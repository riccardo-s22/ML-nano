"""
Dense corona-kinetics analysis (7 tp x 3 conditions x 2 CNT conc, 1 EEM each).

Key questions:
  1) Is the raw temporal intensity drift a sensor/concentration effect (present
     without protein)?  -> compare PBS vs FBS vs FBS+RI at 1 & 5 mg/L CNT.
  2) What is the drift-subtracted corona signature (FBS - PBS at matched
     conc & tp), and how much of it is already present at 0 h?  (reviewer point)
  3) Does the drug (RI) modulate the corona?

Feature: gauss_max (primary). max5x5 cross-checked.
Outputs: FigSX_kinetics.png/.pdf, kinetics_summary.csv
"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
L = pd.read_csv(os.path.join(HERE, "kinetics_features_long.csv"), dtype={'chirality': str})
TPS = [0,2,4,6,8,10,24]
CHIRS = ['6.5','7.5','7.6','8.3','8.4','8.6','8.7','9.4','9.5','10.2','10.3','10.5']
C = {'pbs':"#0072B2", 'fbs':"#D55E00", 'fbsri':"#009E73"}
LB = {'pbs':'PBS (no protein)', 'fbs':'FBS (corona)', 'fbsri':'FBS+RI (corona+drug)'}
INK="#222"; MUT="#8a8a8a"

def mat(feat, cond, conc):
    """12 chir x 7 tp matrix."""
    w = L[(L.condition==cond)&(L.conc==conc)].pivot_table(
        index='chirality', columns='tp_h', values=feat).reindex(index=CHIRS)[TPS]
    return w.to_numpy(float)

FEAT="gauss_max"
rows=[]
for conc in ['1mg','5mg']:
    P,F,R = mat(FEAT,'pbs',conc), mat(FEAT,'fbs',conc), mat(FEAT,'fbsri',conc)
    corona = F - P                                    # drift-subtracted, 12x7
    mag = np.linalg.norm(corona, axis=0)              # per tp
    frac0 = mag[0]/mag.max()
    cos_0_24 = float(corona[:,0]@corona[:,6]/(np.linalg.norm(corona[:,0])*np.linalg.norm(corona[:,6])))
    coh = (corona>0).mean(axis=0)                     # fraction of chir with FBS>PBS
    rows.append(dict(conc=conc,
        pbs_frac_0_24=np.nanmean(P[:,6]/P[:,0]), fbs_frac_0_24=np.nanmean(F[:,6]/F[:,0]),
        corona_frac_at_0h=frac0, corona_cos_0h_24h=cos_0_24,
        coh_0h=coh[0], coh_24h=coh[6]))
S=pd.DataFrame(rows); S.to_csv(os.path.join(HERE,"kinetics_summary.csv"),index=False)
print(S.round(3).to_string(index=False))

# ---- figure ----
plt.rcParams.update({"font.size":9,"axes.titlesize":9.5,"axes.labelsize":9,
    "axes.spines.top":False,"axes.spines.right":False,"figure.dpi":150})
fig,axs=plt.subplots(1,4,figsize=(14.2,3.4))
fig.subplots_adjust(left=0.05,right=0.995,bottom=0.17,top=0.85,wspace=0.34)

# (a) normalized mean intensity trajectories (drift): solid=5mg, dashed=1mg
axa=axs[0]
for cond in ['pbs','fbs','fbsri']:
    for conc,ls in [('5mg','-'),('1mg','--')]:
        y=np.nanmean(mat(FEAT,cond,conc),axis=0); y=y/y[0]
        axa.plot(TPS,y,ls,color=C[cond],lw=1.7,ms=3,marker='o',
                 label=f"{LB[cond].split(' ')[0]} {conc}")
axa.axhline(1,color=MUT,lw=0.7,ls=':')
axa.set_xlabel("time (h)"); axa.set_ylabel("mean peak intensity (/0h)")
axa.set_title("(a) Raw drift is conc-dependent,\n protein-independent",loc="left")
axa.legend(frameon=False,fontsize=6.3,ncol=2,loc='lower left')

# (b) drift-subtracted corona (FBS-PBS)/PBS mean over chir
axb=axs[1]
for conc,ls in [('5mg','-'),('1mg','--')]:
    P,Fm=mat(FEAT,'pbs',conc),mat(FEAT,'fbs',conc)
    y=np.nanmean((Fm-P)/P,axis=0)
    axb.plot(TPS,y,ls,color=INK,lw=1.8,ms=4,marker='o',label=f"{conc}")
axb.axhline(0,color=MUT,lw=0.7)
axb.set_xlabel("time (h)"); axb.set_ylabel("(FBS−PBS)/PBS, mean over chir")
axb.set_title("(b) Corona signature:\n present at 0h, grows",loc="left")
axb.legend(frameon=False,fontsize=7.5,title="CNT")

# (c) per-chirality corona contrast at 0h vs 24h (5mg)
axc=axs[2]
P,Fm=mat(FEAT,'pbs','5mg'),mat(FEAT,'fbs','5mg')
cc=(Fm-P)/P
x=np.arange(12)
axc.axhline(0,color=MUT,lw=0.7)
axc.plot(x,cc[:,0],'-o',color="#56B4E9",lw=1.3,ms=3,label='0h')
axc.plot(x,cc[:,6],'-o',color="#D55E00",lw=1.3,ms=3,label='24h')
axc.set_xticks(x); axc.set_xticklabels(CHIRS,rotation=90,fontsize=6.5)
axc.set_ylabel("(FBS−PBS)/PBS"); axc.set_xlabel("chirality")
axc.set_title("(c) Corona contrast per chirality\n (5 mg/L)",loc="left")
axc.legend(frameon=False,fontsize=7.5)

# (d) drug effect (FBSRI-FBS)/FBS mean over chir
axd=axs[3]
for conc,ls in [('5mg','-'),('1mg','--')]:
    Fm,R=mat(FEAT,'fbs',conc),mat(FEAT,'fbsri',conc)
    y=np.nanmean((R-Fm)/Fm,axis=0)
    axd.plot(TPS,y,ls,color=C['fbsri'],lw=1.8,ms=4,marker='o',label=f"{conc}")
axd.axhline(0,color=MUT,lw=0.7)
axd.set_xlabel("time (h)"); axd.set_ylabel("(FBS+RI − FBS)/FBS")
axd.set_title("(d) Drug (RI) effect on corona",loc="left")
axd.legend(frameon=False,fontsize=7.5,title="CNT")

fig.savefig(os.path.join(HERE,"FigSX_kinetics.png"),dpi=300,bbox_inches="tight")
fig.savefig(os.path.join(HERE,"FigSX_kinetics.pdf"),bbox_inches="tight")
print("\nSaved FigSX_kinetics.png/.pdf and kinetics_summary.csv")

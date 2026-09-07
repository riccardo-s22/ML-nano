"""
exp4 technical controls (FIT_controls_ex4.xlsx) — the strongest control set:
3 conditions x 5 replicates x 12 chiralities x 4 timepoints (0/3/6/24h),
with fitted intensity + excitation/emission centers.

  PBS      (NEG)  = no-protein sensor baseline
  FBS      (POS)  = reference corona
  FBS+HEP  (DRUG) = corona + heparin

Tests, with replication (n=5):
  1) Does the bare sensor drift over time (PBS)?           paired 0h vs 24h
  2) Does the corona suppress that drift (FBS/FBS+HEP)?    condition x time
  3) Drift-subtracted corona signal (FBS-PBS) per chir     Welch t at 24h
  4) Is the PBS drift diameter-dependent (paper's time x diameter claim,
     but in protein-free buffer)?                          drift vs d(n,m)
  5) Emission-center shift over time per condition (solvatochromic check)

Outputs (exp5/tech_controls/analysis/): exp4_controls_summary.csv, FigSX_exp4_controls.png/.pdf
"""
import os, numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP4 = r"/mnt/c/Users/riccardo-s/Documents/CNT/targetALS/exp4"
HERE = os.path.dirname(os.path.abspath(__file__))
TPS = ['0h','3h','6h','24h']; TN = [0,3,6,24]
COND = ['PBS','FBS','FBS+HEP']
C = {'PBS':"#0072B2",'FBS':"#D55E00",'FBS+HEP':"#009E73"}
INK="#222"; MUT="#8a8a8a"

df = pd.read_excel(os.path.join(EXP4,"FIT_controls_ex4.xlsx"))
df['cond'] = df['sample'].str.replace(r'_\d+$','',regex=True)
df['chir'] = df['chir'].astype(int); df['chir.1']=df['chir.1'].astype(int)
df['ch'] = df['chir'].astype(str)+'.'+df['chir.1'].astype(str)
def diam(n,m): return 0.0783*np.sqrt(n*n+n*m+m*m)   # nm
df['d'] = [diam(n,m) for n,m in zip(df['chir'],df['chir.1'])]
CHIRS = df.groupby('ch')['d'].first().sort_values().index.tolist()

# ---- 1&2: intensity trajectories + paired tests ----
rows=[]
print("=== intensity: paired 0h vs 24h (per-sample mean over chir, n=5) ===")
for cond in COND:
    g=df[df.cond==cond].groupby('sample')[[f'intesnity_{t}' for t in TPS]].mean()
    gn=g.div(g['intesnity_0h'],axis=0)
    t,p=stats.ttest_rel(g['intesnity_24h'],g['intesnity_0h'])
    frac=gn['intesnity_24h'].mean()-1
    rows.append(dict(cond=cond, intensity_frac_0_24=frac, intensity_paired_p=p))
    print(f"  {cond:8s} +{frac*100:5.1f}%  t={t:.2f} p={p:.3f}   traj="+
          " ".join(f"{gn[f'intesnity_{t2}'].mean():.2f}" for t2 in TPS))

# ---- 3: drift-subtracted corona per chirality at 24h (Welch t) ----
print("\n=== corona (PBS - FBS) intensity at 24h, per chirality (Welch t, n=5 vs 5) ===")
corona_rows=[]
for ch in CHIRS:
    pbs=df[(df.cond=='PBS')&(df.ch==ch)]['intesnity_24h'].to_numpy()
    fbs=df[(df.cond=='FBS')&(df.ch==ch)]['intesnity_24h'].to_numpy()
    t,p=stats.ttest_ind(pbs,fbs,equal_var=False)
    corona_rows.append(dict(ch=ch, d=df[df.ch==ch]['d'].iloc[0],
        pbs24=pbs.mean(), fbs24=fbs.mean(), diff=pbs.mean()-fbs.mean(), p=p))
CR=pd.DataFrame(corona_rows)
nsig=(CR.p<0.05).sum()
print(CR.round(3).to_string(index=False))
print(f"  {nsig}/12 chiralities: PBS != FBS at 24h (p<0.05)")

# ---- 4: is PBS drift diameter-dependent? ----
pbs_drift=[]
for ch in CHIRS:
    s=df[(df.cond=='PBS')&(df.ch==ch)]
    fr=(s['intesnity_24h'].to_numpy()/s['intesnity_0h'].to_numpy())-1
    pbs_drift.append(fr.mean())
pbs_drift=np.array(pbs_drift); dvec=np.array([df[df.ch==ch]['d'].iloc[0] for ch in CHIRS])
r_dd,p_dd=stats.pearsonr(dvec,pbs_drift)
print(f"\n=== PBS 0->24h fractional drift vs tube diameter: r={r_dd:+.2f} p={p_dd:.3f} ===")
print("   (paper claims larger-diameter tubes brighten more 'during corona formation')")

# ---- 5: emission-center shift over time ----
print("\n=== emission-center shift 0h->24h (nm, mean over chir & reps) ===")
em_rows=[]
for cond in COND:
    s=df[df.cond==cond]
    shift=[ (s[f'emission_{t}']-s['emission_0h']).mean() for t in TPS]
    em_rows.append(dict(cond=cond, em_shift_24h=shift[-1], em_shift_max=max(shift,key=abs)))
    print(f"  {cond:8s} "+" ".join(f"{t}={v:+.2f}" for t,v in zip(TPS,shift)))

pd.DataFrame(rows).merge(pd.DataFrame(em_rows),on='cond').to_csv(
    os.path.join(HERE,"exp4_controls_summary.csv"),index=False)
CR.to_csv(os.path.join(HERE,"exp4_corona_per_chirality.csv"),index=False)

# ===================== FIGURE =====================
plt.rcParams.update({"font.size":9,"axes.titlesize":9.5,"axes.labelsize":9,
    "axes.spines.top":False,"axes.spines.right":False,"figure.dpi":150})
fig,axs=plt.subplots(1,4,figsize=(14.2,3.4))
fig.subplots_adjust(left=0.05,right=0.995,bottom=0.17,top=0.85,wspace=0.36)

# (a) intensity trajectories mean±sd (n=5)
axa=axs[0]
for cond in COND:
    g=df[df.cond==cond].groupby('sample')[[f'intesnity_{t}' for t in TPS]].mean()
    gn=g.div(g['intesnity_0h'],axis=0).to_numpy()
    axa.errorbar(TN,gn.mean(0),yerr=gn.std(0),color=C[cond],lw=1.8,marker='o',ms=4,
                 capsize=2,label=cond)
axa.axhline(1,color=MUT,lw=0.7,ls=':')
axa.set_xlabel("time (h)"); axa.set_ylabel("mean peak intensity (/0h)")
axa.set_title("(a) PBS drifts +29%, corona\n suppresses it (n=5)",loc="left")
axa.legend(frameon=False,fontsize=7.5)

# (b) drift-subtracted corona per chirality at 24h
axb=axs[1]
x=np.arange(12)
axb.bar(x,CR['diff'].to_numpy(),color=['#333' if p<0.05 else MUT for p in CR['p']],alpha=0.85)
axb.axhline(0,color=INK,lw=0.8)
axb.set_xticks(x); axb.set_xticklabels(CR['ch'],rotation=90,fontsize=6.5)
axb.set_ylabel("PBS − FBS intensity @24h"); axb.set_xlabel("chirality")
axb.set_title(f"(b) Corona suppression per chir\n ({nsig}/12 sig, dark)",loc="left")

# (c) PBS drift vs diameter
axc=axs[2]
axc.scatter(dvec,pbs_drift*100,s=30,color=C['PBS'],zorder=3)
b,a=np.polyfit(dvec,pbs_drift*100,1)
xs=np.linspace(dvec.min(),dvec.max(),50); axc.plot(xs,b*xs+a,color=INK,lw=1.2,ls='--')
for xi,yi,ch in zip(dvec,pbs_drift*100,CR['ch']):
    axc.annotate(ch,(xi,yi),fontsize=6,xytext=(2,2),textcoords='offset points',color=MUT)
axc.set_xlabel("tube diameter (nm)"); axc.set_ylabel("PBS 0→24h drift (%)")
axc.set_title(f"(c) Bare-sensor drift vs diameter\n r={r_dd:+.2f}, p={p_dd:.2f}",loc="left")

# (d) emission-center shift over time
axd=axs[3]
for cond in COND:
    s=df[df.cond==cond]
    shift=[ (s[f'emission_{t}']-s['emission_0h']).mean() for t in TPS]
    sd=[ (s[f'emission_{t}']-s['emission_0h']).std()/np.sqrt(5) for t in TPS]
    axd.errorbar(TN,shift,yerr=sd,color=C[cond],lw=1.7,marker='o',ms=4,capsize=2,label=cond)
axd.axhline(0,color=MUT,lw=0.7)
axd.set_xlabel("time (h)"); axd.set_ylabel("emission-center shift (nm)")
axd.set_title("(d) Emission shift not\n corona-specific",loc="left")
axd.legend(frameon=False,fontsize=7.3)

fig.savefig(os.path.join(HERE,"FigSX_exp4_controls.png"),dpi=300,bbox_inches="tight")
fig.savefig(os.path.join(HERE,"FigSX_exp4_controls.pdf"),bbox_inches="tight")
print("\nSaved FigSX_exp4_controls.png/.pdf, exp4_controls_summary.csv, exp4_corona_per_chirality.csv")

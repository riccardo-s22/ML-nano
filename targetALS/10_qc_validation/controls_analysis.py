"""
Technical-control analysis (A-D) for the reviewer comment:
  "since the 0 h timepoint is after serum addition, earlier measurements might
   help capture the initial exchange regime."

Controls (same 0/6/24 h as the study):
  P (PBS, n=4)  = no-protein sensor baseline  -> intrinsic drift
  F (FBS, n=3)  = homogeneous reference corona -> reference maturation / 0h-yardstick

A  PBS baseline drift (is the no-protein baseline static?)
B  Corona vs baseline (is the serum temporal change bigger / distinct from drift?)
C  How much corona is already present at 0 h (baseline-corrected F-P signature)
D  Does patient serum mature along the FBS reference axis or the PBS drift axis?

Two feature bases, both from the paper: max5x5 and gauss_max.
Outputs: several CSVs + a multi-panel figure, in this folder.
"""
import os, sys
import numpy as np
import pandas as pd
from scipy import stats

EXP5 = r"/mnt/c/Users/riccardo-s/Documents/CNT/targetALS/exp5"
MPX  = os.path.join(EXP5, "multiplexing_results")
PD_  = os.path.join(EXP5, "physical_descriptors")
HERE = os.path.dirname(os.path.abspath(__file__))
CHIRS = ['6.5','7.5','7.6','8.3','8.4','8.6','8.7','9.4','9.5','10.2','10.3','10.5']
TPS = ['0h','6h','24h']

# ----------------------------------------------------------------------
# Load controls (long) -> per (group, sample): (3 tp, 12 chir) cube
# ----------------------------------------------------------------------
L = pd.read_csv(os.path.join(HERE, "controls_features_long.csv"), dtype={'chirality': str})

def ctrl_cube(feat, group):
    samples = sorted(L[L.group == group]['sample'].unique())
    C = np.full((len(samples), 3, 12), np.nan)
    for si, s in enumerate(samples):
        for ti, tp in enumerate(TPS):
            for ci, c in enumerate(CHIRS):
                v = L[(L.group==group)&(L['sample']==s)&(L.tp==tp)&(L.chirality==c)][feat]
                if len(v): C[si, ti, ci] = float(v.iloc[0])
    return samples, C

# ----------------------------------------------------------------------
# Load serum
# ----------------------------------------------------------------------
def serum_cube_max5x5():
    dfs = {tp: pd.read_csv(os.path.join(MPX, f"features_max5x5_{tp}.csv")) for tp in TPS}
    common = sorted(set(dfs['0h'].sample_id) & set(dfs['6h'].sample_id) & set(dfs['24h'].sample_id))
    for tp in TPS: dfs[tp] = dfs[tp].set_index('sample_id').loc[common]
    C = np.stack([np.stack([dfs[tp][c].to_numpy(float) for c in CHIRS], axis=1) for tp in TPS], axis=1)
    return common, C  # (N,3,12)

def serum_cube_gauss():
    df = pd.read_csv(os.path.join(PD_, "chirality_gaussian_peak_descriptors.csv"))
    N = len(df); C = np.full((N,3,12), np.nan)
    for ti, tp in enumerate(TPS):
        for ci, c in enumerate(CHIRS):
            col = f"ch{c.replace('.','_')}_gauss_max_{tp}"
            C[:, ti, ci] = df[col].to_numpy(float)
    keep = np.isfinite(C).all(axis=(1,2))
    return df.get('sample_id', pd.Series(range(N)))[keep].tolist(), C[keep]

# ----------------------------------------------------------------------
# helpers: relative-change trajectory (scale-robust, referenced to own 0h)
# ----------------------------------------------------------------------
def rel_traj(C):
    """(N,3,12) -> fractional change vs 0h at 6h and 24h: (N,12) each."""
    I0 = C[:, 0, :]
    d6  = (C[:, 1, :] - I0) / I0
    d24 = (C[:, 2, :] - I0) / I0
    return d6, d24

def cos(a, b):
    a, b = np.asarray(a,float), np.asarray(b,float)
    return float(a @ b / (np.linalg.norm(a)*np.linalg.norm(b)))

# ======================================================================
report = {}
for FEAT in ("max5x5", "gauss_max"):
    print("\n" + "="*70 + f"\nFEATURE = {FEAT}\n" + "="*70)
    psamp, P = ctrl_cube(FEAT, 'P')     # (4,3,12)
    fsamp, F = ctrl_cube(FEAT, 'F')     # (3,3,12)
    if FEAT == "max5x5":
        ssamp, S = serum_cube_max5x5()
    else:
        ssamp, S = serum_cube_gauss()
    print(f"n: PBS={P.shape[0]}  FBS={F.shape[0]}  serum={S.shape[0]}")

    # ---- A. PBS baseline drift ----
    P_d6, P_d24 = rel_traj(P)
    F_d6, F_d24 = rel_traj(F)
    S_d6, S_d24 = rel_traj(S)
    pbs_mean_change = np.nanmean(P_d24)          # avg fractional 0->24h across chir
    fbs_mean_change = np.nanmean(F_d24)
    # paired 0h vs 24h across PBS replicates, per-chirality mean intensity
    P0 = np.nanmean(P[:,0,:],axis=1); P24 = np.nanmean(P[:,2,:],axis=1)
    tP, pP = stats.ttest_rel(P24, P0)
    F0 = np.nanmean(F[:,0,:],axis=1); F24 = np.nanmean(F[:,2,:],axis=1)
    tF, pF = stats.ttest_rel(F24, F0)
    print(f"\n[A] PBS mean fractional change 0->24h = {pbs_mean_change:+.3f} "
          f"(paired t={tP:.2f}, p={pP:.3f}, n=4)")
    print(f"    FBS mean fractional change 0->24h = {fbs_mean_change:+.3f} "
          f"(paired t={tF:.2f}, p={pF:.3f}, n=3)")

    # ---- B. magnitude of 0->24h shift: PBS vs FBS vs serum ----
    magP = np.linalg.norm(P_d24, axis=1)
    magF = np.linalg.norm(F_d24, axis=1)
    magS = np.linalg.norm(S_d24, axis=1)
    print(f"\n[B] ||fractional 0->24h shift|| (mean+/-sd):")
    print(f"    PBS  {magP.mean():.3f} +/- {magP.std():.3f}")
    print(f"    FBS  {magF.mean():.3f} +/- {magF.std():.3f}")
    print(f"    serum{magS.mean():.3f} +/- {magS.std():.3f}")
    # is serum shift larger than PBS drift?
    uS = stats.mannwhitneyu(magS, magP, alternative='greater')
    print(f"    serum>PBS Mann-Whitney p={uS.pvalue:.4f}")

    # direction: serum vs PBS drift, serum vs FBS
    mS24 = np.nanmean(S_d24,axis=0); mP24 = np.nanmean(P_d24,axis=0); mF24 = np.nanmean(F_d24,axis=0)
    print(f"    cos(mean serum shift, PBS drift) = {cos(mS24,mP24):+.3f}")
    print(f"    cos(mean serum shift, FBS shift) = {cos(mS24,mF24):+.3f}")

    # ---- C. corona already present at 0h (baseline-corrected F - P) ----
    # per-chirality group means at each tp
    Pm = np.nanmean(P, axis=0)   # (3,12)
    Fm = np.nanmean(F, axis=0)
    corona = (Fm - Pm) / Pm      # fractional corona-induced offset from bare sensor, per tp
    mag_c = np.linalg.norm(corona, axis=1)   # per tp
    frac_at_0h = mag_c[0] / mag_c.max()
    print(f"\n[C] baseline-corrected corona signature ||(F-P)/P|| by tp: "
          f"0h={mag_c[0]:.3f} 6h={mag_c[1]:.3f} 24h={mag_c[2]:.3f}")
    print(f"    fraction of max corona signature already present at 0h = {frac_at_0h:.2f}")
    # divergence check: |F-P| grows or is already there?
    print(f"    cos(corona@0h, corona@24h) = {cos(corona[0],corona[2]):+.3f}")

    # ---- D. patient serum maturation axis vs FBS vs PBS ----
    # per-sample cosine of serum 0->24h shift to reference axes
    cosF = np.array([cos(S_d24[i], mF24) for i in range(len(S_d24))])
    cosP = np.array([cos(S_d24[i], mP24) for i in range(len(S_d24))])
    print(f"\n[D] serum 0->24h direction  cos to FBS axis = {np.nanmean(cosF):+.3f} +/- {np.nanstd(cosF):.3f}")
    print(f"    serum 0->24h direction  cos to PBS axis = {np.nanmean(cosP):+.3f} +/- {np.nanstd(cosP):.3f}")

    # ---- B2. common-mode-removed "shape": is corona pattern distinct from drift? ----
    def demean(V): return V - np.nanmean(V, axis=1, keepdims=True)
    Ss, Ps, Fs = demean(S_d24), demean(P_d24), demean(F_d24)
    mSs, mPs, mFs = np.nanmean(Ss,0), np.nanmean(Ps,0), np.nanmean(Fs,0)
    # how much of the serum temporal change is bulk common-mode vs chirality-specific shape?
    cm = np.nanmean(S_d24, axis=1)                       # per-sample common-mode
    frac_cm = np.nanmean(np.abs(cm)) / np.nanmean(np.linalg.norm(S_d24,axis=1))
    print(f"\n[B2] shape (common-mode removed):")
    print(f"     cos(serum shape, PBS shape) = {cos(mSs,mPs):+.3f}")
    print(f"     cos(serum shape, FBS shape) = {cos(mSs,mFs):+.3f}")
    print(f"     serum ||shape||/||full|| = {np.nanmean(np.linalg.norm(Ss,axis=1))/np.nanmean(np.linalg.norm(S_d24,axis=1)):.3f} "
          f"(rest is common-mode brightening)")

    report[FEAT] = dict(
        Sm=np.nanmean(S,axis=0),
        mSs=mSs,mPs=mPs,mFs=mFs,cos_shape_PBS=cos(mSs,mPs),cos_shape_FBS=cos(mSs,mFs),
        P=P,F=F,S=S,psamp=psamp,fsamp=fsamp,
        P_d24=P_d24,F_d24=F_d24,S_d24=S_d24,
        Pm=Pm,Fm=Fm,corona=corona,mag_c=mag_c,frac_at_0h=frac_at_0h,
        magP=magP,magF=magF,magS=magS,mS24=mS24,mP24=mP24,mF24=mF24,
        pbs_mean_change=pbs_mean_change,fbs_mean_change=fbs_mean_change,
        pP=pP,pF=pF, cosF=cosF,cosP=cosP)

# save summary table
rows=[]
for FEAT,r in report.items():
    rows.append(dict(feature=FEAT,
        pbs_frac_change_0_24=r['pbs_mean_change'], pbs_paired_p=r['pP'],
        fbs_frac_change_0_24=r['fbs_mean_change'], fbs_paired_p=r['pF'],
        mag_shift_PBS=r['magP'].mean(), mag_shift_FBS=r['magF'].mean(), mag_shift_serum=r['magS'].mean(),
        corona_frac_at_0h=r['frac_at_0h'],
        cos_serum_FBS=np.nanmean(r['cosF']), cos_serum_PBS=np.nanmean(r['cosP'])))
pd.DataFrame(rows).to_csv(os.path.join(HERE,"controls_summary.csv"), index=False)
print("\nSaved controls_summary.csv")

# stash per-chirality trajectories for the figure
# ======================================================================
# FIGURE  (uses gauss_max as primary; max5x5 agrees)
# ======================================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
C_P="#0072B2"; C_F="#D55E00"; C_S="#009E73"; C_INK="#222222"; C_MUT="#8a8a8a"
plt.rcParams.update({"font.size":9,"axes.titlesize":9.5,"axes.labelsize":9,
    "axes.spines.top":False,"axes.spines.right":False,"figure.dpi":150})
tp_x=[0,6,24]
fig,axs=plt.subplots(1,4,figsize=(13.6,3.4))
fig.subplots_adjust(left=0.05,right=0.995,bottom=0.17,top=0.86,wspace=0.42)

FEAT="gauss_max"; r=report[FEAT]
# (a) mean intensity trajectory (normalized to own 0h) PBS/FBS/serum
axa=axs[0]
for M,c,lab in [(r['Pm'],C_P,'PBS (no protein)'),(r['Fm'],C_F,'FBS (ref. serum)'),(r['Sm'],C_S,'patient serum')]:
    y=np.nanmean(M,axis=1); y=y/y[0]
    axa.plot(tp_x,y,'-o',color=c,label=lab,lw=1.8,ms=5)
axa.axhline(1,color=C_MUT,lw=0.7,ls=':')
axa.set_xticks(tp_x); axa.set_xlabel("time (h)"); axa.set_ylabel("mean peak intensity (/0h)")
axa.set_title("(a) Baseline drift vs corona",loc="left"); axa.legend(frameon=False,fontsize=7.3,loc="upper left")

# (b) magnitude of 0->24h fractional shift
axb=axs[1]
data=[r['magP'],r['magF'],r['magS']]; labs=['PBS','FBS','serum']; cols=[C_P,C_F,C_S]
for i,(d,c) in enumerate(zip(data,cols)):
    x=np.random.default_rng(0).normal(i,0.06,len(d))
    axb.scatter(x,d,s=22,color=c,alpha=0.75,edgecolor='white',lw=0.4,zorder=3)
    axb.plot([i-0.22,i+0.22],[np.mean(d)]*2,color=C_INK,lw=2,zorder=4)
axb.set_xticks(range(3)); axb.set_xticklabels(labs); axb.set_ylabel("||0→24h fractional shift||")
axb.set_title("(b) Magnitude of temporal shift",loc="left")

# (c) direction cosines: serum temporal SHAPE vs PBS drift vs FBS corona
axc=axs[2]
vals=[report['gauss_max']['cos_shape_PBS'],report['gauss_max']['cos_shape_FBS'],
      report['max5x5']['cos_shape_PBS'],report['max5x5']['cos_shape_FBS']]
xlab=['vs PBS\n(gauss)','vs FBS\n(gauss)','vs PBS\n(max5x5)','vs FBS\n(max5x5)']
cols2=[C_P,C_F,C_P,C_F]
axc.bar(range(4),vals,color=cols2,alpha=0.85,edgecolor='white')
axc.axhline(0,color=C_INK,lw=0.8)
axc.set_xticks(range(4)); axc.set_xticklabels(xlab,fontsize=7.2)
axc.set_ylabel("cosine (serum temporal shape)"); axc.set_ylim(-1,1)
axc.set_title("(c) Serum pattern tracks bare-sensor drift",loc="left")

# (d) per-chirality 0->24h fractional change, PBS vs FBS vs serum
axd=axs[3]
xc=np.arange(12)
axd.axhline(0,color=C_MUT,lw=0.7)
axd.plot(xc,np.nanmean(r['Pm'],axis=0)*0+np.nanmean((r['Pm'][2]-r['Pm'][0])/r['Pm'][0]),alpha=0)  # noop keep scale
for M,c,lab in [(r['Pm'],C_P,'PBS'),(r['Fm'],C_F,'FBS'),(r['Sm'],C_S,'serum')]:
    frac=(M[2]-M[0])/M[0]
    axd.plot(xc,frac,'-o',color=c,label=lab,lw=1.4,ms=3)
axd.set_xticks(xc); axd.set_xticklabels(CHIRS,rotation=90,fontsize=6.5)
axd.set_ylabel("0→24h fractional change"); axd.set_xlabel("chirality")
axd.set_title("(d) Per-chirality 0→24h change",loc="left"); axd.legend(frameon=False,fontsize=7.3)

fig.savefig(os.path.join(HERE,"FigSX_tech_controls.png"),dpi=300,bbox_inches="tight")
fig.savefig(os.path.join(HERE,"FigSX_tech_controls.pdf"),bbox_inches="tight")
print("Saved FigSX_tech_controls.png/.pdf")

np.savez(os.path.join(HERE,"controls_traj.npz"),
    **{f"{k}_{FEAT}": (report[FEAT][k] if isinstance(report[FEAT][k],np.ndarray) else np.array(report[FEAT][k]))
       for FEAT in report for k in ('Pm','Fm','corona','mag_c','magP','magF','magS','mS24','mP24','mF24')})
print("Saved controls_traj.npz")

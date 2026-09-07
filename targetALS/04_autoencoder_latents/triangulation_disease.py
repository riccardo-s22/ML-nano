"""
Triangulation: is the ALS-vs-CTRL signal carried by the DRIFT relative-movement
(shared with protein-free PBS) or by the CORONA-SPECIFIC relative movement
(FBS - PBS), in the patients' 0->24h trajectories?

Axes (from exp4 controls, relative share = intensity / mean-over-12-chir):
   drift  axis = mean PBS relative-Delta(0->24h)              (u_drift)
   corona axis = (FBS - PBS) relative-Delta, orthogonalized to u_drift (u_corona)

Patient test (exp5 serum, max5x5, 39 labelled samples):
   per patient relative-Delta share (0->24h) -> project onto u_drift, u_corona
   ALS vs CTRL separation on each projection (AUC + Mann-Whitney)
   plus per-chirality relative-Delta ALS vs CTRL (which chiralities discriminate,
   are they the corona-specific ones 6.5/8.6/9.5?)
"""
import os, numpy as np, pandas as pd
from scipy import stats

EXP4 = r"/mnt/c/Users/riccardo-s/Documents/CNT/targetALS/exp4"
MPX  = r"/mnt/c/Users/riccardo-s/Documents/CNT/targetALS/exp5/multiplexing_results"
HERE = os.path.dirname(os.path.abspath(__file__))
CHIRS = ['6.5','7.5','7.6','8.3','8.4','8.6','8.7','9.4','9.5','10.2','10.3','10.5']

def auc(pos, neg):
    U = stats.mannwhitneyu(pos, neg, alternative='two-sided').statistic
    return U/(len(pos)*len(neg))

def unit(v): return v/np.linalg.norm(v)

# ---- control axes from exp4 ----
c = pd.read_excel(os.path.join(EXP4,"FIT_controls_ex4.xlsx"))
c['cond']=c['sample'].str.replace(r'_\d+$','',regex=True)
c['ch']=c['chir'].astype(int).astype(str)+'.'+c['chir.1'].astype(int).astype(str)
def rel_delta(cond):
    samps=sorted(c[c.cond==cond]['sample'].unique()); D=[]
    for s in samps:
        sub=c[c['sample']==s].set_index('ch')
        i0=sub.loc[CHIRS,'intesnity_0h'].to_numpy(float); i0=i0/i0.mean()
        i24=sub.loc[CHIRS,'intesnity_24h'].to_numpy(float); i24=i24/i24.mean()
        D.append(i24-i0)
    return np.array(D)
mP = rel_delta('PBS').mean(0)
mF = rel_delta('FBS').mean(0)
u_drift = unit(mP)
corona = (mF-mP)
corona_perp = corona - (corona@u_drift)*u_drift        # orthogonal to drift
u_corona = unit(corona_perp)
print("control axes (relative-share change 0->24h):")
print("  u_drift  :", np.round(u_drift,2))
print("  u_corona :", np.round(u_corona,2), "(FBS-PBS, drift-orthogonalized)")
print(f"  drift/corona axes orthogonal? dot={u_drift@u_corona:+.3f}")

# ---- patient relative-Delta (exp5 serum, max5x5) ----
d0 =pd.read_csv(os.path.join(MPX,"features_max5x5_0h.csv")).set_index('sample_id')
d24=pd.read_csv(os.path.join(MPX,"features_max5x5_24h.csv")).set_index('sample_id')
lab=pd.read_csv(os.path.join(MPX,"sample_labels.csv")).set_index('sample_id')['group']
ids=[s for s in lab.index if s in d0.index and s in d24.index]
print(f"\npatients: {len(ids)}  ({(lab[ids]=='ALS').sum()} ALS / {(lab[ids]=='CTRL').sum()} CTRL)")
X0 =d0.loc[ids,CHIRS].to_numpy(float); X24=d24.loc[ids,CHIRS].to_numpy(float)
S0 =X0/X0.mean(1,keepdims=True); S24=X24/X24.mean(1,keepdims=True)
dShare = S24-S0                                          # (n,12) relative trajectory
y = (lab[ids]=='ALS').to_numpy()

# ---- project onto control axes, test ALS vs CTRL ----
proj_drift  = dShare@u_drift
proj_corona = dShare@u_corona
print("\n=== ALS vs CTRL on control-derived relative axes ===")
for name,pr in [("drift  projection",proj_drift),("corona projection",proj_corona)]:
    a,b=pr[y],pr[~y]; t,p=stats.ttest_ind(a,b); A=auc(a,b)
    print(f"  {name}: AUC={A:.3f}  meanALS={a.mean():+.3f} meanCTRL={b.mean():+.3f}  t={t:.2f} p={p:.3f}")

# ---- per-chirality relative-Delta ALS vs CTRL (self-contained) ----
print("\n=== per-chirality relative-Delta(0->24h): ALS vs CTRL ===")
rows=[]
for j,ch in enumerate(CHIRS):
    a,b=dShare[y,j],dShare[~y,j]; t,p=stats.ttest_ind(a,b); A=auc(a,b)
    rows.append(dict(chirality=ch, auc=A, p=p, als=a.mean(), ctrl=b.mean(),
                     corona_marker=ch in ('6.5','8.6','9.5','9.4')))
R=pd.DataFrame(rows).sort_values('p')
print(R.round(3).to_string(index=False))
R.to_csv(os.path.join(HERE,"triangulation_per_chirality.csv"),index=False)

# how aligned is the ALS-CTRL discriminative direction with corona vs drift axis?
disc = dShare[y].mean(0)-dShare[~y].mean(0)             # ALS-CTRL mean relative-Delta diff
cos_dc = disc@u_corona/np.linalg.norm(disc)
cos_dd = disc@u_drift /np.linalg.norm(disc)
print(f"\nALS-CTRL discriminative direction alignment:")
print(f"  cos(disc, corona axis) = {cos_dc:+.3f}")
print(f"  cos(disc, drift  axis) = {cos_dd:+.3f}")

"""
Where does the conv-AE disease signal live?
Correlate the AE classifier's OOF P_ALS (the actual ~0.96-AUC disease score) with
interpretable per-sample descriptors of the 0->24h trajectory:
   - drift  projection  (relative movement along PBS axis, from exp4 controls)
   - corona projection  (relative movement along FBS-PBS axis, drift-orthogonal)
   - bulk brightening   (common-mode: mean fractional 0->24h intensity change)
   - absolute 0h / 24h mean level
If P_ALS tracks drift/corona -> AE leans on relative dynamics; if it tracks bulk
or neither -> the disease signal lives elsewhere (absolute/spectral/nonlinear).
"""
import os, numpy as np, pandas as pd
from scipy import stats

EXP4=r"/mnt/c/Users/riccardo-s/Documents/CNT/targetALS/exp4"
EXP5=r"/mnt/c/Users/riccardo-s/Documents/CNT/targetALS/exp5"
MPX=os.path.join(EXP5,"multiplexing_results"); HERE=os.path.dirname(os.path.abspath(__file__))
CHIRS=['6.5','7.5','7.6','8.3','8.4','8.6','8.7','9.4','9.5','10.2','10.3','10.5']
def unit(v): return v/np.linalg.norm(v)
def auc(pos,neg): return stats.mannwhitneyu(pos,neg,alternative='two-sided').statistic/(len(pos)*len(neg))

# control axes (exp4)
c=pd.read_excel(os.path.join(EXP4,"FIT_controls_ex4.xlsx"))
c['cond']=c['sample'].str.replace(r'_\d+$','',regex=True)
c['ch']=c['chir'].astype(int).astype(str)+'.'+c['chir.1'].astype(int).astype(str)
def reld(cond):
    D=[]
    for s in sorted(c[c.cond==cond]['sample'].unique()):
        sub=c[c['sample']==s].set_index('ch')
        i0=sub.loc[CHIRS,'intesnity_0h'].to_numpy(float); i24=sub.loc[CHIRS,'intesnity_24h'].to_numpy(float)
        D.append(i24/i24.mean()-i0/i0.mean())
    return np.array(D).mean(0)
mP,mF=reld('PBS'),reld('FBS')
u_drift=unit(mP); cor=mF-mP; u_corona=unit(cor-(cor@u_drift)*u_drift)

# patient trajectories + AE score
d0=pd.read_csv(os.path.join(MPX,"features_max5x5_0h.csv")).set_index('sample_id')
d24=pd.read_csv(os.path.join(MPX,"features_max5x5_24h.csv")).set_index('sample_id')
ae=pd.read_csv(os.path.join(EXP5,"model_metrics_eval/existing_models_oof_predictions.csv")).set_index('code')
ids=[s for s in ae.index if s in d0.index and s in d24.index]
X0=d0.loc[ids,CHIRS].to_numpy(float); X24=d24.loc[ids,CHIRS].to_numpy(float)
S0=X0/X0.mean(1,keepdims=True); S24=X24/X24.mean(1,keepdims=True); dS=S24-S0
P=ae.loc[ids,'P_ALS'].to_numpy(float); y=(ae.loc[ids,'true_label']=='ALS').to_numpy()

desc={
 'drift_proj':  dS@u_drift,
 'corona_proj': dS@u_corona,
 'bulk_brighten': ((X24-X0)/X0).mean(1),
 'mean_0h': X0.mean(1),
 'mean_24h': X24.mean(1),
}
print(f"n={len(ids)}  AE P_ALS AUC vs truth = {auc(P[y],P[~y]):.3f}  (sanity)\n")
print(f"{'descriptor':14s} {'r(P_ALS)':>9s} {'p':>7s} {'rho':>7s} {'AUC ALS/CTRL':>12s}")
rows=[]
for k,v in desc.items():
    r,p=stats.pearsonr(P,v); rho,_=stats.spearmanr(P,v); A=auc(v[y],v[~y])
    rows.append(dict(descriptor=k,r_PALS=r,p=p,rho=rho,auc=A))
    print(f"{k:14s} {r:+9.3f} {p:7.3f} {rho:+7.3f} {A:12.3f}")
pd.DataFrame(rows).to_csv(os.path.join(HERE,"ae_signal_location.csv"),index=False)

# multivariate: regress P_ALS on standardized drift+corona+bulk
import numpy as np
Zx=np.column_stack([ (desc[k]-desc[k].mean())/desc[k].std() for k in ['drift_proj','corona_proj','bulk_brighten']])
Zx=np.column_stack([np.ones(len(P)),Zx])
beta,*_=np.linalg.lstsq(Zx,(P-P.mean())/P.std(),rcond=None)
print(f"\nstd betas predicting P_ALS: drift={beta[1]:+.3f} corona={beta[2]:+.3f} bulk={beta[3]:+.3f}")
pred=Zx@beta; R2=1-((( (P-P.mean())/P.std())-pred)**2).sum()/len(P)/1
ss=(((P-P.mean())/P.std())**2).sum(); R2=1-(((( (P-P.mean())/P.std())-pred)**2).sum())/ss
print(f"R^2 of P_ALS explained by these 3 trajectory descriptors = {R2:.3f}")

# per-chirality: which chir relative-move aligns with AE decision
print("\nper-chirality corr(P_ALS, relative Δshare):")
for j,ch in enumerate(CHIRS):
    r,p=stats.pearsonr(P,dS[:,j])
    mark=' <-corona' if ch in ('6.5','8.6','9.5') else (' <-drift' if abs(u_drift[j])>0.3 else '')
    print(f"  {ch:5s} r={r:+.3f} p={p:.3f}{mark}")

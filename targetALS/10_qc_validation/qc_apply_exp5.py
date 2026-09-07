"""
Apply the PBS-anchored QC framework (qc_metrics.py) retrospectively to the exp5
patient plates.

  TIER 1  on the exp5 PBS wells (P1-P4, 0/6/24h) -- was the exp5 sensor sound?
          QC1 brightness, QC2 fingerprint fidelity (vs exp4 reference AND vs the
          exp5 batch's own median), QC4 replicate %CV, QC5 drift.
          QC3 (peak-center RMS) not available: the exp5 control extraction stored
          amplitudes only, no fitted centers.
  TIER 2  bonus -- exp5 also has F (FBS) control wells, so the assay window is
          computable for this batch. CAVEAT: the tech-control wells are not
          demonstrably on the same physical plate as the patient samples, so this
          is a batch-level, not plate-level, window.
  TIER 3  LOD from the exp5 PBS replicates -> which chiralities are usable here;
          then per-PATIENT QC: robust fingerprint-outlier z against the cohort,
          and whether flagged samples relate to label / classifier correctness.

Feature choice matters and is split on purpose:
  * QC2 CROSS-BATCH fidelity uses gauss_max (split-Gaussian fitted amplitude).
    The raw 5x5 pixel max does NOT transfer between runs -- exp5 PBS vs the exp4
    reference gives r=0.850 with max5x5 but r=0.994 with gauss_max -- because
    max5x5 samples a fixed grid location and so absorbs sub-pixel peak-position
    and grid-alignment differences between runs. Only the fitted amplitude is
    run-invariant, so only it can carry a cross-batch acceptance threshold.
  * WITHIN-batch work (replicate CV, drift, LOD, per-patient outliers) uses
    max5x5, which is what exists for the patient samples and is perfectly valid
    when every well shares one grid.

Outputs: qc_exp5_tier1.csv, qc_exp5_tier3_lod.csv, qc_exp5_sample_flags.csv,
         FigSX_qc_exp5.png/.pdf
"""
import os, numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
EXP5 = r"/mnt/c/Users/riccardo-s/Documents/CNT/targetALS/exp5"
MPX  = os.path.join(EXP5, "multiplexing_results")
CHIRS = ['6.5', '7.5', '7.6', '8.3', '8.4', '8.6', '8.7', '9.4', '9.5', '10.2', '10.3', '10.5']
TPS = ['0h', '6h', '24h']; TN = [0, 6, 24]
FEAT = 'max5x5'
C = {'PBS': "#0072B2", 'FBS': "#D55E00", 'patient': "#8a8a8a"}
INK, MUT = "#222", "#8a8a8a"

def share(v): return v/np.nanmean(v)
def unit(v):  return v/np.linalg.norm(v)
def robust_z(X):
    """column-wise robust z using median/MAD."""
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X-med), axis=0)*1.4826
    mad = np.where(mad <= 0, np.nan, mad)
    return (X-med)/mad

# ------------------------------------------------------------ exp5 controls
L = pd.read_csv(os.path.join(HERE, "controls_features_long.csv"), dtype={'chirality': str})
def cwells(group, tp, feat=FEAT):
    s = L[(L.group == group) & (L.tp == tp)]
    names = sorted(s['sample'].unique())
    return names, np.array([s[s['sample'] == w].set_index('chirality').reindex(CHIRS)[feat]
                            .to_numpy(float) for w in names])

# exp4 reference fingerprint, PER TIMEPOINT.  The intrinsic drift reshapes the
# profile over 24 h, so a 24 h read must be judged against a 24 h reference --
# comparing everything to a 0 h reference would penalise a perfectly good plate.
REF = pd.read_csv(os.path.join(HERE, "qc_reference_profile.csv"), dtype={'chirality': str}
                  ).set_index('chirality').reindex(CHIRS)
DVEC = REF['d_nm'].to_numpy(float)

_e4 = pd.read_excel(r"/mnt/c/Users/riccardo-s/Documents/CNT/targetALS/exp4/FIT_controls_ex4.xlsx")
_e4['cond'] = _e4['sample'].str.replace(r'_\d+$', '', regex=True)
_e4['ch'] = _e4['chir'].astype(int).astype(str)+'.'+_e4['chir.1'].astype(int).astype(str)
def _ref4(tp):
    s = _e4[_e4.cond == 'PBS']
    M = np.array([s[s['sample'] == w].set_index('ch').reindex(CHIRS)[f'intesnity_{tp}'].to_numpy(float)
                  for w in sorted(s['sample'].unique())])
    return np.median(np.array([share(r) for r in M]), axis=0)
REF_BY_TP = {tp: _ref4(tp) for tp in TPS}          # exp4 0h/6h/24h PBS references
REF_SHARE_EXP4 = REF_BY_TP['0h']

print("="*76)
print("TIER 1 -- exp5 PBS wells (P1-P4)")
print("="*76)
pnames, P0 = cwells('P', '0h')
REF_SHARE_EXP5 = np.median(np.array([share(r) for r in P0]), axis=0)

print("  extractor dependence of the cross-batch QC2 criterion (exp5 PBS vs exp4 ref):")
for feat in ['max5x5', 'gauss_max']:
    _, Pg = cwells('P', '0h', feat)
    prof = np.median(np.array([share(r) for r in Pg]), axis=0)
    print(f"    {feat:10s} r = {stats.pearsonr(prof, REF_SHARE_EXP4)[0]:+.4f}")
print("    -> QC2 must be computed on gauss_max; max5x5 does not transfer between runs.")

t1 = []
for tp in TPS:
    nm, I = cwells('P', tp)
    _, Ig = cwells('P', tp, 'gauss_max')
    for k, w in enumerate(nm):
        sh, shg = share(I[k]), share(Ig[k])
        t1.append(dict(well=w, tp=tp, brightness_1e15=I[k].mean()*1e15,
                       QC2_r_vs_exp4_tpmatched=stats.pearsonr(shg, REF_BY_TP[tp])[0],
                       QC2_r_vs_exp4_0h=stats.pearsonr(shg, REF_SHARE_EXP4)[0],
                       QC2_r_vs_exp5_batch=stats.pearsonr(sh, REF_SHARE_EXP5)[0],
                       QC2_max_dev=np.abs(sh-REF_SHARE_EXP5).max()))
T1 = pd.DataFrame(t1)
print("\n  per-well QC2 (fingerprint fidelity):")
print(T1.round(4).to_string(index=False))
_p = T1['QC2_r_vs_exp4_tpmatched']; _q = T1['QC2_r_vs_exp4_0h']
print(f"\n  timepoint-matched : r={_p.min():.4f}-{_p.max():.4f}")
print(f"  vs a 0h reference : r={_q.min():.4f}-{_q.max():.4f}  <- wrong comparison,"
      f" the drift reshapes the profile; matching by timepoint is required")
bad = T1[_p < 0.99]
print(f"\n  QC2 verdict (threshold r>=0.99, timepoint-matched): "
      f"{len(T1)-len(bad)}/{len(T1)} PBS well-reads PASS")
if len(bad):
    for _, r in bad.iterrows():
        print(f"    FAIL: {r.well} @ {r.tp}  r={r.QC2_r_vs_exp4_tpmatched:.4f}  "
              f"max_dev={r.QC2_max_dev:.3f}")
    print("    -> same signature as exp4: it is replicate #1 at the 0h read.")

# QC4 with and without the first-read well, to size that effect
print("\n  QC4 impact of the first-read well (0h, relative share):")
_, I0a = cwells('P', '0h'); Sa = np.array([share(r) for r in I0a])
cv_all = 100*Sa.std(0, ddof=1)/Sa.mean(0)
Sb = Sa[1:]; cv_wo = 100*Sb.std(0, ddof=1)/Sb.mean(0)
print(f"    all 4 wells   median %CV = {np.median(cv_all):5.2f}%  max = {cv_all.max():5.2f}%")
print(f"    dropping P1   median %CV = {np.median(cv_wo):5.2f}%  max = {cv_wo.max():5.2f}%")

print("\n  QC4 replicate %CV across the 4 PBS wells:")
for tp in TPS:
    _, I = cwells('P', tp)
    S = np.array([share(r) for r in I])
    cvI = 100*I.std(0, ddof=1)/I.mean(0); cvS = 100*S.std(0, ddof=1)/S.mean(0)
    print(f"    {tp:4s} intensity med={np.median(cvI):5.1f}% max={cvI.max():5.1f}%   "
          f"share med={np.median(cvS):5.2f}% max={cvS.max():5.2f}%")

_, I0 = cwells('P', '0h'); _, I24 = cwells('P', '24h')
dr = I24.mean(1)/I0.mean(1)
r_dd, p_dd = stats.pearsonr(DVEC, ((I24/I0)-1).mean(0))
print(f"\n  QC5 drift I24/I0 per well: {np.round(dr,3)}  mean={dr.mean():.3f}"
      f"  [flag outside 0.5-2.0]")
print(f"      drift-vs-diameter: r={r_dd:+.2f} p={p_dd:.3f}")
T1.to_csv(os.path.join(HERE, "qc_exp5_tier1.csv"), index=False)

# ------------------------------------------------------------ TIER 2 (bonus)
print("\n"+"="*76)
print("TIER 2 (bonus) -- exp5 assay window, P (n=4) vs F (n=3) control wells")
print("="*76)
def ssmd(pos, neg): return (pos.mean()-neg.mean())/np.sqrt(pos.std(ddof=1)**2+neg.std(ddof=1)**2)
_, F24 = cwells('F', '24h'); _, P24 = cwells('P', '24h')
axis = unit(np.array([share(r) for r in F24]).mean(0) - np.array([share(r) for r in P24]).mean(0))
for tp in TPS:
    _, Pi = cwells('P', tp); _, Fi = cwells('F', tp)
    ciP = np.array([share(r) for r in Pi]) @ axis
    ciF = np.array([share(r) for r in Fi]) @ axis
    sdP = np.array([share(r) for r in Pi]).std(0, ddof=1).mean()
    sdF = np.array([share(r) for r in Fi]).std(0, ddof=1).mean()
    print(f"  {tp:4s} |SSMD(CI)|={abs(ssmd(ciF,ciP)):5.2f}   SD(PBS)={sdP:.4f} SD(FBS)={sdF:.4f}"
          f"  ratio={sdF/sdP:4.1f}x   [require |SSMD|>=2]")

# ------------------------------------------------------------ TIER 3 LOD
print("\n"+"="*76)
print("TIER 3 -- LOD from exp5 PBS replicates (n=4)")
print("="*76)
lod = []
for tp in TPS:
    _, Pi = cwells('P', tp); _, Fi = cwells('F', tp)
    SP = np.array([share(r) for r in Pi]); SF = np.array([share(r) for r in Fi])
    for j, ch in enumerate(CHIRS):
        eff = SF[:, j].mean()-SP[:, j].mean(); sd = SP[:, j].std(ddof=1)
        lod.append(dict(tp=tp, chirality=ch, d_nm=DVEC[j], pbs_sd=sd,
                        corona_effect=eff, LOD_3sd=3*sd, SNR=abs(eff)/sd,
                        usable=abs(eff) > 3*sd))
LOD = pd.DataFrame(lod)
piv = LOD.pivot(index='chirality', columns='tp', values='SNR').reindex(CHIRS)[TPS]
print("\n  SNR = |corona effect| / SD(PBS), per chirality (usable if >3):")
print(piv.round(2).to_string())
for tp in TPS:
    print(f"    {tp:4s} usable: {LOD[LOD.tp==tp]['usable'].sum():2d}/12")
LOD.to_csv(os.path.join(HERE, "qc_exp5_tier3_lod.csv"), index=False)

# ------------------------------------------------------------ per-patient QC
print("\n"+"="*76)
print("PER-PATIENT SAMPLE QC (robust fingerprint outlier vs the cohort)")
print("="*76)
pat = {tp: pd.read_csv(os.path.join(MPX, f"features_{FEAT}_{tp}.csv")).set_index('sample_id')
       for tp in TPS}
ids = sorted(set.intersection(*[set(pat[tp].index) for tp in TPS]))
print(f"  rows with all 3 timepoints: {len(ids)}")
SPIKE = [s for s in ids if s.startswith(('F1.', 'F2.', 'F3.'))]
print(f"  NOTE: {len(SPIKE)} of these are FBS control wells sitting in the patient")
print(f"        feature table, not patients: {SPIKE}")
print("        They are left in deliberately as a positive control for the QC itself:")
print("        a working sample-QC should rank them as outliers.")

lab = pd.read_csv(os.path.join(MPX, "sample_labels.csv")).set_index('sample_id')['group']
try:
    ae = pd.read_csv(os.path.join(EXP5, "model_metrics_eval/existing_models_oof_predictions.csv")
                     ).set_index('code')
except Exception:
    ae = None

rows = []
for tp in TPS:
    X = pat[tp].loc[ids, CHIRS].to_numpy(float)
    S = np.array([share(r) for r in X])
    Z = robust_z(S)                                   # per-chirality robust z
    zmax = np.nanmax(np.abs(Z), axis=1)
    # brightness outlier (robust, on log scale)
    b = np.log2(X.mean(1)); bz = robust_z(b[:, None])[:, 0]
    # fidelity to the PATIENT cohort median profile (serum, so not the PBS profile)
    medS = np.median(S, axis=0)
    rfit = np.array([stats.pearsonr(S[i], medS)[0] for i in range(len(ids))])
    for i, s in enumerate(ids):
        rows.append(dict(sample_id=s, tp=tp, zmax_fingerprint=zmax[i],
                         z_brightness=bz[i], r_vs_cohort=rfit[i],
                         worst_chir=CHIRS[int(np.nanargmax(np.abs(Z[i])))]))
Q = pd.DataFrame(rows)
W = Q.pivot(index='sample_id', columns='tp', values='zmax_fingerprint')[TPS]
B = Q.pivot(index='sample_id', columns='tp', values='z_brightness')[TPS]
R = Q.pivot(index='sample_id', columns='tp', values='r_vs_cohort')[TPS]

flag = pd.DataFrame({
    'zmax_any_tp': W.max(1),
    'n_tp_z_gt5': (W > 5).sum(1),
    'bright_z_absmax': B.abs().max(1),
    'r_min': R.min(1),
})
flag['group'] = lab.reindex(flag.index)
if ae is not None:
    flag['P_ALS'] = ae['P_ALS'].reindex(flag.index)
    flag['correct'] = ae['correct'].reindex(flag.index)
# acceptance: robust z <= 5 at every tp, |brightness z| <= 5, cohort r >= 0.95
flag['QC_FAIL'] = (flag.zmax_any_tp > 5) | (flag.bright_z_absmax > 5) | (flag.r_min < 0.95)
flag = flag.sort_values('zmax_any_tp', ascending=False)
print("\n  top 12 samples by fingerprint-outlier score:")
print(flag.head(12).round(3).to_string())
nfail = int(flag.QC_FAIL.sum())
print(f"\n  QC_FAIL: {nfail}/{len(flag)} rows "
      f"(criteria: robust z<=5 all tp, |brightness z|<=5, cohort r>=0.95)")

# does the QC recover the spiked FBS control wells?
rank = {s: i+1 for i, s in enumerate(flag.index)}
print("\n  positive control -- rank of the spiked FBS wells (1 = most outlying of 42):")
for s in SPIKE:
    print(f"    {s.split('__')[0]:8s} rank {rank[s]:2d}/{len(flag)}   "
          f"z={flag.loc[s,'zmax_any_tp']:.2f}  flagged={bool(flag.loc[s,'QC_FAIL'])}")

# threshold sensitivity for the patient-only cohort
pats = flag.drop(index=SPIKE)
print("\n  flag-rate sensitivity to the robust-z cutoff (patients only, n=%d):" % len(pats))
for thr in [3, 4, 5, 6, 8]:
    print(f"    z>{thr}: {(pats.zmax_any_tp > thr).sum():2d}/{len(pats)} "
          f"({(pats.zmax_any_tp > thr).mean():5.1%})")

lb = flag['group'].notna()
if lb.any():
    a = flag[lb & (flag.group == 'ALS')]['QC_FAIL'].mean()
    c = flag[lb & (flag.group == 'CTRL')]['QC_FAIL'].mean()
    tab = pd.crosstab(flag[lb]['group'], flag[lb]['QC_FAIL'])
    print(f"\n  flag rate by label: ALS {a:.1%}  CTRL {c:.1%}")
    if tab.shape == (2, 2):
        print(f"  Fisher exact p = {stats.fisher_exact(tab.to_numpy())[1]:.3f}  "
              f"(QC flags should NOT track diagnosis)")
if ae is not None and flag['correct'].notna().any():
    m = flag['correct'].notna()
    cf = flag[m & flag.QC_FAIL]['correct'].astype(str)
    cp = flag[m & ~flag.QC_FAIL]['correct'].astype(str)
    print(f"  classifier accuracy: QC-fail {(cf=='True').mean() if len(cf) else float('nan'):.2f} "
          f"(n={len(cf)})   QC-pass {(cp=='True').mean():.2f} (n={len(cp)})")
    r_zp = stats.spearmanr(flag[m]['zmax_any_tp'], flag[m]['P_ALS'])
    print(f"  rho(fingerprint-outlier score, P_ALS) = {r_zp.statistic:+.3f} p={r_zp.pvalue:.3f}")
flag.to_csv(os.path.join(HERE, "qc_exp5_sample_flags.csv"))

# ------------------------------------------------------------ FIGURE
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
fig, axs = plt.subplots(1, 4, figsize=(14.2, 3.4))
fig.subplots_adjust(left=0.05, right=0.995, bottom=0.19, top=0.84, wspace=0.36)
x = np.arange(12)

a = axs[0]
_, Pg0 = cwells('P', '0h', 'gauss_max')
Sg = np.array([share(r) for r in Pg0]); prof_g = np.median(Sg, axis=0)
Sm = np.array([share(r) for r in P0]);  prof_m = np.median(Sm, axis=0)
a.plot(x, REF_SHARE_EXP4, '-o', color=INK, lw=1.6, ms=4, label='exp4 reference')
a.errorbar(x, prof_g, yerr=Sg.std(0, ddof=1), color=C['PBS'], lw=1.5, marker='s', ms=4,
           capsize=2, label=f"exp5 PBS gauss_max (r={stats.pearsonr(prof_g,REF_SHARE_EXP4)[0]:.3f})")
a.plot(x, prof_m, '--^', color="#CC79A7", lw=1.2, ms=4,
       label=f"exp5 PBS max5x5 (r={stats.pearsonr(prof_m,REF_SHARE_EXP4)[0]:.3f})")
a.set_xticks(x); a.set_xticklabels(CHIRS, rotation=90, fontsize=6.5)
a.set_ylabel("PBS relative share"); a.set_xlabel("chirality")
a.set_title("(a) QC2 transfers only with\n the fitted amplitude", loc="left")
a.legend(frameon=False, fontsize=6.3)

b = axs[1]
for tp, mk in zip(TPS, ['o', 's', '^']):
    b.plot(x, LOD[LOD.tp == tp].set_index('chirality').reindex(CHIRS)['SNR'], '-'+mk,
           ms=4, lw=1.2, label=tp)
b.axhline(3, color=MUT, ls='--', lw=0.9)
b.text(0.1, 3.2, "LOD (3xSD)", fontsize=6.5, color=MUT)
b.set_xticks(x); b.set_xticklabels(CHIRS, rotation=90, fontsize=6.5)
b.set_ylabel("SNR = |corona| / SD(PBS)"); b.set_xlabel("chirality")
b.set_title("(b) Tier-3 usable readouts\n in the exp5 batch", loc="left")
b.legend(frameon=False, fontsize=7)

c = axs[2]
for i, tp in enumerate(TPS):
    v = W[tp].to_numpy()
    c.scatter(np.full(len(v), i)+np.random.RandomState(0).uniform(-.12, .12, len(v)),
              v, s=14, color=C['patient'], alpha=.8, zorder=3)
c.axhline(5, color="#D55E00", ls='--', lw=1)
c.text(-0.35, 5.2, "flag", fontsize=7, color="#D55E00")
c.set_xticks(range(3)); c.set_xticklabels(TPS)
c.set_xlabel("timepoint"); c.set_ylabel("robust fingerprint z (max over chir)")
c.set_title(f"(c) Per-sample QC\n {nfail}/{len(flag)} flagged", loc="left")

d = axs[3]
m = flag['group'].notna()
for g, col in [('CTRL', "#0072B2"), ('ALS', "#D55E00")]:
    s = flag[m & (flag.group == g)]
    d.scatter(s['zmax_any_tp'], s['bright_z_absmax'], s=26, color=col, alpha=.85, label=g, zorder=3)
d.axvline(5, color=MUT, ls='--', lw=0.9); d.axhline(5, color=MUT, ls='--', lw=0.9)
d.set_xlabel("fingerprint outlier z"); d.set_ylabel("|brightness z|")
d.set_title("(d) QC flags vs diagnosis\n (should be unrelated)", loc="left")
d.legend(frameon=False, fontsize=7)

fig.savefig(os.path.join(HERE, "FigSX_qc_exp5.png"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(HERE, "FigSX_qc_exp5.pdf"), bbox_inches="tight")
print("\nSaved FigSX_qc_exp5.png/.pdf + qc_exp5_tier1.csv, qc_exp5_tier3_lod.csv, "
      "qc_exp5_sample_flags.csv")

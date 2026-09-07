"""
Assay QC framework anchored on the PBS negative control.

Three tiers:
  TIER 1  SENSOR QC  (PBS wells only) -- is the SWCNT prep itself sound this run?
      QC1 brightness          mean peak intensity over 12 chir
      QC2 fingerprint fidelity r(relative-share vector, reference PBS profile)   <-- scale-free
      QC3 peak-center accuracy RMS |emission - reference| (nm)
      QC4 replicate precision  %CV across PBS replicates, per chirality
      QC5 drift magnitude      mean I(24h)/I(0h)  (intrinsic, protein-free)
  TIER 2  ASSAY-WINDOW QC (PBS=neg vs FBS=pos) -- can this run resolve corona at all?
      SSMD (primary) and Z'-factor (secondary), per chirality and on a 1-D corona
      index (CI).  SSMD is primary because the two controls have very unequal
      variance here (FBS wells inherit protein variability, PBS does not), which
      Z' -- built for symmetric-variance HTS controls -- penalises unfairly.
  TIER 3  SAMPLE QC -- PBS-anchored noise floor -> which chiralities are usable
      LOD per chirality: |corona effect| vs 3*SD(PBS); per-sample fingerprint outlier z

Reference set = exp4 controls (n=5 PBS / 5 FBS / 5 FBS+HEP, 0/3/6/24h, fitted peaks).
Cross-batch transferability of the fingerprint checked against exp5 P1-4 and the
dense corona-kinetics PBS wells.

Outputs (this dir): qc_tier1_sensor.csv, qc_tier2_window.csv, qc_tier3_lod.csv,
                    qc_reference_profile.csv, FigSX_qc.png/.pdf
"""
import os, numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP4 = r"/mnt/c/Users/riccardo-s/Documents/CNT/targetALS/exp4"
HERE = os.path.dirname(os.path.abspath(__file__))
TPS  = ['0h', '3h', '6h', '24h']; TN = [0, 3, 6, 24]
COND = ['PBS', 'FBS', 'FBS+HEP']
C    = {'PBS': "#0072B2", 'FBS': "#D55E00", 'FBS+HEP': "#009E73"}
INK, MUT = "#222", "#8a8a8a"

def diam(n, m): return 0.0783*np.sqrt(n*n + n*m + m*m)      # nm
def share(v):   return v/np.nanmean(v)                       # scale-free profile
def unit(v):    return v/np.linalg.norm(v)

# ---------------------------------------------------------------- load exp4
df = pd.read_excel(os.path.join(EXP4, "FIT_controls_ex4.xlsx"))
df['cond'] = df['sample'].str.replace(r'_\d+$', '', regex=True)
df['ch']   = df['chir'].astype(int).astype(str)+'.'+df['chir.1'].astype(int).astype(str)
df['d']    = [diam(n, m) for n, m in zip(df['chir'].astype(int), df['chir.1'].astype(int))]
CHIRS = df.groupby('ch')['d'].first().sort_values().index.tolist()
DVEC  = np.array([df[df.ch == ch]['d'].iloc[0] for ch in CHIRS])

def wells(cond, tp, col='intesnity'):
    """(n_reps, 12) matrix of `col` at timepoint tp for a condition."""
    s = df[df.cond == cond]
    out = []
    for w in sorted(s['sample'].unique()):
        sub = s[s['sample'] == w].set_index('ch')
        out.append(sub.loc[CHIRS, f'{col}_{tp}'].to_numpy(float))
    return np.array(out)

# ================================================================ TIER 1
print("="*74)
print("TIER 1 -- SENSOR QC (PBS negative control alone)")
print("="*74)

P0 = wells('PBS', '0h')
REF_SHARE = np.median(np.array([share(r) for r in P0]), axis=0)   # reference fingerprint
REF_EM    = np.median(wells('PBS', '0h', 'emission'), axis=0)
REF_EX    = np.median(wells('PBS', '0h', 'excitation'), axis=0)
REF_I     = np.median(P0.mean(1))

pd.DataFrame({'chirality': CHIRS, 'd_nm': DVEC.round(3), 'ref_share': REF_SHARE.round(4),
              'ref_emission_nm': REF_EM.round(2), 'ref_excitation_nm': REF_EX.round(2)}
             ).to_csv(os.path.join(HERE, "qc_reference_profile.csv"), index=False)

t1 = []
for tp in TPS:
    I  = wells('PBS', tp); EM = wells('PBS', tp, 'emission')
    for k, w in enumerate(sorted(df[df.cond == 'PBS']['sample'].unique())):
        sh = share(I[k])
        t1.append(dict(well=w, tp=tp,
                       QC1_brightness=I[k].mean(),
                       QC1_log2_vs_ref=np.log2(I[k].mean()/REF_I),
                       QC2_fingerprint_r=stats.pearsonr(sh, REF_SHARE)[0],
                       QC2_max_share_dev=np.abs(sh-REF_SHARE).max(),
                       QC3_em_rms_nm=np.sqrt(((EM[k]-REF_EM)**2).mean())))
T1 = pd.DataFrame(t1)
print("\nQC1/QC2/QC3 per PBS well (mean +/- sd over 5 wells):")
print(T1.groupby('tp')[['QC1_brightness', 'QC2_fingerprint_r', 'QC2_max_share_dev',
                        'QC3_em_rms_nm']].agg(['mean', 'std']).reindex(TPS).round(4).to_string())

# QC4 replicate precision, per chirality
print("\nQC4  replicate %CV across 5 PBS wells:")
cv_rows = []
for tp in TPS:
    I = wells('PBS', tp)
    cv_abs = 100*I.std(0, ddof=1)/I.mean(0)                       # raw intensity
    S = np.array([share(r) for r in I])
    cv_sh = 100*S.std(0, ddof=1)/S.mean(0)                        # relative share
    cv_rows.append(dict(tp=tp, cv_intensity_med=np.median(cv_abs), cv_intensity_max=cv_abs.max(),
                        cv_share_med=np.median(cv_sh), cv_share_max=cv_sh.max()))
    if tp == '0h':
        CV_SHARE_0H = cv_sh; CV_ABS_0H = cv_abs
CV = pd.DataFrame(cv_rows)
print(CV.round(2).to_string(index=False))
print("  -> raw intensity is well-to-well noisy; the relative-share profile is the stable readout")

# QC5 drift
I0, I24 = wells('PBS', '0h'), wells('PBS', '24h')
drift = (I24.mean(1)/I0.mean(1))
r_dd, p_dd = stats.pearsonr(DVEC, ((I24/I0)-1).mean(0))
print(f"\nQC5  drift I(24h)/I(0h) per PBS well: {np.round(drift,3)}  "
      f"mean={drift.mean():.3f} sd={drift.std(ddof=1):.3f}")
print(f"     drift-vs-diameter slope: r={r_dd:+.2f} p={p_dd:.3f}  (structured, not random)")
T1.to_csv(os.path.join(HERE, "qc_tier1_sensor.csv"), index=False)

# QC6 read-order / equilibration: is the 0h read itself less reproducible?
print("\nQC6  read stability by timepoint (leave-one-out spread of the share profile):")
for tp in TPS:
    line = f"    {tp:4s} "
    for cond in COND:
        S = np.array([share(r) for r in wells(cond, tp)])
        tot = S.std(0, ddof=1).mean()
        loo = min(np.delete(S, k, 0).std(0, ddof=1).mean() for k in range(len(S)))
        w1 = np.delete(S, 0, 0).std(0, ddof=1).mean()
        line += f" {cond}: SD={tot:.4f} (drop rep1 -> {100*(w1-tot)/tot:+3.0f}%) "
    print(line)
print("     -> 0h is the least reproducible read in EVERY arm, and replicate #1 drives it;")
print("        the effect is condition-independent => plate read-order / equilibration,")
print("        not protein. By 6-24h the profile has settled.")

# ---- cross-batch transferability of the fingerprint ----
print("\nCross-batch check of the QC2 reference fingerprint (different runs, different extractors):")
xb = {'exp4 PBS (fitted)': REF_SHARE}
try:
    c5 = pd.read_csv(os.path.join(HERE, "controls_features_long.csv"), dtype={'chirality': str})
    p5 = c5[(c5.group == 'P') & (c5.tp == '0h')].pivot_table(index='chirality', columns='sample',
                                                             values='gauss_max').reindex(CHIRS)
    xb['exp5 PBS (gauss_max)'] = share(p5.mean(1).to_numpy(float))
except Exception as e:
    print("  exp5 controls unavailable:", e)
try:
    kk = pd.read_csv(os.path.join(HERE, "kinetics_features_long.csv"), dtype={'chirality': str})
    for conc in ['1mg', '5mg']:
        v = kk[(kk.condition == 'pbs') & (kk.conc == conc) & (kk.tp_h == 0)].set_index(
            'chirality').reindex(CHIRS)['gauss_max'].to_numpy(float)
        xb[f'kinetics PBS {conc}'] = share(v)
except Exception as e:
    print("  kinetics unavailable:", e)
names = list(xb)
for nm in names[1:]:
    r = stats.pearsonr(xb[nm], REF_SHARE)[0]
    print(f"  r(exp4 ref, {nm:22s}) = {r:+.3f}")

# ================================================================ TIER 2
print("\n"+"="*74)
print("TIER 2 -- ASSAY-WINDOW QC  (PBS = negative, FBS = positive)")
print("="*74)

def zprime(pos, neg):
    sep = abs(pos.mean()-neg.mean())
    return 1-3*(pos.std(ddof=1)+neg.std(ddof=1))/sep if sep > 0 else -np.inf
def ssmd(pos, neg):
    return (pos.mean()-neg.mean())/np.sqrt(pos.std(ddof=1)**2+neg.std(ddof=1)**2)

# per-chirality, on the relative-share readout (scale-free -> plate-transferable)
t2 = []
for tp in TPS:
    Pn = np.array([share(r) for r in wells('PBS', tp)])
    Fp = np.array([share(r) for r in wells('FBS', tp)])
    for j, ch in enumerate(CHIRS):
        t2.append(dict(tp=tp, chirality=ch, d_nm=DVEC[j],
                       zprime=zprime(Fp[:, j], Pn[:, j]), ssmd=ssmd(Fp[:, j], Pn[:, j]),
                       p=stats.ttest_ind(Fp[:, j], Pn[:, j], equal_var=False)[1]))
T2 = pd.DataFrame(t2)
print("\nPer-chirality |SSMD| on relative share (FBS vs PBS), by timepoint:")
print("  (Zhang criteria: |SSMD|>3 excellent, >2 good, >1 weak)")
piv = T2.assign(a=T2.ssmd.abs()).pivot(index='chirality', columns='tp', values='a').reindex(CHIRS)[TPS]
print(piv.round(2).to_string())
for tp in TPS:
    s = T2[T2.tp == tp]['ssmd'].abs(); z = T2[T2.tp == tp]['zprime']
    print(f"    {tp:4s}  |SSMD|>3: {(s > 3).sum():2d}/12  >2: {(s > 2).sum():2d}/12   "
          f"[Z'>0: {(z > 0).sum():2d}/12]")

# variance asymmetry -- why Z' is not the right metric here
print("\n  control variance asymmetry (mean over chir of SD of relative share):")
for tp in TPS:
    sp = np.array([share(r) for r in wells('PBS', tp)]).std(0, ddof=1).mean()
    sf = np.array([share(r) for r in wells('FBS', tp)]).std(0, ddof=1).mean()
    print(f"    {tp:4s}  SD(PBS)={sp:.4f}  SD(FBS)={sf:.4f}   ratio={sf/sp:.1f}x")

# 1-D corona index: project the well's share profile on the PBS->FBS axis (built at 24h)
axis = unit(np.array([share(r) for r in wells('FBS', '24h')]).mean(0)
            - np.array([share(r) for r in wells('PBS', '24h')]).mean(0))
print("\nCorona index CI = (relative-share profile) . (PBS->FBS axis).  Assay-window Z' on CI:")
ci_rows = []
for tp in TPS:
    ci = {c: np.array([share(r) for r in wells(c, tp)]) @ axis for c in COND}
    z, s = zprime(ci['FBS'], ci['PBS']), ssmd(ci['FBS'], ci['PBS'])
    ci_rows.append(dict(tp=tp, zprime_CI=z, ssmd_CI=s,
                        CI_PBS=ci['PBS'].mean(), CI_FBS=ci['FBS'].mean(), CI_HEP=ci['FBS+HEP'].mean(),
                        sd_PBS=ci['PBS'].std(ddof=1), sd_FBS=ci['FBS'].std(ddof=1)))
CI = pd.DataFrame(ci_rows)
print(CI.round(3).to_string(index=False))
T2.to_csv(os.path.join(HERE, "qc_tier2_window.csv"), index=False)
CI.to_csv(os.path.join(HERE, "qc_tier2_corona_index.csv"), index=False)

# ================================================================ TIER 3
print("\n"+"="*74)
print("TIER 3 -- PBS-ANCHORED NOISE FLOOR / LOD per chirality")
print("="*74)
t3 = []
for j, ch in enumerate(CHIRS):
    pn = np.array([share(r) for r in wells('PBS', '24h')])[:, j]
    fp = np.array([share(r) for r in wells('FBS', '24h')])[:, j]
    eff = fp.mean()-pn.mean(); lod = 3*pn.std(ddof=1)
    t3.append(dict(chirality=ch, d_nm=DVEC[j], pbs_share=pn.mean(), pbs_sd=pn.std(ddof=1),
                   pbs_cv_pct=100*pn.std(ddof=1)/pn.mean(), corona_effect=eff,
                   LOD_3sd=lod, SNR=abs(eff)/pn.std(ddof=1), usable=abs(eff) > lod))
T3 = pd.DataFrame(t3).sort_values('SNR', ascending=False)
print(T3.round(4).to_string(index=False))
print(f"\n  {T3.usable.sum()}/12 chiralities carry a corona effect above 3xSD(PBS) at 24h")
T3.to_csv(os.path.join(HERE, "qc_tier3_lod.csv"), index=False)

# ================================================================ acceptance rule
print("\n"+"="*74)
print("PROPOSED RUN-ACCEPTANCE RULE  (derived from the 5 exp4 PBS wells)")
print("="*74)
r_all = T1[T1.tp == '0h']['QC2_fingerprint_r']
dev_all = T1[T1.tp == '0h']['QC2_max_share_dev']
em_all = T1[T1.tp == '0h']['QC3_em_rms_nm']
print(f"  QC2 fingerprint r     observed {r_all.min():.4f}-{r_all.max():.4f}  -> threshold r >= 0.99")
print(f"  QC2 max share dev     observed {dev_all.min():.3f}-{dev_all.max():.3f}  -> threshold <= 0.15")
print(f"  QC3 emission RMS      observed {em_all.min():.2f}-{em_all.max():.2f} nm -> threshold <= 2 nm")
print(f"  QC4 share %CV (median) observed {np.median(CV_SHARE_0H):.1f}%          -> threshold median <= 10%")
print(f"  QC5 drift I24/I0      observed {drift.min():.2f}-{drift.max():.2f}     -> flag if outside 0.5-2.0")
print(f"  QC6 window |SSMD(CI)| observed {CI['ssmd_CI'].abs().min():.2f}-{CI['ssmd_CI'].abs().max():.2f}"
      f"    -> require |SSMD| >= 2 on plates carrying FBS")

# ================================================================ FIGURE
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
fig, axs = plt.subplots(1, 4, figsize=(14.2, 3.4))
fig.subplots_adjust(left=0.05, right=0.995, bottom=0.19, top=0.84, wspace=0.36)
x = np.arange(12)

# (a) reference PBS fingerprint + replicate spread
a = axs[0]
S = np.array([share(r) for r in P0])
a.errorbar(x, REF_SHARE, yerr=S.std(0, ddof=1), color=C['PBS'], lw=1.6, marker='o', ms=4, capsize=2)
a.set_xticks(x); a.set_xticklabels(CHIRS, rotation=90, fontsize=6.5)
a.set_ylabel("PBS relative share (I / mean)"); a.set_xlabel("chirality")
a.set_title(f"(a) QC2 reference fingerprint\n median %CV = {np.median(CV_SHARE_0H):.1f}% (n=5)", loc="left")

# (b) per-chirality |SSMD| at each tp
b = axs[1]
for tp, mk in zip(TPS, ['o', 's', '^', 'D']):
    b.plot(x, T2[T2.tp == tp].set_index('chirality').reindex(CHIRS)['ssmd'].abs(), '-'+mk,
           ms=3.5, lw=1.2, label=tp)
b.axhline(3, color=MUT, ls='--', lw=0.8); b.axhline(2, color=MUT, ls=':', lw=0.8)
b.text(0.1, 3.15, "excellent", fontsize=6.5, color=MUT)
b.text(0.1, 2.1, "good", fontsize=6.5, color=MUT)
b.set_xticks(x); b.set_xticklabels(CHIRS, rotation=90, fontsize=6.5)
b.set_ylabel("|SSMD| (FBS vs PBS)"); b.set_xlabel("chirality")
b.set_title("(b) Tier-2 assay window\n per chirality", loc="left")
b.legend(frameon=False, fontsize=7, ncol=2)

# (c) corona index separation
c = axs[2]
for i, tp in enumerate(TPS):
    for cond, off in zip(COND, [-0.22, 0, 0.22]):
        v = np.array([share(r) for r in wells(cond, tp)]) @ axis
        c.scatter(np.full(len(v), i+off), v, s=16, color=C[cond], zorder=3,
                  label=cond if i == 0 else None)
        c.plot([i+off-0.09, i+off+0.09], [v.mean()]*2, color=INK, lw=1.2, zorder=4)
c.set_xticks(range(4)); c.set_xticklabels(TPS)
c.set_xlabel("time"); c.set_ylabel("corona index CI")
_best = CI.loc[CI.ssmd_CI.abs().idxmax()]
c.set_title(f"(c) 1-D QC readout\n best |SSMD|={abs(_best.ssmd_CI):.1f} @ {_best.tp}", loc="left")
c.legend(frameon=False, fontsize=7)

# (d) LOD: corona effect vs 3sd(PBS)
d = axs[3]
T3s = T3.set_index('chirality').reindex(CHIRS)
d.bar(x, T3s['corona_effect'].abs(), color=[C['FBS'] if u else MUT for u in T3s['usable']], alpha=.85,
      label="|corona effect|")
d.plot(x, T3s['LOD_3sd'], 'k_', ms=12, mew=1.6, label="LOD = 3xSD(PBS)")
d.set_xticks(x); d.set_xticklabels(CHIRS, rotation=90, fontsize=6.5)
d.set_ylabel("relative-share units"); d.set_xlabel("chirality")
d.set_title(f"(d) Tier-3 usable readouts\n {T3.usable.sum()}/12 above LOD @24h", loc="left")
d.legend(frameon=False, fontsize=7)

fig.savefig(os.path.join(HERE, "FigSX_qc.png"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(HERE, "FigSX_qc.pdf"), bbox_inches="tight")
print("\nSaved FigSX_qc.png/.pdf + qc_tier{1,2,3}*.csv, qc_reference_profile.csv")

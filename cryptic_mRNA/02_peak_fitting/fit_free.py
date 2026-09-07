#!/usr/bin/env python3
"""
Fit-FREE spectral descriptors (no Gaussian, no curve_fit, no R2 gate).

Per (well, chirality), on the +/-25 nm emission window:
  - linear baseline subtraction (connect the two window endpoints)
  - Ipos = clip(spectrum - baseline, 0)
  - integ   = trapz(Ipos)                       -> brightness (analog of gauss_max/auc)
  - peak    = max(Ipos)
  - centroid= trapz(lambda*Ipos)/trapz(Ipos)    -> position (fit-free Delta-lambda)
  - width   = sqrt(2nd moment)                  -> fit-free FWHM-like
  - snr     = peak / robust noise (MAD of diff)

These are well-defined even when a Gaussian fit fails. We then redo the two
questions model-free: (1) does brightness rise with dose? (2) does centroid shift?
"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr

HERE = "/mnt/c/Users/riccardo-s/Documents/CNT/cryptic_mRNA/exp1"
EXC = [570, 640, 670, 750, 780]
WIN = 25.0
CHIR = {"ch8_3":973.98,"ch6_5":987.82,"ch7_5":1047.81,"ch10_2":1080.60,
        "ch9_4":1131.96,"ch8_4":1130.34,"ch7_6":1138.19,"ch8_6":1200.03,
        "ch8_7":1288.27,"ch9_5":1262.98,"ch10_3":1267.70,"ch10_5":1282.97}
EXOF  = {"ch8_3":673.94,"ch6_5":577.12,"ch7_5":653.32,"ch10_2":745.92,
         "ch9_4":731.39,"ch8_4":599.78,"ch7_6":659.79,"ch8_6":727.40,
         "ch8_7":740.87,"ch9_5":685.15,"ch10_3":648.97,"ch10_5":801.23}
EXMAP = {c: min(EXC, key=lambda e: abs(e-EXOF[c])) for c in CHIR}
COPIES = [0,10,100,1000,10000,100000]

def parse(name):
    col=int(name[1:]); idx=(col-1) if col<=6 else (col-7)
    return ("GT15-STMN2" if name[0] in "ABCDEF" else "GT15",
            "water" if col<=6 else "serum", col, COPIES[idx])

def moments(em, sp, emc):
    sel = np.abs(em-emc) <= WIN
    x, y = em[sel], sp[sel]
    # linear baseline through endpoints (median of 3 pts each end for robustness)
    yl, yr = np.median(y[:3]), np.median(y[-3:])
    base = np.interp(x, [x[0], x[-1]], [yl, yr])
    ip = np.clip(y - base, 0, None)
    integ = np.trapz(ip, x)
    if integ <= 0:
        return dict(ff_integ=0.0, ff_peak=float(max(y.max()-min(yl,yr),0)),
                    ff_centroid=np.nan, ff_width=np.nan, ff_snr=0.0)
    cen = np.trapz(x*ip, x)/integ
    var = np.trapz((x-cen)**2*ip, x)/integ
    noise = 1.4826*np.median(np.abs(np.diff(y)-np.median(np.diff(y))))/np.sqrt(2)+1e-30
    return dict(ff_integ=float(integ), ff_peak=float(ip.max()),
                ff_centroid=float(cen), ff_width=float(np.sqrt(max(var,0))),
                ff_snr=float(ip.max()/noise))

em = pd.read_csv(f"{HERE}/emission.txt")["Emission"].to_numpy(float)
spectra = {e: pd.read_csv(f"{HERE}/{e}.txt", sep="\t") for e in EXC}
wells = list(spectra[570].columns)

rows=[]
for w in wells:
    s,m,col,cop = parse(w)
    r=dict(well=w, sensor=s, matrix=m, col=col, copies=cop)
    for c,emc in CHIR.items():
        d = moments(em, spectra[EXMAP[c]][w].to_numpy(float), emc)
        for k,v in d.items(): r[f"{c}_{k}"]=v
    rows.append(r)
ff = pd.DataFrame(rows)
ff["logc"]=np.log10(ff["copies"]+1)
ff.to_csv(f"{HERE}/fit_free_descriptors.csv", index=False)

CLEAN=["ch6_5","ch7_5","ch8_3"]
ff["integ3"]=ff[[f"{c}_ff_integ" for c in CLEAN]].sum(axis=1)
sw=ff[(ff.sensor=="GT15-STMN2")&(ff.matrix=="water")]
cw=ff[(ff.sensor=="GT15")&(ff.matrix=="water")]

print("="*64); print("FIT-FREE (spectral moments) — no Gaussian, no R2 gate"); print("="*64)

print("\n(1) BRIGHTNESS: integrated intensity (clean 3 tubes) vs dose, water")
for lab,g in [("GT15-STMN2",sw),("GT15 ctrl",cw)]:
    r,p=spearmanr(g["logc"],g["integ3"])
    fc=g[g.copies==g.copies.max()].integ3.mean()/g[g.copies==0].integ3.mean()
    print(f"  {lab:11}: rho={r:+.2f} p={p:.1e}  fold(max/0)={fc:.2f}x")

print("\n(2) POSITION: fit-free centroid (Delta-lambda) vs dose, GT15-STMN2/water")
print(f"  {'tube':7} {'rho':>6} {'p':>8}  centroid span across conc (nm)")
for c in CLEAN+["ch7_6","ch9_5"]:
    s=sw[[f"{c}_ff_centroid","logc","copies"]].dropna()
    r,p=spearmanr(s["logc"],s[f"{c}_ff_centroid"])
    span=s.groupby("copies")[f"{c}_ff_centroid"].mean()
    print(f"  {c:7} {r:+6.2f} {p:8.1e}  {span.max()-span.min():.2f}  "
          f"({span.min():.1f}->{span.max():.1f})")

print("\n(3) cross-check: fit-free centroid vs Gaussian em_center (all wells, clean tubes)")
g=pd.read_csv(f"{HERE}/chirality_gaussian_descriptors.csv")
for c in CLEAN:
    m=ff[f"{c}_ff_centroid"].notna()&g[f"{c}_gauss_em_center"].notna()
    r=np.corrcoef(ff.loc[m,f"{c}_ff_centroid"], g.loc[m,f"{c}_gauss_em_center"])[0,1]
    print(f"  {c}: r(centroid, gauss_center) = {r:+.2f}")

print("\nSaved fit_free_descriptors.csv")

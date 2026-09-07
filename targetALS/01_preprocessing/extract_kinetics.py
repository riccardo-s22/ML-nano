"""
Extract 12-chirality features (max5x5 + gauss_max) from the dense corona-kinetics
series: 7 timepoints (0,2,4,6,8,10,24h) x 3 conditions (pbs, fbs, fbsri) x
2 CNT concentrations (1mg, 5mg /L). One EEM each.

File layout differs from the exp5 EEMs: columns are
    N | Emission range | EX500 | EX505 | ... | EX850
Same emission grid (852-1676 nm) and excitation grid (500-850 nm).

Output: kinetics_features_long.csv
    condition(pbs/fbs/fbsri) conc(1mg/5mg) tp_h(int) chirality max5x5 gauss_max
"""
import os, re, sys, glob
import numpy as np
import pandas as pd

EXP5 = r"/mnt/c/Users/riccardo-s/Documents/CNT/targetALS/exp5"
RAW  = os.path.join(EXP5, "tech_controls",
    "Protein_corona_kinetics-20260811T162107Z-1-001", "Protein_corona_kinetics", "raw_data")
OUT  = os.path.join(EXP5, "tech_controls", "analysis")
sys.path.insert(0, os.path.join(EXP5, "physical_descriptors"))
import chirality_interpolated_extraction as X

CHIRS = ['6.5','7.5','7.6','8.3','8.4','8.6','8.7','9.4','9.5','10.2','10.3','10.5']
CHIRALITY_CENTERS = {  # (excitation_nm, emission_nm) - max5x5 method
    '6.5': (573, 975), '7.5': (647, 1024), '7.6': (645, 1115), '8.3': (667, 952),
    '8.4': (726, 1100), '8.6': (718, 1170), '8.7': (726, 1260), '9.4': (720, 1100),
    '9.5': (800, 1240), '10.2': (740, 1050), '10.3': (800, 1100), '10.5': (850, 1250),
}
GAUSS_KEY = {c: "ch" + c.replace('.', '_') for c in CHIRS}

# canonical emission axis (512 pts) taken from a headered 0h file; exc = 500..850 step 5
CANON_EM = pd.read_excel(os.path.join(RAW, "0h", "fbs_1mg_0h.xlsx"))["Emission range"].to_numpy(float)
EXC = np.arange(500, 851, 5).astype(float)   # 71 excitation values

def load_eem(path):
    """Handle two layouts: 0h files have a header (N|Emission range|EX500..);
    2-24h files are headerless 512x71 intensity blocks (emission axis implicit)."""
    raw = pd.read_excel(path, header=None)
    if raw.shape[1] == 73:                    # headered 0h layout
        df = pd.read_excel(path).rename(columns={'Emission range': 'Emission'})
        em = df['Emission'].to_numpy(float)
        excols = [c for c in df.columns if str(c).startswith('EX')]
        exc = np.array([float(re.findall(r'\d+', c)[0]) for c in excols], float)
        order = np.argsort(exc); exc = exc[order]; excols = [excols[i] for i in order]
        mat = df[excols].to_numpy(float)
        return mat, em, exc, df, excols
    # headerless: 512 x 71 intensities, columns already EX500..EX850 ascending
    mat = raw.to_numpy(float)
    assert mat.shape == (len(CANON_EM), len(EXC)), f"unexpected shape {mat.shape} in {path}"
    excols = [f"EX{int(v)}" for v in EXC]
    df = pd.DataFrame(mat, columns=excols); df['Emission'] = CANON_EM
    return mat, CANON_EM, EXC, df, excols

def gauss_max_all(mat, em, exc):
    spline, em_f, ex_f = X.build_interpolated_surface(mat, em, exc, factor=X.INTERP_FACTOR)
    out = {}
    for name, coords in X.CHIRALITY_POSITIONS.items():
        be, bx, mx = X.find_local_max_in_radius(spline, em_f, ex_f,
                        coords['emission_nm'], coords['excitation_nm'], X.SEARCH_RADIUS_NM)
        em_pts, int_pts = X.extract_consistent_gradient_points(spline, em_f, be, bx)
        d = X.fit_gaussian_and_extract(em_pts, int_pts, be, mx)
        out[name] = float(d['gauss_max'])
    return out

def max5x5(df, excols, exc, exc_t, em_t, win=2):
    em = df['Emission'].to_numpy(float)
    ie = int(np.argmin(np.abs(exc - exc_t))); im = int(np.argmin(np.abs(em - em_t)))
    ci = np.arange(max(0, ie-win), min(len(exc)-1, ie+win)+1)
    ri = np.arange(max(0, im-win), min(len(em)-1, im+win)+1)
    sub = np.nan_to_num(df.loc[ri, [excols[i] for i in ci]].to_numpy(float))
    return float(np.max(sub))

def parse_name(fp):
    b = os.path.basename(fp).lower()
    m = re.match(r'(fbsri|fbs|pbs)_(\d)mg_(\d+)h', b)
    return m.group(1), m.group(2)+"mg", int(m.group(3))

def main():
    files = sorted(glob.glob(os.path.join(RAW, "*", "*.xlsx")))
    print(f"{len(files)} files")
    rows = []
    for fp in files:
        cond, conc, tp = parse_name(fp)
        mat, em, exc, df, excols = load_eem(fp)
        gm = gauss_max_all(mat, em, exc)
        for c in CHIRS:
            ex_t, em_t = CHIRALITY_CENTERS[c]
            rows.append(dict(condition=cond, conc=conc, tp_h=tp, chirality=c,
                             max5x5=max5x5(df, excols, exc, ex_t, em_t),
                             gauss_max=gm[GAUSS_KEY[c]]))
        print(f"  {cond:6s} {conc} {tp:>2d}h done")
    L = pd.DataFrame(rows)
    L.to_csv(os.path.join(OUT, "kinetics_features_long.csv"), index=False)
    print("\nSaved kinetics_features_long.csv", L.shape)
    for feat in ("max5x5","gauss_max"):
        piv = L.groupby(['condition','conc','tp_h'])[feat].mean().unstack('tp_h')
        print(f"\nmean {feat} (over 12 chir) by condition x conc x tp:\n{piv.round(3)*1e14} (x1e-14)")

if __name__ == "__main__":
    main()

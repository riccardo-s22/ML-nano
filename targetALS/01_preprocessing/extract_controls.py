"""
Extract 12-chirality peak features from the PBS (P) and FBS (F) technical
controls at 0/6/24 h, using EXACTLY the two feature definitions the paper uses:

  max5x5   - max intensity in a 5x5 raw-pixel window at each chirality center
             (method + centers from multiplexing_results/validate_multiplexing_ple.py)
  gauss_max- split-Gaussian fitted amplitude on the bicubic-interpolated surface
             (method + DNA coordinates from
              physical_descriptors/chirality_interpolated_extraction.py)

Output (long tidy table): controls_features_long.csv
    group(P/F)  sample(P1..P4/F1..F3)  tp(0h/6h/24h)  chirality  max5x5  gauss_max

Note on artifact masking: the 12 chirality ROIs sit at ex 577-801 nm / em 974-1288
nm, well away from the excitation edge (ex>820) and the 2nd-order scatter line
(em = 2*ex). Masking those regions therefore does not touch any peak window, so
the per-chirality features are already artifact-clean (unlike whole-EEM sums).
"""
import os, re, sys, glob
import numpy as np
import pandas as pd

EXP5 = r"/mnt/c/Users/riccardo-s/Documents/CNT/targetALS/exp5"
CTRL = os.path.join(EXP5, "tech_controls")
OUT  = os.path.join(CTRL, "analysis")
sys.path.insert(0, os.path.join(EXP5, "physical_descriptors"))

# --- gauss_max pipeline (import, guarded main so safe) ---
from chirality_interpolated_extraction import extract_sample_tp, CHIRALITY_POSITIONS

# --- max5x5 centers (excitation_nm, emission_nm) + extractor, verbatim ---
CHIRALITY_CENTERS = {
    '6.5': (573, 975), '7.5': (647, 1024), '7.6': (645, 1115), '8.3': (667, 952),
    '8.4': (726, 1100), '8.6': (718, 1170), '8.7': (726, 1260), '9.4': (720, 1100),
    '9.5': (800, 1240), '10.2': (740, 1050), '10.3': (800, 1100), '10.5': (850, 1250),
}
# canonical chirality order & the gauss column name for each
CHIRS = ['6.5','7.5','7.6','8.3','8.4','8.6','8.7','9.4','9.5','10.2','10.3','10.5']
GAUSS_KEY = {c: "ch" + c.replace('.', '_') for c in CHIRS}   # '6.5' -> 'ch6_5'

def parse_exc(df):
    cols = [c for c in df.columns if str(c).startswith("Excitation_")]
    vals = []
    for c in cols:
        m = re.search(r"Excitation_(\d+)\.(\d+)", str(c))
        vals.append(float(m.group(1)) + float("0." + m.group(2)) if m
                    else float(re.findall(r"[\d.]+", str(c))[0]))
    vals = np.array(vals, float); order = np.argsort(vals)
    return [cols[i] for i in order], vals[order]

def max5x5(df, exc_t, em_t, win=2):
    em = df["Emission"].to_numpy(float)
    cols, exc = parse_exc(df)
    ie, im = int(np.argmin(np.abs(exc - exc_t))), int(np.argmin(np.abs(em - em_t)))
    ci = np.arange(max(0, ie-win), min(len(exc)-1, ie+win)+1)
    ri = np.arange(max(0, im-win), min(len(em)-1, im+win)+1)
    sub = np.nan_to_num(df.loc[ri, [cols[i] for i in ci]].to_numpy(float))
    return float(np.max(sub))

# --- locate every (group, sample, tp) file ------------------------------
def controls_files():
    """Return list of (group, sample, tp, filepath)."""
    out = []
    # 0h: F1-3 + P1 at root, P2-4 in 0h/
    for p in glob.glob(os.path.join(CTRL, "[FP]*.xlsx")):
        m = re.match(r"([FP])(\d)", os.path.basename(p))
        if m: out.append((m.group(1), m.group(1)+m.group(2), "0h", p))
    for p in glob.glob(os.path.join(CTRL, "0h", "*.xlsx")):
        m = re.match(r"([FP])(\d)", os.path.basename(p))
        if m: out.append((m.group(1), m.group(1)+m.group(2), "0h", p))
    for tp in ("6h", "24h"):
        for p in glob.glob(os.path.join(CTRL, tp, "*.xlsx")):
            m = re.match(r"([FP])(\d)", os.path.basename(p))
            if m: out.append((m.group(1), m.group(1)+m.group(2), tp, p))
    return out

def main():
    files = controls_files()
    print(f"Found {len(files)} control EEM files")
    by = {}
    for g, s, tp, _ in files: by.setdefault(tp, []).append(s)
    for tp in ("0h","6h","24h"):
        print(f"  {tp}: {sorted(by.get(tp,[]))}")

    rows = []
    for g, s, tp, fp in sorted(files, key=lambda x: (x[2], x[1])):
        df = pd.read_excel(fp)                       # for max5x5
        gres = extract_sample_tp(fp, CHIRALITY_POSITIONS)   # for gauss_max
        for c in CHIRS:
            ex_t, em_t = CHIRALITY_CENTERS[c]
            m5 = max5x5(df, ex_t, em_t)
            gm = float(gres[GAUSS_KEY[c]]["gauss_max"])
            rows.append(dict(group=g, sample=s, tp=tp, chirality=c,
                             max5x5=m5, gauss_max=gm))
        print(f"  done {g} {s} {tp}")
    L = pd.DataFrame(rows)
    L.to_csv(os.path.join(OUT, "controls_features_long.csv"), index=False)
    print("\nSaved controls_features_long.csv", L.shape)
    # quick sanity: mean per group x tp
    for feat in ("max5x5", "gauss_max"):
        piv = L.groupby(['group','tp'])[feat].mean().unstack('tp')[['0h','6h','24h']]
        print(f"\nmean {feat} by group x tp:\n{piv}")

if __name__ == "__main__":
    main()

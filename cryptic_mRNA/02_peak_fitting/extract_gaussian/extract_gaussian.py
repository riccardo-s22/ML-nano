#!/usr/bin/env python3
"""
extract_gaussian.py
===================
Gaussian fitting + feature extraction for the GT15-STMN2 / STMN2-CE plate (exp1).

Data layout (per excitation file 570/640/670/750/780.txt):
  - tab-separated; header row = well names (A1 absent: water/no-SWCNT blank,
    already background-subtracted by the instrument)
  - 512 emission-intensity rows, wavelengths listed in emission.txt

Each chirality is read at the excitation file nearest its resonance peak, then a
split (asymmetric) Gaussian + baseline is fit on a local emission window around
the chirality's known emission center (method mirrors targetALS exp5
chirality_interpolated_extraction.py).

Output: chirality_gaussian_descriptors.csv  (one row per well)
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import skew as sp_skew, kurtosis as sp_kurtosis

HERE = "/mnt/c/Users/riccardo-s/Documents/CNT/cryptic_mRNA/exp1"
EXCITATIONS = [570, 640, 670, 750, 780]
WINDOW_NM = 25.0  # local fit half-window around each chirality emission center

# Chirality emission/excitation peaks (Coordinates_DNA.txt)
CHIRALITY = {
    "ch8_3":  {"em": 973.98,  "ex": 673.94},
    "ch6_5":  {"em": 987.82,  "ex": 577.12},
    "ch7_5":  {"em": 1047.81, "ex": 653.32},
    "ch10_2": {"em": 1080.60, "ex": 745.92},
    "ch9_4":  {"em": 1131.96, "ex": 731.39},
    "ch8_4":  {"em": 1130.34, "ex": 599.78},
    "ch7_6":  {"em": 1138.19, "ex": 659.79},
    "ch8_6":  {"em": 1200.03, "ex": 727.40},
    "ch8_7":  {"em": 1288.27, "ex": 740.87},
    "ch9_5":  {"em": 1262.98, "ex": 685.15},
    "ch10_3": {"em": 1267.70, "ex": 648.97},
    "ch10_5": {"em": 1282.97, "ex": 801.23},
}
DESC = ["gauss_max", "gauss_em_center", "gauss_fwhm", "gauss_auc",
        "gauss_skew", "gauss_kurt", "gauss_r2", "gauss_npts", "ex_used"]

COPIES = [0, 10, 100, 1000, 10000, 100000]


def split_gaussian_with_baseline(x, amplitude, center, sigma_left, sigma_right, baseline):
    out = np.empty_like(x, dtype=float)
    left = x <= center
    right = ~left
    out[left] = baseline + amplitude * np.exp(-0.5 * ((x[left] - center) / sigma_left) ** 2)
    out[right] = baseline + amplitude * np.exp(-0.5 * ((x[right] - center) / sigma_right) ** 2)
    return out


def assign_excitation():
    """Map each chirality to the nearest available excitation file."""
    m = {}
    for ch, c in CHIRALITY.items():
        m[ch] = min(EXCITATIONS, key=lambda e: abs(e - c["ex"]))
    return m


def fit_chirality(em_axis, spectrum, em_center):
    """Fit split Gaussian + baseline on a local window; return descriptor dict."""
    sel = np.abs(em_axis - em_center) <= WINDOW_NM
    em_pts = em_axis[sel]
    int_pts = spectrum[sel]
    res = {k: np.nan for k in DESC[:-1]}
    res["gauss_npts"] = float(len(em_pts))
    if len(em_pts) < 5:
        return res

    # intensity-weighted skew/kurt
    w = int_pts - int_pts.min()
    ws = w.sum()
    if ws > 1e-15:
        wn = w / ws
        wm = np.sum(em_pts * wn)
        wv = np.sum(wn * (em_pts - wm) ** 2)
        wstd = np.sqrt(wv) if wv > 0 else 1.0
        if wstd > 1e-12:
            z = (em_pts - wm) / wstd
            res["gauss_skew"] = float(np.sum(wn * z ** 3))
            res["gauss_kurt"] = float(np.sum(wn * z ** 4) - 3.0)
    else:
        res["gauss_skew"] = float(sp_skew(int_pts, bias=True))
        res["gauss_kurt"] = float(sp_kurtosis(int_pts, fisher=True, bias=True))

    scale = float(int_pts.max())
    if scale < 1e-30:
        return res
    yn = int_pts / scale
    bl = float(yn.min())
    amp = max(1.0 - bl, 0.01)

    half = bl + 0.5 * amp
    pk = int(np.argmax(yn))
    pem = em_pts[pk]
    sl0 = sr0 = 8.0
    for i in range(pk, -1, -1):
        if yn[i] <= half:
            sl0 = max((pem - em_pts[i]) / 1.177, 1.0); break
    for i in range(pk, len(yn)):
        if yn[i] <= half:
            sr0 = max((em_pts[i] - pem) / 1.177, 1.0); break

    try:
        popt, _ = curve_fit(
            split_gaussian_with_baseline, em_pts, yn,
            p0=[amp, pem, sl0, sr0, bl],
            bounds=([0, em_pts.min() - 5, 0.5, 0.5, 0],
                    [2.0, em_pts.max() + 5, 80.0, 80.0, 1.0]),
            maxfev=10000,
        )
        a, c, sl, sr, b = popt
        res["gauss_max"] = float((a + b) * scale)
        res["gauss_em_center"] = float(c)
        res["gauss_fwhm"] = float(1.17741 * (sl + sr))
        res["gauss_auc"] = float(a * np.sqrt(2 * np.pi) * (sl + sr) / 2.0 * scale)
        fit = split_gaussian_with_baseline(em_pts, *popt)
        ssr = float(np.sum((yn - fit) ** 2))
        sst = float(np.sum((yn - yn.mean()) ** 2))
        res["gauss_r2"] = float(1 - ssr / sst) if sst > 1e-30 else np.nan
    except (RuntimeError, ValueError, TypeError):
        res["gauss_max"] = float(int_pts.max())
        res["gauss_em_center"] = float(em_center)
    return res


def parse_well(name):
    row = name[0]
    col = int(name[1:])
    sensor = "GT15-STMN2" if row in "ABCDEF" else "GT15"
    matrix = "water" if col <= 6 else "serum"
    idx = (col - 1) if col <= 6 else (col - 7)
    return dict(well=name, row=row, col=col, sensor=sensor,
                matrix=matrix, conc_idx=idx, copies=COPIES[idx])


def main():
    em_axis = pd.read_csv(f"{HERE}/emission.txt")["Emission"].to_numpy(float)
    spectra = {e: pd.read_csv(f"{HERE}/{e}.txt", sep="\t") for e in EXCITATIONS}
    wells = list(spectra[570].columns)
    ex_map = assign_excitation()

    rows = []
    for well in wells:
        meta = parse_well(well)
        for ch, c in CHIRALITY.items():
            ex = ex_map[ch]
            spec = spectra[ex][well].to_numpy(float)
            d = fit_chirality(em_axis, spec, c["em"])
            d["ex_used"] = ex
            for k in DESC:
                meta[f"{ch}_{k}"] = d[k]
        rows.append(meta)

    df = pd.DataFrame(rows)
    out = f"{HERE}/chirality_gaussian_descriptors.csv"
    df.to_csv(out, index=False)

    # sanity
    r2 = df[[f"{ch}_gauss_r2" for ch in CHIRALITY]].to_numpy().ravel()
    r2 = r2[~np.isnan(r2)]
    print(f"Saved {out}  shape={df.shape}  ({len(wells)} wells)")
    print(f"R2: median={np.median(r2):.4f}  min={r2.min():.4f}  "
          f"frac>0.9={np.mean(r2 > 0.9):.2f}")
    print("\nFitted em-center vs nominal (mean over wells):")
    for ch, c in CHIRALITY.items():
        fc = df[f"{ch}_gauss_em_center"].mean()
        print(f"  {ch:7s} ex{ex_map[ch]}  nominal={c['em']:7.1f}  "
              f"fitted={fc:7.1f}  d={fc - c['em']:+5.1f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
blueshift_predict.py — combine Hyb vs Unhyb radial densities into a predicted
emission-shift SIGN, reproducing Harvey's mechanistic conclusion (Fig 1f,g text):

  On hybridisation:  local water density near the wall INCREASES  -> red-shift term
                     local phosphate density near the wall DECREASES -> blue-shift term
  Net blue-shift observed => phosphate removal out-competes the water increase.

We integrate each density over a near-surface shell (surface .. surface+cutoff) and
report d(rho_water) and d(rho_PO4) between Hyb and Unhyb, plus the predicted sign.

Usage:
  blueshift_predict.py --hyb results/hyb/radial_density.csv \
                       --unhyb results/unhyb/radial_density.csv \
                       --out results/blueshift_prediction.json [--shell_nm 1.0]
"""
import argparse, json
from pathlib import Path
import numpy as np

def near_surface_integral(csv, shell_nm):
    d = np.loadtxt(csv)
    r, w, p = d[:, 0], d[:, 1], d[:, 2]
    # near-surface window: first radius where phosphate/water first appear .. +shell
    r0 = r[np.argmax((w + p) > 0)] if np.any((w + p) > 0) else r[0]
    m = (r >= r0) & (r <= r0 + shell_nm)
    dr = np.diff(r).mean()
    return float(np.sum(w[m]) * dr), float(np.sum(p[m]) * dr)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyb", required=True); ap.add_argument("--unhyb", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--shell_nm", type=float, default=1.0)
    a = ap.parse_args()
    wH, pH = near_surface_integral(a.hyb,   a.shell_nm)
    wU, pU = near_surface_integral(a.unhyb, a.shell_nm)
    dW, dP = wH - wU, pH - pU
    # red-shift contribution ~ +dW ; blue-shift contribution ~ -dP (charge removal)
    # net predicted shift sign: negative => blue-shift (as Harvey observed)
    score = dW - dP                     # >0 red, <0 blue (arbitrary units)
    pred = "BLUE-shift" if score < 0 else "RED-shift"
    res = dict(near_surface_shell_nm=a.shell_nm,
               water_integral_hyb=wH, water_integral_unhyb=wU, d_water=dW,
               phosphate_integral_hyb=pH, phosphate_integral_unhyb=pU, d_phosphate=dP,
               predicted_shift=pred, score=score,
               interpretation=("d_water>0 (red term) and d_phosphate<0 (blue term); "
                               "net %s. Harvey: phosphate removal out-competes water "
                               "increase -> net blue-shift." % pred))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
radial_density.py — radial number-density profiles of WATER (OW) and PHOSPHATE (P)
from the tube axis, reproducing Harvey Fig 1f (water) and Fig 1g (phosphate).

Usage:
  radial_density.py --tpr run/<sys>/prod.tpr --xtc run/<sys>/prod.xtc \
                    --out results/<sys>/radial_density.csv [--begin_ps N]

Tube axis = mean (x,y) of the CNT residue, tube along z. Density is normalised
per-frame-averaged count in each cylindrical shell divided by shell volume.
"""
import argparse
from pathlib import Path
import numpy as np
import MDAnalysis as mda
import yaml

ROOT = Path(__file__).resolve().parents[2]
CFG  = yaml.safe_load(open(ROOT / "config" / "md_config.yaml"))["analysis"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tpr", required=True); ap.add_argument("--xtc", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--begin_ps", type=float, default=0)
    a = ap.parse_args()

    u = mda.Universe(a.tpr, a.xtc)
    cnt = u.select_atoms("resname CNT")
    if len(cnt) == 0:
        raise SystemExit("no CNT atoms found (resname CNT)")
    water = u.select_atoms("resname SOL TIP3 and name %s OW" % CFG["water_atom"])
    if len(water) == 0:
        water = u.select_atoms("name OW OH2")
    phos = u.select_atoms("name %s" % CFG["phosphate_atom"])

    dr = CFG["radial_bin_nm"] * 10.0                 # nm -> A
    rmax = CFG["r_max_nm"] * 10.0
    edges = np.arange(0, rmax + dr, dr)
    rmid = 0.5 * (edges[:-1] + edges[1:])
    w_hist = np.zeros(len(rmid)); p_hist = np.zeros(len(rmid)); nfr = 0

    for ts in u.trajectory:
        if ts.time < a.begin_ps:
            continue
        axis = cnt.positions[:, :2].mean(0)          # (x,y) of tube axis this frame
        Lz = u.dimensions[2]
        for grp, hist in ((water, w_hist), (phos, p_hist)):
            if len(grp) == 0:
                continue
            r = np.hypot(grp.positions[:, 0] - axis[0], grp.positions[:, 1] - axis[1])
            h, _ = np.histogram(r, bins=edges)
            hist += h
        nfr += 1

    Lz = float(u.dimensions[2])
    shell_vol = np.pi * (edges[1:]**2 - edges[:-1]**2) * Lz     # A^3
    w_dens = (w_hist / max(nfr, 1)) / shell_vol
    p_dens = (p_hist / max(nfr, 1)) / shell_vol
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out, np.column_stack([rmid/10.0, w_dens, p_dens]),
               header="r_nm  water_density_A-3  phosphate_density_A-3",
               fmt="%.4f")
    print("[radial_density] %d frames -> %s" % (nfr, out))

if __name__ == "__main__":
    main()

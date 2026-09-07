#!/usr/bin/env python3
"""
base_geometry.py — per-nucleobase radial distance from the tube surface and base
stacking angle relative to the tube, reproducing Harvey Fig S9.

For each residue: radial distance = |base-ring centroid - tube axis| - tube_radius;
stacking angle = angle between the base-ring normal and the LOCAL RADIAL vector
(0 deg => base plane parallel to the sidewall = adsorbed/stacked).

Usage:
  base_geometry.py --tpr run/<sys>/prod.tpr --xtc run/<sys>/prod.xtc \
                   --out results/<sys>/base_geometry.csv [--begin_ps N]
Outputs per-residue mean radial distance (nm) and mean stacking angle (deg).
"""
import argparse
from pathlib import Path
import numpy as np
import MDAnalysis as mda

# ring atoms defining each base plane (purine 6+5 ring / pyrimidine 6 ring)
RING = {"DA":"N9 C8 N7 C5 C4", "DG":"N9 C8 N7 C5 C4",
        "DT":"N1 C2 N3 C4 C5 C6", "DC":"N1 C2 N3 C4 C5 C6",
        "RA":"N9 C8 N7 C5 C4", "RG":"N9 C8 N7 C5 C4",
        "RU":"N1 C2 N3 C4 C5 C6", "RC":"N1 C2 N3 C4 C5 C6"}

def plane_normal(pos):
    c = pos.mean(0); u, s, vt = np.linalg.svd(pos - c)
    return vt[2], c                    # smallest-variance dir = ring normal

def main():
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--tpr", required=True); ap.add_argument("--xtc", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--begin_ps", type=float, default=0)
    a = ap.parse_args()

    u = mda.Universe(a.tpr, a.xtc)
    cnt = u.select_atoms("resname CNT")
    meta = json.load(open(Path(a.tpr).resolve().parents[2] / "system" / "cnt_meta.json"))
    Rtube = meta["radius_A"]
    nucl = u.select_atoms("resname " + " ".join(RING))
    resids = list(dict.fromkeys(zip(nucl.resnames, nucl.resids, nucl.segids)))

    acc = {}                          # (resname,resid) -> [sum_radial, sum_angle, n]
    nfr = 0
    for ts in u.trajectory:
        if ts.time < a.begin_ps:
            continue
        axis = cnt.positions[:, :2].mean(0)
        for rn, rid, seg in resids:
            sel = nucl.select_atoms("resname %s and resid %d and segid %s and name %s"
                                    % (rn, rid, seg, RING[rn]))
            if len(sel) < 3:
                continue
            n, c = plane_normal(sel.positions)
            radial_vec = np.array([c[0]-axis[0], c[1]-axis[1], 0.0])
            rr = np.linalg.norm(radial_vec)
            dist = rr - Rtube
            ang = np.degrees(np.arccos(np.clip(abs(np.dot(n, radial_vec/max(rr,1e-6))), 0, 1)))
            # ang ~ 90 deg when base normal perpendicular to radial (plane faces wall)
            k = (rn, rid)
            s = acc.setdefault(k, [0.0, 0.0, 0])
            s[0] += dist; s[1] += ang; s[2] += 1
        nfr += 1

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        fh.write("resname,resid,mean_radial_dist_nm,mean_stacking_angle_deg,adsorbed\n")
        for (rn, rid), (sd, sa, nn) in sorted(acc.items(), key=lambda x: x[0][1]):
            d = sd/nn/10.0; ang = sa/nn
            adsorbed = int(d < 0.45)      # within ~4.5 A of wall = adsorbed (Harvey ~stacked)
            fh.write("%s,%d,%.3f,%.1f,%d\n" % (rn, rid, d, ang, adsorbed))
    print("[base_geometry] %d frames, %d residues -> %s" % (nfr, len(acc), out))

if __name__ == "__main__":
    main()

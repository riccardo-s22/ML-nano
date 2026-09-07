#!/usr/bin/env python3
"""Conceptual full-sensor starting structures (§12, Gate 5 = conceptual).

Builds GT15-junction20, GT15-junction22, and GT15-only sensors as full ssDNA
(tleap), then places each near the chosen SWCNT sidewall in >=20 orientations
(axial shift, rotation, radial offset). Rejects genuine nonbonded INTERMOLECULAR
clashes only (DNA vs CNT heavy atoms < threshold); does NOT treat normal covalent
geometry as a clash. Saves transforms + construction metadata.

Default status (§12.1): conceptual / docking-anchor-informed starting ensemble —
NOT energy-minimized or dynamically equilibrated. MD is disabled (run_md=false),
and full-oligo rigid docking is clash-dominated, so structures are builder-placed
near the surface and explicitly labeled.
"""
import os, sys, subprocess, tempfile
import numpy as np
from scipy.spatial.distance import cdist
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_common import ROOT, load_configs, write_json, utc_now

OUT = os.path.join(ROOT, "results", "structures")
SM = os.path.join(OUT, "starting_models")
os.makedirs(SM, exist_ok=True)
SEED = int(os.environ.get("PIPELINE_SEED", "20260625"))

SENSORS = {
    "GT15_junction20": "GTGTGTGTGTGTGTGTGTGTGTGTGTGTGT" + "CTGCCGAGTCCCATTGCTGT",
    "GT15_junction22": "GTGTGTGTGTGTGTGTGTGTGTGTGTGTGT" + "TCTGCCGAGTCCCATTGCTGTT",
    "GT15_only":       "GTGTGTGTGTGTGTGTGTGTGTGTGTGTGT",
}

def build_sensor_pdb(seq, out_pdb):
    bmap = {"A": "DA", "C": "DC", "G": "DG", "T": "DT"}
    res = []
    for i, b in enumerate(seq):
        nm = bmap[b]
        if i == 0: nm += "5"
        elif i == len(seq)-1: nm += "3"
        res.append(nm)
    with tempfile.NamedTemporaryFile("w", suffix=".in", delete=False) as tf:
        tf.write("source leaprc.DNA.OL15\n")
        tf.write("m = sequence { " + " ".join(res) + " }\n")
        tf.write(f"savepdb m {out_pdb}\nquit\n")
        leapin = tf.name
    subprocess.run(["tleap", "-f", leapin], capture_output=True, text=True)
    return os.path.exists(out_pdb)

def read_heavy(pdb):
    coords, names, elems, anchor_mask = [], [], [], []
    with open(pdb) as fh:
        for l in fh:
            if l.startswith(("ATOM", "HETATM")):
                el = l[76:78].strip() or l[12:16].strip()[0]
                if el == "H":
                    continue
                coords.append([float(l[30:38]), float(l[38:46]), float(l[46:54])])
                names.append(l[12:16].strip())
                resid = int(l[22:26])
                anchor_mask.append(resid <= 15)   # GT15 anchor = first 15 residues
    return np.array(coords), np.array(anchor_mask)

def read_cnt(pdb):
    c = []
    with open(pdb) as fh:
        for l in fh:
            if l.startswith("ATOM"):
                c.append([float(l[30:38]), float(l[38:46]), float(l[46:54])])
    return np.array(c)

def rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def main():
    cfg = load_configs()
    clash_thresh = 1.8  # Å, intermolecular nonbonded heavy-atom (§12.3)
    chirality = "7_5"   # representative experimentally relevant chirality
    cnt = read_cnt(os.path.join(ROOT, "results", "swcnt", f"{chirality}.pdb"))
    cnt_center = cnt.mean(axis=0)
    cnt_radius = np.sqrt(((cnt[:, :2] - cnt[:, :2].mean(0))**2).sum(1)).mean()
    rng = np.random.default_rng(SEED)

    manifest = []
    clash_rows = []
    for sensor, seq in SENSORS.items():
        pdb = os.path.join(SM, f"{sensor}_base.pdb")
        if not build_sensor_pdb(seq, pdb):
            continue
        coords0, anchor_mask = read_heavy(pdb)
        # orient sensor long axis -> z (PCA)
        cc = coords0 - coords0.mean(0)
        u, s, vt = np.linalg.svd(cc, full_matrices=False)
        long_axis = vt[0]
        # rotation aligning long_axis to z
        z = np.array([0, 0, 1.0])
        v = np.cross(long_axis, z); sval = np.linalg.norm(v); cval = np.dot(long_axis, z)
        if sval < 1e-6:
            Ralign = np.eye(3)
        else:
            vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            Ralign = np.eye(3) + vx + vx @ vx * ((1 - cval) / (sval**2))
        oriented = cc @ Ralign.T

        n_models = 0
        attempts = 0
        while n_models < 20 and attempts < 200:
            attempts += 1
            theta = rng.uniform(0, 2*np.pi)
            # axis offset must clear the ssDNA strand half-width so the near edge
            # (not the strand axis) sits near the surface
            radial_off = rng.uniform(10.0, 18.0)
            axial_shift = rng.uniform(-15, 15)
            tilt = rng.uniform(-0.15, 0.15)
            P = oriented @ rot_z(theta).T
            # tilt about x slightly
            ct_, st_ = np.cos(tilt), np.sin(tilt)
            Rt = np.array([[1, 0, 0], [0, ct_, -st_], [0, st_, ct_]])
            P = P @ Rt.T
            # place: anchor near +x sidewall
            P = P + np.array([cnt_center[0] + cnt_radius + radial_off, cnt_center[1],
                              cnt_center[2] + axial_shift])
            # clash: min intermolecular distance DNA-heavy vs CNT
            dmin = cdist(P, cnt).min()
            if dmin < clash_thresh:
                clash_rows.append({"sensor": sensor, "model": "rejected", "min_dist_A": round(float(dmin), 3)})
                continue
            n_models += 1
            mid = f"{sensor}_{chirality}_m{n_models:02d}"
            # write a combined PDB (CNT + sensor) starting model
            outp = os.path.join(SM, f"{mid}.pdb")
            with open(outp, "w") as fh:
                fh.write(f"REMARK Representative starting geometry derived from builder placement;\n")
                fh.write(f"REMARK GT15 anchor near ({chirality}) sidewall; NOT energy-minimized or\n")
                fh.write(f"REMARK dynamically equilibrated. Full-oligo docking was clash-dominated.\n")
                for i, c in enumerate(cnt, 1):
                    fh.write(f"ATOM  {i:5d}  C   CNT A{1:4d}    {c[0]:8.3f}{c[1]:8.3f}{c[2]:8.3f}  1.00  0.00           C\n")
                for j, c in enumerate(P, 1):
                    fh.write(f"ATOM  {len(cnt)+j:5d}  C   DNA B{2:4d}    {c[0]:8.3f}{c[1]:8.3f}{c[2]:8.3f}  1.00  0.00           C\n")
                fh.write("END\n")
            n_anchor_contact = int((cdist(P[anchor_mask], cnt).min(axis=1) < 4.5).sum())
            manifest.append({"model_id": mid, "sensor": sensor, "chirality": chirality,
                             "rotation_rad": round(theta, 4), "radial_offset_A": round(radial_off, 3),
                             "axial_shift_A": round(axial_shift, 3), "tilt_rad": round(tilt, 4),
                             "min_intermol_dist_A": round(float(dmin), 3),
                             "n_anchor_atoms_within_4.5A": n_anchor_contact,
                             "status": "conceptual_not_equilibrated"})
            clash_rows.append({"sensor": sensor, "model": mid, "min_dist_A": round(float(dmin), 3)})
        print(f"{sensor}: {n_models} starting models ({attempts} attempts)")

    def write_tsv(path, rows):
        if not rows: return
        cols = list(rows[0].keys())
        with open(path, "w") as fh:
            fh.write("\t".join(cols) + "\n")
            for r in rows:
                fh.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")
    write_tsv(os.path.join(OUT, "starting_model_manifest.tsv"), manifest)
    write_tsv(os.path.join(OUT, "clash_validation.tsv"), clash_rows)

    write_json(os.path.join(ROOT, "results", "gates", "gate5_structure.json"),
               {"gate": "Gate5_structural_exposure", "passed": False,
                "status": "structures retained as CONCEPTUAL (MD disabled; no validated "
                          "SWCNT carbon force field supplied; full-oligo docking clash-dominated)",
                "n_starting_models": len(manifest),
                "quantitative_exposure_claim_permitted": False,
                "label": "Representative starting geometry; not energy-minimized or "
                         "dynamically equilibrated.",
                "utc": utc_now()})
    print(f"Total starting models: {len(manifest)}; Gate 5 = conceptual (no exposure claim)")

if __name__ == "__main__":
    main()

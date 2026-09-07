"""Controlled Vina docking helpers (§11): sidewall-offset box, map reuse,
pose metrics (base vs phosphate contacts, end-exclusion), bootstrap.

Vina scores here are CONTROLLED COMPARATIVE steric/stacking scores on an
uncharged graphitic receptor — NOT binding free energies.
"""
import os, numpy as np

def tube_frame(pos):
    cx, cy = float(pos[:, 0].mean()), float(pos[:, 1].mean())
    cz = float(pos[:, 2].mean())
    rad = 2 * np.sqrt(((pos[:, :2] - pos[:, :2].mean(0))**2).sum(1)).mean() / 2
    zmin, zmax = float(pos[:, 2].min()), float(pos[:, 2].max())
    return {"cx": cx, "cy": cy, "cz": cz, "radius": float(rad),
            "zmin": zmin, "zmax": zmax, "axial": zmax - zmin}

def sidewall_box(pos, axial_window=16.0, radial_extent=14.0, tangential=16.0,
                 surface_offset=2.0):
    """Box whose INNER radial face sits at (or just outside) the near sidewall and
    extends OUTWARD, so the search volume excludes the tube interior and the far
    wall entirely. The ligand can approach the surface from outside but cannot be
    placed inside the carbon wall. Centred axially on the tube middle (excludes ends).

    near surface at x = cx + radius; box spans x in
    [cx+radius+surface_offset, cx+radius+surface_offset+radial_extent].
    """
    f = tube_frame(pos)
    inner = f["cx"] + f["radius"] + surface_offset
    center = [inner + radial_extent / 2.0, f["cy"], f["cz"]]
    box = [float(radial_extent), float(tangential), float(axial_window)]
    return [float(c) for c in center], box, f

class DockTimeout(Exception):
    pass

def _with_timeout(fn, seconds):
    import signal
    def handler(signum, frame):
        raise DockTimeout()
    old = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        return fn()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)

def dock_one(maps_prefix, ligand_pdbqt, out_pdbqt, seed, exhaustiveness=32,
             n_poses=20, cpu=1):
    from vina import Vina
    v = Vina(sf_name="vina", cpu=cpu, seed=seed, verbosity=0)
    v.load_maps(maps_prefix)
    v.set_ligand_from_file(ligand_pdbqt)
    v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)
    e = v.energies(n_poses=n_poses)
    if not os.path.exists(out_pdbqt):           # never overwrite raw (§6 rule 16)
        v.write_poses(out_pdbqt, n_poses=n_poses, overwrite=False)
    return e

def compute_and_save_maps(receptor_pdbqt, ligand_pdbqt, center, box, maps_prefix):
    """Compute maps once (ligand defines atom types) and save for reuse."""
    import glob
    from vina import Vina
    for f in glob.glob(maps_prefix + ".*.map") + glob.glob(maps_prefix + ".maps.fld"):
        try: os.remove(f)
        except OSError: pass
    v = Vina(sf_name="vina", cpu=1, seed=1, verbosity=0)
    v.set_receptor(receptor_pdbqt)
    v.set_ligand_from_file(ligand_pdbqt)
    v.compute_vina_maps(center=center, box_size=box, force_even_voxels=True)
    v.write_maps(maps_prefix)
    return maps_prefix

# ---- pose metrics ----
PHOS_NAMES = "name P OP1 OP2 OP3 O5' O3' O1P O2P"
BASE_NAMES = "name N1 N2 N3 N4 N6 N7 N9 C2 C4 C5 C6 C8 O2 O4 O6"

def pose_metrics(receptor_pos, pose_pdbqt, frame, cutoff=4.5, end_buffer=10.0):
    import MDAnalysis as mda
    from MDAnalysis.analysis import distances
    lig = mda.Universe(pose_pdbqt)
    metrics = []
    rec = receptor_pos
    # iterate poses (models) in the pose file
    for ts in lig.trajectory:
        dna = lig.atoms
        d = distances.distance_array(dna.positions, rec)
        mind = float(d.min())
        within = (d < cutoff).any(axis=1)
        nheavy = int(within.sum())
        phos = lig.select_atoms(PHOS_NAMES)
        base = lig.select_atoms(BASE_NAMES)
        nb = int((distances.distance_array(base.positions, rec) < cutoff).any(axis=1).sum()) if len(base) else 0
        nph = int((distances.distance_array(phos.positions, rec) < cutoff).any(axis=1).sum()) if len(phos) else 0
        zmin_d = float(dna.positions[:, 2].min() - frame["zmin"])
        zmax_d = float(frame["zmax"] - dna.positions[:, 2].max())
        closest_end = min(zmin_d, zmax_d)
        axial_span = float(dna.positions[:, 2].ptp())
        metrics.append({
            "min_DNA_CNT_distance_A": round(mind, 3),
            "n_heavy_within_cut": nheavy,
            "n_base_within_cut": nb,
            "n_phosphate_within_cut": nph,
            "closest_distance_to_tube_end_A": round(closest_end, 3),
            "axial_contact_span_A": round(axial_span, 3),
            "sidewall_pose_pass": bool(closest_end > end_buffer),
        })
    return metrics

def bootstrap_median(scores, n=10000, seed=0, ci=(2.5, 97.5)):
    scores = np.asarray(scores, float)
    if len(scores) == 0:
        return None, (None, None)
    rng = np.random.default_rng(seed)
    meds = [np.median(rng.choice(scores, size=len(scores), replace=True)) for _ in range(n)]
    return float(np.median(scores)), tuple(float(x) for x in np.percentile(meds, ci))

def bootstrap_median_diff(a, b, n=10000, seed=0, ci=(2.5, 97.5)):
    a = np.asarray(a, float); b = np.asarray(b, float)
    rng = np.random.default_rng(seed)
    diffs = [np.median(rng.choice(a, len(a), replace=True)) -
             np.median(rng.choice(b, len(b), replace=True)) for _ in range(n)]
    lo, hi = np.percentile(diffs, ci)
    return float(np.median(diffs)), float(lo), float(hi)

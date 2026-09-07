#!/usr/bin/env python3
"""Controlled Vina docking driver (§11) — benchmark + chirality stages.

Vina scores are CONTROLLED COMPARATIVE steric/stacking scores (uncharged
graphitic receptor) — NOT binding free energies.

Scale is configurable. Because a 14-mer rigid ssDNA is an atypically large Vina
ligand (~143 s/dock at exhaustiveness 1), the default scale here is REDUCED and
its results are labeled EXPLORATORY (below the spec's Gate-3 minimums of
>=10 conformers x >=10 seeds x exh>=32). The full-scale command is documented in
reports/. Each dock is cpu=1 (deterministic); independent docks run in parallel.

Usage: run_docking.py --stage benchmark|chirality [--conformers N --seeds N --exh E]
"""
import os, sys, glob, json, argparse, itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_common import ROOT, load_configs, write_json, utc_now
import lib_dock as ld

DOCK = os.path.join(ROOT, "results", "docking")

def receptor_pos(pdbqt):
    import MDAnalysis as mda
    return mda.Universe(pdbqt).atoms.positions

def conformer_list(anchor, n):
    cs = sorted(glob.glob(os.path.join(ROOT, "results", "anchor_conformers", anchor, "conformer_*.pdbqt")))
    return cs[:n]

def _dock_chunk(args):
    """Worker: dock a chunk of jobs, loading each receptor's maps ONCE and reusing
    the Vina object across all that receptor's ligands (eliminates per-dock map I/O)."""
    chunk, exh, n_poses = args
    from vina import Vina
    out = []
    vmap = {}
    for (maps_prefix, lig, out_pdbqt, seed) in chunk:
        try:
            v = vmap.get(maps_prefix)
            if v is None:
                v = Vina(sf_name="vina", cpu=1, seed=seed, verbosity=0)
                v.load_maps(maps_prefix)
                vmap[maps_prefix] = v
            v.set_ligand_from_file(lig)
            ld._with_timeout(lambda: v.dock(exhaustiveness=exh, n_poses=n_poses), 90)
            e = v.energies(n_poses=n_poses)
            if not os.path.exists(out_pdbqt):
                v.write_poses(out_pdbqt, n_poses=n_poses, overwrite=False)
            out.append((out_pdbqt, [float(x) for x in e[:, 0]], None))
        except Exception as ex:
            out.append((out_pdbqt, None, f"{type(ex).__name__}: {ex}"))
    return out

def run_stage(stage, n_conf, n_seeds, exh, n_poses, max_workers):
    cfg = load_configs()
    dcfg = cfg["docking"]
    seeds = [cfg["config"]["pipeline_seed"] + i for i in range(n_seeds)]
    # Compact dinucleotide repeat units (GTd/CTd/ATd): full-oligo rigid docking on
    # the curved sidewall is clash-dominated (revision A1, empirically confirmed),
    # so the controlled steric/stacking screen uses the base-pair repeat units that
    # constitute the anchors — probing base-surface stacking, the physical basis of
    # the anchor preference.
    REP = "GTd"
    if stage == "benchmark":
        anchors = ["GTd", "CTd", "ATd"]
        receptors = {"7_5": [7, 5]}
        rawdir = os.path.join(DOCK, "benchmark", "raw")
        outdir = os.path.join(DOCK, "benchmark")
    else:
        anchors = ["GTd"]
        receptors = {f"{n}_{m}": [n, m] for n, m in cfg["config"]["chiralities"]}
        rawdir = os.path.join(DOCK, "chirality", "raw")
        outdir = os.path.join(DOCK, "chirality")
    os.makedirs(rawdir, exist_ok=True)

    # build job list; compute maps once per receptor
    jobs = []
    frames = {}
    for rkey, (n, m) in receptors.items():
        rec_pdbqt = os.path.join(ROOT, "results", "swcnt", f"{rkey}.pdbqt")
        pos = receptor_pos(rec_pdbqt)
        center, box, frame = ld.sidewall_box(pos)
        frames[rkey] = (frame, pos)
        maps_prefix = os.path.join("/tmp", f"maps_{rkey}")
        # representative dinucleotide conformer defines the atom types for the maps
        rep_lig = conformer_list(REP, 1)[0]
        ld.compute_and_save_maps(rec_pdbqt, rep_lig, center, box, maps_prefix)
        for anchor in anchors:
            for ci, lig in enumerate(conformer_list(anchor, n_conf), 1):
                for seed in seeds:
                    out = os.path.join(rawdir, f"{rkey}__{anchor}__c{ci:02d}__s{seed}.pdbqt")
                    jobs.append((rkey, anchor, ci, seed, maps_prefix, lig, out))

    print(f"[{stage}] {len(jobs)} docks (conf={n_conf} seeds={n_seeds} exh={exh}) on {max_workers} workers")

    # partition jobs into per-worker chunks (each worker loads maps once, reuses Vina)
    meta = {jb[2]: jb for jb in [(rkey, anchor, ci, seed, mp, lig, out)
                                 for (rkey, anchor, ci, seed, mp, lig, out) in jobs]}
    meta = {out: (rkey, anchor, ci, seed) for (rkey, anchor, ci, seed, mp, lig, out) in jobs}
    chunks = [[] for _ in range(max_workers)]
    for i, (rkey, anchor, ci, seed, mp, lig, out) in enumerate(jobs):
        chunks[i % max_workers].append((mp, lig, out, seed))

    rows = []
    done = 0
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_dock_chunk, (ch, exh, n_poses)) for ch in chunks if ch]
        for f in as_completed(futs):
            for out_pdbqt, scores, err in f.result():
                done += 1
                if err or scores is None:
                    continue
                rkey, anchor, ci, seed = meta[out_pdbqt]
                frame, pos = frames[rkey]
                try:
                    pm = ld.pose_metrics(pos, out_pdbqt, frame)
                except Exception:
                    pm = [{} for _ in scores]
                for prank, (sc, met) in enumerate(zip(scores, pm), 1):
                    row = {"stage": stage, "chirality": rkey, "anchor": anchor,
                           "conformer_id": ci, "seed": seed, "cpu": 1, "pose_rank": prank,
                           "vina_score": sc, "raw_pose_path": os.path.relpath(out_pdbqt, ROOT)}
                    row.update(met)
                    rows.append(row)
            print(f"  chunk done; {done}/{len(jobs)} docks processed")

    # write pose metrics
    cols = sorted({k for r in rows for k in r})
    pm_tsv = os.path.join(outdir, "pose_metrics.tsv")
    with open(pm_tsv, "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")
    print(f"[{stage}] wrote {len(rows)} pose rows -> {pm_tsv}")
    return rows, outdir

def summarize(rows, group_key, outdir, scale_meta):
    """Per-group median + bootstrap CI over ACCEPTED SIDEWALL poses."""
    groups = {}
    for r in rows:
        if r.get("sidewall_pose_pass") in (True, "True"):
            groups.setdefault(r[group_key], []).append(r["vina_score"])
    summary = []
    for g, scores in sorted(groups.items()):
        med, (lo, hi) = ld.bootstrap_median(scores, n=5000, seed=0)
        n_all = sum(1 for r in rows if r[group_key] == g)
        n_side = len(scores)
        summary.append({group_key: g, "n_accepted_sidewall_poses": n_side,
                        "n_total_poses": n_all,
                        "frac_sidewall": round(n_side / n_all, 3) if n_all else 0,
                        "median_score": round(med, 3) if med else None,
                        "ci_lo": round(lo, 3) if lo else None,
                        "ci_hi": round(hi, 3) if hi else None})
    cols = list(summary[0].keys()) if summary else []
    with open(os.path.join(outdir, "summary.tsv"), "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for s in summary:
            fh.write("\t".join(str(s.get(c, "")) for c in cols) + "\n")
    write_json(os.path.join(outdir, "summary.json"),
               {"summary": summary, "scale": scale_meta, "utc": utc_now()})
    return summary

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["benchmark", "chirality"], required=True)
    ap.add_argument("--conformers", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--exh", type=int, default=8)
    ap.add_argument("--poses", type=int, default=10)
    ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args()
    scale_meta = {"conformers": a.conformers, "seeds": a.seeds, "exhaustiveness": a.exh,
                  "poses": a.poses, "meets_gate3_minimums": (a.conformers >= 10 and a.seeds >= 10 and a.exh >= 32),
                  "label": "EXPLORATORY (reduced scale)" if not (a.conformers >= 10 and a.seeds >= 10 and a.exh >= 32) else "full"}
    rows, outdir = run_stage(a.stage, a.conformers, a.seeds, a.exh, a.poses, a.workers)
    gk = "anchor" if a.stage == "benchmark" else "chirality"
    summ = summarize(rows, gk, outdir, scale_meta)
    print(f"[{a.stage}] summary:")
    for s in summ:
        print(f"  {s[gk]}: median={s['median_score']} CI[{s['ci_lo']},{s['ci_hi']}] "
              f"n_sidewall={s['n_accepted_sidewall_poses']} frac={s['frac_sidewall']}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Benchmark gate (Gate 3, §11.3) — correct bootstrap median-difference logic.

Score component passes ONLY when:
  median(GT) < median(CT) AND median(GT) < median(AT) AND
  upper_95CI(median(GT)-median(CT)) < 0 AND upper_95CI(median(GT)-median(AT)) < 0
(more-negative Vina scores are more favorable). Plus predeclared stability and
sidewall-fraction criteria. A failed gate does NOT delete the chirality screen;
it stamps every chirality conclusion EXPLORATORY.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_common import ROOT, load_configs, write_json, utc_now
import lib_dock as ld

BM = os.path.join(ROOT, "results", "docking", "benchmark")
GATES = os.path.join(ROOT, "results", "gates")

def load_rows(path):
    rows = []
    with open(path) as fh:
        cols = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            vals = line.rstrip("\n").split("\t")
            rows.append(dict(zip(cols, vals)))
    return rows

def accepted_scores(rows, anchor):
    return [float(r["vina_score"]) for r in rows
            if r["anchor"] == anchor and r.get("sidewall_pose_pass") == "True"]

def between_conformer_std(rows, anchor):
    per = {}
    for r in rows:
        if r["anchor"] == anchor and r.get("sidewall_pose_pass") == "True":
            per.setdefault(r["conformer_id"], []).append(float(r["vina_score"]))
    meds = [np.median(v) for v in per.values() if v]
    return float(np.std(meds)) if len(meds) > 1 else 0.0

def main():
    cfg = load_configs()
    th = cfg["docking"]["gate3_thresholds"]
    rows = load_rows(os.path.join(BM, "pose_metrics.tsv"))
    gt = accepted_scores(rows, "GTd")
    ct = accepted_scores(rows, "CTd")
    at = accepted_scores(rows, "ATd")

    med_gt, ci_gt = ld.bootstrap_median(gt, n=10000, seed=0)
    med_ct, ci_ct = ld.bootstrap_median(ct, n=10000, seed=0)
    med_at, ci_at = ld.bootstrap_median(at, n=10000, seed=0)
    d_gt_ct, lo_ct, hi_ct = ld.bootstrap_median_diff(gt, ct, n=10000, seed=0)
    d_gt_at, lo_at, hi_at = ld.bootstrap_median_diff(gt, at, n=10000, seed=0)

    score_pass = (med_gt < med_ct and med_gt < med_at and hi_ct < 0 and hi_at < 0)
    bcs = between_conformer_std(rows, "GTd")
    stable = bcs <= th["between_conformer_std_max"]
    n_total = sum(1 for r in rows if r["anchor"] == "GTd")
    n_side = len(gt)
    sidewall_frac = (n_side / n_total) if n_total else 0
    sidewall_ok = sidewall_frac >= th["sidewall_fraction_min"]

    passed = bool(score_pass and stable and sidewall_ok)
    status = {
        "gate": "Gate3_docking_benchmark",
        "passed": passed,
        "ligand_basis": "dinucleotide repeat units (GTd/CTd/ATd) — base-surface stacking; "
                        "full-oligo rigid docking is clash-dominated (revision A1)",
        "median_GTd": round(med_gt, 3), "ci_GTd": [round(x, 3) for x in ci_gt],
        "median_CTd": round(med_ct, 3), "ci_CTd": [round(x, 3) for x in ci_ct],
        "median_ATd": round(med_at, 3), "ci_ATd": [round(x, 3) for x in ci_at],
        "diff_GT_CT_median": round(d_gt_ct, 3), "diff_GT_CT_ci_upper": round(hi_ct, 3),
        "diff_GT_AT_median": round(d_gt_at, 3), "diff_GT_AT_ci_upper": round(hi_at, 3),
        "score_component_pass": bool(score_pass),
        "between_conformer_std": round(bcs, 3), "stable_across_conformers": bool(stable),
        "sidewall_fraction": round(sidewall_frac, 3), "sidewall_ok": bool(sidewall_ok),
        "n_accepted_GTd": n_side, "n_accepted_CTd": len(ct), "n_accepted_ATd": len(at),
        "gate_logic": "median(GT)<median(CT,AT) AND upper95CI(GT-CT)<0 AND upper95CI(GT-AT)<0",
        "consequence_if_failed": "chirality screen runs but ALL chirality conclusions EXPLORATORY",
        "utc": utc_now(),
    }
    write_json(os.path.join(BM, "gate_status.json"), status)
    write_json(os.path.join(GATES, "gate3_benchmark.json"), status)
    print(f"Gate 3 benchmark passed={passed}")
    print(f"  median GTd={med_gt:.2f} CTd={med_ct:.2f} ATd={med_at:.2f}")
    print(f"  GT-CT diff median={d_gt_ct:.2f} ci_upper={hi_ct:.2f}; "
          f"GT-AT diff median={d_gt_at:.2f} ci_upper={hi_at:.2f}")
    print(f"  score_pass={score_pass} stable={stable} sidewall_frac={sidewall_frac:.2f}")

if __name__ == "__main__":
    main()

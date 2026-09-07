#!/usr/bin/env python3
"""Chirality docking analysis (§11.6, Gate 4). Reads chirality pose_metrics,
computes per-chirality median + bootstrap CI over accepted sidewall poses,
cluster occupancy proxy, contact metrics, and consumes the benchmark gate to
stamp ranking_status. Gate 4 passes only if a unique winner is supported and the
benchmark passed; otherwise EXPLORATORY or 'no unique winner'."""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_common import ROOT, write_json, utc_now
import lib_dock as ld

CH = os.path.join(ROOT, "results", "docking", "chirality")
GATES = os.path.join(ROOT, "results", "gates")

def load_rows(path):
    with open(path) as fh:
        cols = fh.readline().rstrip("\n").split("\t")
        return [dict(zip(cols, l.rstrip("\n").split("\t"))) for l in fh]

def main():
    rows = load_rows(os.path.join(CH, "pose_metrics.tsv"))
    bench = {}
    bp = os.path.join(GATES, "gate3_benchmark.json")
    if os.path.exists(bp):
        bench = json.load(open(bp))
    benchmark_passed = bool(bench.get("passed", False))

    by = {}
    for r in rows:
        if r.get("sidewall_pose_pass") == "True":
            by.setdefault(r["chirality"], []).append(r)
    summary = []
    for chir, rs in sorted(by.items()):
        scores = [float(r["vina_score"]) for r in rs]
        med, (lo, hi) = ld.bootstrap_median(scores, n=10000, seed=0)
        nbase = np.mean([float(r.get("n_base_within_cut", 0) or 0) for r in rs])
        nphos = np.mean([float(r.get("n_phosphate_within_cut", 0) or 0) for r in rs])
        # between-conformer std
        per = {}
        for r in rs:
            per.setdefault(r["conformer_id"], []).append(float(r["vina_score"]))
        bcs = float(np.std([np.median(v) for v in per.values()])) if len(per) > 1 else 0.0
        summary.append({"chirality": chir, "n_accepted": len(scores),
                        "median_score": round(med, 3), "ci_lo": round(lo, 3), "ci_hi": round(hi, 3),
                        "mean_base_contacts": round(nbase, 2), "mean_phosphate_contacts": round(nphos, 2),
                        "between_conformer_std": round(bcs, 3),
                        "benchmark_passed": benchmark_passed})
    summary.sort(key=lambda s: s["median_score"])  # most favorable first
    for i, s in enumerate(summary):
        s["rank"] = i + 1

    # unique winner test: best CI does not overlap 2nd-best CI
    unique = False
    if len(summary) >= 2:
        unique = summary[0]["ci_hi"] < summary[1]["ci_lo"]
    ranking_status = ("conditional" if (benchmark_passed and unique)
                      else "no unique winner" if not unique else "exploratory")
    if not benchmark_passed:
        ranking_status = "EXPLORATORY (benchmark did not pass)"

    cols = list(summary[0].keys())
    with open(os.path.join(CH, "chirality_ranking.tsv"), "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for s in summary:
            fh.write("\t".join(str(s[c]) for c in cols) + "\n")

    gate4 = {"gate": "Gate4_chirality_robustness",
             "passed": bool(benchmark_passed and unique),
             "benchmark_passed": benchmark_passed,
             "unique_winner_by_CI": unique,
             "ranking_status": ranking_status,
             "top_chirality": summary[0]["chirality"] if summary else None,
             "note": "Vina scores are controlled comparative base-stacking scores, NOT binding "
                     "energies; best-ranked chirality is NOT claimed to be the best optical sensor.",
             "ranking": summary, "utc": utc_now()}
    write_json(os.path.join(GATES, "gate4_chirality.json"), gate4)
    write_json(os.path.join(CH, "bootstrap_intervals.json"), {"ranking": summary})
    print(f"Gate 4 passed={gate4['passed']} status={ranking_status}")
    for s in summary:
        print(f"  {s['chirality']}: median={s['median_score']} CI[{s['ci_lo']},{s['ci_hi']}] n={s['n_accepted']}")

if __name__ == "__main__":
    main()

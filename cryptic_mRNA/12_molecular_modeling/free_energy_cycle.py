#!/usr/bin/env python3
"""
free_energy_cycle.py — Harvey Suppl. Fig S11 thermodynamic cycle, adapted to the
STMN2-CE DNA:RNA hybrid.

Two reference states for hybridisation ON the nanotube:
  Case A: capture strand pre-adsorbed on tube, target arrives from SOLUTION.
          dG_A = dG_hyb(solution)  +  dG_ads(duplex)  -  dG_ads(capture_ss)
  Case B: BOTH capture and target pre-adsorbed on the tube.
          dG_B = dG_hyb(solution)  +  dG_ads(duplex)  -  dG_ads(capture_ss) - dG_ads(target_ss)

dG_ads(ss) = sum of per-base adsorption free energies (config, Jung 2010 / ref 41
order G>A>T~U>C). dG_ads(duplex) is small: MD shows only 1-2 terminal bases of the
duplex contact the wall (Harvey Fig S9), so few bases contribute.

Harvey (17-mer DNA-DNA, Jung params): Case A ~= -135 kcal/mol, Case B ~= +9 kcal/mol.
Experiment = Case A. Adapted here with the STMN2-CE hybrid dG (MELTING).

Usage:
  free_energy_cycle.py [--hybrid_dG kcal] [--duplex_contact_bases 2] \
                       --out results/free_energy_cycle.json
"""
import argparse, json
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]
CFG  = yaml.safe_load(open(ROOT / "config" / "md_config.yaml"))

def ads_sum(seq, table):
    return sum(table.get(b, table.get("T", -5.0)) for b in seq)

def main():
    fe = CFG["free_energy_cycle"]; con = CFG["construct"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--hybrid_dG", type=float, default=fe["fallback_hybrid_dG_kcalmol"],
                    help="solution-phase capture:target hybridisation dG (kcal/mol); "
                         "use the MELTING value from the STMN2 thermo pipeline")
    ap.add_argument("--duplex_contact_bases", type=int, default=2,
                    help="bases of the formed duplex that stay adsorbed (Harvey Fig S9: 1-2)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    tbl = {k: v for k, v in fe["adsorption_dG_per_nt"].items() if isinstance(v, (int, float))}
    capture = con["capture"]
    target  = con["target_RNA"]

    dG_ads_capture = ads_sum(capture, tbl)                 # full capture strand adsorbed
    dG_ads_target  = ads_sum(target,  tbl)
    # duplex barely touches the wall -> only a few bases' worth of adsorption
    per_base_mean  = (dG_ads_capture / len(capture))
    dG_ads_duplex  = per_base_mean * a.duplex_contact_bases

    dG_A = a.hybrid_dG + dG_ads_duplex - dG_ads_capture
    dG_B = a.hybrid_dG + dG_ads_duplex - dG_ads_capture - dG_ads_target

    res = dict(
        hybrid_dG_solution_kcalmol=a.hybrid_dG,
        dG_ads_capture_kcalmol=round(dG_ads_capture, 1),
        dG_ads_target_kcalmol=round(dG_ads_target, 1),
        dG_ads_duplex_kcalmol=round(dG_ads_duplex, 1),
        duplex_contact_bases=a.duplex_contact_bases,
        caseA_dG_kcalmol=round(dG_A, 1),
        caseB_dG_kcalmol=round(dG_B, 1),
        relevant_case="A (target introduced from solution, as in the assay)",
        verdict=("hybridisation favoured on the tube (Case A < 0)"
                 if dG_A < 0 else "hybridisation NOT favoured (Case A >= 0)"),
        note=("Adapted from Harvey Fig S11. Replace adsorption_dG_per_nt with the "
              "cited Jung 2010 values and --hybrid_dG with the MELTING DNA:RNA "
              "hybrid dG for a quantitative estimate. Signs follow the paper: "
              "Case A strongly negative, Case B near zero/positive."))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()

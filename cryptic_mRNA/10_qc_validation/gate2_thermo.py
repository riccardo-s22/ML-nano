#!/usr/bin/env python3
"""Gate 2 (thermodynamic model validity) status, from hybridization outputs."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_common import ROOT, write_json, utc_now

HYB = os.path.join(ROOT, "results", "hybridization")
GATES = os.path.join(ROOT, "results", "gates")

def load_tsv(path):
    if not os.path.exists(path): return []
    with open(path) as fh:
        cols = fh.readline().rstrip("\n").split("\t")
        return [dict(zip(cols, l.rstrip("\n").split("\t"))) for l in fh]

def main():
    perfect = load_tsv(os.path.join(HYB, "melting_perfect_results.tsv"))
    acc = load_tsv(os.path.join(HYB, "nupack_accessibility_short.tsv"))
    have_perfect = len(perfect) >= 2
    have_nupack_rna = len(acc) >= 1
    j = {r["sensor"]: r for r in perfect}
    passed = have_perfect and have_nupack_rna
    gate2 = {
        "gate": "Gate2_thermodynamic_model_validity", "passed": bool(passed),
        "perfect_pathological_engine": "MELTING5 Sugimoto1995 DNA/RNA hybrid",
        "perfect_duplex_crosscheck": "Biopython R_DNA_NN1 (same set); side-by-side, not averaged",
        "salt_labeling": "Tm salt-corrected; dH/dS/dG reference-NN, NOT salt-conditioned",
        "longest_WC_run": "DESCRIPTOR only",
        "mixed_material_hybrid": "NUPACK 4.1 single-material analysis; true DNA/RNA hybrid params UNAVAILABLE",
        "nupack_legitimate_use": "RNA-only accessibility primary; competition EXPLORATORY single-material bracket",
        "junction20_Tm_C": float(j.get("junction20", {}).get("Tm_C_salt_corrected", "nan") or "nan"),
        "junction22_Tm_C": float(j.get("junction22", {}).get("Tm_C_salt_corrected", "nan") or "nan"),
        "utc": utc_now(),
    }
    write_json(os.path.join(GATES, "gate2_thermo.json"), gate2)
    print("Gate 2 passed =", passed)

if __name__ == "__main__":
    main()

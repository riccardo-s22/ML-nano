#!/usr/bin/env python3
"""Timestamped archive (§21): code, config, provenance, result tables, figures,
reports, logs + manifest + README listing completed/failed/skipped stages."""
import os, sys, json, shutil, glob, hashlib
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_common import ROOT

def gate(name):
    p = os.path.join(ROOT, "results", "gates", f"{name}.json")
    return json.load(open(p)) if os.path.exists(p) else {}

def main():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    arc = os.path.join(ROOT, f"STMN2_SWCNT_production_analysis_{stamp}")
    os.makedirs(arc, exist_ok=True)
    for sub in ["code", "config", "provenance", "results_tables", "figures", "reports", "logs", "tests"]:
        os.makedirs(os.path.join(arc, sub), exist_ok=True)
    # copy
    for f in glob.glob(os.path.join(ROOT, "scripts", "**", "*.py"), recursive=True):
        rel = os.path.relpath(f, os.path.join(ROOT, "scripts"))
        dst = os.path.join(arc, "code", rel); os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(f, dst)
    shutil.copy(os.path.join(ROOT, "workflow", "Snakefile"), os.path.join(arc, "code"))
    for f in glob.glob(os.path.join(ROOT, "config", "*")):
        shutil.copy(f, os.path.join(arc, "config"))
    for f in glob.glob(os.path.join(ROOT, "provenance", "*")):
        if os.path.isfile(f): shutil.copy(f, os.path.join(arc, "provenance"))
    for f in glob.glob(os.path.join(ROOT, "results", "**", "*.tsv"), recursive=True):
        rel = os.path.relpath(f, os.path.join(ROOT, "results"))
        dst = os.path.join(arc, "results_tables", rel); os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(f, dst)
    for f in glob.glob(os.path.join(ROOT, "results", "gates", "*.json")):
        shutil.copy(f, os.path.join(arc, "results_tables"))
    for f in glob.glob(os.path.join(ROOT, "figures", "*")):
        shutil.copy(f, os.path.join(arc, "figures"))
    for f in glob.glob(os.path.join(ROOT, "reports", "*")):
        shutil.copy(f, os.path.join(arc, "reports"))
    for f in glob.glob(os.path.join(ROOT, "logs", "*")):
        if os.path.isfile(f): shutil.copy(f, os.path.join(arc, "logs"))

    gates = {g: gate(g) for g in ["gate0_smoke", "gate1_sequence", "gate2_thermo",
             "swcnt_geometry", "gate3_benchmark", "gate4_chirality", "gate5_structure", "gate6_repro"]}
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "gates": {g: d.get("passed") for g, d in gates.items()},
                "n_figures": len(glob.glob(os.path.join(arc, "figures", "*.svg"))),
                "raw_docking_poses_referenced_by_path": "results/docking/*/raw/ (not duplicated)"}
    json.dump(manifest, open(os.path.join(arc, "manifest.json"), "w"), indent=2)

    with open(os.path.join(arc, "README.md"), "w") as fh:
        fh.write(f"# STMN2 RNA-SWCNT production analysis archive ({stamp})\n\n")
        fh.write("## Stage status\n")
        fh.write("- COMPLETED: sequence verification (Gate 1), hybridization perfect-duplex + "
                 "RNA accessibility (Gate 2), SWCNT construction, anchor conformers, controlled "
                 "dinucleotide docking benchmark + chirality screen, conceptual structures, figures, reports.\n")
        fh.write("- NOT RUN / LIMITED: NUPACK true DNA/RNA hybrid analysis (single-material only -> "
                 "competition is exploratory bracket); full-oligo Vina docking (clash-dominated -> "
                 "dinucleotide base-stacking screen used); MD (disabled, no carbon force field); "
                 "experimental NIR integration (no schema-conforming input).\n\n")
        fh.write("## Gate outcomes\n")
        for g, d in gates.items():
            fh.write(f"- {g}: passed={d.get('passed')}\n")
        fh.write("\n## Rerun\n```\nsnakemake -s workflow/Snakefile --cores 8 --use-conda "
                 "--rerun-incomplete --printshellcmds --config run_md=false\n```\n")
        fh.write("\nLarge raw docking poses are referenced by path, not duplicated.\n")
    print(f"Archive written: {arc}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Output validation (§16, Gate 6 component). Verifies required outputs/gates
exist, figures have SVG+PNG, no Vina figure/report says 'binding energy', no
conceptual structure implies equilibration, experimental stage not fabricated."""
import os, sys, glob, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_common import ROOT, write_json, utc_now

def main():
    checks = []
    def chk(name, cond, detail=""):
        checks.append({"check": name, "passed": bool(cond), "detail": detail})

    # gate files
    for g in ["gate0_smoke", "gate1_sequence", "gate2_thermo", "swcnt_geometry",
              "gate3_benchmark", "gate4_chirality", "gate5_structure"]:
        chk(f"gate:{g}", os.path.exists(os.path.join(ROOT, "results", "gates", f"{g}.json")))

    # required result tables
    for f in ["results/sequence_validation/validated_sequences.tsv",
              "results/hybridization/melting_perfect_results.tsv",
              "results/hybridization/nupack_accessibility_short.tsv",
              "results/swcnt/geometry_validation.tsv",
              "results/structures/starting_model_manifest.tsv"]:
        chk(f"output:{os.path.basename(f)}", os.path.exists(os.path.join(ROOT, f)))

    # figures: svg + png pairs
    svgs = glob.glob(os.path.join(ROOT, "figures", "*.svg"))
    for s in svgs:
        png = s[:-4] + ".png"
        chk(f"figure_pair:{os.path.basename(s)}", os.path.exists(png))
    chk("at_least_4_figures", len(svgs) >= 4, f"{len(svgs)} svg figures")

    # no 'binding energy' next to Vina in reports/figures text
    bad = []
    for rep in glob.glob(os.path.join(ROOT, "reports", "*.md")):
        txt = open(rep, errors="ignore").read().lower()
        if "binding energy" in txt and "not" not in txt[max(0, txt.find("binding energy")-40):txt.find("binding energy")]:
            # allow 'NOT binding free energies' disclaimers
            if "vina" in txt and "not binding" not in txt:
                bad.append(os.path.basename(rep))
    chk("no_binding_energy_mislabel", len(bad) == 0, str(bad))

    # conceptual structures labeled, no equilibration claim
    g5 = os.path.join(ROOT, "results", "gates", "gate5_structure.json")
    if os.path.exists(g5):
        d = json.load(open(g5))
        chk("structures_conceptual_no_exposure_claim",
            d.get("quantitative_exposure_claim_permitted") is False)

    # experimental stage not fabricated
    exp = os.path.join(ROOT, "results", "experimental")
    fabricated = any(f.endswith(".tsv") for f in os.listdir(exp)) if os.path.isdir(exp) else False
    chk("experimental_not_fabricated", not fabricated,
        "no experimental tables (stage not run: no schema-conforming input)")

    passed = all(c["passed"] for c in checks)
    out = {"gate": "Gate6_output_validation", "passed": bool(passed),
           "n_checks": len(checks), "n_failed": sum(1 for c in checks if not c["passed"]),
           "checks": checks, "utc": utc_now()}
    write_json(os.path.join(ROOT, "results", "gates", "gate6_repro.json"), out)
    print(f"Output validation: {sum(c['passed'] for c in checks)}/{len(checks)} checks passed; overall={passed}")
    for c in checks:
        if not c["passed"]:
            print("  FAIL:", c["check"], c["detail"])

if __name__ == "__main__":
    main()

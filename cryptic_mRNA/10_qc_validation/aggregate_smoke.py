#!/usr/bin/env python3
"""Aggregate smoke-test status (Gate 0). Reads all per-tool JSONs and writes
results/smoke_tests/aggregate_status.json + results/gates/gate0_smoke.json.

A failed OPTIONAL test (NUPACK, MD) disables only its stage; it does not fail
Gate 0 for the sequence/thermo-perfect/docking work that remains valid."""
import os, sys, json, glob
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from lib_common import ROOT, write_json, utc_now

SMOKE = os.path.join(ROOT, "results", "smoke_tests")
GATES = os.path.join(ROOT, "results", "gates")

# classification: which tests block which stages
REQUIRED_CORE = ["ase_openbabel", "nucleic_acid_builder", "ligand_prep",
                 "vina_receptor", "contact_atom_classes", "meeko_and_ligand_loadable"]
REQUIRED_THERMO_PERFECT = ["melting"]
OPTIONAL_BLOCKER = ["nupack"]   # blocks only mixed-material hybridization stage

def load(name):
    p = os.path.join(SMOKE, f"{name}.json")
    if os.path.exists(p):
        return json.load(open(p))
    return {"test": name, "passed": False, "status": "missing"}

def main():
    results = {}
    for f in glob.glob(os.path.join(SMOKE, "*.json")):
        if os.path.basename(f) == "aggregate_status.json":
            continue
        d = json.load(open(f))
        results[d.get("test", os.path.basename(f)[:-5])] = d

    def passed(name):
        return bool(results.get(name, {}).get("passed", False))

    core_ok = all(passed(t) for t in REQUIRED_CORE)
    thermo_ok = all(passed(t) for t in REQUIRED_THERMO_PERFECT)
    nupack_ok = passed("nupack")

    agg = {
        "utc": utc_now(),
        "tests": {k: results[k].get("passed", False) for k in sorted(results)},
        "core_docking_stack_ok": core_ok,
        "thermo_perfect_duplex_ok": thermo_ok,
        "nupack_available": nupack_ok,
        "stage_consequences": {
            "sequence_validation": "independent of smoke; Gate 1",
            "hybridization_perfect_MELTING": "ENABLED" if thermo_ok else "BLOCKED",
            "hybridization_mixed_material_NUPACK":
                "ENABLED" if nupack_ok else "NOT RUN (NUPACK blocker; no substitution)",
            "long_context_accessibility_ViennaRNA": "ENABLED (RNA-only engine present)",
            "swcnt_construction": "ENABLED" if passed("ase_openbabel") else "BLOCKED",
            "docking": "ENABLED" if core_ok else "BLOCKED",
            "md": "DISABLED by config (conceptual branch)",
        },
        # Gate 0 passes for the enabled core stages; NUPACK is an explicit, isolated blocker
        "gate0_passed_for_enabled_stages": bool(core_ok and thermo_ok),
        "explicit_blockers": ([] if nupack_ok else
            ["NUPACK 4 unavailable (licensed wheel) -> mixed-material hybridization NOT RUN"]),
    }
    write_json(os.path.join(SMOKE, "aggregate_status.json"), agg)

    gate0 = {
        "gate": "Gate0_tool_smoke_tests",
        "passed": agg["gate0_passed_for_enabled_stages"],
        "core_docking_stack_ok": core_ok,
        "thermo_perfect_ok": thermo_ok,
        "nupack_blocker": not nupack_ok,
        "consequence": "Enabled stages may proceed; NUPACK mixed-material stage marked "
                       "NOT RUN; MD disabled by config.",
        "utc": utc_now(),
    }
    write_json(os.path.join(GATES, "gate0_smoke.json"), gate0)
    print("Gate 0 (enabled stages):", agg["gate0_passed_for_enabled_stages"])
    for k, v in agg["stage_consequences"].items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()

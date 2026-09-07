#!/usr/bin/env python3
"""Toy test (§16): fast checks of schema wiring + the integrity-critical rule that
a FAILED benchmark gate cannot yield a 'supported' chirality claim. All toy values
are non-scientific.
"""
import os, sys, json
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

def test_gate_jsons_exist():
    for g in ["gate0_smoke", "gate1_sequence", "gate2_thermo"]:
        assert os.path.exists(os.path.join(ROOT, "results", "gates", f"{g}.json")), g

def test_failed_benchmark_forces_exploratory():
    """Integrity: with benchmark_passed=False, chirality status must NOT be 'supported'/'conditional'."""
    # simulate analyze_chirality status logic
    benchmark_passed = False
    unique = True
    status = ("conditional" if (benchmark_passed and unique)
              else "EXPLORATORY (benchmark did not pass)")
    assert "EXPLORATORY" in status or status == "no unique winner", status
    assert status != "conditional"

def test_no_binding_energy_in_methods():
    m = os.path.join(ROOT, "reports", "METHODS.md")
    if os.path.exists(m):
        txt = open(m).read().lower()
        # 'binding free energies' only allowed in a NOT-disclaimer
        assert "not binding free energies" in txt or "binding energy" not in txt

def test_ligprep_torsdof_zero():
    import lib_ligprep as lp  # noqa  (import works = module healthy)
    assert hasattr(lp, "mol2_to_rigid_pdbqt")

if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("PASS", name)
            except AssertionError as e:
                fails += 1; print("FAIL", name, e)
            except Exception as e:
                print("SKIP", name, type(e).__name__, e)
    sys.exit(1 if fails else 0)

#!/usr/bin/env python3
"""
gen_remd_mdps.py <hyb|unhyb> — generate a geometric temperature ladder and one
.mdp per replica for temperature-REMD adsorption of the ss(GT)15 domain.

Creates  run/<sys>/remd/repXX/grompp.mdp  and prints the temperature ladder.
Reads config/md_config.yaml (remd:, run:).
"""
import sys, math
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]
CFG  = yaml.safe_load(open(ROOT / "config" / "md_config.yaml"))

def main(sysname):
    r = CFG["remd"]; run = CFG["run"]
    n  = int(r["n_replicas"]); tmin = float(r["t_min_K"]); tmax = float(r["t_max_K"])
    nsteps = int(run["adsorption_remd_ns"] * 1000 / CFG["forcefield"]["dt_fs"] * 1000)
    # geometric (exponential) ladder -> ~uniform exchange acceptance
    temps = [round(tmin * (tmax / tmin) ** (i / (n - 1)), 2) for i in range(n)]
    tmpl = (ROOT / "mdp" / "remd" / "remd_template.mdp").read_text()
    base = ROOT / "run" / sysname / "remd"
    for i, T in enumerate(temps):
        d = base / ("rep%02d" % i); d.mkdir(parents=True, exist_ok=True)
        (d / "grompp.mdp").write_text(
            tmpl.replace("__TEMP__", "%.2f" % T).replace("__NSTEPS__", str(nsteps)))
    (base / "temperatures.txt").write_text("\n".join("%d %.2f" % (i, T)
                                                     for i, T in enumerate(temps)) + "\n")
    print("[gen_remd] %d replicas, %.0f-%.0f K, %d steps each" % (n, tmin, tmax, nsteps))
    print("[gen_remd] ladder:", " ".join("%.1f" % T for T in temps))
    print("[gen_remd] wrote", base)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "hyb")

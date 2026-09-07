#!/usr/bin/env bash
# 01_build_oligos.sh — build the STMN2-CE sensor nucleic acids with AmberTools
# (paper ref 59), then translate names for the CHARMM36 GROMACS build.
# Produces:
#   system/duplex_hybrid.pdb   capture(DNA):target(RNA) A-form duplex   (nab)
#   system/sensor_amber.pdb    (GT)15 + capture + target, joined        (tleap)
#   system/sensor_charmm.pdb   same, CHARMM36-named, no H                (translator)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p system

# --- require the NAB compiler (paper ref 59). conda-forge ambertools DROPS the
#     nab binary; a classic Amber install / `module load amber` has it. If nab is
#     absent, use the CHARMM-GUI path (docs/PROTOCOL.md §1b) which is equally valid
#     and emits GROMACS+CHARMM36 files directly.
if ! command -v nab >/dev/null 2>&1; then
  cat >&2 <<'MSG'
[01] ERROR: `nab` not found.
     conda-forge `ambertools` does not include the nab binary. Options:
       (A) module load amber        # classic Amber build ships nab   <-- try this on Unity
       (B) Use the CHARMM-GUI route (docs/PROTOCOL.md §1b): build the
           capture:target DNA:RNA hybrid duplex + (GT)15 as a partially double-
           stranded structure in "Nucleic Acid Builder", download the GROMACS
           CHARMM36 output, drop conf.gro/topol.top into run/<sys>/, then jump to
           merge_cnt_topology.py + 03_solvate_ions.sh (skip 01/02).
     Aborting the scripted AmberTools path.
MSG
  exit 3
fi

echo "[01] NAB: building capture:target hybrid duplex ..."
nab scripts/nab/build_duplex.nab -o /tmp/nab_duplex.x
/tmp/nab_duplex.x

echo "[01] tleap: building (GT)15, joining to duplex ..."
tleap -f scripts/nab/leap_build.in

echo "[01] CHECKPOINT — verify the phosphodiester junction (GT)15(30)->capture(31):"
echo "     parmed system/sensor_amber.prmtop  ->  printBonds :30-31"
echo "     (or open system/sensor_amber.pdb in VMD and inspect the O3'-P bond)"

echo "[01] translating Amber -> CHARMM36 names (dropping H for pdb2gmx -ignh) ..."
python3 scripts/amber2charmm_names.py system/sensor_amber.pdb system/sensor_charmm.pdb

echo "[01] DONE. Next: scripts/02_assemble_system.py"

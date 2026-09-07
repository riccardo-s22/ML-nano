#!/usr/bin/env bash
# 04_make_index.sh <hyb|unhyb> — build index groups used by the .mdp files and
# by analysis: CNT (freeze group), DNA, RNA, Water, Ions, and Solute.
set -euo pipefail
cd "$(dirname "$0")/.."
SYSNAME="${1:?usage: 04_make_index.sh <hyb|unhyb>}"
GMX="${GMX:-gmx}"
cd run/$SYSNAME

# CNT is residue name CNT; nucleic acids are the default DNA/RNA groups.
$GMX make_ndx -f solv_ions.gro -o index.ndx <<'EOF'
r CNT
name (last) CNT
r DA DT DG DC
name (last) DNA
r RA RU RG RC
name (last) RNA
"Water"
"Ion"
q
EOF

echo "[04] index.ndx written. Confirm a 'CNT' group exists (freeze group):"
$GMX make_ndx -f solv_ions.gro -n index.ndx <<'EOF'
q
EOF
echo "[04] DONE. Next: minimise -> equilibrate -> (REMD adsorb) -> production."

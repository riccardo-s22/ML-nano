#!/usr/bin/env bash
# 03_solvate_ions.sh <hyb|unhyb> — CHARMM36 topology for the nucleic acids,
# merge the frozen CNT, set the Harvey box, solvate with TIP3P, neutralise with Na+.
# Produces (per system) in  run/<sys>/ :  solv_ions.gro  topol.top
set -euo pipefail
cd "$(dirname "$0")/.."
SYSNAME="${1:?usage: 03_solvate_ions.sh <hyb|unhyb>}"
export GMXLIB="$PWD/forcefield"          # so pdb2gmx finds charmm36-*.ff
FF="$(python3 -c "import yaml;print(yaml.safe_load(open('config/md_config.yaml'))['forcefield']['gmx_ff_dir'].replace('.ff',''))")"
read BX BY BZ < <(python3 -c "import yaml;b=yaml.safe_load(open('config/md_config.yaml'))['box'];print(b['x_nm'],b['y_nm'],b['z_nm'])")
GMX="${GMX:-gmx}"
mkdir -p run/$SYSNAME && cd run/$SYSNAME

SENSOR="../../system/sensor_${SYSNAME}_only.pdb"

echo "[03] pdb2gmx (CHARMM36) on nucleic acids ..."
# Termini: for each chain pdb2gmx prompts. DNA chain: 5'-OH-ish, 3'-OH; RNA same.
# We answer with the neutral 5'/3'-OH/-H termini set. VERIFY prompt order for your
# ff version; adjust the here-doc if pdb2gmx asks differently.
$GMX pdb2gmx -f "$SENSOR" -o na.gro -p topol.top -i posre.itp \
     -ff "$FF" -water tip3p -ignh -ter <<'EOF'
1
1
1
1
EOF

echo "[03] merging frozen CNT into topology + coordinates ..."
python3 ../../scripts/merge_cnt_topology.py \
        topol.top na.gro ../../system/cnt.pdb ../../system/cnt.itp \
        topol.top complex.gro

echo "[03] setting box to Harvey dimensions ${BX} ${BY} ${BZ} nm ..."
$GMX editconf -f complex.gro -o boxed.gro -box $BX $BY $BZ -noc

echo "[03] solvating with TIP3P ..."
$GMX solvate -cp boxed.gro -cs spc216.gro -o solv.gro -p topol.top

echo "[03] adding Na+ to neutralise (Harvey: 74 Na+) ..."
cat > ions.mdp <<'EOF'
integrator = steep
nsteps     = 0
EOF
$GMX grompp -f ions.mdp -c solv.gro -p topol.top -o ions.tpr -maxwarn 2
printf "SOL\n" | $GMX genion -s ions.tpr -o solv_ions.gro -p topol.top \
        -pname NA -nname CL -neutral

echo "[03] DONE for '$SYSNAME'. Files: run/$SYSNAME/{solv_ions.gro,topol.top}"
echo "[03] Next: scripts/04_make_index.sh $SYSNAME"

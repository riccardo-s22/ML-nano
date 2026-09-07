#!/usr/bin/env bash
# Provenance capture (§6.4). Run from the pipeline root.
set -euo pipefail
OUT=provenance
mkdir -p "$OUT" "$OUT/run_manifests"

{
  echo "## host";      uname -a; echo
  echo "## date_utc";  date -u +%Y-%m-%dT%H:%M:%SZ; echo
  echo "## cpu";       lscpu | grep -E 'Model name|Socket|Core|Thread|^CPU\(s\)' || true; echo
  echo "## mem";       free -h || true; echo
  echo "## disk";      df -h . || true; echo
  echo "## gpu";       (nvidia-smi -L 2>/dev/null || echo "no gpu"); echo
  echo "## conda";     conda --version 2>/dev/null || echo "no conda"; echo
  echo "## conda envs"; conda env list 2>/dev/null || true; echo
  echo "## pipeline_seed"; echo "${PIPELINE_SEED:-unset}"; echo
} > "$OUT/software_versions.txt"

# Per-env exports (only those that exist)
for env in stmn2-seq stmn2-struct stmn2-docking stmn2-report stmn2-md stmn2-smk; do
  if conda env list 2>/dev/null | grep -q "$env"; then
    conda env export -n "$env" --no-builds > "$OUT/environment_export_${env}.yml" 2>/dev/null || true
  fi
done

# Input checksums (§6.7): references, configs, and project reference PDFs
{
  find references config -type f 2>/dev/null -print0 | xargs -0 sha256sum 2>/dev/null || true
  for pdf in ../../STMN2_final_RNA_SWCNT_sensor_order_report.pdf \
             ../../kinetically-tuned-single-walled-carbon-nanotube-corona-for-selective-detection-of-circulating-tumor-dna-point-mutations.pdf \
             ../../hiv-detection-via-a-carbon-nanotube-rna-sensor.pdf \
             ../../science.abq5622.pdf ../../13024_2023_Article_608.pdf ../../s00401-023-02655-0.pdf; do
    [ -f "$pdf" ] && sha256sum "$pdf" 2>/dev/null || true
  done
} > "$OUT/input_checksums.sha256"

echo "record_versions: wrote $OUT/software_versions.txt and input_checksums.sha256"

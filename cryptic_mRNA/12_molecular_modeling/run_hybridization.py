#!/usr/bin/env python3
"""Hybridization thermodynamics (§8, Gate 2) — MELTING + Biopython + descriptors.

- MELTING 5 (Sugimoto DNA/RNA hybrid) for the two PERFECT pathological duplexes:
  Tm (salt-corrected), ΔH, ΔS, ΔG(assay T) = ΔH − T·ΔS (reference NN value, NOT
  salt-conditioned — labeled honestly).
- Biopython R_DNA_NN1 (same Sugimoto set) Tm cross-check (reported side by side,
  not averaged).
- Descriptive metrics (longest contiguous Watson–Crick run, mismatch count) for
  ALL probe×target pairs — descriptors only, never a full heteroduplex energy.
- Condition sensitivity grid (Tm) for the perfect duplexes — labeled sensitivity.
"""
import os, sys, re, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_common import ROOT, load_configs, dna_to_rna, rna_to_dna, write_json, utc_now

OUT = os.path.join(ROOT, "results", "hybridization")
RAW = os.path.join(OUT, "melting_raw")
GATES = os.path.join(ROOT, "results", "gates")
os.makedirs(RAW, exist_ok=True); os.makedirs(GATES, exist_ok=True)

MELTING_JAR = os.path.join(ROOT, "tools", "MELTING5.2.0", "executable", "melting5.jar")
NN_PATH = os.path.join(ROOT, "tools", "MELTING5.2.0", "Data")

WC = {"A": "U", "U": "A", "T": "A", "G": "C", "C": "G"}

def load_seqs():
    seqs = {}
    with open(os.path.join(ROOT, "results", "sequence_validation", "validated_sequences.tsv")) as fh:
        next(fh)
        for line in fh:
            name, seq, *_ = line.rstrip("\n").split("\t")
            seqs[name] = seq
    return seqs

def parse_num(txt, key):
    m = re.search(key + r"\s*:\s*([-\d,\.]+)", txt)
    if not m:
        return None
    raw = m.group(1)
    raw = raw.replace(",", "") if re.search(r",\d{3}\b", raw) else raw.replace(",", ".")
    try: return float(raw)
    except ValueError: return None

def run_melting(probe_dna, target_rna, na_M, mg_M, conc_M, label):
    """probe 5'->3' DNA; target 5'->3' RNA. -C must be the per-position complement
    (target reversed) so MELTING aligns the antiparallel duplex correctly."""
    out = os.path.join(RAW, f"{label}.out")
    env = dict(os.environ, NN_PATH=NN_PATH)
    cmd = ["java", "-cp", MELTING_JAR, "melting.Main",
           "-S", probe_dna, "-C", target_rna[::-1], "-H", "dnarna",
           "-E", f"Na={na_M}" + (f":Mg={mg_M}" if mg_M else ""),
           "-P", str(conc_M), "-O", out]
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=120)
    txt = p.stdout + "\n" + p.stderr
    if os.path.exists(out):
        txt += "\n" + open(out).read()
    dH = parse_num(txt, "Enthalpy")
    dS = parse_num(txt, "Entropy")
    tm = parse_num(txt, "Melting temperature")
    return {"Tm_C": tm, "deltaH_cal_mol": dH, "deltaS_cal_mol_K": dS,
            "returncode": p.returncode, "raw_path": os.path.relpath(out, ROOT)}

def biopython_tm(probe_dna, na_mM, conc_nM):
    from Bio.SeqUtils import MeltingTemp as mt
    tm150 = float(mt.Tm_NN(probe_dna, nn_table=mt.R_DNA_NN1, Na=na_mM, dnac1=conc_nM,
                           dnac2=0, saltcorr=5))
    tmref = float(mt.Tm_NN(probe_dna, nn_table=mt.R_DNA_NN1, dnac1=conc_nM, dnac2=0, saltcorr=0))
    return round(tm150, 2), round(tmref, 2)

def longest_wc_run(probe_dna, target_rna):
    """Longest contiguous antiparallel Watson–Crick run between probe (DNA 5'->3')
    and target (RNA 5'->3') over all alignment registers. DESCRIPTOR ONLY."""
    P = dna_to_rna(probe_dna)
    T = target_rna
    Trev = T[::-1]  # align antiparallel: P[i] pairs with T[len-1-i]
    best = 0; best_off = 0
    for off in range(-(len(Trev) - 1), len(P)):
        run = 0
        for i in range(len(P)):
            j = i - off
            if 0 <= j < len(Trev):
                if WC.get(P[i]) == Trev[j]:
                    run += 1
                    if run > best:
                        best = run; best_off = off
                else:
                    run = 0
            else:
                run = 0
    return best

def total_wc_in_register(probe_dna, target_rna):
    """WC matches in the design register (probe vs revcomp target, end-aligned)."""
    P = dna_to_rna(probe_dna); Trev = target_rna[::-1]
    n = min(len(P), len(Trev))
    matches = sum(1 for i in range(n) if WC.get(P[i]) == Trev[i])
    return matches, n

def main():
    cfg = load_configs()
    assay = cfg["assay"]
    seqs = load_seqs()
    T_assay = assay["temperature_c"]
    na_M = assay["sodium_mM"] / 1000.0
    mg_M = assay["magnesium_mM"] / 1000.0
    probe_conc_M = assay["probe_concentration_nM"] * 1e-9

    perfect_pairs = [
        ("junction20", "junction20_capture", "pathological20_RNA"),
        ("junction22", "junction22_capture", "pathological22_RNA"),
    ]
    R = 1.98720425864083  # cal/mol/K

    # ---- MELTING perfect duplexes + Biopython cross-check ----
    perfect_rows = []; model_cmp_rows = []
    for sensor, pk, tk in perfect_pairs:
        probe = seqs[pk]; target = seqs[tk]
        m = run_melting(probe, target, na_M, mg_M, probe_conc_M, f"{sensor}_perfect")
        dG = None
        if m["deltaH_cal_mol"] is not None and m["deltaS_cal_mol_K"] is not None:
            dG = (m["deltaH_cal_mol"] - (T_assay + 273.15) * m["deltaS_cal_mol_K"]) / 1000.0
        tm150, tmref = biopython_tm(probe, assay["sodium_mM"], assay["probe_concentration_nM"])
        perfect_rows.append({
            "sensor": sensor, "target": tk, "target_class": "pathological_perfect",
            "probe_sequence_5to3": probe, "target_sequence_5to3": target,
            "probe_length": len(probe), "target_length": len(target),
            "engine": "MELTING5", "software_version": "5.2.0",
            "model_name": "Sugimoto1995_DNA_RNA_hybrid", "parameter_set": "dnarna",
            "ionic_conditions": f"Na={na_M}M;Mg={mg_M}M", "strand_conc_M": probe_conc_M,
            "Tm_C_salt_corrected": m["Tm_C"],
            "deltaH_kcal_mol_refNN": round(m["deltaH_cal_mol"]/1000, 2) if m["deltaH_cal_mol"] else None,
            "deltaS_cal_mol_K_refNN": m["deltaS_cal_mol_K"],
            "deltaG_kcal_mol_at_assayT_refNN_NOT_salt_conditioned": round(dG, 2) if dG else None,
            "assay_temperature_C": T_assay,
            "longest_contiguous_WC_run": len(probe),
            "raw_output_path": m["raw_path"],
        })
        model_cmp_rows.append({
            "sensor": sensor, "MELTING_Tm_C_150mM": m["Tm_C"],
            "Biopython_Tm_C_150mM_saltcorr5": tm150,
            "Biopython_Tm_C_referenceNN_saltcorr0": tmref,
            "abs_diff_150mM_C": round(abs(m["Tm_C"] - tm150), 2) if m["Tm_C"] else None,
            "note": "same Sugimoto1995 set; residual gap = salt-correction/initiation "
                    "convention; NOT averaged (§8.5).",
        })

    # ---- descriptors for ALL probe x target pairs ----
    probes = [("junction20", "junction20_capture"), ("junction22", "junction22_capture")]
    targets = ["pathological20_RNA", "pathological22_RNA", "normal20_RNA",
               "normal22_RNA", "cryptic_exon_positive26_RNA", "negative_RNA"]
    desc_rows = []
    for sensor, pk in probes:
        probe = seqs[pk]
        for tk in targets:
            target = seqs[tk]
            lwc = longest_wc_run(probe, target)
            tw, n = total_wc_in_register(probe, target)
            tclass = ("pathological_perfect" if tk.startswith("pathological") and
                      ((sensor == "junction20" and tk == "pathological20_RNA") or
                       (sensor == "junction22" and tk == "pathological22_RNA"))
                      else "normal" if tk.startswith("normal")
                      else "cryptic" if "cryptic" in tk else "negative"
                      if tk == "negative_RNA" else "cross")
            desc_rows.append({
                "sensor": sensor, "probe_5to3": probe, "target": tk,
                "target_class": tclass, "target_5to3": target,
                "longest_contiguous_WC_run_DESCRIPTOR": lwc,
                "WC_matches_in_design_register": tw, "register_length": n,
                "fraction_register_paired_DESCRIPTOR": round(tw / n, 3) if n else None,
                "note": "longest_contiguous_WC_run is a mechanistic DESCRIPTOR, "
                        "NOT a full heteroduplex free energy.",
            })

    # ---- condition sensitivity (Tm) for perfect duplexes ----
    sens_rows = []
    for sensor, pk, tk in perfect_pairs:
        probe = seqs[pk]; target = seqs[tk]
        for T in assay["temperature_grid_c"]:
            for na in assay["sodium_grid_mM"]:
                for mg in assay["magnesium_grid_mM"]:
                    m = run_melting(probe, target, na/1000, mg/1000, probe_conc_M,
                                    f"sens_{sensor}_T{int(T)}_Na{int(na)}_Mg{int(mg)}")
                    sens_rows.append({"sensor": sensor, "temperature_c": T,
                                      "sodium_mM": na, "magnesium_mM": mg,
                                      "Tm_C_salt_corrected": m["Tm_C"],
                                      "label": "SENSITIVITY (not primary assay)"})

    # ---- write TSVs ----
    def write_tsv(path, rows):
        if not rows: return
        cols = list(rows[0].keys())
        with open(path, "w") as fh:
            fh.write("\t".join(cols) + "\n")
            for r in rows:
                fh.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")

    write_tsv(os.path.join(OUT, "melting_perfect_results.tsv"), perfect_rows)
    write_tsv(os.path.join(OUT, "model_comparison.tsv"), model_cmp_rows)
    write_tsv(os.path.join(OUT, "duplex_descriptors.tsv"), desc_rows)
    write_tsv(os.path.join(OUT, "condition_sensitivity.tsv"), sens_rows)

    summary = {"perfect_duplexes": perfect_rows, "model_comparison": model_cmp_rows,
               "n_descriptor_pairs": len(desc_rows), "n_sensitivity_points": len(sens_rows),
               "utc": utc_now()}
    write_json(os.path.join(OUT, "hybridization_melting_summary.json"), summary)
    print("MELTING perfect duplexes:")
    for r in perfect_rows:
        print(f"  {r['sensor']}: Tm={r['Tm_C_salt_corrected']}C "
              f"dG(37C,refNN)={r['deltaG_kcal_mol_at_assayT_refNN_NOT_salt_conditioned']} kcal/mol")
    print("Specificity descriptor (longest contiguous WC run):")
    for r in desc_rows:
        if r["target_class"] in ("normal", "negative") or "pathological" in r["target_class"]:
            print(f"  {r['sensor']} vs {r['target']} [{r['target_class']}]: "
                  f"run={r['longest_contiguous_WC_run_DESCRIPTOR']} "
                  f"paired={r['fraction_register_paired_DESCRIPTOR']}")

if __name__ == "__main__":
    main()

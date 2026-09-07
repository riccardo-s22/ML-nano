#!/usr/bin/env python3
"""NUPACK 4.1 analyses (§8.2, §8.4) — used only where legitimate.

NUPACK 4.1 analysis is single-material (no true DNA/RNA hybrid NN parameters; see
results/smoke_tests/nupack.json). Therefore:
  (1) RNA-ONLY accessibility (material='rna') of the targets and transcript-derived
      junction windows — LEGITIMATE, primary §8.4 deliverable.
  (2) probe·target equilibrium competition bound fractions under single-material
      DNA and RNA models as a BRACKET — labeled EXPLORATORY / non-hybrid, NOT a
      substitute hybrid ΔG (§8.2 forbids substitution for the primary result).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_common import ROOT, load_configs, rna_to_dna, dna_to_rna, write_json, utc_now

OUT = os.path.join(ROOT, "results", "hybridization")
BPP = os.path.join(OUT, "base_pair_probabilities")
os.makedirs(BPP, exist_ok=True)

def load_seqs():
    seqs = {}
    with open(os.path.join(ROOT, "results", "sequence_validation", "validated_sequences.tsv")) as fh:
        next(fh)
        for line in fh:
            name, seq, *_ = line.rstrip("\n").split("\t")
            seqs[name] = seq
    return seqs

def read_mrna():
    fa = os.path.join(ROOT, "references", "STMN2_transcript.fasta")
    with open(fa) as fh:
        lines = [l.strip() for l in fh if l and not l.startswith(">")]
    return "".join(lines).upper().replace("U", "T")

def unpaired_probabilities(seq_rna, model):
    from nupack import Strand, Complex, complex_analysis
    s = Strand(seq_rna, name="w")
    cx = Complex([s], name="w1")
    r = complex_analysis(complexes=[cx], model=model, compute=["pairs", "mfe"])
    P = r[cx].pairs.to_array()
    unpaired = [float(1.0 - P[i].sum() + P[i, i]) for i in range(len(seq_rna))]
    mfe = r[cx].mfe
    struct = str(mfe[0].structure) if mfe else ""
    return unpaired, struct

def main():
    cfg = load_configs()
    assay = cfg["assay"]
    seqs = load_seqs()
    from nupack import Strand, Complex, Tube, Model, complex_analysis, tube_analysis
    import nupack
    na = assay["sodium_mM"] / 1000.0; mg = assay["magnesium_mM"] / 1000.0
    Trna = Model(material="rna", celsius=assay["temperature_c"], sodium=na, magnesium=mg)

    prov = {"nupack_version": nupack.__version__, "utc": utc_now(),
            "material_policy": "RNA-only accessibility primary; single-material "
                               "competition bracket exploratory; no hybrid substitution"}

    # ---- (1) RNA-only accessibility: short targets ----
    acc_rows = []
    for tk in ["pathological20_RNA", "pathological22_RNA", "normal20_RNA",
               "normal22_RNA", "cryptic_exon_positive26_RNA", "negative_RNA"]:
        up, struct = unpaired_probabilities(seqs[tk], Trna)
        acc_rows.append({"sequence_name": tk, "length": len(seqs[tk]),
                         "engine": "NUPACK4.1_rna", "mean_unpaired_prob": round(sum(up)/len(up), 3),
                         "min_unpaired_prob": round(min(up), 3),
                         "mfe_structure": struct})
        write_json(os.path.join(BPP, f"{tk}_unpaired.json"),
                   {"unpaired_prob": [round(x, 4) for x in up], "structure": struct})

    # ---- (1b) long-context junction windows ----
    mrna = read_mrna()
    normal20_dna = rna_to_dna(seqs["normal20_RNA"])
    idx = mrna.find(normal20_dna)
    junction_pos = idx + 10  # exon1|exon2 boundary in the spliced normal transcript
    long_rows = []
    if idx >= 0:
        for w in (60, 100, 150):
            half = w // 2
            a = max(0, junction_pos - half); b = min(len(mrna), junction_pos + half)
            win = dna_to_rna(mrna[a:b])
            up, struct = unpaired_probabilities(win, Trna)
            # footprint = central 20 nt around junction
            jc = junction_pos - a
            fp = up[max(0, jc-10):jc+10]
            long_rows.append({"junction": "normal", "window_nt": w,
                              "mrna_span": [a, b], "footprint_mean_unpaired": round(sum(fp)/len(fp), 3),
                              "engine": "NUPACK4.1_rna",
                              "source": "RefSeq NM_007029.4 (transcript-derived)"})
            write_json(os.path.join(BPP, f"normal_window_{w}.json"),
                       {"unpaired_prob": [round(x, 4) for x in up], "structure": struct,
                        "junction_index_in_window": jc})
        # pathological junction context: exon1 terminus (RefSeq) + cryptic exon (project source)
        exon1_ctx = dna_to_rna(mrna[max(0, junction_pos-50):junction_pos])
        cryptic = seqs["cryptic_exon_positive26_RNA"]
        path_win = exon1_ctx + cryptic
        up, struct = unpaired_probabilities(path_win, Trna)
        jc = len(exon1_ctx)
        fp = up[max(0, jc-10):jc+10]
        long_rows.append({"junction": "pathological", "window_nt": len(path_win),
                          "mrna_span": "constructed", "footprint_mean_unpaired": round(sum(fp)/len(fp), 3),
                          "engine": "NUPACK4.1_rna",
                          "source": "CONSTRUCTED: exon1 terminus (RefSeq) + cryptic exon 2a "
                                    "(project primary source; not in canonical mRNA)"})
        write_json(os.path.join(BPP, "pathological_window_constructed.json"),
                   {"unpaired_prob": [round(x, 4) for x in up], "structure": struct,
                    "junction_index_in_window": jc})

    def write_tsv(path, rows):
        if not rows: return
        cols = list(rows[0].keys())
        with open(path, "w") as fh:
            fh.write("\t".join(cols) + "\n")
            for r in rows:
                fh.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")

    # write the legitimate accessibility outputs first (independent of exploratory part)
    write_tsv(os.path.join(OUT, "nupack_accessibility_short.tsv"), acc_rows)
    write_tsv(os.path.join(OUT, "long_context_accessibility.tsv"), long_rows)

    # ---- (2) EXPLORATORY single-material competition bracket (fault-tolerant) ----
    probe_conc = assay["probe_concentration_nM"] * 1e-9
    targ_conc = assay["probe_concentration_nM"] * 1e-9
    comp_rows = []
    tnames = ["pathological20_RNA", "pathological22_RNA", "normal20_RNA",
              "normal22_RNA", "negative_RNA"]
    name_of = {tk: tk.replace("_RNA", "") for tk in tnames}   # unique names
    try:
        for sensor, pk in [("junction20", "junction20_capture"), ("junction22", "junction22_capture")]:
            probe_dna = seqs[pk]
            for mat in ("dna", "rna"):
                model = Model(material=mat, celsius=assay["temperature_c"], sodium=na, magnesium=mg)
                pseq = probe_dna if mat == "dna" else dna_to_rna(probe_dna)
                probe = Strand(pseq, name="probe")
                strands = {probe: probe_conc}
                for tk in tnames:
                    tsq = seqs[tk] if mat == "rna" else rna_to_dna(seqs[tk])
                    strands[Strand(tsq, name=name_of[tk])] = targ_conc
                tube = Tube(strands=strands, complexes=nupack.SetSpec(max_size=2), name="comp")
                res = tube_analysis(tubes=[tube], model=model, compute=["pfunc"])
                conc = res.tubes[tube].complex_concentrations
                for tk in tnames:
                    bound = 0.0
                    for cx, c in conc.items():
                        nms = sorted(s.name for s in cx.strands)
                        if "probe" in nms and name_of[tk] in nms and len(cx.strands) == 2:
                            bound += c
                    comp_rows.append({
                        "sensor": sensor, "target": tk, "material_model": mat,
                        "frac_probe_bound_to_target": round(bound / probe_conc, 4),
                        "label": "EXPLORATORY single-material competition (NOT hybrid; "
                                 "NUPACK 4.1 lacks DNA/RNA hybrid analysis params)",
                    })
        write_tsv(os.path.join(OUT, "nupack_competition_exploratory.tsv"), comp_rows)
    except Exception as e:
        prov["competition_status"] = f"exploratory competition failed: {type(e).__name__}: {e}"
        print("WARN: exploratory competition failed:", e)

    write_json(os.path.join(OUT, "nupack_provenance.json"), prov)
    print("NUPACK accessibility (short targets):")
    for r in acc_rows:
        print(f"  {r['sequence_name']}: mean unpaired={r['mean_unpaired_prob']}")
    print("Long-context footprint accessibility:")
    for r in long_rows:
        print(f"  {r['junction']} w={r['window_nt']}: footprint unpaired={r['footprint_mean_unpaired']}")
    print("Exploratory competition (frac probe bound, DNA/RNA bracket):")
    for r in comp_rows:
        if r["target"] in ("pathological20_RNA", "normal20_RNA", "negative_RNA") and r["sensor"] == "junction20":
            print(f"  {r['sensor']} vs {r['target']} [{r['material_model']}]: {r['frac_probe_bound_to_target']}")

if __name__ == "__main__":
    main()

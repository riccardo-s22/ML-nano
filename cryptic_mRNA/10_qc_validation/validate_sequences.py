#!/usr/bin/env python3
"""Sequence validation gate (§5.3, §5.4, Gate 1).

Derives normal22 from the authoritative RefSeq mRNA (§4.5) and runs the full
unit-test battery. Writes validated_sequences.tsv, sequence_alignment.txt,
sequence_validation.json, and the Gate 1 status file. Exits non-zero on any
failure so Snakemake halts the DAG.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_common import (ROOT, load_configs, revcomp_dna, dna_to_rna, rna_to_dna,
                        write_json, utc_now, sha256_str)

OUTDIR = os.path.join(ROOT, "results", "sequence_validation")
GATEDIR = os.path.join(ROOT, "results", "gates")
REFDIR = os.path.join(ROOT, "references")
os.makedirs(OUTDIR, exist_ok=True); os.makedirs(GATEDIR, exist_ok=True)


def read_mrna():
    fa = os.path.join(REFDIR, "STMN2_transcript.fasta")
    with open(fa) as fh:
        lines = [l.strip() for l in fh if l and not l.startswith(">")]
    return "".join(lines).upper().replace("U", "T")  # mRNA stored as DNA alphabet


def derive_normal22(seqs, mrna, errors, prov):
    """normal20 = exon1[-10:]+exon2[:10] is a contiguous mRNA substring.
    Extend +/-1 nt -> normal22 = exon1[-11:]+exon2[:11]. Cross-validate the
    shared exon-1 11-mer against pathological22."""
    normal20_dna = rna_to_dna(seqs["normal20_RNA"])
    idx = mrna.find(normal20_dna)
    prov["normal20_dna_query"] = normal20_dna
    prov["normal20_mrna_index"] = idx
    if idx < 0:
        errors.append("normal20 not found in RefSeq mRNA — cannot derive normal22 from reference")
        return None
    if idx == 0 or idx + 20 >= len(mrna):
        errors.append("normal20 too close to mRNA terminus to extend to 22 nt")
        return None
    normal22_dna = mrna[idx - 1: idx + 21]      # exon1[-11:] + exon2[:11]
    normal22_rna = dna_to_rna(normal22_dna)
    prov["normal22_mrna_span"] = [idx - 1, idx + 21]
    # cross-validate shared exon-1 11-mer with pathological22 (both start at exon1[-11:])
    shared_exon1_11 = normal22_rna[:11]
    if shared_exon1_11 != seqs["pathological22_RNA"][:11]:
        errors.append(
            f"normal22 exon-1 11-mer {shared_exon1_11} != pathological22 exon-1 "
            f"11-mer {seqs['pathological22_RNA'][:11]} — boundary inconsistent")
    prov["shared_exon1_11mer"] = shared_exon1_11
    prov["normal_exon2_first11"] = normal22_rna[11:]
    prov["normal22_RNA_derived"] = normal22_rna
    prov["normal22_source"] = f"RefSeq mRNA substring [{idx-1}:{idx+21}] (reference-derived)"
    return normal22_rna


def run_checks(seqs, errors):
    def check(cond, msg):
        if not cond:
            errors.append(msg)

    # alphabet & strand
    check(set(seqs["junction20_capture"]) <= set("ACGT"), "capture20 has non-DNA letters")
    check(set(seqs["junction22_capture"]) <= set("ACGT"), "capture22 has non-DNA letters")
    check("U" not in seqs["junction20_full_sensor"], "U found in junction20 DNA sensor")
    check("U" not in seqs["junction22_full_sensor"], "U found in junction22 DNA sensor")
    for k in ("pathological20_RNA", "pathological22_RNA", "normal20_RNA",
              "cryptic_exon_positive26_RNA", "negative_RNA"):
        check(set(seqs[k]) <= set("ACGU"), f"{k} not pure RNA alphabet")
        check("T" not in seqs[k], f"T found in RNA target {k}")

    # exact lengths
    check(len(seqs["anchor_full_GT15"]) == 30, "GT15 != 30 nt")
    for a in ("GT7", "CT7", "AT7"):
        check(len(seqs[a]) == 14, f"{a} != 14 nt")
    check(len(seqs["junction20_capture"]) == 20, "capture20 != 20 nt")
    check(len(seqs["junction22_capture"]) == 22, "capture22 != 22 nt")
    check(len(seqs["pathological20_RNA"]) == 20, "pathological20 != 20 nt")
    check(len(seqs["pathological22_RNA"]) == 22, "pathological22 != 22 nt")

    # anchor composition
    check(seqs["anchor_full_GT15"] == "GT" * 15, "GT15 is not (GT)15")
    check(seqs["GT7"] == "GT" * 7, "GT7 is not (GT)7")
    check(seqs["CT7"] == "CT" * 7, "CT7 is not (CT)7")
    check(seqs["AT7"] == "AT" * 7, "AT7 is not (AT)7")

    # full sensor = anchor + capture
    check(seqs["junction20_full_sensor"] == seqs["anchor_full_GT15"] + seqs["junction20_capture"],
          "junction20 full sensor != GT15 + capture20")
    check(seqs["junction22_full_sensor"] == seqs["anchor_full_GT15"] + seqs["junction22_capture"],
          "junction22 full sensor != GT15 + capture22")

    # reverse-complement / antiparallel complementarity (probe vs pathological RNA)
    check(dna_to_rna(revcomp_dna(seqs["junction20_capture"])) == seqs["pathological20_RNA"],
          "capture20 is not the antiparallel complement of pathological20")
    check(dna_to_rna(revcomp_dna(seqs["junction22_capture"])) == seqs["pathological22_RNA"],
          "capture22 is not the antiparallel complement of pathological22")

    # pathological junction composition: exon1 + cryptic-exon halves
    # 20-mer: exon1[-10:] shared with normal20; cryptic first10 = pathological20[10:]
    # 22-mer: exon1[-11:] = pathological22[:11]; cryptic first11 = pathological22[11:]
    check(seqs["cryptic_exon_positive26_RNA"].startswith(seqs["pathological22_RNA"][11:]),
          "cryptic-exon-positive control does not start with the CE2a 11-mer of pathological22")

    # normal targets diverge after shared exon-1 run
    shared = 0
    for x, y in zip(seqs["normal20_RNA"], seqs["pathological20_RNA"]):
        if x == y:
            shared += 1
        else:
            break
    check(0 < shared < 20, f"normal20 shared run with pathological20 = {shared} (suspicious)")
    seqs["_normal20_shared_run_nt"] = shared

    # normal22 must be present and source-verified (not the placeholder)
    check(seqs.get("normal22_RNA") not in (None, "REQUIRED_FROM_REFERENCE"),
          "normal22 unresolved — pipeline must not proceed")
    if seqs.get("normal22_RNA") not in (None, "REQUIRED_FROM_REFERENCE"):
        check(len(seqs["normal22_RNA"]) == 22, "normal22 != 22 nt")
        check(set(seqs["normal22_RNA"]) <= set("ACGU"), "normal22 not pure RNA")

    # gene-confusion guard
    check("SMN" not in seqs.get("_gene_notes", ""), "possible SMN1/SMN2 confusion")
    return shared


def main():
    cfg = load_configs()
    s = cfg["sequences"]
    seqs = {}
    for group in ("anchors", "capture_probes", "full_sensor_DNA", "rna_targets"):
        seqs.update(s[group])

    errors = []
    prov = {"utc": utc_now()}

    # derive normal22 from reference
    try:
        mrna = read_mrna()
        prov["mrna_length"] = len(mrna)
        n22 = derive_normal22(seqs, mrna, errors, prov)
        if n22:
            seqs["normal22_RNA"] = n22
    except FileNotFoundError:
        errors.append("RefSeq mRNA FASTA not found — run fetch_reference first")

    shared = run_checks(seqs, errors)

    # ---- outputs ----
    # validated_sequences.tsv
    order = ["anchor_full_GT15", "GT7", "CT7", "AT7",
             "junction20_capture", "junction22_capture",
             "junction20_full_sensor", "junction22_full_sensor",
             "pathological20_RNA", "pathological22_RNA",
             "normal20_RNA", "normal22_RNA",
             "cryptic_exon_positive26_RNA", "negative_RNA"]
    with open(os.path.join(OUTDIR, "validated_sequences.tsv"), "w") as fh:
        fh.write("name\tsequence_5to3\tlength\tstrand_type\tsha256\n")
        for k in order:
            v = seqs.get(k, "NA")
            stype = "RNA" if "RNA" in k else "DNA"
            fh.write(f"{k}\t{v}\t{len(v) if v!='NA' else 0}\t{stype}\t{sha256_str(str(v))}\n")

    # sequence_alignment.txt — show the shared exon-1 run mechanism (revision A4)
    with open(os.path.join(OUTDIR, "sequence_alignment.txt"), "w") as fh:
        fh.write("STMN2 junction alignment (shared exon-1 run drives normal-target off-pairing)\n")
        fh.write("=" * 70 + "\n\n")
        p20 = seqs["pathological20_RNA"]; n20 = seqs["normal20_RNA"]
        fh.write("20-nt junction targets (5'->3'):\n")
        fh.write(f"  pathological20: {p20}\n")
        fh.write(f"  normal20:       {n20}\n")
        fh.write("  match:          " + "".join("|" if a == b else " " for a, b in zip(p20, n20)) + "\n")
        fh.write(f"  shared exon-1 run = {shared} nt ({p20[:shared]})\n")
        fh.write(f"  divergence: pathological -> CE2a ({p20[shared:]}); normal -> exon2 ({n20[shared:]})\n\n")
        if seqs.get("normal22_RNA") not in (None, "REQUIRED_FROM_REFERENCE", "NA"):
            p22 = seqs["pathological22_RNA"]; n22 = seqs["normal22_RNA"]
            fh.write("22-nt junction targets (5'->3'):\n")
            fh.write(f"  pathological22: {p22}\n")
            fh.write(f"  normal22:       {n22}  [reference-derived]\n")
            fh.write("  match:          " + "".join("|" if a == b else " " for a, b in zip(p22, n22)) + "\n")
            sh22 = 0
            for a, b in zip(p22, n22):
                if a == b: sh22 += 1
                else: break
            fh.write(f"  shared exon-1 run = {sh22} nt\n\n")
        fh.write("capture/target antiparallel complementarity (probe 3'<-5' under RNA 5'->3'):\n")
        for cap, tgt in [("junction20_capture", "pathological20_RNA"),
                         ("junction22_capture", "pathological22_RNA")]:
            fh.write(f"  {tgt}:  5'-{seqs[tgt]}-3'\n")
            fh.write(f"  {cap}:  3'-{seqs[cap][::-1]}-5'\n")
            fh.write(f"  perfect duplex: {dna_to_rna(revcomp_dna(seqs[cap]))==seqs[tgt]}\n\n")

    passed = len(errors) == 0
    result = {"passed": passed, "errors": errors,
              "normal20_shared_run_nt": shared,
              "normal22_provenance": prov,
              "utc": utc_now()}
    write_json(os.path.join(OUTDIR, "sequence_validation.json"), result)

    # Gate 1
    gate1 = {"gate": "Gate1_sequence_provenance",
             "passed": passed,
             "normal22_reference_derived": prov.get("normal22_RNA_derived") is not None,
             "normal22_source": prov.get("normal22_source", "UNRESOLVED"),
             "shared_exon1_run_nt": shared,
             "errors": errors,
             "consequence_if_failed": "halt; no hybridization or structural analysis",
             "utc": utc_now()}
    write_json(os.path.join(GATEDIR, "gate1_sequence.json"), gate1)

    if errors:
        print("Sequence validation FAILED:", file=sys.stderr)
        for e in errors:
            print("  -", e, file=sys.stderr)
        sys.exit(1)
    print(f"Sequence validation PASSED. normal22 (reference-derived) = {seqs['normal22_RNA']}")
    print(f"  shared exon-1 run (normal20 vs pathological20) = {shared} nt")

if __name__ == "__main__":
    main()

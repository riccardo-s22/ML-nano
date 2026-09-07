#!/usr/bin/env python3
"""Fetch the authoritative STMN2 transcript live (§5.1).

Queries NCBI E-utilities for the RefSeq mRNA and Ensembl REST for the exon
structure. Records the *returned* accession version + checksum (not the
requested one). Writes references/ and the §5.2 provenance table.

This stage does NOT assume NM_007029.4 is current; it records whatever NCBI
serves and fails loudly if the network/record is unavailable.
"""
import os, sys, json, time
import requests
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_common import ROOT, utc_now, sha256_str, write_json, load_configs

REFDIR = os.path.join(ROOT, "references")
OUTDIR = os.path.join(ROOT, "results", "sequence_validation")
os.makedirs(REFDIR, exist_ok=True)
os.makedirs(OUTDIR, exist_ok=True)

def fetch_refseq(accession, email):
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {"db": "nuccore", "id": accession, "rettype": "gb",
              "retmode": "text", "email": email}
    r = requests.get(base, params=params, timeout=60)
    r.raise_for_status()
    return r.text

def fetch_refseq_fasta(accession, email):
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {"db": "nuccore", "id": accession, "rettype": "fasta",
              "retmode": "text", "email": email}
    r = requests.get(base, params=params, timeout=60)
    r.raise_for_status()
    return r.text

def parse_gb_version(gb_text):
    for line in gb_text.splitlines():
        if line.startswith("VERSION"):
            return line.split()[1]
    return None

def fasta_seq(fasta_text):
    lines = [l.strip() for l in fasta_text.splitlines() if l and not l.startswith(">")]
    return "".join(lines).upper()

def fetch_ensembl_exons(tx):
    server = "https://rest.ensembl.org"
    # strip version for overlap query
    tx_base = tx.split(".")[0]
    r = requests.get(f"{server}/overlap/id/{tx_base}?feature=exon",
                     headers={"Content-Type": "application/json"}, timeout=60)
    r.raise_for_status()
    return r.json()

def main():
    cfg = load_configs()
    email = cfg["config"]["reference_email"]
    cand = cfg["sequences"]["reference_candidates"]
    acc = cand["refseq"]

    manifest = {"retrieval_date_utc": utc_now(), "requested_refseq": acc,
                "requested_ensembl": cand["ensembl"], "assembly": cand["assembly"]}

    # --- RefSeq GenBank + FASTA ---
    gb = fetch_refseq(acc, email)
    time.sleep(0.4)
    fa = fetch_refseq_fasta(acc, email)
    version = parse_gb_version(gb)
    seq = fasta_seq(fa)

    with open(os.path.join(REFDIR, "STMN2_transcript.gb"), "w") as fh:
        fh.write(gb)
    with open(os.path.join(REFDIR, "STMN2_transcript.fasta"), "w") as fh:
        fh.write(fa)

    manifest["returned_refseq_version"] = version
    manifest["refseq_record_sha256"] = sha256_str(gb)
    manifest["mrna_length_nt"] = len(seq)

    # --- Ensembl exon structure (cross-check; may fail without aborting RefSeq path) ---
    exons = None
    try:
        exons = fetch_ensembl_exons(cand["ensembl"])
        with open(os.path.join(REFDIR, "STMN2_ensembl_exons.json"), "w") as fh:
            json.dump(exons, fh, indent=2)
        manifest["ensembl_exon_count"] = len(exons)
        manifest["ensembl_status"] = "ok"
    except Exception as e:
        manifest["ensembl_status"] = f"failed: {type(e).__name__}: {e}"

    write_json(os.path.join(OUTDIR, "reference_fetch_manifest.json"), manifest)

    # §5.2 provenance table
    prov_path = os.path.join(OUTDIR, "reference_provenance.tsv")
    cols = ["source", "assembly", "accession", "accession_version",
            "retrieval_date_utc", "record_sha256", "exon_1_coordinates",
            "normal_exon_2_coordinates", "cryptic_exon_2a_coordinates",
            "strand", "notes"]
    rows = [[
        "RefSeq (NCBI nuccore)", cand["assembly"], acc, version or "NA",
        manifest["retrieval_date_utc"], manifest["refseq_record_sha256"],
        "from_mRNA_substring_match", "from_mRNA_substring_match",
        "from_project_primary_source_(TDP43_cryptic_exon;not_in_canonical_mRNA)",
        "+", "Exon coords resolved by substring localization of validated junction "
             "oligos in the spliced mRNA; cryptic exon 2a is intronic and not in "
             "canonical RefSeq (Melamed 2019 / Klim 2019)."
    ]]
    with open(prov_path, "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(map(str, r)) + "\n")

    print(f"fetch_reference: RefSeq {version} ({len(seq)} nt) cached. "
          f"Ensembl: {manifest.get('ensembl_status')}")

if __name__ == "__main__":
    main()

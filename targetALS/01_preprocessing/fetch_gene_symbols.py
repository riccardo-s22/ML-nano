"""Map every UniProt accession in the three Welch result files to a gene symbol.

Queries the UniProt REST API in batches and caches the result to
uniprot_gene_symbols.tsv so the volcano script can run offline afterwards.
"""
import os
import time
import urllib.parse
import urllib.request

import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(BASE, "uniprot_gene_symbols.tsv")
FILES = [f"{e}_ALS_vs_CTRL_medianNormalized_welch.xlsx" for e in ("ex1", "ex2", "ex3")]

accs = set()
for f in FILES:
    d = pd.read_excel(os.path.join(BASE, f), sheet_name="Differential_all")
    for g in d["PG.ProteinGroups"].dropna():
        accs.update(a.strip() for a in str(g).split(";") if a.strip())
accs = sorted(accs)
print(f"{len(accs)} unique accessions")

def query(chunk):
    """Fetch one batch; UniProt rejects very long boolean queries, so split on 400."""
    q = " OR ".join(f"accession:{a}" for a in chunk)
    url = ("https://rest.uniprot.org/uniprotkb/search?"
           + urllib.parse.urlencode({
               "query": q,
               "fields": "accession,gene_primary,reviewed,protein_name",
               "format": "tsv", "size": "500"}))
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            text = r.read().decode()
    except urllib.error.HTTPError:
        if len(chunk) == 1:
            return []
        mid = len(chunk) // 2
        return query(chunk[:mid]) + query(chunk[mid:])
    return [ln.split("\t")[:4] for ln in text.strip().split("\n")[1:]
            if len(ln.split("\t")) >= 3]


rows = []
BATCH = 50
for i in range(0, len(accs), BATCH):
    got = query(accs[i:i + BATCH])
    rows += got
    print(f"  batch {i // BATCH + 1}/{-(-len(accs) // BATCH)}: {len(got)} entries")
    time.sleep(0.3)

out = pd.DataFrame(rows, columns=["accession", "gene", "reviewed", "protein_name"])
out = out.drop_duplicates("accession")
out.to_csv(CACHE, sep="\t", index=False)
missing = set(accs) - set(out["accession"])
print(f"wrote {CACHE}: {len(out)} mapped, {len(missing)} unmapped, "
      f"{(out['gene'].fillna('') == '').sum()} with no gene symbol")

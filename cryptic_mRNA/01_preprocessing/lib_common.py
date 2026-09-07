"""Shared helpers: config loading, provenance, sequence utilities."""
import json, os, hashlib, socket, time, sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def utc_now():
    return datetime.now(timezone.utc).isoformat()

def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def load_yaml(path):
    import yaml  # PyYAML ships with snakemake/conda envs
    with open(path) as fh:
        return yaml.safe_load(fh)

def load_configs():
    cfgdir = os.path.join(ROOT, "config")
    return {
        "config": load_yaml(os.path.join(cfgdir, "config.yaml")),
        "sequences": load_yaml(os.path.join(cfgdir, "sequences.yaml")),
        "assay": load_yaml(os.path.join(cfgdir, "assay_conditions.yaml")),
        "docking": load_yaml(os.path.join(cfgdir, "docking.yaml")),
        "md": load_yaml(os.path.join(cfgdir, "md.yaml")),
    }

def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)

def write_manifest(rule, extra=None):
    rec = {
        "rule": rule,
        "host": socket.gethostname(),
        "utc": utc_now(),
        "seed_env": os.environ.get("PIPELINE_SEED", "unset"),
    }
    if extra:
        rec.update(extra)
    path = os.path.join(ROOT, "provenance", "run_manifests", f"{rule}.manifest.json")
    write_json(path, rec)
    return path

# ---- sequence utilities ----
DNA_COMP = str.maketrans("ACGT", "TGCA")
RNA_COMP = str.maketrans("ACGU", "UGCA")

def revcomp_dna(s):
    return s.translate(DNA_COMP)[::-1]

def dna_to_rna(s):
    return s.replace("T", "U")

def rna_to_dna(s):
    return s.replace("U", "T")

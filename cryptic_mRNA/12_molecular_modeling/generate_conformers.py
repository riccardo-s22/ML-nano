#!/usr/bin/env python3
"""Anchor conformer ensembles (§10) — nucleic-acid-aware, NOT RDKit/MMFF.

Route per anchor: tleap (DNA.OL15) -> sander minimization -> Langevin MD
(implicit solvent igb=1, elevated T for backbone sampling) -> cpptraj RMSD
clustering -> >=10 medoids -> controlled rigid PDBQT (lib_ligprep). Identical
settings across GT7/CT7/AT7 (controlled comparison).
"""
import os, sys, subprocess, tempfile, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_common import ROOT, write_json, utc_now
import lib_ligprep as lp

OUTBASE = os.path.join(ROOT, "results", "anchor_conformers")
SEED = int(os.environ.get("PIPELINE_SEED", "20260625"))

ANCHOR_SEQS = {
    "GT7": "GTGTGTGTGTGTGT", "CT7": "CTCTCTCTCTCTCT", "AT7": "ATATATATATATAT",
    "GT15": "GTGTGTGTGTGTGTGTGTGTGTGTGTGTGT",
    # dinucleotide repeat units for the compact base-stacking docking screen (§11);
    # full-oligo rigid docking on the curved sidewall is clash-dominated (revision A1).
    "GTd": "GT", "CTd": "CT", "ATd": "AT",
}

def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)

def build_prmtop(seq, workdir):
    bmap = {"A": "DA", "C": "DC", "G": "DG", "T": "DT"}
    res = []
    for i, b in enumerate(seq):
        nm = bmap[b]
        if i == 0: nm += "5"
        elif i == len(seq)-1: nm += "3"
        res.append(nm)
    leapin = os.path.join(workdir, "leap.in")
    with open(leapin, "w") as fh:
        fh.write("source leaprc.DNA.OL15\n")
        fh.write("m = sequence { " + " ".join(res) + " }\n")
        fh.write(f"saveamberparm m {workdir}/m.prmtop {workdir}/m.inpcrd\n")
        fh.write(f"savepdb m {workdir}/m.pdb\nquit\n")
    p = sh(["tleap", "-f", leapin])
    return os.path.exists(f"{workdir}/m.prmtop"), p.stdout

def run_md(workdir, md_ps=100, save_every_ps=2, temp=400.0):
    nstlim = int(md_ps / 0.002); ntwx = int(save_every_ps / 0.002)
    with open(f"{workdir}/min.in", "w") as fh:
        fh.write(" min\n &cntrl\n  imin=1,maxcyc=1000,ncyc=400,igb=1,cut=999.0,ntb=0,\n /\n")
    with open(f"{workdir}/md.in", "w") as fh:
        fh.write(" langevin sampling\n &cntrl\n")
        fh.write(f"  imin=0,irest=0,ntx=1,nstlim={nstlim},dt=0.002,\n")
        fh.write(f"  igb=1,cut=999.0,ntb=0,gamma_ln=2.0,ntt=3,temp0={temp},tempi={temp},\n")
        fh.write(f"  ntpr=5000,ntwx={ntwx},ntwr={nstlim},ig={SEED},\n /\n")
    r1 = sh(["sander", "-O", "-i", f"{workdir}/min.in", "-p", f"{workdir}/m.prmtop",
             "-c", f"{workdir}/m.inpcrd", "-r", f"{workdir}/min.rst", "-o", f"{workdir}/min.out"])
    r2 = sh(["sander", "-O", "-i", f"{workdir}/md.in", "-p", f"{workdir}/m.prmtop",
             "-c", f"{workdir}/min.rst", "-r", f"{workdir}/md.rst",
             "-x", f"{workdir}/md.nc", "-o", f"{workdir}/md.out"])
    return os.path.exists(f"{workdir}/md.nc")

def cluster_medoids(workdir, n_clusters=10):
    cptin = os.path.join(workdir, "cluster.in")
    with open(cptin, "w") as fh:
        fh.write(f"parm {workdir}/m.prmtop\n")
        fh.write(f"trajin {workdir}/md.nc\n")
        fh.write("rms first !@H=\n")
        fh.write(f"cluster C0 hieragglo clusters {n_clusters} rms !@H= "
                 f"repout {workdir}/medoid repfmt pdb out {workdir}/cnumvtime.dat "
                 f"summary {workdir}/cluster_summary.dat\n")
        fh.write("run\nquit\n")
    p = sh(["cpptraj", "-i", cptin])
    medoids = sorted(glob.glob(f"{workdir}/medoid.c*.pdb"))
    return medoids, p.stdout

def pdb_to_mol2(workdir, pdb, out_mol2):
    """Re-derive a bonded mol2 (with charges) for a medoid frame via tleap."""
    leapin = os.path.join(workdir, "m2.in")
    with open(leapin, "w") as fh:
        fh.write("source leaprc.DNA.OL15\n")
        fh.write(f"x = loadpdb {pdb}\n")
        fh.write(f"savemol2 x {out_mol2} 1\nquit\n")
    sh(["tleap", "-f", leapin])
    return os.path.exists(out_mol2)

def process_anchor(anchor, n_clusters=10, md_ps=100):
    seq = ANCHOR_SEQS[anchor]
    outdir = os.path.join(OUTBASE, anchor)
    pool = os.path.join(outdir, "source_pool")
    os.makedirs(pool, exist_ok=True)
    work = tempfile.mkdtemp(prefix=f"conf_{anchor}_")
    rec = {"anchor": anchor, "sequence": seq, "workdir": work, "seed": SEED,
           "method": "tleap OL15 -> sander min+Langevin(igb=1,400K) -> cpptraj RMSD cluster",
           "md_ps": md_ps, "utc": utc_now()}
    ok, _ = build_prmtop(seq, work)
    if not ok:
        rec["status"] = "tleap failed"; write_json(os.path.join(outdir, "atom_integrity.tsv.json"), rec); return rec
    if not run_md(work, md_ps=md_ps):
        rec["status"] = "sander MD failed"; write_json(os.path.join(outdir, "status.json"), rec); return rec
    medoids, _ = cluster_medoids(work, n_clusters)
    rec["n_medoids"] = len(medoids)
    manifest = []
    for i, mp in enumerate(medoids, 1):
        cid = f"conformer_{i:03d}"
        pdb = os.path.join(outdir, f"{cid}.pdb")
        mol2 = os.path.join(work, f"{cid}.mol2")
        pdbqt = os.path.join(outdir, f"{cid}.pdbqt")
        # copy medoid pdb
        with open(mp) as src, open(pdb, "w") as dst:
            dst.write(src.read())
        n_res = len(set(l[22:26] for l in open(pdb) if l.startswith("ATOM")))
        if pdb_to_mol2(work, pdb, mol2):
            info = lp.mol2_to_rigid_pdbqt(mol2, pdbqt)
            manifest.append({"conformer_id": cid, "source": os.path.basename(mp),
                             "n_residues": n_res, "n_heavy": info["n_heavy"],
                             "torsdof": info["torsdof"], "pdbqt": os.path.relpath(pdbqt, ROOT)})
    # write manifest + integrity
    if manifest:
        cols = list(manifest[0].keys())
        with open(os.path.join(outdir, "conformer_manifest.tsv"), "w") as fh:
            fh.write("\t".join(cols) + "\n")
            for r in manifest:
                fh.write("\t".join(str(r[c]) for c in cols) + "\n")
        n_res_set = {r["n_residues"] for r in manifest}
        with open(os.path.join(outdir, "atom_integrity.tsv"), "w") as fh:
            fh.write("anchor\tn_conformers\tresidues_consistent\texpected_residues\tall_torsdof_0\n")
            fh.write(f"{anchor}\t{len(manifest)}\t{len(n_res_set)==1}\t{len(seq)}\t"
                     f"{all(r['torsdof']==0 for r in manifest)}\n")
    rec["status"] = "ok"; rec["n_conformers"] = len(manifest)
    write_json(os.path.join(outdir, "status.json"), rec)
    print(f"{anchor}: {len(manifest)} conformers, residues consistent="
          f"{len({r['n_residues'] for r in manifest})==1 if manifest else False}")
    return rec

if __name__ == "__main__":
    anchors = sys.argv[1:] or ["GT7", "CT7", "AT7"]
    for a in anchors:
        process_anchor(a)

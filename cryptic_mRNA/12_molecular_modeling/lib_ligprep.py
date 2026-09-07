"""Controlled rigid-ligand PDBQT preparation for ssDNA conformers (§10.3).

Meeko 0.5/0.7 fail to prepare bare ssDNA in this environment (RDKit bond-
perception crash / query-bond rejection). Rather than fall back silently to
deprecated MGLTools, this module prepares a RIGID ligand PDBQT from an
AmberTools `tleap` mol2 with AMBER atom types + AMBER partial charges. AutoDock
atom types are assigned from element + standard DNA atom naming (primed = sugar/
backbone, unprimed ring = nucleobase). Every heavy atom is preserved; nonpolar
hydrogens are merged (standard AutoDock practice); TORSDOF is 0 (rigid). All
rules are documented and reproducible. The receptor is uncharged graphite, so
H-bond/electrostatic terms vanish and the screen is steric/stacking-dominated.
"""
import os

# base nucleobase ring carbons (unprimed) -> aromatic 'A'; sugar carbons -> 'C'
BASE_RING_C = {"C2", "C4", "C5", "C6", "C8"}
# aromatic ring nitrogens (acceptor-capable) -> 'NA'; exocyclic/amine -> 'N'
BASE_RING_N = {"N1", "N3", "N7"}

def _element(amber_type, name=""):
    s = (amber_type or name).strip()
    if not s:
        return "C"
    c = s[0].upper()
    return {"C": "C", "N": "N", "O": "O", "P": "P", "H": "H", "S": "S"}.get(c, "C")

def _ad_type(name, el):
    nm = name.strip()
    if el == "P":
        return "P"
    if el == "O":
        return "OA"
    if el == "S":
        return "SA"
    if el == "C":
        primed = nm.endswith("'") or nm.endswith("*")
        return "A" if (not primed and nm in BASE_RING_C) else "C"
    if el == "N":
        primed = nm.endswith("'") or nm.endswith("*")
        return "NA" if (not primed and nm in BASE_RING_N) else "N"
    return "C"

def parse_mol2(path):
    atoms, bonds = [], []
    sec = None
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if s.startswith("@<TRIPOS>"):
                sec = s.split(">")[1]
                continue
            if not s:
                continue
            if sec == "ATOM":
                t = s.split()
                atoms.append({"id": int(t[0]), "name": t[1],
                              "x": float(t[2]), "y": float(t[3]), "z": float(t[4]),
                              "sybyl": t[5], "resid": t[6] if len(t) > 6 else "1",
                              "resname": t[7] if len(t) > 7 else "LIG",
                              "charge": float(t[8]) if len(t) > 8 else 0.0})
            elif sec == "BOND":
                t = s.split()
                bonds.append((int(t[1]), int(t[2])))
    return atoms, bonds

def mol2_to_rigid_pdbqt(mol2_path, out_pdbqt, resname="DNA"):
    atoms, bonds = parse_mol2(mol2_path)
    by_id = {a["id"]: a for a in atoms}
    neigh = {a["id"]: [] for a in atoms}
    for a, b in bonds:
        neigh[a].append(b); neigh[b].append(a)

    # classify hydrogens: polar (bonded to N/O/S) kept as HD; nonpolar (bonded to C) merged
    merge_into = {}   # nonpolar H id -> parent heavy id
    keep = []
    for a in atoms:
        el = _element(a["sybyl"], a["name"])
        if el == "H":
            parents = [by_id[n] for n in neigh[a["id"]]]
            pel = _element(parents[0]["sybyl"], parents[0]["name"]) if parents else "C"
            if pel in ("N", "O", "S"):
                a["ad_type"] = "HD"; keep.append(a)
            else:
                merge_into[a["id"]] = neigh[a["id"]][0] if neigh[a["id"]] else None
        else:
            a["ad_type"] = _ad_type(a["name"], el)
            keep.append(a)

    # merge nonpolar-H charge into parent
    qmap = {a["id"]: a["charge"] for a in atoms}
    for h_id, p_id in merge_into.items():
        if p_id is not None:
            qmap[p_id] += qmap[h_id]

    n_heavy = sum(1 for a in keep if _element(a["sybyl"], a["name"]) != "H")
    lines = []
    lines.append("REMARK  rigid ssDNA ligand; SYBYL->AutoDock typed from AmberTools mol2;\n")
    lines.append("REMARK  nonpolar H merged; AMBER partial charges; TORSDOF 0 (rigid).\n")
    lines.append("ROOT\n")
    serial = 0
    for a in keep:
        serial += 1
        q = qmap[a["id"]]
        lines.append(
            f"ATOM  {serial:5d} {a['name'][:4]:<4s} {resname:<3s} A{1:4d}    "
            f"{a['x']:8.3f}{a['y']:8.3f}{a['z']:8.3f}  1.00  0.00    "
            f"{q:6.3f} {a['ad_type']:<2s}\n")
    lines.append("ENDROOT\n")
    lines.append("TORSDOF 0\n")
    os.makedirs(os.path.dirname(os.path.abspath(out_pdbqt)), exist_ok=True)
    with open(out_pdbqt, "w") as fh:
        fh.writelines(lines)
    type_counts = {}
    for a in keep:
        type_counts[a["ad_type"]] = type_counts.get(a["ad_type"], 0) + 1
    return {"n_input_atoms": len(atoms), "n_heavy": n_heavy,
            "n_written": serial, "n_nonpolar_H_merged": len(merge_into),
            "ad_type_counts": type_counts, "torsdof": 0}

def build_ssdna_mol2(seq, out_mol2, leaprc="leaprc.DNA.OL15", tmpdir="/tmp"):
    """Build ssDNA via AmberTools tleap and save a mol2 with bonds + charges."""
    import subprocess, tempfile
    bmap = {"A": "DA", "C": "DC", "G": "DG", "T": "DT"}
    res = []
    for i, b in enumerate(seq):
        nm = bmap[b]
        if i == 0: nm += "5"
        elif i == len(seq) - 1: nm += "3"
        res.append(nm)
    with tempfile.NamedTemporaryFile("w", suffix=".in", dir=tmpdir, delete=False) as tf:
        tf.write(f"source {leaprc}\n")
        tf.write("m = sequence { " + " ".join(res) + " }\n")
        tf.write(f"savemol2 m {out_mol2} 1\n")
        tf.write("quit\n")
        leapin = tf.name
    p = subprocess.run(["tleap", "-f", leapin], capture_output=True, text=True)
    return p.returncode == 0 and os.path.exists(out_mol2), p.stdout

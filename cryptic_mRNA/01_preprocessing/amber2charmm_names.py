#!/usr/bin/env python3
"""
amber2charmm_names.py  —  translate an Amber-named nucleic-acid PDB into names
the CHARMM36 GROMACS port (charmm36-*.ff) accepts, and drop hydrogens so that
`gmx pdb2gmx -ignh` rebuilds them cleanly.

Usage:  amber2charmm_names.py in_amber.pdb out_charmm.pdb

VERIFY the residue map against your ff:  grep '^\[' charmm36-*.ff/{dna,rna}.rtp
Adjust RESMAP below if your port uses A/U/G/C instead of RA/RU/RG/RC, etc.
"""
import sys

# Amber residue -> CHARMM36-port residue (5'/3' handled later by pdb2gmx -ter).
RESMAP = {
    # DNA
    "DA5":"DA","DA":"DA","DA3":"DA","DAN":"DA",
    "DT5":"DT","DT":"DT","DT3":"DT","DTN":"DT",
    "DG5":"DG","DG":"DG","DG3":"DG","DGN":"DG",
    "DC5":"DC","DC":"DC","DC3":"DC","DCN":"DC",
    # RNA  (charmm36 port commonly uses RA/RU/RG/RC; switch to A/U/G/C if needed)
    "RA5":"RA","RA":"RA","RA3":"RA","A":"RA","A5":"RA","A3":"RA",
    "RU5":"RU","RU":"RU","RU3":"RU","U":"RU","U5":"RU","U3":"RU",
    "RG5":"RG","RG":"RG","RG3":"RG","G":"RG","G5":"RG","G3":"RG",
    "RC5":"RC","RC":"RC","RC3":"RC","C":"RC","C5":"RC","C3":"RC",
}
# Heavy-atom renames (Amber -> CHARMM). Phosphate oxygens differ by convention.
ATOMMAP = {"O1P":"OP1", "O2P":"OP2"}

def main(src, dst):
    n_at = n_drop = 0
    with open(src) as fi, open(dst, "w") as fo:
        for ln in fi:
            if not ln.startswith(("ATOM", "HETATM")):
                if ln.startswith(("TER", "END")):
                    fo.write(ln)
                continue
            atom = ln[12:16].strip()
            elem = ln[76:78].strip() or atom[:1]
            if elem == "H" or atom.startswith("H") or atom.startswith(("1H","2H","3H")):
                n_drop += 1
                continue
            res = ln[17:20].strip()
            res_c = RESMAP.get(res, res)
            atom_c = ATOMMAP.get(atom, atom)
            # right-justify atom name in cols 13-16 the PDB way
            if len(atom_c) < 4:
                atom_field = " %-3s" % atom_c
            else:
                atom_field = atom_c[:4]
            ln = ln[:12] + atom_field + ln[16:17] + "%-3s" % res_c + ln[20:]
            fo.write(ln)
            n_at += 1
        fo.write("END\n")
    print("[amber2charmm] wrote %s : %d heavy atoms kept, %d H dropped"
          % (dst, n_at, n_drop))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])

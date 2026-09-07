#!/usr/bin/env python3
"""Generate publication figures (§13) as SVG + 300 dpi PNG. Each function is
defensive: it skips if its input data is not present. Vina scores are never
labeled 'binding energy'."""
import os, sys, json, glob
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_common import ROOT
import figstyle
figstyle.apply_style()

FIG = os.path.join(ROOT, "figures")
HYB = os.path.join(ROOT, "results", "hybridization")
BPP = os.path.join(HYB, "base_pair_probabilities")
os.makedirs(FIG, exist_ok=True)

def load_seqs():
    s = {}
    with open(os.path.join(ROOT, "results", "sequence_validation", "validated_sequences.tsv")) as fh:
        next(fh)
        for line in fh:
            n, seq, *_ = line.rstrip("\n").split("\t"); s[n] = seq
    return s

def load_tsv(path):
    if not os.path.exists(path): return []
    with open(path) as fh:
        cols = fh.readline().rstrip("\n").split("\t")
        return [dict(zip(cols, l.rstrip("\n").split("\t"))) for l in fh]

# ---- Figure A: sequence architecture ----
def fig_sequence_architecture(seqs):
    fig, ax = plt.subplots(figsize=(9, 3.2))
    anchor = seqs["anchor_full_GT15"]; cap20 = seqs["junction20_capture"]
    # bar segments: GT15 anchor | exon1-comp (first 10) | CE2a-comp (rest)
    segs = [("(GT)15 SWCNT anchor", len(anchor), "#4575b4"),
            ("exon1-complementary", 10, "#fdae61"),
            ("cryptic exon 2a-complementary", len(cap20)-10, "#d73027")]
    x = 0
    for label, w, col in segs:
        ax.barh(1.0, w, left=x, color=col, edgecolor="k", height=0.5)
        ax.text(x + w/2, 1.0, label, ha="center", va="center", fontsize=8, color="white")
        x += w
    ax.axvline(len(anchor)+10, color="k", ls="--", lw=1)
    ax.text(len(anchor)+10, 1.45, "splice boundary\n(exon1|cryptic 2a)", ha="center", fontsize=7)
    # pathological vs normal RNA junctions
    ax.text(0, 0.3, f"pathological20 RNA 5'-{seqs['pathological20_RNA']}-3'", fontsize=7, family="monospace")
    ax.text(0, 0.05, f"normal20 RNA      5'-{seqs['normal20_RNA']}-3'", fontsize=7, family="monospace")
    ax.text(0, -0.2, "shared exon-1 run = 10 nt (ACAGCAAUGG) -> drives normal-target off-pairing",
            fontsize=7, color="#b30000")
    ax.set_xlim(-1, len(anchor)+len(cap20)+1); ax.set_ylim(-0.4, 1.7)
    ax.set_yticks([]); ax.set_xlabel("nucleotide position (5'->3' DNA sensor)")
    ax.set_title("Figure A. STMN2 RNA-SWCNT sensor sequence architecture")
    fig.tight_layout(); figstyle.save_both(fig, os.path.join(FIG, "sequence_architecture"))
    print("Figure A written")

# ---- Figure D: hybridization ----
def fig_hybridization():
    perfect = load_tsv(os.path.join(HYB, "melting_perfect_results.tsv"))
    desc = load_tsv(os.path.join(HYB, "duplex_descriptors.tsv"))
    comp = load_tsv(os.path.join(HYB, "nupack_competition_exploratory.tsv"))
    if not perfect: return
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    # panel 1: perfect duplex Tm + dG
    sens = [r["sensor"] for r in perfect]
    tm = [float(r["Tm_C_salt_corrected"]) for r in perfect]
    dg = [float(r["deltaG_kcal_mol_at_assayT_refNN_NOT_salt_conditioned"]) for r in perfect]
    ax = axes[0]; x = np.arange(len(sens))
    ax.bar(x-0.2, tm, 0.4, label="Tm (°C, salt-corr)", color="#4575b4")
    ax2 = ax.twinx(); ax2.bar(x+0.2, dg, 0.4, label="ΔG 37°C (kcal/mol, ref-NN)", color="#d73027")
    ax.set_xticks(x); ax.set_xticklabels(sens); ax.set_ylabel("Tm (°C)")
    ax2.set_ylabel("ΔG (kcal/mol, NN ref-cond)")
    ax.set_title("Perfect pathological duplex\n(MELTING Sugimoto hybrid)")
    # panel 2: longest WC run descriptor across targets (junction20)
    ax = axes[1]
    d20 = [r for r in desc if r["sensor"] == "junction20"]
    tgts = [r["target"].replace("_RNA", "") for r in d20]
    runs = [int(r["longest_contiguous_WC_run_DESCRIPTOR"]) for r in d20]
    ax.barh(range(len(tgts)), runs, color="#777")
    ax.set_yticks(range(len(tgts))); ax.set_yticklabels(tgts, fontsize=7)
    ax.set_xlabel("longest contiguous WC run (nt) — DESCRIPTOR")
    ax.set_title("Specificity descriptor\n(junction20 probe vs targets)")
    # panel 3: NUPACK exploratory competition bound fraction
    ax = axes[2]
    if comp:
        c20 = [r for r in comp if r["sensor"] == "junction20" and r["material_model"] == "dna"]
        t = [r["target"].replace("_RNA", "") for r in c20]
        fb = [float(r["frac_probe_bound_to_target"]) for r in c20]
        ax.bar(range(len(t)), fb, color="#1a9850")
        ax.set_xticks(range(len(t))); ax.set_xticklabels(t, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("frac probe bound (NUPACK)")
        ax.set_title("EXPLORATORY competition\n(NUPACK DNA model; non-hybrid)")
    fig.suptitle("Figure D. Hybridization thermodynamics (free-solution duplex stability contrast)")
    fig.tight_layout(); figstyle.save_both(fig, os.path.join(FIG, "hybridization"))
    print("Figure D written")

# ---- Figure E: long-context accessibility ----
def fig_accessibility():
    nfile = os.path.join(BPP, "normal_window_100.json")
    pfile = os.path.join(BPP, "pathological_window_constructed.json")
    if not os.path.exists(nfile): return
    nd = json.load(open(nfile))
    fig, ax = plt.subplots(figsize=(10, 3.6))
    up = nd["unpaired_prob"]; jc = nd["junction_index_in_window"]
    ax.plot(range(len(up)), up, color="#4575b4", label="normal junction (100-nt window)")
    ax.axvline(jc, color="k", ls="--", lw=1)
    ax.axvspan(jc-10, jc, color="#fdae61", alpha=0.4, label="shared exon-1 run (10 nt)")
    if os.path.exists(pfile):
        pdj = json.load(open(pfile)); pup = pdj["unpaired_prob"]; pj = pdj["junction_index_in_window"]
        # align pathological junction at same x as normal junction
        shift = jc - pj
        ax.plot([i+shift for i in range(len(pup))], pup, color="#d73027",
                label="pathological junction (constructed)")
    ax.set_xlabel("position in window"); ax.set_ylabel("per-nucleotide unpaired probability")
    ax.set_ylim(0, 1.05); ax.legend(fontsize=7, loc="lower right")
    ax.set_title("Figure E. Long-context junction accessibility (NUPACK RNA model)\n"
                 "shared exon-1 run highlighted (revision A4)")
    fig.tight_layout(); figstyle.save_both(fig, os.path.join(FIG, "long_context_accessibility"))
    print("Figure E written")

# ---- Figure F: conceptual full-sensor starting model ----
def fig_structure():
    models = sorted(glob.glob(os.path.join(ROOT, "results", "structures", "starting_models",
                                            "GT15_junction20_*_m01.pdb")))
    if not models: return
    cnt, dna = [], []
    with open(models[0]) as fh:
        for l in fh:
            if l.startswith("ATOM"):
                xyz = [float(l[30:38]), float(l[38:46]), float(l[46:54])]
                (cnt if l[21] == "A" else dna).append(xyz)
    cnt = np.array(cnt); dna = np.array(dna)
    fig = plt.figure(figsize=(9, 4))
    ax = fig.add_subplot(111)
    ax.scatter(cnt[:, 2], cnt[:, 0], s=4, c="#555", label="SWCNT (7,5)")
    ax.scatter(dna[:, 2], dna[:, 0], s=6, c="#d73027", label="GT15-junction20 sensor")
    ax.set_xlabel("z (Å, tube axis)"); ax.set_ylabel("x (Å)")
    ax.legend(fontsize=8); ax.set_aspect("equal")
    ax.set_title("Figure F. Representative starting geometry (GT15-junction20 on (7,5))\n"
                 "Builder-placed; NOT energy-minimized or dynamically equilibrated")
    fig.tight_layout(); figstyle.save_both(fig, os.path.join(FIG, "full_sensor_starting_models"))
    print("Figure F written")

# ---- Figure B: docking benchmark ----
def fig_docking_benchmark():
    rows = load_tsv(os.path.join(ROOT, "results", "docking", "benchmark", "pose_metrics.tsv"))
    if not rows: return
    acc = [r for r in rows if r.get("sidewall_pose_pass") == "True"]
    anchors = ["GTd", "CTd", "ATd"]
    data = {a: [float(r["vina_score"]) for r in acc if r["anchor"] == a] for a in anchors}
    data = {a: v for a, v in data.items() if v}
    if not data: return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    ax.violinplot([data[a] for a in data], showmedians=True)
    ax.set_xticks(range(1, len(data)+1)); ax.set_xticklabels(list(data.keys()))
    ax.set_ylabel("controlled Vina score (more negative = more favorable)")
    ax.set_title("Figure B. Anchor benchmark on (7,5)\nbase-stacking score distribution "
                 f"(n={sum(len(v) for v in data.values())} accepted sidewall poses)")
    ax = axes[1]
    base = {a: np.mean([float(r.get("n_base_within_cut", 0) or 0) for r in acc if r["anchor"] == a]) for a in data}
    phos = {a: np.mean([float(r.get("n_phosphate_within_cut", 0) or 0) for r in acc if r["anchor"] == a]) for a in data}
    x = np.arange(len(data))
    ax.bar(x-0.2, [base[a] for a in data], 0.4, label="base contacts", color="#4575b4")
    ax.bar(x+0.2, [phos[a] for a in data], 0.4, label="phosphate contacts", color="#d73027")
    ax.set_xticks(x); ax.set_xticklabels(list(data.keys())); ax.legend(fontsize=8)
    ax.set_ylabel("mean atoms within 4.5 Å of sidewall")
    ax.set_title("Base vs phosphate surface contacts")
    fig.text(0.5, 0.01, "Vina scores are controlled comparative steric/stacking scores, NOT binding energies.",
             ha="center", fontsize=7, style="italic")
    fig.tight_layout(rect=[0, 0.03, 1, 1]); figstyle.save_both(fig, os.path.join(FIG, "docking_benchmark"))
    print("Figure B written")

# ---- Figure C: chirality comparison ----
def fig_chirality():
    rows = load_tsv(os.path.join(ROOT, "results", "docking", "chirality", "pose_metrics.tsv"))
    rank = load_tsv(os.path.join(ROOT, "results", "docking", "chirality", "chirality_ranking.tsv"))
    if not rows or not rank: return
    acc = [r for r in rows if r.get("sidewall_pose_pass") == "True"]
    chirs = [r["chirality"] for r in rank]
    data = {c: [float(r["vina_score"]) for r in acc if r["chirality"] == c] for c in chirs}
    data = {c: v for c, v in data.items() if v}
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.violinplot([data[c] for c in data], showmedians=True)
    for i, c in enumerate(data, 1):
        rr = next((x for x in rank if x["chirality"] == c), {})
        ax.errorbar(i, float(rr["median_score"]),
                    yerr=[[float(rr["median_score"])-float(rr["ci_lo"])],
                          [float(rr["ci_hi"])-float(rr["median_score"])]],
                    fmt="o", color="k", capsize=4)
        ax.text(i, max(data[c])+0.2, f"n={len(data[c])}", ha="center", fontsize=7)
    ax.set_xticks(range(1, len(data)+1))
    ax.set_xticklabels([c.replace("_", ",") for c in data])
    ax.set_ylabel("controlled Vina score (more negative = more favorable)")
    status = rank[0].get("benchmark_passed", "False")
    ax.set_title(f"Figure C. GT base-unit on chiralities (controlled screen)\n"
                 f"distributions + bootstrap 95% CI; benchmark_passed={status} "
                 f"(EXPLORATORY if False)")
    fig.text(0.5, 0.01, "Controlled comparative base-stacking scores, NOT binding energies; "
             "not an optical-readout ranking.", ha="center", fontsize=7, style="italic")
    fig.tight_layout(rect=[0, 0.03, 1, 1]); figstyle.save_both(fig, os.path.join(FIG, "chirality_docking"))
    print("Figure C written")

if __name__ == "__main__":
    seqs = load_seqs()
    fig_sequence_architecture(seqs)
    fig_hybridization()
    fig_accessibility()
    fig_structure()
    fig_docking_benchmark()
    fig_chirality()

"""Publication-quality volcano plots for the ALS vs CTRL MS experiments.

Reads the *_ALS_vs_CTRL_medianNormalized_welch.xlsx 'Differential_all' sheets and
renders one volcano per experiment plus a combined 3-panel figure.
"""
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D

BASE = "/mnt/c/Users/riccardo-s/Documents/CNT/targetALS/MS"
OUT = BASE

# --- palette (diverging polarity: blue = down in ALS, red = up in ALS) --------
C_UP = "#e34948"
C_DOWN = "#2a78d6"
C_NS = "#c3c2b7"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

FDR_CUT = 0.05
FC_CUT = 1.0          # |log2FC| >= 1  (2-fold)
N_LABEL = 25
GENE_TSV = os.path.join(BASE, "uniprot_gene_symbols.tsv")   # from fetch_gene_symbols.py

# Entries matching these description terms are never called out with a label.
# They stay in the plot and in the counts; they are simply not named. Samples are
# serum protein corona, so RBC-, platelet- and clotting-derived proteins are
# prep artifacts rather than corona signal.
NO_LABEL_CATEGORIES = {
    "epithelial/skin": (
        r"keratin|filaggrin|hornerin|desmoglein|desmocollin|corneodesmosin|"
        r"cornulin|involucrin|loricrin|trichohyalin"),
    # 'erythrocyt' already covers the erythrocytic spectrins, so bare 'spectrin'
    # is left out — it would catch spectrin-repeat proteins such as Nesprin-1
    "hemolysis/RBC": (
        r"h[ae]moglobin|erythrocyt|erythroid|glycophorin|ankyrin-1|"
        r"band 3|anion exchange protein 1|bisphosphoglycerate mutase|"
        r"carbonic anhydrase 1\b|peroxiredoxin-2|biliverdin reductase B|"
        r"flavin reductase|\bstomatin(?!-like)"),
    "platelet": (
        r"platelet|thrombospondin|von Willebrand factor(?!\s+type)|multimerin-1|"
        r"thromboglobulin|integrin alpha-IIb|selectin P|P-selectin"),
    # \b before 'plasmin' keeps ceruloplasmin and endoplasmin out
    "coagulation": (
        r"fibrinogen(?!-like)|fibrin\b|prothrombin|thrombin|coagulation factor|"
        r"antithrombin|\bplasminogen\b|\bplasmin\b|antiplasmin|kininogen|"
        r"kallikrein|vitamin K-dependent protein|heparin cofactor|"
        r"thrombomodulin|tissue factor"),
    # reagents and handling contaminants; the boundary and look-ahead keep the
    # genuine serum serpins (alpha-1-antitrypsin, alpha-1-antichymotrypsin,
    # inter-alpha-trypsin inhibitor)
    "reagent/handling": (
        r"\btrypsin(?!\w*\s*inhibitor)|dermcidin|casein|streptavidin"),
}
NO_LABEL_TERMS = re.compile("|".join(NO_LABEL_CATEGORIES.values()), re.I)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth": 0.8,
})


_gene = pd.read_csv(GENE_TSV, sep="\t")
GENE = dict(zip(_gene["accession"], _gene["gene"].fillna("")))
REVIEWED = dict(zip(_gene["accession"], _gene["reviewed"] == "reviewed"))


def symbol(groups, desc):
    """Gene symbol(s) for a protein group; falls back to the description.

    A group can carry several accessions: prefer the reviewed (Swiss-Prot) ones,
    then take the distinct symbols in that order (at most two, joined by '/').
    """
    accs = [a.strip() for a in str(groups).split(";") if a.strip()]
    accs.sort(key=lambda a: not REVIEWED.get(a, False))       # reviewed first
    syms = []
    for a in accs:
        s = GENE.get(a, "")
        if s and s not in syms:
            syms.append(s)
    if not syms:
        # no gene assigned (unreviewed fragments): the accession is the identifier
        return accs[0] if accs else short_name(desc, groups)
    return "/".join(syms[:2]) + ("…" if len(syms) > 2 else "")


def short_name(desc, groups, maxlen=34):
    """One readable protein name from the ';'-joined description list."""
    if not isinstance(desc, str) or not desc.strip():
        return str(groups).split(";")[0]
    name = desc.split(";")[0].strip()
    name = re.sub(r"\s*\(Fragment\)\s*$", "", name)
    # "Keratin, type I cytoskeletal 9" -> keep; just cap length on a word break
    if len(name) > maxlen:
        cut = name[:maxlen].rsplit(" ", 1)[0]
        name = (cut if len(cut) > maxlen * 0.6 else name[:maxlen]).rstrip(",") + "…"
    return name


def load(exp):
    path = os.path.join(BASE, f"{exp}_ALS_vs_CTRL_medianNormalized_welch.xlsx")
    d = pd.read_excel(path, sheet_name="Differential_all")
    d = d.dropna(subset=["log2FC_ALS_vs_CTRL", "p_value"]).copy()
    d["nlp"] = -np.log10(d["p_value"].clip(lower=np.nextafter(0, 1)))
    d["label"] = [symbol(g, desc) for g, desc in
                  zip(d["PG.ProteinGroups"], d["PG.ProteinDescriptions"])]
    d["sig"] = (d["FDR_BH"] < FDR_CUT) & (d["log2FC_ALS_vs_CTRL"].abs() >= FC_CUT)
    d["dirn"] = np.where(d["log2FC_ALS_vs_CTRL"] > 0, "up", "down")
    d["labelable"] = ~d["PG.ProteinDescriptions"].fillna("").str.contains(NO_LABEL_TERMS)
    return d


def fdr_line(d):
    """p-value at which BH-FDR crosses the cutoff (None if nothing passes)."""
    passing = d.loc[d["FDR_BH"] < FDR_CUT, "p_value"]
    return float(passing.max()) if len(passing) else None


def place_labels(ax, fig, rows, xlim, ylim):
    """Non-overlapping labels with leader lines.

    Labels are pinned to the side of the plot their point sits on and stacked
    vertically; overlaps are resolved in y only, which converges (a free 2-D
    repulsion oscillates and leaves residual collisions).
    """
    if not len(rows):
        return
    anns = []
    for _, r in rows.iterrows():
        anns.append(ax.annotate(
            r["label"], xy=(r["log2FC_ALS_vs_CTRL"], r["nlp"]),
            xytext=(0, 0), textcoords="offset points",
            ha="center", va="center", fontsize=8.0, color=INK,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6,
                            shrinkA=0, shrinkB=3.0),
            annotation_clip=False, zorder=6,
            # halo keeps a label legible where it passes over a mark
            path_effects=[pe.withStroke(linewidth=2.2, foreground=SURFACE)],
        ))
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    ax_bb = ax.get_window_extent()

    # fixed text half-extents (px) and anchor positions (px)
    hw = np.array([a.get_window_extent(rend).width / 2 for a in anns])
    hh = np.array([a.get_window_extent(rend).height / 2 for a in anns])
    anchor = ax.transData.transform(
        np.column_stack([rows["log2FC_ALS_vs_CTRL"], rows["nlp"]]))
    ax_px, ay_px = anchor[:, 0], anchor[:, 1]
    # place the label outward (away from x=0), but flip inward when the outward
    # side has no room — otherwise the clamp drags the text over its own point
    gap = 18.0                       # leader-line length, px
    side = np.where(rows["log2FC_ALS_vs_CTRL"].to_numpy() >= 0, 1.0, -1.0)
    room_out = np.where(side > 0, ax_bb.x1 - ax_px, ax_px - ax_bb.x0)
    side = np.where(room_out < 2 * hw + gap, -side, side)

    cx = ax_px + side * (hw + gap)   # preferred: just outside the point
    cy = ay_px + 10.0
    padx, pady = 6.0, 2.5

    for _ in range(600):
        moved = False
        for i in range(len(anns)):
            for j in range(i + 1, len(anns)):
                ox = (hw[i] + hw[j] + padx) - abs(cx[i] - cx[j])
                oy = (hh[i] + hh[j] + pady) - abs(cy[i] - cy[j])
                if ox > 0 and oy > 0:                      # overlapping
                    push = oy / 2 + 0.01
                    if cy[i] >= cy[j]:
                        cy[i] += push; cy[j] -= push
                    else:
                        cy[i] -= push; cy[j] += push
                    moved = True
        # weak spring back toward the anchor's height, then clamp to the axes
        cy += 0.06 * ((ay_px + 10.0) - cy)
        cx = np.clip(cx, ax_bb.x0 + hw + 2, ax_bb.x1 - hw - 2)
        cy = np.clip(cy, ax_bb.y0 + hh + 2, ax_bb.y1 - hh - 2)
        if not moved:
            break

    # a label must never sit on top of the point it names: if the clamp pulled it
    # over its own anchor, lift it clear so the leader line reads as a leader line
    for i in range(len(anns)):
        if abs(cx[i] - ax_px[i]) < hw[i] + 4:
            up = 1.0 if cy[i] >= ay_px[i] else -1.0
            cy[i] = ay_px[i] + up * (hh[i] + 9)

    for i, a in enumerate(anns):
        right = cx[i] >= ax_px[i]
        a.set_ha("left" if right else "right")
        edge = cx[i] - hw[i] if right else cx[i] + hw[i]
        a.set_position(((edge - ax_px[i]) * 72 / fig.dpi,
                        (cy[i] - ay_px[i]) * 72 / fig.dpi))


def volcano(ax, d, title, fig, label=True, n_label=N_LABEL, xlim=None, ylim=None):
    p_fdr = fdr_line(d)
    n_up = int((d["sig"] & (d["dirn"] == "up")).sum())
    n_dn = int((d["sig"] & (d["dirn"] == "down")).sum())

    if xlim is None:
        xmax = np.ceil(np.abs(d["log2FC_ALS_vs_CTRL"]).max() * 1.06)
        xlim = (-xmax, xmax)
    if ylim is None:
        ylim = (-0.12, d["nlp"].max() * 1.12)
    ymax = ylim[1]

    ax.set_axisbelow(True)
    ax.grid(True, color=GRID, lw=0.6)
    ax.axvline(0, color=AXIS, lw=0.8, zorder=1)
    for v in (-FC_CUT, FC_CUT):
        ax.axvline(v, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
    if p_fdr is not None:
        y = -np.log10(p_fdr)
        ax.axhline(y, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
        ax.text(xlim[1] - 0.08, y + ymax * 0.012, f"FDR {FDR_CUT:g}",
                ha="right", va="bottom", fontsize=7, color=MUTED)

    ns = d[~d["sig"]]
    ax.scatter(ns["log2FC_ALS_vs_CTRL"], ns["nlp"], s=16, c=C_NS,
               linewidths=0, alpha=0.75, zorder=2, rasterized=True)
    for dirn, col in (("down", C_DOWN), ("up", C_UP)):
        s = d[d["sig"] & (d["dirn"] == dirn)]
        ax.scatter(s["log2FC_ALS_vs_CTRL"], s["nlp"], s=34, c=col,
                   linewidths=0.7, edgecolors=SURFACE, zorder=4)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("log$_2$ fold change (ALS / CTRL)", fontsize=9.5, color=INK_2)
    ax.set_ylabel("−log$_{10}$ $p$", fontsize=9.5, color=INK_2)
    ax.set_title(title, fontsize=11, color=INK, loc="left", pad=10)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    ax.text(0.015, 0.985, f"▼ {n_dn} down in ALS", transform=ax.transAxes,
            ha="left", va="top", fontsize=8, color=C_DOWN, weight="bold")
    ax.text(0.985, 0.985, f"▲ {n_up} up in ALS", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color=C_UP, weight="bold")

    if label:
        sig = d[d["sig"] & d["labelable"]].copy()
        if len(sig):
            sig["rank"] = sig["nlp"] * sig["log2FC_ALS_vs_CTRL"].abs()
            top = sig.nlargest(min(n_label, len(sig)), "rank")
            place_labels(ax, fig, top, xlim, ylim)
    return n_up, n_dn, p_fdr


def volcano_fdr10(ax, d, title, fig, fdr_relaxed=0.10):
    """Volcano marking the whole relaxed-FDR set, with the strict tier filled.

    Ring  = FDR < fdr_relaxed (any fold change)
    Fill  = FDR < FDR_CUT and |log2FC| >= FC_CUT  (the strict tier)
    """
    d = d.copy()
    d["rel"] = d["FDR_BH"] < fdr_relaxed
    p_strict = fdr_line(d)
    p_rel = d.loc[d["rel"], "p_value"].max() if d["rel"].any() else None

    xmax = np.ceil(np.abs(d["log2FC_ALS_vs_CTRL"]).max() * 1.06)
    ylim = (-0.12, d["nlp"].max() * 1.12)
    xlim, ymax = (-xmax, xmax), ylim[1]

    ax.set_axisbelow(True)
    ax.grid(True, color=GRID, lw=0.6)
    ax.axvline(0, color=AXIS, lw=0.8, zorder=1)
    for v in (-FC_CUT, FC_CUT):
        ax.axvline(v, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
    for p_line, lab in ((p_strict, f"FDR {FDR_CUT:g}"), (p_rel, f"FDR {fdr_relaxed:g}")):
        if p_line is not None:
            y = -np.log10(p_line)
            ax.axhline(y, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
            ax.text(xlim[1] - 0.08, y + ymax * 0.012, lab,
                    ha="right", va="bottom", fontsize=7, color=MUTED)

    ns = d[~d["rel"] & ~d["sig"]]
    ax.scatter(ns["log2FC_ALS_vs_CTRL"], ns["nlp"], s=16, c=C_NS,
               linewidths=0, alpha=0.75, zorder=2, rasterized=True)
    for dirn, col in (("down", C_DOWN), ("up", C_UP)):
        ring = d[d["rel"] & ~d["sig"] & (d["dirn"] == dirn)]
        ax.scatter(ring["log2FC_ALS_vs_CTRL"], ring["nlp"], s=26,
                   facecolors=SURFACE, edgecolors=col, linewidths=1.1, zorder=3)
        fill = d[d["sig"] & (d["dirn"] == dirn)]
        ax.scatter(fill["log2FC_ALS_vs_CTRL"], fill["nlp"], s=34, c=col,
                   linewidths=0.7, edgecolors=SURFACE, zorder=4)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("log$_2$ fold change (ALS / CTRL)", fontsize=9.5, color=INK_2)
    ax.set_ylabel("−log$_{10}$ $p$", fontsize=9.5, color=INK_2)
    ax.set_title(title, fontsize=11, color=INK, loc="left", pad=10)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    n_dn = int((d["rel"] & (d["dirn"] == "down")).sum())
    n_up = int((d["rel"] & (d["dirn"] == "up")).sum())
    ax.text(0.015, 0.985, f"▼ {n_dn} down at FDR < {fdr_relaxed:g}",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=8, color=C_DOWN, weight="bold")
    ax.text(0.985, 0.985, f"▲ {n_up} up at FDR < {fdr_relaxed:g}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color=C_UP, weight="bold")

    def dot(col, fill):
        return Line2D([], [], marker="o", ls="", ms=6,
                      mfc=col if fill else SURFACE,
                      mec=SURFACE if fill else col, mew=0.7 if fill else 1.1)

    # paired swatches: each row shows both directions, so the key matches the plot
    handles = [(dot(C_DOWN, True), dot(C_UP, True)),
               (dot(C_DOWN, False), dot(C_UP, False)),
               Line2D([], [], marker="o", ls="", ms=5, mfc=C_NS, mec="none")]
    labels = [f"FDR < {FDR_CUT:g} and |log$_2$FC| ≥ {FC_CUT:g}",
              f"FDR < {fdr_relaxed:g} only",
              "not significant"]
    ax.legend(handles, labels, loc="lower left", frameon=False, fontsize=7.5,
              labelcolor=INK_2, handletextpad=0.7, borderpad=0.2,
              handler_map={tuple: HandlerTuple(ndivide=None, pad=0.5)})

    # label the relaxed set that also clears the fold-change cut
    lab = d[d["rel"] & (d["log2FC_ALS_vs_CTRL"].abs() >= FC_CUT) & d["labelable"]].copy()
    if len(lab):
        lab["rank"] = lab["nlp"] * lab["log2FC_ALS_vs_CTRL"].abs()
        place_labels(ax, fig, lab.nlargest(min(26, len(lab)), "rank"), xlim, ylim)
    return n_up, n_dn


TITLES = {
    "ex1": "Experiment 1 — ALS vs CTRL  (3 vs 3)",
    "ex2": "Experiment 2 — ALS vs CTRL  (3 vs 3)",
    "ex3": "Experiment 3 — ALS vs CTRL  (3 vs 3)",
}

summary = []
data = {}
for exp in ("ex1", "ex2", "ex3"):
    d = load(exp)
    data[exp] = d
    fig, ax = plt.subplots(figsize=(7.6, 6.2), dpi=300)
    nu, nd, pf = volcano(ax, d, TITLES[exp], fig)
    ax.text(0.0, -0.115,
            f"{len(d):,} protein groups · significant = BH-FDR < {FDR_CUT:g} "
            f"and |log$_2$FC| ≥ {FC_CUT:g}"
            + ("" if pf is not None else "  ·  no protein reaches FDR < 0.05"),
            transform=ax.transAxes, ha="left", va="top", fontsize=7.5, color=MUTED)
    fig.tight_layout()
    out = os.path.join(OUT, f"{exp}_volcano_ALS_vs_CTRL.png")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    summary.append((exp, len(d), nd, nu, pf))
    print(f"{exp}: {len(d)} groups, {nd} down, {nu} up, FDR0.05 p-cut={pf}")

def benjamini_hochberg(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1].clip(max=1)
    out = np.empty(n)
    out[order] = q
    return out


def render_fdr10(d, out_name, note):
    fig, ax = plt.subplots(figsize=(7.6, 6.2), dpi=300)
    nu, nd = volcano_fdr10(ax, d, "Experiment 3 — ALS vs CTRL  (3 vs 3)", fig)
    n_both = int(((d["FDR_BH"] < 0.10) & (d["log2FC_ALS_vs_CTRL"].abs() >= FC_CUT)).sum())
    ax.text(0.0, -0.115,
            f"{len(d):,} protein groups · {nu + nd} at BH-FDR < 0.10, of which "
            f"{n_both} also reach |log$_2$FC| ≥ {FC_CUT:g}"
            "  ·  red = up in ALS, blue = down in ALS\n" + note,
            transform=ax.transAxes, ha="left", va="top", fontsize=7.5, color=MUTED)
    fig.tight_layout()
    out = os.path.join(OUT, out_name)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"{out_name}: {nd} down, {nu} up ({n_both} also |log2FC|>=1)")


d3 = data["ex3"]

# (a) all proteins, FDR as computed on the full set
render_fdr10(d3, "ex3_volcano_ALS_vs_CTRL_FDR0.10_unfiltered.png",
             "All protein groups retained; BH computed over the full set.")

# (b) flagged categories dropped BEFORE the BH correction, so the FDR of every
#     surviving protein is recomputed over the smaller set of tests
d3f = d3[d3["labelable"]].copy()
d3f["FDR_BH"] = benjamini_hochberg(d3f["p_value"].values)
d3f["sig"] = (d3f["FDR_BH"] < FDR_CUT) & (d3f["log2FC_ALS_vs_CTRL"].abs() >= FC_CUT)
n_drop = len(d3) - len(d3f)
render_fdr10(d3f, "ex3_volcano_ALS_vs_CTRL_FDR0.10.png",
             f"{n_drop} flagged protein groups (epithelial/skin, hemolysis/RBC, "
             f"platelet, coagulation, reagent) removed before BH; FDR recomputed "
             f"on the remaining {len(d3f):,} tests.")

# combined 3-panel — small multiples share both scales so the panels compare
gx = np.ceil(max(np.abs(d["log2FC_ALS_vs_CTRL"]).max() for d in data.values()) * 1.06)
gy = max(d["nlp"].max() for d in data.values()) * 1.12
fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.6), dpi=300, sharex=True, sharey=True)
for ax, exp in zip(axes, ("ex1", "ex2", "ex3")):
    volcano(ax, data[exp], TITLES[exp], fig, label=True, n_label=12,
            xlim=(-gx, gx), ylim=(-0.12, gy))
handles = [
    Line2D([], [], marker="o", ls="", ms=6, mfc=C_UP, mec=SURFACE, label="Up in ALS"),
    Line2D([], [], marker="o", ls="", ms=6, mfc=C_DOWN, mec=SURFACE, label="Down in ALS"),
    Line2D([], [], marker="o", ls="", ms=5, mfc=C_NS, mec="none", label="Not significant"),
]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
           fontsize=9, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("ALS vs CTRL differential protein abundance — median-normalized log$_2$ "
             "PG.Quantity, Welch $t$-test, BH-FDR",
             fontsize=12, x=0.008, ha="left", y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "volcano_all_experiments.png"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "volcano_all_experiments.pdf"), bbox_inches="tight")
plt.close(fig)
print("\n".join(str(s) for s in summary))

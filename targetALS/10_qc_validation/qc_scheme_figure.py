"""
Schematic summary of the three-tier PBS-anchored QC framework, with the proposed
thresholds, as a publication figure.

Each tier is a panel; each row is one QC metric with (a) what is computed,
(b) what a pass asserts about the assay, (c) the proposed gate. The left rail of
each panel encodes which control wells the tier requires, so the dependency
(PBS alone -> PBS+FBS -> PBS+samples) is readable at a glance. Arrows between
panels carry the reason the cascade is not redundant.

Outputs: FigSX_qc_scheme.png / .pdf
"""
import os, textwrap
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))

# Okabe-Ito, matching the rest of the figure set
BLUE, VERM, GREEN, GREY = "#0072B2", "#D55E00", "#009E73", "#8a8a8a"
INK, MUT = "#1b1f24", "#5c6673"
PANEL, HEADBG, RULE = "#FFFFFF", "#EEF3F9", "#D5DCE5"

# ---------------------------------------------------------------- content
TIERS = [
    dict(num="TIER 1", name="Sensor QC", wells=["PBS"],
         q="Is the nanotube sensor the one the reference was built on, and did it behave?",
         rows=[
      ("QC1", "Brightness", "log$_2$(I / I$_{ref}$)",
       "Nanotubes present at the intended concentration; the well is live and the "
       "excitation path is intact.",
       "|log$_2$| $\\leq$ 1"),
      ("QC2", "Fingerprint fidelity", "r vs s$_{ref}$ ; max |$\\Delta$s|",
       "Material unchanged from reference — same lot, still individualised, not "
       "bundled or aggregated. This is what lets runs be compared at all.",
       "r $\\geq$ 0.99\nmax |$\\Delta$s| $\\leq$ 0.15"),
      ("QC3", "Peak-centre accuracy", "RMS |$\\lambda_{em}$ − ref|",
       "Each peak is the species it is labelled as, and the wavelength axis has not moved.",
       "$\\leq$ 2 nm"),
      ("QC4", "Replicate precision", "%CV, raw and share",
       "Plate reproducibility well-to-well, and how much of that spread is pure "
       "common-mode scaling rather than real difference.",
       "share %CV $\\leq$ 10%"),
      ("QC5", "Drift magnitude", "I(24 h) / I(0 h)",
       "The sensor stayed photophysically stable, and its drift kept the normal "
       "diameter structure.",
       "0.5 – 2.0\nr(drift, d) $\\approx$ +0.8"),
      ("QC6", "Read stability", "leave-one-out spread\nby timepoint",
       "The plate had equilibrated before the first read. Condition-independent excess "
       "spread is read order, not protein.",
       "no 0 h excess"),
    ]),
    dict(num="TIER 2", name="Assay window", wells=["PBS", "FBS"],
         q="Even with a well-behaved sensor, can this run resolve corona from no corona?",
         rows=[
      ("W1", "Corona index", "CI = s $\\cdot$ unit(s$_{FBS}$−s$_{PBS}$)",
       "The run has a usable signal window: buffer and serum land in separable places on "
       "the assay's own intended readout. Single release number.",
       "|SSMD| $\\geq$ 2"),
      ("W2", "Per-chirality SSMD", "12 $\\times$ per timepoint",
       "Which individual channels are doing the separating, and at which timepoint.",
       "diagnostic"),
      ("W3", "Control variance ratio", "SD(FBS) / SD(PBS)",
       "Whether the spread is instrument imprecision or genuine protein heterogeneity. "
       "A high ratio is why Z$'$ is inapplicable and SSMD is reported instead.",
       "diagnostic\n(Z$'$ invalid if $\\gg$1)"),
    ]),
    dict(num="TIER 3", name="Detection limit & sample QC", wells=["PBS", "patient"],
         q="How wide is the multiplex at a given read, and which wells can be trusted?",
         rows=[
      ("L1", "Channel detection limit", "LOD = 3 $\\cdot$ SD(PBS)",
       "The effective multiplex width at this timepoint. Usability is a property of "
       "chirality $\\times$ timepoint — twelve is an upper bound, not the width at a given read.",
       "|effect| > LOD"),
      ("S1", "Sample outlier screen", "robust z ; brightness z ;\nr vs cohort",
       "An individual patient well is technically sound. Scored against the cohort "
       "profile, not PBS, since patient wells legitimately contain serum.",
       "z $\\leq$ 6\nr $\\geq$ 0.95"),
      ("S2", "Diagnosis blindness", "flag rate by class",
       "QC is neutral to the biological question — not discarding one class "
       "preferentially and manufacturing the separation being measured.",
       "Fisher p > 0.05"),
    ]),
]
GATES = [
    "Precision is not discrimination: a sensor that responds identically to buffer and "
    "serum passes Tier 1 completely.",
    "A whole-plate signal window does not mean all twelve channels carry it — "
    "channels must be counted one at a time against the buffer noise.",
]
WELLC = {"PBS": BLUE, "FBS": VERM, "patient": GREY}

# ---------------------------------------------------------------- layout
# column edges in axis units (0-100 across)
X0, X1 = 1.5, 98.5
RAIL = 9.0                      # left rail width
CID   = X0 + RAIL + 1.2         # id chip
CMET  = CID + 5.0               # metric name/formula
CTELL = CMET + 22.5             # what it tells you
CGATE = X1 - 17.0               # threshold
W_MET, W_TELL, W_GATE = 21.0, CGATE - CTELL - 2.0, 16.0

def wrap(s, width_units, fs):
    """wrap to a column width given in axis units (100 units = fig width)."""
    chars = max(8, int(width_units / (fs * 0.0680)))
    return "\n".join(textwrap.fill(p, chars) for p in s.split("\n"))

FS_TELL, FS_MET, FS_GATE = 8.4, 8.6, 8.2
LINE = 1.62                      # axis units per text line
PAD_ROW = 1.55                   # vertical padding inside a row

# pre-wrap and measure
for t in TIERS:
    t['wrapped'] = []
    for rid, name, form, tell, gate in t['rows']:
        wt = wrap(tell, W_TELL, FS_TELL)
        wn = wrap(name, W_MET, FS_MET)
        nl = max(wt.count("\n")+1,
                 wn.count("\n")+1 + form.count("\n")+1,
                 gate.count("\n")+1)
        t['wrapped'].append((rid, wn, form, wt, gate, nl))

HEAD_H, QLINE_H, GATE_H = 5.6, 3.2, 5.4
total = 14.9                                        # title + principle band
for i, t in enumerate(TIERS):
    total += HEAD_H + QLINE_H + sum(r[5]*LINE + PAD_ROW for r in t['wrapped']) + 2.0
    if i < len(GATES):
        total += GATE_H
total += 4.0                                        # footer

FIG_W = 11.0
FIG_H = FIG_W * total / 100.0 * 1.02
fig = plt.figure(figsize=(FIG_W, FIG_H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(total, 0); ax.axis("off")

def T(x, y, s, fs=8.4, w="normal", c=INK, ha="left", va="top", fam="DejaVu Sans", **kw):
    return ax.text(x, y, s, fontsize=fs, fontweight=w, color=c, ha=ha, va=va,
                   family=fam, linespacing=1.42, **kw)

# ---------------------------------------------------------------- title
y = 2.0
T(X0, y, "Quality control for the SWCNT protein-corona assay", fs=15.5, w="bold")
y += 3.3
T(X0, y, "Three tiers, each gated on the one below it, all anchored on the PBS negative control.",
  fs=9.6, c=MUT)
y += 3.4

# principle band
PRIN_H = 7.4
ax.add_patch(FancyBboxPatch((X0, y), X1-X0, PRIN_H, boxstyle="round,pad=0,rounding_size=0.35",
                            fc=HEADBG, ec=BLUE, lw=1.3))
T(X0+1.4, y+1.15, "PRINCIPLE", fs=7.6, w="bold", c=BLUE)
T(X0+1.4, y+2.75,
  "Divide each well's 12 peak intensities by their own mean $\\rightarrow$ scale-free relative-share "
  "fingerprint.\nConcentration, laser power and drift move all peaks together and are discarded;\n"
  "aggregation and material identity move them relative to one another and are kept.",
  fs=8.7)
T(X1-1.4, y+1.15, "PBS well-to-well %CV:  15.9  $\\rightarrow$  3.0", fs=8.7, w="bold",
  c=GREEN, ha="right")
y += PRIN_H + 2.6

# ---------------------------------------------------------------- tiers
for ti, t in enumerate(TIERS):
    rows_h = sum(r[5]*LINE + PAD_ROW for r in t['wrapped'])
    panel_h = HEAD_H + QLINE_H + rows_h + 1.2
    ax.add_patch(FancyBboxPatch((X0, y), X1-X0, panel_h,
                                boxstyle="round,pad=0,rounding_size=0.35",
                                fc=PANEL, ec=RULE, lw=1.1, zorder=1))
    # left rail
    ax.add_patch(FancyBboxPatch((X0, y), RAIL, panel_h,
                                boxstyle="round,pad=0,rounding_size=0.35",
                                fc=HEADBG, ec="none", zorder=2))
    ax.add_patch(plt.Rectangle((X0+RAIL-0.06, y), 0.06, panel_h, fc=RULE, ec="none", zorder=3))
    # header strip
    ax.add_patch(plt.Rectangle((X0+RAIL, y), X1-X0-RAIL, HEAD_H, fc=HEADBG, ec="none", zorder=2))
    ax.add_patch(plt.Rectangle((X0+RAIL, y+HEAD_H-0.06), X1-X0-RAIL, 0.06, fc=RULE, ec="none", zorder=3))

    T(X0+RAIL/2, y+1.35, t['num'], fs=8.0, w="bold", c=BLUE, ha="center",
      fam="DejaVu Sans Mono", zorder=4)
    T(X0+RAIL+1.5, y+1.5, t['name'], fs=12.2, w="bold", zorder=4)

    # required wells, as dots on the rail
    wy = y + 3.6
    for w in t['wells']:
        ax.add_patch(Circle((X0+2.0, wy+0.42), 0.52, fc=WELLC[w], ec="none", zorder=4))
        T(X0+3.1, wy, w, fs=7.5, c=MUT, zorder=4)
        wy += 2.1
    T(X1-1.5, y+2.0, "requires: " + " + ".join(t['wells']), fs=8.0, c=MUT, ha="right", zorder=4)

    yy = y + HEAD_H + 0.9
    T(X0+RAIL+1.5, yy, t['q'], fs=8.7, c=MUT, style="italic", zorder=4)
    yy += QLINE_H

    for k, (rid, wn, form, wt, gate, nl) in enumerate(t['wrapped']):
        h = nl*LINE + PAD_ROW
        if k:
            ax.add_patch(plt.Rectangle((X0+RAIL+1.2, yy-0.35), X1-X0-RAIL-2.6, 0.045,
                                       fc=RULE, ec="none", zorder=3))
        ax.add_patch(FancyBboxPatch((CID-0.55, yy+0.12), 4.0, 1.75,
                                    boxstyle="round,pad=0,rounding_size=0.28",
                                    fc="#E8EDF3", ec="#BCC7D4", lw=0.8, zorder=3))
        T(CID+1.45, yy+0.42, rid, fs=7.6, w="bold", ha="center", fam="DejaVu Sans Mono", zorder=4)
        T(CMET, yy+0.25, wn, fs=FS_MET, w="bold", zorder=4)
        T(CMET, yy+0.25+(wn.count("\n")+1)*LINE, form, fs=7.5, c=MUT,
          fam="DejaVu Sans Mono", zorder=4)
        T(CTELL, yy+0.25, wt, fs=FS_TELL, c="#3d4650", zorder=4)
        gc = GREEN if gate.startswith("diagnostic") is False else MUT
        T(CGATE, yy+0.25, gate, fs=FS_GATE, w="bold", c=gc, fam="DejaVu Sans Mono", zorder=4)
        yy += h

    y += panel_h

    if ti < len(GATES):
        ax.add_patch(FancyArrowPatch((X0+RAIL/2, y+0.5), (X0+RAIL/2, y+GATE_H-0.6),
                                     arrowstyle="-|>", mutation_scale=13,
                                     color=BLUE, lw=1.5, zorder=5))
        T(X0+RAIL+1.5, y+1.1, wrap(GATES[ti], X1-X0-RAIL-4, 8.6), fs=8.6, c=INK, zorder=4)
        y += GATE_H

# ---------------------------------------------------------------- footer
y += 0.6
ax.add_patch(plt.Rectangle((X0, y), X1-X0, 0.045, fc=RULE, ec="none"))
T(X0, y+0.9,
  "Thresholds derived from replicate PBS wells in the exp4 reference set (5 PBS / 5 FBS / 5 FBS+HEP, 0–24 h); "
  "validated on the exp5 patient batch.\nQC2 requires the fitted amplitude (gauss_max) and a timepoint-matched "
  "reference. Code: qc_metrics.py, qc_apply_exp5.py",
  fs=7.6, c=MUT)

fig.savefig(os.path.join(HERE, "FigSX_qc_scheme.png"), dpi=300, bbox_inches="tight",
            facecolor="white")
fig.savefig(os.path.join(HERE, "FigSX_qc_scheme.pdf"), bbox_inches="tight", facecolor="white")
print(f"Saved FigSX_qc_scheme.png/.pdf  ({FIG_W:.1f} x {FIG_H:.1f} in)")

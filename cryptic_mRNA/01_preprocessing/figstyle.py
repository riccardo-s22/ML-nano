"""Shared figure style (§13): 300 dpi, editable SVG text, no rainbow colormap."""
import matplotlib as mpl

def apply_style():
    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300,
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "image.cmap": "viridis", "svg.fonttype": "none",
        "axes.spines.top": False, "axes.spines.right": False,
    })

def save_both(fig, stem):
    fig.savefig(f"{stem}.svg")
    fig.savefig(f"{stem}.png", dpi=300)
    import matplotlib.pyplot as plt
    plt.close(fig)

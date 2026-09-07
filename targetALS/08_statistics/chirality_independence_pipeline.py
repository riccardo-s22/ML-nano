import os
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.covariance import GraphicalLassoCV
import networkx as nx


# =========================
# USER SETTINGS
# =========================
DATA_DIR = r"C:\Users\riccardo-s\Documents\CNT\targetALS\exp5\multiplexing_results"
FILES = {
    "0h":  "features_max5x5_0h.csv",
    "6h":  "features_max5x5_6h.csv",
    "24h": "features_max5x5_24h.csv",
}
OUT_DIR = os.path.join(DATA_DIR, "chirality_independence_out")
os.makedirs(OUT_DIR, exist_ok=True)

# Module cutoff for hierarchical clustering (|r| >= cutoff)
MODULE_CUTOFF_ABS_R = 0.6

# Graphical lasso edge display thresholds
GLASSO_THRESHOLDS = [0.15, 0.25, 0.30]

# Redundancy thresholds for |r| fractions
REDUND_THRESHOLDS = [0.6, 0.7, 0.8, 0.9]

RANDOM_SEED_LAYOUT = 1


# =========================
# HELPERS
# =========================
def load_features(path):
    df = pd.read_csv(path)
    if "sample_id" not in df.columns:
        raise ValueError(f"Missing sample_id in {path}")
    chir_cols = [c for c in df.columns if c != "sample_id"]
    return df, chir_cols

def corr_mat(X):
    return np.corrcoef(X, rowvar=False)

def offdiag_abs(R):
    iu = np.triu_indices(R.shape[0], k=1)
    return np.abs(R[iu])

def plot_dendrogram(Z, labels, title, outpath):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    dendrogram(Z, labels=labels, leaf_rotation=45, leaf_font_size=9, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)

def plot_clustered_heatmap(R, labels, Z, title, outpath):
    d = dendrogram(Z, labels=labels, no_plot=True)
    order = d["leaves"]
    R_ord = R[np.ix_(order, order)]
    labels_ord = [labels[i] for i in order]

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(R_ord, vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(len(labels_ord)))
    ax.set_yticks(range(len(labels_ord)))
    ax.set_xticklabels(labels_ord, rotation=45, ha="right")
    ax.set_yticklabels(labels_ord)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)

def partial_corr_from_precision(precision):
    P = precision
    d = np.sqrt(np.diag(P))
    d[d == 0] = np.nan
    pcorr = -P / np.outer(d, d)
    np.fill_diagonal(pcorr, 1.0)
    return pcorr

def plot_glasso_network(edges_df, labels, title, outpath, thresh, seed=1):
    G = nx.Graph()
    for lab in labels:
        G.add_node(lab)

    sel = edges_df[edges_df["abs_partial_corr"] >= thresh]
    for _, r in sel.iterrows():
        G.add_edge(r["chirality_i"], r["chirality_j"], weight=r["partial_corr"])

    pos = nx.spring_layout(G, seed=seed)

    fig, ax = plt.subplots(figsize=(7.8, 7.0))
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=900, alpha=0.9)

    edgelist = list(G.edges(data=True))
    widths = [2.0 + 6.0 * abs(d["weight"]) for (_, _, d) in edgelist] if edgelist else []
    nx.draw_networkx_edges(G, pos, ax=ax, width=widths, alpha=0.8)

    nx.draw_networkx_labels(G, pos, ax=ax, font_size=9)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)

    return int(len(sel))


# =========================
# LOAD + ALIGN
# =========================
df0, chir_cols = load_features(os.path.join(DATA_DIR, FILES["0h"]))
df6, _ = load_features(os.path.join(DATA_DIR, FILES["6h"]))
df24, _ = load_features(os.path.join(DATA_DIR, FILES["24h"]))

common_ids = sorted(set(df0["sample_id"]).intersection(df6["sample_id"]).intersection(df24["sample_id"]))
df0 = df0.set_index("sample_id").loc[common_ids].reset_index()
df6 = df6.set_index("sample_id").loc[common_ids].reset_index()
df24 = df24.set_index("sample_id").loc[common_ids].reset_index()

X0 = df0[chir_cols].astype(float).to_numpy()
X6 = df6[chir_cols].astype(float).to_numpy()
X24 = df24[chir_cols].astype(float).to_numpy()

# Tensor (N, 3, 12)
T = np.stack([X0, X6, X24], axis=1)
N, k, p = T.shape


# =========================
# RESIDUALIZE + STANDARDIZE
# =========================
# Per observation: subtract mean across chiralities (common-mode removal)
T_res = T - T.mean(axis=2, keepdims=True)

# Global z-score after residualization across all observations
Tz = StandardScaler().fit_transform(T_res.reshape(-1, p)).reshape(N, k, p)


# =========================
# DYNAMICS: L and C
# =========================
c_lin = np.array([-1.0, 0.0, 1.0])   # 0h -> 24h linear drift
c_curv = np.array([1.0, -2.0, 1.0])  # 6h intermediate curvature

L = (Tz * c_lin[None, :, None]).sum(axis=1)    # (N, 12)
C = (Tz * c_curv[None, :, None]).sum(axis=1)   # (N, 12)

# Strict: remove per-sample dynamic common-mode (mean across chiralities)
L_cm = L - L.mean(axis=1, keepdims=True)
C_cm = C - C.mean(axis=1, keepdims=True)


# =========================
# 1) REDUNDANCY THRESHOLDS
# =========================
R_L = corr_mat(L)
R_C = corr_mat(C)
R_Lcm = corr_mat(L_cm)
R_Ccm = corr_mat(C_cm)

rows = []
for name, R in [
    ("L_raw", R_L),
    ("C_raw", R_C),
    ("L_commonmode_removed", R_Lcm),
    ("C_commonmode_removed", R_Ccm),
]:
    vals = offdiag_abs(R)
    for th in REDUND_THRESHOLDS:
        rows.append({
            "matrix": name,
            "threshold_abs_r": th,
            "n_pairs": int(len(vals)),
            "n_pairs_ge_threshold": int(np.sum(vals >= th)),
            "fraction_ge_threshold": float(np.mean(vals >= th)),
            "max_abs_r": float(vals.max()),
            "median_abs_r": float(np.median(vals)),
            "mean_abs_r": float(np.mean(vals)),
        })

redundancy_df = pd.DataFrame(rows)
redundancy_path = os.path.join(OUT_DIR, "redundancy_thresholds_absr.csv")
redundancy_df.to_csv(redundancy_path, index=False)


# =========================
# 2) HIERARCHICAL CLUSTERING MODULES
# =========================
# distance = 1 - |r|
def cluster_modules(R, labels, cutoff_abs_r=0.6, method="average"):
    D = 1.0 - np.abs(R)
    iu = np.triu_indices(D.shape[0], k=1)
    dist = D[iu]
    Z = linkage(dist, method=method)
    t = 1.0 - cutoff_abs_r
    cl = fcluster(Z, t=t, criterion="distance")
    return Z, cl

module_rows = []
for name, R in [
    ("L_commonmode_removed", R_Lcm),
    ("C_commonmode_removed", R_Ccm),
]:
    Z, cl = cluster_modules(R, chir_cols, cutoff_abs_r=MODULE_CUTOFF_ABS_R, method="average")

    # Save module assignments
    for lab, m in zip(chir_cols, cl):
        module_rows.append({
            "matrix": name,
            "cutoff_abs_r": MODULE_CUTOFF_ABS_R,
            "chirality": lab,
            "module": int(m)
        })

    # Plots
    plot_dendrogram(
        Z, chir_cols,
        title=f"Dendrogram (distance=1-|r|): {name}",
        outpath=os.path.join(OUT_DIR, f"dendrogram_{name}.png")
    )
    plot_clustered_heatmap(
        R, chir_cols, Z,
        title=f"Clustered correlation heatmap: {name}\n(distance=1-|r|, cutoff |r|≥{MODULE_CUTOFF_ABS_R})",
        outpath=os.path.join(OUT_DIR, f"clustered_heatmap_{name}.png")
    )

modules_df = pd.DataFrame(module_rows).sort_values(["matrix", "module", "chirality"])
modules_path = os.path.join(OUT_DIR, "chirality_modules_hclust.csv")
modules_df.to_csv(modules_path, index=False)


# =========================
# 3) PARTIAL CORRELATION / GRAPHICAL LASSO
# =========================
def fit_glasso_and_export(X, labels, tag):
    # per-feature z-score across samples (important for GL stability)
    Xs = StandardScaler().fit_transform(X)

    model = GraphicalLassoCV(alphas=10, cv=5).fit(Xs)
    prec = model.precision_
    pcorr = partial_corr_from_precision(prec)

    # Save matrices
    pcorr_df = pd.DataFrame(pcorr, index=labels, columns=labels)
    prec_df = pd.DataFrame(prec, index=labels, columns=labels)

    pcorr_path = os.path.join(OUT_DIR, f"partial_corr_{tag}.csv")
    prec_path = os.path.join(OUT_DIR, f"precision_{tag}.csv")
    pcorr_df.to_csv(pcorr_path)
    prec_df.to_csv(prec_path)

    # Edge list
    rows = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            w = pcorr[i, j]
            rows.append({
                "matrix": tag,
                "chirality_i": labels[i],
                "chirality_j": labels[j],
                "partial_corr": float(w),
                "abs_partial_corr": float(abs(w)),
            })
    edges = pd.DataFrame(rows).sort_values("abs_partial_corr", ascending=False).reset_index(drop=True)
    edges_path = os.path.join(OUT_DIR, f"partial_corr_edges_{tag}.csv")
    edges.to_csv(edges_path, index=False)

    # Networks at multiple thresholds
    net_counts = {}
    for th in GLASSO_THRESHOLDS:
        outpng = os.path.join(OUT_DIR, f"glasso_network_{tag}_thresh{th:.2f}.png")
        n_edges = plot_glasso_network(
            edges, labels,
            title=f"Graphical Lasso partial-correlation network\n{tag} (|partial| ≥ {th}, CV alpha={model.alpha_:.4g})",
            outpath=outpng,
            thresh=th,
            seed=RANDOM_SEED_LAYOUT
        )
        net_counts[f"n_edges_|partial|>={th}"] = n_edges

    return {
        "matrix": tag,
        "alpha_cv": float(model.alpha_),
        "partial_corr_csv": pcorr_path,
        "precision_csv": prec_path,
        "edges_csv": edges_path,
        **net_counts
    }

gl_L = fit_glasso_and_export(L_cm, chir_cols, "L_commonmode_removed")
gl_C = fit_glasso_and_export(C_cm, chir_cols, "C_commonmode_removed")

gl_summary = pd.DataFrame([gl_L, gl_C])
gl_summary_path = os.path.join(OUT_DIR, "glasso_network_summary.csv")
gl_summary.to_csv(gl_summary_path, index=False)


# =========================
# ZIP BUNDLE
# =========================
zip_path = os.path.join(OUT_DIR, "chirality_independence_bundle.zip")
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for fn in os.listdir(OUT_DIR):
        if fn.endswith((".csv", ".png")):
            z.write(os.path.join(OUT_DIR, fn), arcname=fn)

print("Done.")
print("Outputs folder:", OUT_DIR)
print("Bundle zip:", zip_path)

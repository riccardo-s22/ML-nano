"""
Self-consistent PERMANOVA on the Gaussian-fit peak amplitudes + metric-robustness.

Background
----------
The paper's headline GLOBAL PERMANOVA (F=23.05, R2=0.27) was computed on the
split-Gaussian fit amplitudes (ch*_gauss_max_{tp} in
physical_descriptors/chirality_gaussian_peak_descriptors.csv), whereas the
deposited within-subject script and the reported pairwise values used the 5x5
window-max features. This script puts everything on ONE feature set (the
Gaussian-fit amplitudes) and ONE model (within-subject repeated-measures
permutation), and additionally re-runs the distance-metric sensitivity there so
the non-normality robustness statement defends the published feature set.

Outputs (this folder):
    gaussfit_permanova_global.csv     repeated-measures Euclidean global
    gaussfit_permanova_pairwise.csv   repeated-measures Euclidean pairwise + Holm
    gaussfit_metric_sensitivity_global.csv
    gaussfit_metric_sensitivity_pairwise.csv
"""
import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
FIT_CSV = os.path.normpath(os.path.join(
    HERE, "..", "..", "physical_descriptors", "chirality_gaussian_peak_descriptors.csv"))
CH = ["8_3", "6_5", "7_5", "10_2", "9_4", "8_4", "7_6", "8_6", "8_7", "9_5", "10_3", "10_5"]
TPS = ["0h", "6h", "24h"]
N_PERM = 9999
SEED = 42

# ----------------------------------------------------------------------
# Build observation matrix (row k = subject s, timepoint t -> k = s*3+t)
# ----------------------------------------------------------------------
df = pd.read_csv(FIT_CSV)
cube = np.empty((len(df), 3, 12))            # (subjects, timepoints, chiralities)
for j, ch in enumerate(CH):
    for t, tp in enumerate(TPS):
        cube[:, t, j] = df[f"ch{ch}_gauss_max_{tp}"].to_numpy(float)
keep = np.isfinite(cube).all(axis=(1, 2))
cube = cube[keep]
N = cube.shape[0]
raw = cube.reshape(N * 3, 12)                # subject-blocked observations
z = (raw - raw.mean(0)) / raw.std(0, ddof=0)
ranks = np.column_stack([rankdata(raw[:, j]) for j in range(12)])
print(f"Gaussian-fit amplitudes: N subjects={N}, observations={N*3}, features=12, perms={N_PERM}\n")

# ----------------------------------------------------------------------
# Distance-based pseudo-F (Anderson 2001) + within-subject permutation
# ----------------------------------------------------------------------
def pseudo_F(D2, labels):
    n = D2.shape[0]
    groups = np.unique(labels)
    a = len(groups)
    ss_total = D2[np.triu_indices(n, 1)].sum() / n
    ss_within = 0.0
    for g in groups:
        idx = np.where(labels == g)[0]
        ng = len(idx)
        sub = D2[np.ix_(idx, idx)]
        ss_within += sub[np.triu_indices(ng, 1)].sum() / ng
    ss_between = ss_total - ss_within
    F = (ss_between / (a - 1)) / (ss_within / (n - a))
    R2 = ss_between / ss_total if ss_total > 0 else np.nan
    return float(F), float(R2)

PERMS3 = np.array([[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]])

def sqdist(M, metric):
    return squareform(pdist(M, metric)) ** 2

def perm_p(D2, base_labels, idx_map, ncond, seed):
    F_obs, R2_obs = pseudo_F(D2, base_labels)
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(N_PERM):
        lab = np.empty(len(base_labels), int)
        if ncond == 3:
            ch = rng.integers(0, 6, size=N)
            for s in range(N):
                perm = PERMS3[ch[s]]
                for i in range(3):
                    lab[idx_map[s][i]] = perm[i]
        else:
            ch = rng.integers(0, 2, size=N)
            for s in range(N):
                perm = (1, 0) if ch[s] else (0, 1)
                lab[idx_map[s][0]] = perm[0]
                lab[idx_map[s][1]] = perm[1]
        Fp, _ = pseudo_F(D2, lab)
        if Fp >= F_obs:
            ge += 1
    return F_obs, R2_obs, (1 + ge) / (1 + N_PERM)

def holm(p):
    p = np.asarray(p, float); m = len(p); order = np.argsort(p)
    adj = np.empty(m); prev = 0.0
    for i, idx in enumerate(order):
        prev = max(prev, min(1.0, (m - i) * p[idx])); adj[idx] = prev
    return adj

# feature matrix by metric
def matrix_for(metric):
    return {"euclidean_z": z, "manhattan_z": z, "rank_eucl": ranks,
            "correlation": raw, "braycurtis": raw}[metric]
def scipy_metric(metric):
    return {"euclidean_z": "euclidean", "manhattan_z": "cityblock",
            "rank_eucl": "euclidean", "correlation": "correlation",
            "braycurtis": "braycurtis"}[metric]

idx_map3 = [[s * 3 + 0, s * 3 + 1, s * 3 + 2] for s in range(N)]
base3 = np.tile([0, 1, 2], N)
PAIRS = [("0h", "6h"), ("6h", "24h"), ("0h", "24h")]

def pairwise_subset(M, a_tp, b_tp):
    ai, bi = TPS.index(a_tp), TPS.index(b_tp)
    order = []
    for s in range(N):
        order += [s * 3 + ai, s * 3 + bi]
    return M[np.array(order)]

# ======================================================================
# PART 1 - metric sensitivity (global + pairwise) on Gaussian-fit features
# ======================================================================
METRICS = ["euclidean_z", "manhattan_z", "rank_eucl", "correlation", "braycurtis"]
DESC = {"euclidean_z": "Euclidean (z) - PRIMARY", "manhattan_z": "Manhattan L1 (z)",
        "rank_eucl": "Rank-Euclidean (robust)", "correlation": "1-Pearson (pattern)",
        "braycurtis": "Bray-Curtis (raw)"}

print("=== PART 1: metric sensitivity, GLOBAL (Gaussian-fit amplitudes) ===")
grows = []
for m in METRICS:
    D2 = sqdist(matrix_for(m), scipy_metric(m))
    F, R2, p = perm_p(D2, base3, idx_map3, 3, SEED)
    grows.append({"metric": m, "desc": DESC[m], "F": F, "R2": R2, "p": p})
    print(f"  {DESC[m]:<26} F={F:8.3f}  R2={R2:6.3f}  p={p:.4f}")
Gs = pd.DataFrame(grows)
Gs.to_csv(os.path.join(HERE, "gaussfit_metric_sensitivity_global.csv"), index=False)

print("\n=== PART 1: metric sensitivity, PAIRWISE (unadjusted p) ===")
prows = []
for m in METRICS:
    Mfull = matrix_for(m)
    for a, b in PAIRS:
        sub = pairwise_subset(Mfull, a, b)
        D2 = sqdist(sub, scipy_metric(m))
        base2 = np.tile([0, 1], N)
        idx_map2 = [[s * 2 + 0, s * 2 + 1] for s in range(N)]
        F, R2, p = perm_p(D2, base2, idx_map2, 2, SEED + 1)
        prows.append({"metric": m, "desc": DESC[m], "comparison": f"{a} vs {b}",
                      "F": F, "R2": R2, "p": p})
Ps = pd.DataFrame(prows)
Ps.to_csv(os.path.join(HERE, "gaussfit_metric_sensitivity_pairwise.csv"), index=False)
for m in METRICS:
    s = Ps[Ps.metric == m]
    print("  " + f"{DESC[m]:<26} " +
          "  ".join(f"{r.comparison}: F={r.F:6.2f} p={r.p:.4f}" for r in s.itertuples()))

# ======================================================================
# PART 2 - self-consistent repeated-measures Euclidean global + pairwise
# ======================================================================
print("\n=== PART 2: self-consistent repeated-measures PERMANOVA (Euclidean z) ===")
D2z = sqdist(z, "euclidean")
Fg, R2g, pg = perm_p(D2z, base3, idx_map3, 3, SEED)
gdf = pd.DataFrame([{"n_subjects": N, "n_obs": N * 3, "F": Fg, "R2": R2g, "p": pg,
                     "df_between": 2, "df_within": N * 3 - 3, "n_perm": N_PERM}])
gdf.to_csv(os.path.join(HERE, "gaussfit_permanova_global.csv"), index=False)
print(f"  GLOBAL: F={Fg:.3f}  R2={R2g:.3f}  p={pg:.4f}")

rows = []
for a, b in PAIRS:
    sub = pairwise_subset(z, a, b)
    D2 = sqdist(sub, "euclidean")
    base2 = np.tile([0, 1], N)
    idx_map2 = [[s * 2 + 0, s * 2 + 1] for s in range(N)]
    F, R2, p = perm_p(D2, base2, idx_map2, 2, SEED + 1)
    rows.append({"comparison": f"{a} vs {b}", "F": F, "R2": R2, "p": p})
pdf = pd.DataFrame(rows)
pdf["p_holm"] = holm(pdf["p"].values)
pdf.to_csv(os.path.join(HERE, "gaussfit_permanova_pairwise.csv"), index=False)
for r in pdf.itertuples():
    print(f"  {r.comparison:<10} F={r.F:7.3f}  R2={r.R2:6.3f}  p={r.p:.4f}  p_holm={r.p_holm:.4f}")

print("\nSaved: gaussfit_permanova_global.csv, gaussfit_permanova_pairwise.csv,")
print("       gaussfit_metric_sensitivity_global.csv, gaussfit_metric_sensitivity_pairwise.csv")

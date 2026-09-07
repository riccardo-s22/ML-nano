"""
Distance-metric sensitivity analysis for the timepoint PERMANOVA.

The primary analysis (permanova_full12_within_subject.py) uses Euclidean
sums-of-squares. Because the chirality intensities are heavy-tailed / non-normal
(see normality_intensity_supp.py), Euclidean SS could in principle be influenced
by tail observations. This script re-runs the SAME repeated-measures,
within-subject permutation test under several alternative dissimilarities using
Anderson's (2001) distance-based PERMANOVA formulation:

    SS_total  = (1/n) * sum_{i<j} d_ij^2
    SS_within = sum_g (1/n_g) * sum_{i<j in g} d_ij^2
    F = (SS_between/(a-1)) / (SS_within/(n-a)),  SS_between = SS_total - SS_within

With a Euclidean d on the standardized features this is algebraically identical
to the primary script -> used as a correctness check (must reproduce F=23.05).

Metrics:
    euclidean_z   Euclidean on z-scored features        (primary; magnitude-sensitive)
    manhattan_z   L1 on z-scored features               (down-weights tails vs L2)
    rank_eucl     Euclidean on column-rank features     (robust to outliers, keeps order)
    correlation   1 - Pearson across the 12 chiralities (pattern-only, magnitude-free)
    braycurtis    Bray-Curtis on raw non-negative ints  (ratio-based, magnitude-free)
"""
import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(HERE, ".."))
FILES = {"0h": "features_max5x5_0h.csv", "6h": "features_max5x5_6h.csv",
         "24h": "features_max5x5_24h.csv"}
TIMEPOINTS = ["0h", "6h", "24h"]
N_PERM = 9999
SEED = 42

# ----------------------------------------------------------------------
# Load + align (identical to primary script)
# ----------------------------------------------------------------------
dfs = {tp: pd.read_csv(os.path.join(DATA_DIR, FILES[tp])) for tp in TIMEPOINTS}
chir = [c for c in dfs["0h"].columns if c != "sample_id"]
common = sorted(set(dfs["0h"].sample_id) & set(dfs["6h"].sample_id) & set(dfs["24h"].sample_id))
for tp in TIMEPOINTS:
    dfs[tp] = dfs[tp].set_index("sample_id").loc[common].reset_index()
N = len(common)

# Observation matrix: row k = subject s, timepoint t  ->  k = s*3 + t
raw = np.empty((N * 3, len(chir)), float)
for s in range(N):
    for t, tp in enumerate(TIMEPOINTS):
        raw[s * 3 + t] = dfs[tp][chir].iloc[s].to_numpy(float)
n_obs = raw.shape[0]

# feature representations
z = (raw - raw.mean(0)) / raw.std(0, ddof=0)                 # StandardScaler equiv
ranks = np.column_stack([rankdata(raw[:, j]) for j in range(raw.shape[1])])

def sqdist(metric):
    if metric == "euclidean_z":
        d = pdist(z, "euclidean")
    elif metric == "manhattan_z":
        d = pdist(z, "cityblock")
    elif metric == "rank_eucl":
        d = pdist(ranks, "euclidean")
    elif metric == "correlation":
        d = pdist(raw, "correlation")          # 1 - Pearson r across 12 chiralities
    elif metric == "braycurtis":
        d = pdist(raw, "braycurtis")           # raw intensities are all > 0
    else:
        raise ValueError(metric)
    return squareform(d) ** 2                   # squared-distance matrix

# ----------------------------------------------------------------------
# Distance-based pseudo-F for a group labelling
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

PERMS3 = np.array([[0, 1, 2], [0, 2, 1], [1, 0, 2],
                   [1, 2, 0], [2, 0, 1], [2, 1, 0]])

def within_subject_labels(rng, groups_per_obs_base, nsub, ncond, idx_map):
    """Return a fresh label vector permuting condition labels within each subject."""
    lab = np.empty(nsub * ncond, int)
    choice = rng.integers(0, len(PERMS3) if ncond == 3 else 2, size=nsub)
    for s in range(nsub):
        if ncond == 3:
            perm = PERMS3[choice[s]]
        else:
            perm = np.array([1, 0]) if choice[s] else np.array([0, 1])
        for i in range(ncond):
            lab[idx_map[s][i]] = perm[i]
    return lab

def run_global(metric):
    D2 = sqdist(metric)
    base_labels = np.tile([0, 1, 2], N)
    idx_map = [[s * 3 + 0, s * 3 + 1, s * 3 + 2] for s in range(N)]
    F_obs, R2_obs = pseudo_F(D2, base_labels)
    rng = np.random.default_rng(SEED)
    ge = 0
    for _ in range(N_PERM):
        lab = within_subject_labels(rng, base_labels, N, 3, idx_map)
        Fp, _ = pseudo_F(D2, lab)
        if Fp >= F_obs:
            ge += 1
    p = (1 + ge) / (1 + N_PERM)
    return F_obs, R2_obs, p

def run_pairwise(metric, a_tp, b_tp):
    ai, bi = TIMEPOINTS.index(a_tp), TIMEPOINTS.index(b_tp)
    keep = np.array([s * 3 + ai for s in range(N)] + [s * 3 + bi for s in range(N)])
    # reorder as subject-blocked pairs so within-subject swap is easy
    order = []
    for s in range(N):
        order += [s * 3 + ai, s * 3 + bi]
    order = np.array(order)
    if metric == "euclidean_z":
        M = z[order]; D2 = squareform(pdist(M, "euclidean")) ** 2
    elif metric == "manhattan_z":
        M = z[order]; D2 = squareform(pdist(M, "cityblock")) ** 2
    elif metric == "rank_eucl":
        M = ranks[order]; D2 = squareform(pdist(M, "euclidean")) ** 2
    elif metric == "correlation":
        M = raw[order]; D2 = squareform(pdist(M, "correlation")) ** 2
    elif metric == "braycurtis":
        M = raw[order]; D2 = squareform(pdist(M, "braycurtis")) ** 2
    base = np.tile([0, 1], N)
    idx_map = [[s * 2 + 0, s * 2 + 1] for s in range(N)]
    F_obs, R2_obs = pseudo_F(D2, base)
    rng = np.random.default_rng(SEED + 1)
    ge = 0
    for _ in range(N_PERM):
        lab = within_subject_labels(rng, base, N, 2, idx_map)
        Fp, _ = pseudo_F(D2, lab)
        if Fp >= F_obs:
            ge += 1
    p = (1 + ge) / (1 + N_PERM)
    return F_obs, R2_obs, p

# ----------------------------------------------------------------------
METRICS = ["euclidean_z", "manhattan_z", "rank_eucl", "correlation", "braycurtis"]
DESC = {"euclidean_z": "Euclidean (z) - PRIMARY", "manhattan_z": "Manhattan L1 (z)",
        "rank_eucl": "Rank-Euclidean (robust)", "correlation": "1-Pearson (pattern)",
        "braycurtis": "Bray-Curtis (raw)"}

print(f"N subjects={N}, observations={n_obs}, features={len(chir)}, perms={N_PERM}\n")
print("=== GLOBAL PERMANOVA across timepoints ===")
grows = []
for m in METRICS:
    F, R2, p = run_global(m)
    grows.append({"metric": m, "desc": DESC[m], "F": F, "R2": R2, "p": p})
    print(f"  {DESC[m]:<26} F={F:8.3f}  R2={R2:6.3f}  p={p:.4f}")
G = pd.DataFrame(grows)

print("\n=== PAIRWISE PERMANOVA (unadjusted p) ===")
prows = []
for m in METRICS:
    for a, b in [("0h", "6h"), ("6h", "24h"), ("0h", "24h")]:
        F, R2, p = run_pairwise(m, a, b)
        prows.append({"metric": m, "desc": DESC[m], "comparison": f"{a} vs {b}",
                      "F": F, "R2": R2, "p": p})
P = pd.DataFrame(prows)
for m in METRICS:
    sub = P[P.metric == m]
    line = "  ".join(f"{r.comparison}: F={r.F:6.2f} p={r.p:.4f}" for r in sub.itertuples())
    print(f"  {DESC[m]:<26} {line}")

G.to_csv(os.path.join(HERE, "permanova_metric_sensitivity_global.csv"), index=False)
P.to_csv(os.path.join(HERE, "permanova_metric_sensitivity_pairwise.csv"), index=False)
print("\nSaved -> permanova_metric_sensitivity_global.csv / _pairwise.csv")

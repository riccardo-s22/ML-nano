import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# -------------------------
# Inputs
# -------------------------
f0 = pd.read_csv("features_max5x5_0h.csv")
f6 = pd.read_csv("features_max5x5_6h.csv")
f24 = pd.read_csv("features_max5x5_24h.csv")

chir_cols = [c for c in f0.columns if c != "sample_id"]

common = sorted(set(f0.sample_id) & set(f6.sample_id) & set(f24.sample_id))
f0 = f0.set_index("sample_id").loc[common].reset_index()
f6 = f6.set_index("sample_id").loc[common].reset_index()
f24 = f24.set_index("sample_id").loc[common].reset_index()

def to_long(df, time_label):
    x = df.melt(
        id_vars=["sample_id"],
        value_vars=chir_cols,
        var_name="chirality",
        value_name="y",
    )
    x["time"] = time_label
    return x

long = pd.concat(
    [to_long(f0, "0h"), to_long(f6, "6h"), to_long(f24, "24h")],
    ignore_index=True,
)

# Preprocessing
long["y"] = np.log1p(long["y"])
long["time"] = pd.Categorical(long["time"], categories=["0h", "6h", "24h"], ordered=True)
long["chirality"] = long["chirality"].astype("category")

# Within-chirality standardization (optional but reasonable)
# Set observed=False explicitly to silence FutureWarning
long["y"] = long.groupby("chirality", observed=False)["y"].transform(
    lambda x: (x - x.mean()) / x.std(ddof=0)
)

# -------------------------
# Fixed sample effect model
# -------------------------
ols = smf.ols("y ~ time * chirality + C(sample_id)", data=long).fit()
ols_cr = ols.get_robustcov_results(cov_type="cluster", groups=long["sample_id"])

print(ols_cr.summary())

# -------------------------
# Omnibus Wald tests (robust, clustered by sample_id)
# -------------------------
param_names = ols.model.exog_names  # <-- THIS is the key fix

def wald_test_block(term_selector, label):
    """Build an R matrix selecting parameters that match term_selector(name)->bool."""
    idx = [i for i, name in enumerate(param_names) if term_selector(name)]
    if len(idx) == 0:
        print(f"\n{label}: no matching terms found.")
        return

    R = np.zeros((len(idx), len(param_names)))
    for r, j in enumerate(idx):
        R[r, j] = 1.0

    w = ols_cr.wald_test(R)
    print(f"\nWald test: {label}")
    print(w)

# 1) Interaction block: time[T.*]:chirality[T.*]
wald_test_block(
    lambda n: ("time[T." in n) and (":chirality[T." in n),
    "time × chirality interaction (omnibus)",
)

# 2) Time main indicators: time[T.6h], time[T.24h]
wald_test_block(
    lambda n: n.startswith("time[T."),
    "time indicators (main-effect coding; interpret with interaction in mind)",
)

# 3) Chirality main indicators: chirality[T.*]
wald_test_block(
    lambda n: n.startswith("chirality[T."),
    "chirality indicators",
)

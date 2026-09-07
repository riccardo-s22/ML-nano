import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

# -------------------------
# Inputs
# -------------------------
f0 = pd.read_csv("features_max5x5_0h.csv")
f6 = pd.read_csv("features_max5x5_6h.csv")
f24 = pd.read_csv("features_max5x5_24h.csv")

chir_cols = [c for c in f0.columns if c != "sample_id"]

# align sample_ids across timepoints
common = sorted(set(f0.sample_id) & set(f6.sample_id) & set(f24.sample_id))
f0 = f0.set_index("sample_id").loc[common].reset_index()
f6 = f6.set_index("sample_id").loc[common].reset_index()
f24 = f24.set_index("sample_id").loc[common].reset_index()

# -------------------------
# Long-format dataframe
# -------------------------
def to_long(df, time_label):
    x = df.melt(id_vars=["sample_id"], value_vars=chir_cols,
                var_name="chirality", value_name="y")
    x["time"] = time_label
    return x

long = pd.concat([
    to_long(f0, "0h"),
    to_long(f6, "6h"),
    to_long(f24, "24h")
], ignore_index=True)

# optional transform: log1p or standardize
# long["y"] = np.log1p(long["y"])
# or standardize globally:
long["y"] = np.log1p(long["y"])  # optional but often stabilizes
long["y"] = long.groupby("chirality")["y"].transform(
    lambda x: (x - x.mean()) / x.std(ddof=0)
)

# make categorical with explicit order
long["time"] = pd.Categorical(long["time"], categories=["0h", "6h", "24h"], ordered=True)
long["chirality"] = long["chirality"].astype("category")

# -------------------------
# Mixed model (random intercept)
# -------------------------
m_full = smf.mixedlm("y ~ time * chirality", long, groups=long["sample_id"])
fit_full = m_full.fit(reml=False, method="lbfgs")
print(fit_full.summary())

# Reduced models for LR tests
m_no_inter = smf.mixedlm("y ~ time + chirality", long, groups=long["sample_id"])
fit_no_inter = m_no_inter.fit(reml=False, method="lbfgs")

m_no_time = smf.mixedlm("y ~ chirality", long, groups=long["sample_id"])
fit_no_time = m_no_time.fit(reml=False, method="lbfgs")

m_no_chir = smf.mixedlm("y ~ time", long, groups=long["sample_id"])
fit_no_chir = m_no_chir.fit(reml=False, method="lbfgs")

def lr_test(fit_reduced, fit_full):
    lr = 2 * (fit_full.llf - fit_reduced.llf)
    df = fit_full.df_modelwc - fit_reduced.df_modelwc
    p = 1 - stats.chi2.cdf(lr, df)
    return lr, df, p

from scipy import stats

tests = []
tests.append(("time:chirality (interaction)", *lr_test(fit_no_inter, fit_full)))
tests.append(("time (main effect)", *lr_test(fit_no_time, fit_full)))
tests.append(("chirality (main effect)", *lr_test(fit_no_chir, fit_full)))

tests_df = pd.DataFrame(tests, columns=["term", "LR_stat", "df", "p_value"])
print("\nLikelihood Ratio Tests (approx):")
print(tests_df)

tests_df.to_csv("lmm_lr_tests.csv", index=False)

# -------------------------
# Optional: random slope for time
# (can be heavier; try if it converges)
# -------------------------
# long["time_num"] = long["time"].cat.codes  # 0,1,2
# m_rs = smf.mixedlm("y ~ time * chirality", long, groups=long["sample_id"],
#                    re_formula="~time_num")
# fit_rs = m_rs.fit(reml=False, method="lbfgs")
# print(fit_rs.summary())

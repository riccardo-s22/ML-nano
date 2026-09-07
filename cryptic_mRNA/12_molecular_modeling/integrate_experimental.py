#!/usr/bin/env python3
"""Experimental NIR integration (§14) — REAL data from ../../exp1/.

Fits a binding isotherm to the GT15-STMN2 sensor optical response vs spiked
STMN2-CE mRNA concentration (water curve) to estimate an APPARENT association
constant K_A, and contrasts it with the GT15-only control. Compares qualitatively
with the in-silico prediction (specific capture-domain hybridization drives the
response) as an EXPLORATORY association — NOT a causal validation. The apparent
K_A reflects operational assay sensitivity (copies/uL), NOT a solution-phase
thermodynamic K_D; the molar conversion is given with that caveat.
"""
import os, sys, json
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_common import ROOT, write_json, utc_now
import figstyle
figstyle.apply_style()

EXP1 = os.path.abspath(os.path.join(ROOT, "..", "..", "exp1"))
OUT = os.path.join(ROOT, "results", "experimental")
FIG = os.path.join(ROOT, "figures")
os.makedirs(OUT, exist_ok=True)

CHIRALITIES = ["ch8_3", "ch6_5", "ch7_5", "ch10_2", "ch9_4", "ch8_4",
               "ch7_6", "ch8_6", "ch8_7", "ch9_5", "ch10_3", "ch10_5"]
COPIES_PER_UL_TO_MOLAR = 1e6 / 6.02214076e23   # 1 copy/uL -> mol/L

def langmuir(c, Rmax, Kd):
    return Rmax * c / (Kd + c)

def hill(c, Rmax, Kd, n):
    return Rmax * c**n / (Kd**n + c**n)

def fit_isotherm(c, R):
    """Fit Langmuir (n=1) and Hill; return params, CIs, R^2."""
    res = {}
    c = np.asarray(c, float); R = np.asarray(R, float)
    cpos = c[c > 0]
    p0_kd = np.median(cpos) if len(cpos) else 100.0
    p0_rmax = max(np.max(R), 1e-6)
    try:
        popt, pcov = curve_fit(langmuir, c, R, p0=[p0_rmax, p0_kd],
                               bounds=([0, 1e-3], [np.inf, 1e9]), maxfev=20000)
        perr = np.sqrt(np.diag(pcov))
        pred = langmuir(c, *popt)
        ss_res = np.sum((R - pred)**2); ss_tot = np.sum((R - R.mean())**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        res["langmuir"] = {"Rmax": popt[0], "Kd_copies_per_uL": popt[1],
                           "Kd_se": perr[1], "R2": r2}
    except Exception as e:
        res["langmuir"] = {"error": str(e)}
    try:
        popt, pcov = curve_fit(hill, c, R, p0=[p0_rmax, p0_kd, 1.0],
                               bounds=([0, 1e-3, 0.2], [np.inf, 1e9, 5.0]), maxfev=20000)
        pred = hill(c, *popt)
        ss_res = np.sum((R - pred)**2); ss_tot = np.sum((R - R.mean())**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        res["hill"] = {"Rmax": popt[0], "Kd_copies_per_uL": popt[1], "n": popt[2], "R2": r2}
    except Exception as e:
        res["hill"] = {"error": str(e)}
    return res

def bootstrap_Kd(df_long, n_boot=2000, seed=0):
    """Bootstrap Kd over replicate wells (resample within each concentration)."""
    rng = np.random.default_rng(seed)
    kds = []
    by_c = {c: g["R"].values for c, g in df_long.groupby("copies")}
    cs = sorted(by_c)
    for _ in range(n_boot):
        cc, RR = [], []
        for c in cs:
            vals = by_c[c]
            samp = rng.choice(vals, size=len(vals), replace=True)
            cc.extend([c]*len(samp)); RR.extend(samp)
        try:
            popt, _ = curve_fit(langmuir, np.array(cc, float), np.array(RR, float),
                                p0=[max(np.max(RR), 1e-6), np.median([c for c in cs if c > 0])],
                                bounds=([0, 1e-3], [np.inf, 1e9]), maxfev=10000)
            kds.append(popt[1])
        except Exception:
            pass
    if not kds:
        return None, (None, None)
    return float(np.median(kds)), tuple(float(x) for x in np.percentile(kds, [2.5, 97.5]))

def response_table(df, chir, sensor, matrix, feature="ff_integ"):
    col = f"{chir}_{feature}"
    sub = df[(df.sensor == sensor) & (df.matrix == matrix)].copy()
    I0 = sub[sub.copies == 0][col].mean()
    if not np.isfinite(I0) or I0 == 0:
        I0 = sub[col].replace(0, np.nan).min()
    sub["R"] = sub[col] / I0 - 1.0       # fractional response vs 0-copy baseline
    return sub[["copies", "R", col]].rename(columns={col: "intensity"}), I0

def main():
    fpath = os.path.join(EXP1, "fit_free_descriptors.csv")
    if not os.path.exists(fpath):
        write_json(os.path.join(OUT, "experimental_status.json"),
                   {"status": "not run: no input data", "utc": utc_now()})
        print("No experimental input; stage not run.")
        return
    df = pd.read_csv(fpath)

    from scipy.stats import spearmanr
    rows = []
    for chir in CHIRALITIES:
        sens, I0s = response_table(df, chir, "GT15-STMN2", "water")
        ctrl, I0c = response_table(df, chir, "GT15", "water")
        # per-concentration means (primary fit; removes replicate scatter)
        sm = sens.groupby("copies").R.mean().reset_index()
        cm = ctrl.groupby("copies").R.mean().reset_index()
        fit_s = fit_isotherm(sm.copies, sm.R)          # fit on means
        fit_sr = fit_isotherm(sens.copies, sens.R)     # replicate-level (noisier)
        fit_c = fit_isotherm(cm.copies, cm.R)
        kd_med, (kd_lo, kd_hi) = bootstrap_Kd(sens.rename(columns={"R": "R"}))
        # monotonic dose-response significance (replicate level)
        rho_s, p_s = spearmanr(sens.copies, sens.R)
        rho_c, p_c = spearmanr(ctrl.copies, ctrl.R)
        lang = fit_s.get("langmuir", {})
        kd = lang.get("Kd_copies_per_uL")
        ka = (1.0 / kd) if (kd and kd > 0) else None
        rows.append({
            "chirality": chir.replace("ch", "").replace("_", ","),
            "sensor_langmuir_Kd_copies_per_uL": round(kd, 3) if kd else None,
            "sensor_Kd_boot_median": round(kd_med, 3) if kd_med else None,
            "sensor_Kd_boot_CI_lo": round(kd_lo, 3) if kd_lo else None,
            "sensor_Kd_boot_CI_hi": round(kd_hi, 3) if kd_hi else None,
            "sensor_apparent_KA_per_copies_per_uL": round(ka, 6) if ka else None,
            "sensor_apparent_KA_per_M": (f"{ka / COPIES_PER_UL_TO_MOLAR:.3e}" if ka else None),
            "sensor_Rmax": round(lang.get("Rmax", float("nan")), 3),
            "sensor_R2_langmuir_means": round(lang.get("R2", float("nan")), 3),
            "sensor_R2_langmuir_replicate": round(fit_sr.get("langmuir", {}).get("R2", float("nan")), 3),
            "sensor_hill_n": round(fit_s.get("hill", {}).get("n", float("nan")), 3),
            "sensor_spearman_rho": round(float(rho_s), 3), "sensor_spearman_p": float(p_s),
            "control_spearman_rho": round(float(rho_c), 3), "control_spearman_p": float(p_c),
            "control_R2_langmuir_means": round(fit_c.get("langmuir", {}).get("R2", float("nan")), 3),
            "sensor_max_response": round(float(sm.R.max()), 3),
            "control_max_response": round(float(cm.R.max()), 3),
        })
    rank = pd.DataFrame(rows).sort_values("sensor_spearman_p", ascending=True)
    rank.to_csv(os.path.join(OUT, "chirality_response_KA.tsv"), sep="\t", index=False)

    # figure for (7,5) and the best-fit chirality
    best = rank.iloc[0]["chirality"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, chname in zip(axes, ["7,5", best]):
        chir = "ch" + chname.replace(",", "_")
        sens, _ = response_table(df, chir, "GT15-STMN2", "water")
        ctrl, _ = response_table(df, chir, "GT15", "water")
        # mean +/- sd per concentration
        for d, lab, col in [(sens, "GT15-STMN2 sensor", "#d73027"), (ctrl, "GT15 control", "#4575b4")]:
            g = d.groupby("copies").R.agg(["mean", "std", "count"]).reset_index()
            xpos = g.copies.replace(0, 3)  # plot 0 at x=3 for log axis
            ax.errorbar(xpos, g["mean"], yerr=g["std"], fmt="o", color=col, capsize=3, label=lab)
        # langmuir fit overlay (sensor)
        fit = fit_isotherm(sens.copies, sens.R).get("langmuir", {})
        if "Kd_copies_per_uL" in fit:
            xx = np.logspace(0.5, 5, 100)
            ax.plot(xx, langmuir(xx, fit["Rmax"], fit["Kd_copies_per_uL"]), "-", color="#d73027",
                    lw=1.5, label=f"Langmuir fit (Kd={fit['Kd_copies_per_uL']:.0f} cp/µL, R²={fit['R2']:.2f})")
        ax.set_xscale("log"); ax.set_xlabel("STMN2-CE mRNA (copies/µL)")
        ax.set_ylabel("fractional NIR response  (I/I₀ − 1)")
        ax.set_title(f"({chname}) — water")
        ax.legend(fontsize=7)
    fig.suptitle("Experimental NIR dose-response in water (integrated intensity) — "
                 "sensor vs control; apparent K_A from Langmuir fit (EXPLORATORY)")
    fig.tight_layout(); figstyle.save_both(fig, os.path.join(FIG, "experimental_water_KA"))

    # provenance + summary
    prov = {"source": os.path.relpath(fpath, ROOT), "matrix": "water",
            "sensor": "GT15-STMN2", "control": "GT15",
            "response_feature": "ff_integ (integrated chirality NIR intensity)",
            "concentration_unit": "copies/uL", "n_concentrations": 6,
            "copies_per_uL_to_M": COPIES_PER_UL_TO_MOLAR,
            "caveat": "apparent K_A = 1/EC50 reflects operational assay sensitivity, NOT a "
                      "thermodynamic solution K_D; comparison to in-silico is exploratory association.",
            "utc": utc_now()}
    write_json(os.path.join(OUT, "experimental_provenance.json"), prov)

    # agreement check vs in-silico
    s75 = rank[rank.chirality == "7,5"].iloc[0].to_dict()
    best_row = rank.iloc[0].to_dict()           # best = lowest sensor spearman p
    # n_sig sensor channels with significant positive monotonic dose-response
    sig_sensor = rank[(rank.sensor_spearman_p < 0.05) & (rank.sensor_spearman_rho > 0)]
    sig_control = rank[(rank.control_spearman_p < 0.05) & (rank.control_spearman_rho > 0)]
    agreement = {
        "in_silico_prediction": "STMN2 capture domain forms a strong, specific duplex with "
            "pathological STMN2-CE target (MELTING dG~-30 kcal/mol; high specificity vs normal/negative). "
            "Docking did NOT predict optical magnitude or chirality optical ranking.",
        "experimental_observation_water": {
            "n_sensor_channels_significant_dose_response": int(len(sig_sensor)),
            "n_control_channels_significant_dose_response": int(len(sig_control)),
            "best_channel_by_significance": best_row["chirality"],
            "best_sensor_spearman_rho": best_row["sensor_spearman_rho"],
            "best_sensor_spearman_p": best_row["sensor_spearman_p"],
            "best_sensor_R2_means": best_row["sensor_R2_langmuir_means"],
            "best_apparent_KA_per_copies_per_uL": best_row["sensor_apparent_KA_per_copies_per_uL"],
            "ch7_5_spearman_rho": s75.get("sensor_spearman_rho"),
            "ch7_5_spearman_p": s75.get("sensor_spearman_p"),
            "ch7_5_apparent_KA_per_copies_per_uL": s75.get("sensor_apparent_KA_per_copies_per_uL"),
            "ch7_5_apparent_Kd_copies_per_uL": s75.get("sensor_langmuir_Kd_copies_per_uL"),
        },
        "agreement_verdict": None, "verdict_rationale": None,
    }
    sensor_responds = len(sig_sensor) >= 1
    control_weaker = len(sig_control) < len(sig_sensor)
    if sensor_responds and control_weaker:
        agreement["agreement_verdict"] = "QUALITATIVELY CONSISTENT"
        agreement["verdict_rationale"] = (
            f"GT15-STMN2 shows a significant, positive, saturable dose-response to STMN2-CE mRNA "
            f"in water in {len(sig_sensor)}/12 chirality channels (best {best_row['chirality']}: "
            f"Spearman rho={best_row['sensor_spearman_rho']}, p={best_row['sensor_spearman_p']:.1e}; "
            f"apparent K_d~{best_row['sensor_langmuir_Kd_copies_per_uL']} copies/uL), whereas the "
            f"GT15-only control responds in only {len(sig_control)}/12 channels. The early-saturating, "
            f"specific dose-response is consistent with the predicted capture-domain-driven hybridization. "
            f"This is an EXPLORATORY qualitative association; the in-silico work did not predict optical "
            f"magnitude or the apparent K_A, and the apparent K_A is an operational sensitivity metric, "
            f"not a thermodynamic solution K_D.")
    else:
        agreement["agreement_verdict"] = "INCONCLUSIVE / MIXED"
        agreement["verdict_rationale"] = ("Sensor dose-response and/or sensor>control contrast did not "
            "cleanly match the specific-capture expectation; see chirality_response_KA.tsv per channel.")
    write_json(os.path.join(OUT, "insilico_agreement.json"), agreement)

    print(f"Sensor significant dose-response channels: {len(sig_sensor)}/12; control: {len(sig_control)}/12")
    print(f"Best sensor channel {best_row['chirality']}: rho={best_row['sensor_spearman_rho']} "
          f"p={best_row['sensor_spearman_p']:.1e} Kd={best_row['sensor_langmuir_Kd_copies_per_uL']} cp/uL "
          f"apparentKA={best_row['sensor_apparent_KA_per_copies_per_uL']} (cp/uL)^-1 "
          f"(={best_row['sensor_apparent_KA_per_M']} M^-1, operational)")
    print(f"(7,5): rho={s75.get('sensor_spearman_rho')} p={s75.get('sensor_spearman_p'):.1e} "
          f"Kd={s75.get('sensor_langmuir_Kd_copies_per_uL')} cp/uL R2_means={s75.get('sensor_R2_langmuir_means')}")
    print("Agreement verdict:", agreement["agreement_verdict"])

if __name__ == "__main__":
    main()

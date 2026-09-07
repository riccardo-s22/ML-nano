#!/usr/bin/env python3
"""Hill-coefficient cooperativity test for the GT15-STMN2 water dose-response (§14).

Fits the Hill model R(c)=Rmax*c^n/(Kd^n+c^n) per chirality, bootstraps the Hill
coefficient n over replicate wells (CI), and compares Hill (3 params) vs Langmuir
(2 params, n=1) by AICc. Cooperativity is claimed ONLY if (a) n's 95% CI excludes
1 AND (b) AICc favours Hill. Honest caveat: the concentration grid saturates early
(~2 points on the rising edge), so n is weakly identified and the bootstrap CI will
be wide — this is reported, not hidden.
"""
import os, sys, json
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_common import ROOT, write_json, utc_now
import figstyle
figstyle.apply_style()

EXP1 = os.path.abspath(os.path.join(ROOT, "..", "..", "exp1"))
OUT = os.path.join(ROOT, "results", "experimental")
FIG = os.path.join(ROOT, "figures")
CHIRALITIES = ["ch8_3", "ch6_5", "ch7_5", "ch10_2", "ch9_4", "ch8_4",
               "ch7_6", "ch8_6", "ch8_7", "ch9_5", "ch10_3", "ch10_5"]

def langmuir(c, Rmax, Kd):
    return Rmax * c / (Kd + c)

def hill(c, Rmax, Kd, n):
    return Rmax * np.power(c, n) / (np.power(Kd, n) + np.power(c, n))

def aicc(R, pred, k):
    n = len(R)
    rss = np.sum((np.asarray(R) - np.asarray(pred))**2)
    if rss <= 0 or n - k - 1 <= 0:
        return np.nan
    aic = n * np.log(rss / n) + 2 * k
    return aic + 2 * k * (k + 1) / (n - k - 1)

def response(df, chir, sensor, matrix="water"):
    col = f"{chir}_ff_integ"
    sub = df[(df.sensor == sensor) & (df.matrix == matrix)].copy()
    I0 = sub[sub.copies == 0][col].mean()
    sub["R"] = sub[col] / I0 - 1.0
    return sub[["copies", "R"]]

def fit_hill_langmuir(c, R):
    c = np.asarray(c, float); R = np.asarray(R, float)
    rmax0 = max(np.max(R), 1e-6); kd0 = np.median(c[c > 0]) if np.any(c > 0) else 100.0
    out = {}
    try:
        pL, _ = curve_fit(langmuir, c, R, p0=[rmax0, kd0],
                          bounds=([0, 1e-3], [np.inf, 1e9]), maxfev=20000)
        out["langmuir"] = {"Rmax": pL[0], "Kd": pL[1],
                           "aicc": aicc(R, langmuir(c, *pL), 2),
                           "r2": 1 - np.sum((R-langmuir(c,*pL))**2)/np.sum((R-R.mean())**2)}
    except Exception as e:
        out["langmuir"] = {"error": str(e)}
    try:
        pH, pcov = curve_fit(hill, c, R, p0=[rmax0, kd0, 1.0],
                             bounds=([0, 1e-3, 0.2], [np.inf, 1e9, 5.0]), maxfev=20000)
        perr = np.sqrt(np.diag(pcov))
        out["hill"] = {"Rmax": pH[0], "Kd": pH[1], "n": pH[2], "n_se_asym": perr[2],
                       "aicc": aicc(R, hill(c, *pH), 3),
                       "r2": 1 - np.sum((R-hill(c,*pH))**2)/np.sum((R-R.mean())**2)}
    except Exception as e:
        out["hill"] = {"error": str(e)}
    return out

def bootstrap_n(sub, n_boot=3000, seed=0):
    rng = np.random.default_rng(seed)
    by_c = {c: g.R.values for c, g in sub.groupby("copies")}
    cs = sorted(by_c); ns = []
    for _ in range(n_boot):
        cc, RR = [], []
        for c in cs:
            v = by_c[c]; s = rng.choice(v, size=len(v), replace=True)
            cc += [c]*len(s); RR += list(s)
        cc = np.array(cc, float); RR = np.array(RR, float)
        try:
            pH, _ = curve_fit(hill, cc, RR, p0=[max(RR.max(),1e-6), np.median([c for c in cs if c>0]), 1.0],
                              bounds=([0,1e-3,0.2],[np.inf,1e9,5.0]), maxfev=8000)
            ns.append(pH[2])
        except Exception:
            pass
    if len(ns) < 50:
        return None, (None, None), None
    return float(np.median(ns)), tuple(float(x) for x in np.percentile(ns, [2.5, 97.5])), float(np.mean(np.array(ns) > 1))

def main():
    df = pd.read_csv(os.path.join(EXP1, "fit_free_descriptors.csv"))
    rows = []
    for chir in CHIRALITIES:
        sub = response(df, chir, "GT15-STMN2")
        means = sub.groupby("copies").R.mean().reset_index()
        rho, p = spearmanr(sub.copies, sub.R)
        responsive = (p < 0.05 and rho > 0)
        fit = fit_hill_langmuir(means.copies, means.R)
        n_med, (n_lo, n_hi), frac_gt1 = bootstrap_n(sub)
        H = fit.get("hill", {}); L = fit.get("langmuir", {})
        d_aicc = (H.get("aicc", np.nan) - L.get("aicc", np.nan))
        ci_excludes_1 = (n_lo is not None and (n_lo > 1 or n_hi < 1))
        hill_favored = (np.isfinite(d_aicc) and d_aicc < -2)   # >2 AICc units better
        verdict = "no cooperativity evidence"
        if responsive and ci_excludes_1 and hill_favored:
            verdict = ("positive cooperativity" if n_med and n_med > 1 else "negative cooperativity")
        elif responsive and ci_excludes_1:
            verdict = "n!=1 but model comparison not decisive"
        rows.append({
            "chirality": chir.replace("ch","").replace("_",","),
            "responsive": responsive, "spearman_p": float(p),
            "hill_n_pointfit": round(H.get("n", float("nan")), 3),
            "hill_n_boot_median": round(n_med, 3) if n_med else None,
            "hill_n_CI_lo": round(n_lo, 3) if n_lo else None,
            "hill_n_CI_hi": round(n_hi, 3) if n_hi else None,
            "frac_boot_n_gt_1": round(frac_gt1, 3) if frac_gt1 is not None else None,
            "hill_Kd": round(H.get("Kd", float("nan")), 3),
            "hill_R2": round(H.get("r2", float("nan")), 3),
            "langmuir_R2": round(L.get("r2", float("nan")), 3),
            "AICc_hill": round(H.get("aicc", float("nan")), 2),
            "AICc_langmuir": round(L.get("aicc", float("nan")), 2),
            "dAICc_hill_minus_langmuir": round(d_aicc, 2) if np.isfinite(d_aicc) else None,
            "CI_excludes_1": bool(ci_excludes_1), "hill_favored_by_AICc": bool(hill_favored),
            "cooperativity_verdict": verdict,
        })
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(OUT, "hill_cooperativity.tsv"), sep="\t", index=False)

    resp = tab[tab.responsive]
    n_coop = int((resp.cooperativity_verdict.str.contains("positive cooperativity") |
                  resp.cooperativity_verdict.str.contains("negative cooperativity")).sum())
    summary = {
        "stage": "Hill cooperativity test (water, GT15-STMN2)",
        "n_responsive_channels": int(len(resp)),
        "n_channels_with_cooperativity_evidence": n_coop,
        "overall_verdict": ("NO ROBUST COOPERATIVITY: across responsive channels the Hill coefficient's "
            "bootstrap 95%% CI includes n=1 and/or AICc does not favour the 3-parameter Hill model over "
            "Langmuir." if n_coop == 0 else
            f"{n_coop} responsive channel(s) show cooperativity evidence (CI excludes 1 AND AICc favours Hill)."),
        "identifiability_caveat": ("The concentration grid (0,10,100,1e3,1e4,1e5 copies/uL) saturates by "
            "~100 copies/uL (apparent K_d~12), leaving ~2 points on the rising edge; n is therefore weakly "
            "identified and bootstrap CIs are wide. A denser low-copy series (e.g. 1-300 copies/uL) is "
            "needed to estimate n reliably."),
        "ch7_5": resp[resp.chirality == "7,5"].to_dict("records")[:1],
        "utc": utc_now(),
    }
    write_json(os.path.join(OUT, "hill_cooperativity_summary.json"), summary)

    # figure: Hill vs Langmuir on (7,5) and best responsive channel; + bootstrap-n histogram
    best = resp.sort_values("spearman_p").iloc[0]["chirality"] if len(resp) else "8,4"
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4))
    for ax, chname in zip(axes[:2], ["7,5", best]):
        chir = "ch" + chname.replace(",", "_")
        sub = response(df, chir, "GT15-STMN2"); means = sub.groupby("copies").R.agg(["mean","std"]).reset_index()
        ax.errorbar(means.copies.replace(0,3), means["mean"], yerr=means["std"], fmt="o", color="k", capsize=3, label="data (mean±sd)")
        fit = fit_hill_langmuir(means.copies, means["mean"])
        xx = np.logspace(0.5, 5, 200)
        if "Kd" in fit["langmuir"]:
            ax.plot(xx, langmuir(xx, fit["langmuir"]["Rmax"], fit["langmuir"]["Kd"]), "--", color="#4575b4",
                    label=f"Langmuir (n=1) R²={fit['langmuir']['r2']:.2f}")
        if "n" in fit["hill"]:
            ax.plot(xx, hill(xx, fit["hill"]["Rmax"], fit["hill"]["Kd"], fit["hill"]["n"]), "-", color="#d73027",
                    label=f"Hill n={fit['hill']['n']:.2f} R²={fit['hill']['r2']:.2f}")
        ax.set_xscale("log"); ax.set_xlabel("STMN2-CE (copies/µL)"); ax.set_ylabel("fractional NIR response")
        ax.set_title(f"({chname}) water"); ax.legend(fontsize=7)
    # bootstrap n histogram for (7,5)
    sub75 = response(df, "ch7_5", "GT15-STMN2")
    rng = np.random.default_rng(0); by_c = {c: g.R.values for c,g in sub75.groupby("copies")}; cs=sorted(by_c); ns=[]
    for _ in range(3000):
        cc,RR=[],[]
        for c in cs:
            s=rng.choice(by_c[c],size=len(by_c[c]),replace=True); cc+=[c]*len(s); RR+=list(s)
        try:
            pH,_=curve_fit(hill,np.array(cc,float),np.array(RR,float),p0=[max(max(RR),1e-6),11.6,1.0],bounds=([0,1e-3,0.2],[np.inf,1e9,5.0]),maxfev=8000); ns.append(pH[2])
        except Exception: pass
    ax = axes[2]
    ax.hist(ns, bins=30, color="#999"); ax.axvline(1.0, color="k", ls="--", label="n=1 (non-cooperative)")
    if ns: ax.axvline(np.median(ns), color="#d73027", label=f"median n={np.median(ns):.2f}")
    ax.set_xlabel("Hill n (bootstrap, (7,5))"); ax.set_ylabel("count"); ax.legend(fontsize=7)
    ax.set_title("Hill coefficient uncertainty")
    fig.suptitle("Hill cooperativity test — GT15-STMN2 water dose-response (EXPLORATORY; n weakly identified)")
    fig.tight_layout(); figstyle.save_both(fig, os.path.join(FIG, "experimental_hill_cooperativity"))

    print(f"Responsive channels: {len(resp)}; with cooperativity evidence: {n_coop}")
    print(summary["overall_verdict"])
    r75 = resp[resp.chirality=="7,5"]
    if len(r75):
        r=r75.iloc[0]
        print(f"(7,5): Hill n={r['hill_n_boot_median']} CI[{r['hill_n_CI_lo']},{r['hill_n_CI_hi']}] "
              f"frac(n>1)={r['frac_boot_n_gt_1']} dAICc={r['dAICc_hill_minus_langmuir']} -> {r['cooperativity_verdict']}")

if __name__ == "__main__":
    main()

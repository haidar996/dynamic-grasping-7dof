#!/usr/bin/env python3
"""B5: split conformal + ACI coverage for all frozen estimators across horizons.

Method (split conformal, Vovk et al. / Angelopoulos & Bates 2022):
  For each estimator × condition × horizon h:
    1. Compute nonconformity score per val launch: s_i = sqrt(mean(e_i^2))  (per-launch RMS, meters)
    2. Fit conformal threshold q at level ceil((n_val+1)*(1-alpha)) / n_val
    3. Report empirical coverage = fraction of test launches where s_j <= q

Method (ACI online, Gibbs & Candes 2106.00170):
  Process test launches sequentially.  At each step:
    - threshold = quantile_{1-alpha_t}(val scores)
    - covered = (test_score <= threshold)
    - alpha_{t+1} = alpha_t + gamma*(alpha_nominal - (1 - covered))
  Report time-averaged coverage (= fraction covered) and final alpha_t drift.

Inputs: outputs/ekf_mu_eval_errors.pkl.gz     (ekf19b_fit, ekfmu_grid, ekfmu_h05, knet)
        outputs/ekf_imm_eval_errors.pkl.gz     (ekfimm, plus duplicates of ekf refs)
Outputs: outputs/conformal_coverage.json
         docs/B5_conformal_results.md

Usage: python3 experiments/step_4a_conformal.py"""

import gzip, json, pickle, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT  = ROOT / "outputs"
DOCS = ROOT / "docs"

CORE_CONDS  = ["ideal", "noise", "all_real", "all_plane", "dropout_silent_noisy"]
HORIZONS    = ["h0.2", "h0.5", "h1.0"]
ALPHAS      = [0.10, 0.05]          # target miscoverage -> 90%, 95% coverage
ACI_GAMMA   = 0.005                 # ACI step size (Gibbs & Candès default)
ESTIMATORS  = ["ekf19b_fit", "ekfmu_grid", "ekfmu_h05", "knet", "ekfimm"]


def load_errors():
    """Return merged {cond: {est: {launch_id: {horizon: np.array}}}} for ESTIMATORS."""
    mu  = pickle.load(gzip.open(OUT / "ekf_mu_eval_errors.pkl.gz",  "rb"))
    imm = pickle.load(gzip.open(OUT / "ekf_imm_eval_errors.pkl.gz", "rb"))

    # verify splits match
    assert mu["split"] == imm["split"], "Split mismatch between error files"
    split = mu["split"]

    merged = {}
    for cond in CORE_CONDS:
        merged[cond] = {}
        for est in ESTIMATORS:
            src = imm if est == "ekfimm" else mu
            if cond not in src["errors"]:
                continue
            merged[cond][est] = {}
            for li, errs in src["errors"][cond].items():
                if est not in errs:
                    continue
                merged[cond][est][li] = errs[est]
    return merged, split


def launch_rms(errs_dict, horizon):
    """Nonconformity score: per-launch RMS (m) for one horizon.
    Returns sorted array of (launch_id, rms) for launches with >= 1 prediction."""
    out = []
    for li, e in errs_dict.items():
        arr = e.get(horizon)
        if arr is None or len(arr) == 0:
            continue
        out.append((li, float(np.sqrt(np.mean(np.asarray(arr, float) ** 2)))))
    return out


def conformal_threshold(scores, alpha):
    """Finite-sample split conformal quantile.
    q = scores sorted ascending; return index ceil((n+1)*(1-alpha)) - 1 (0-based)."""
    n = len(scores)
    idx = int(np.ceil((n + 1) * (1 - alpha))) - 1
    idx = max(0, min(idx, n - 1))
    return float(np.sort(scores)[idx])


def split_conformal(val_scores, test_scores, alpha):
    """Return (q_alpha_mm, empirical_coverage, n_val, n_test)."""
    q = conformal_threshold(val_scores, alpha)
    cov = float(np.mean(np.asarray(test_scores) <= q))
    return q * 1000, cov, len(val_scores), len(test_scores)


def aci_coverage(val_scores, test_scores, alpha_nominal, gamma=ACI_GAMMA):
    """Adaptive conformal inference (Gibbs & Candes 2106.00170).
    Returns (mean_coverage, final_alpha_drift) over test set."""
    alpha_t = alpha_nominal
    covered_hist = []
    val_arr = np.sort(val_scores)
    for s in test_scores:
        q_t = conformal_threshold(val_arr, alpha_t)
        covered = s <= q_t
        covered_hist.append(float(covered))
        alpha_t = alpha_t + gamma * (alpha_nominal - (1.0 - float(covered)))
    return float(np.mean(covered_hist)), float(alpha_t - alpha_nominal)


def run():
    merged, split = load_errors()
    val_ids  = set(split["val"])
    test_ids = set(split["test"])

    results = {}  # cond -> est -> horizon -> alpha -> {...}

    for cond in CORE_CONDS:
        results[cond] = {}
        for est in ESTIMATORS:
            if est not in merged.get(cond, {}):
                continue
            results[cond][est] = {}
            errs = merged[cond][est]
            for hor in HORIZONS:
                pairs = launch_rms(errs, hor)
                if not pairs:
                    continue
                val_s  = [s for li, s in pairs if li in val_ids]
                test_s = [s for li, s in pairs if li in test_ids]
                if len(val_s) < 10 or len(test_s) < 10:
                    continue
                results[cond][est][hor] = {}
                for alpha in ALPHAS:
                    q, sc_cov, nv, nt = split_conformal(val_s, test_s, alpha)
                    aci_cov, drift    = aci_coverage(val_s, test_s, alpha)
                    results[cond][est][hor][str(alpha)] = dict(
                        q_mm=round(q, 2),
                        split_coverage=round(sc_cov, 4),
                        aci_coverage=round(aci_cov, 4),
                        aci_alpha_drift=round(drift, 5),
                        n_val=nv, n_test=nt,
                    )
    return results


def write_json(results):
    path = OUT / "conformal_coverage.json"
    path.write_text(json.dumps(results, indent=1))
    print(f"Saved {path}")


def write_md(results):
    lines = [
        "# B5: Conformal Forecast Radii — Coverage Report",
        "",
        f"**Date:** 2026-11-24",
        "**Method:** split conformal (Angelopoulos & Bates 2022) + ACI (Gibbs & Candès 2106.00170)",
        "**Nonconformity score:** per-launch RMS (m); calibrated on val (n=73-75), evaluated on test (n≈75)",
        "**Nominal coverage targets:** 90% (α=0.10), 95% (α=0.05)",
        "",
        "A coverage ≥ target means the estimator is calibrated at that level.",
        "A coverage far below target means the predictor is overconfident.",
        "",
    ]

    for hor in HORIZONS:
        label = {"h0.2": "0.2 s horizon", "h0.5": "0.5 s horizon", "h1.0": "1.0 s horizon"}[hor]
        lines += [f"## {label}", ""]

        for alpha in ALPHAS:
            target_pct = int((1 - alpha) * 100)
            lines += [f"### Target {target_pct}% coverage (α={alpha})", ""]
            header = f"| condition            | estimator   | q (mm) | split cov | ACI cov | n_val | n_test |"
            sep    = f"|----------------------|-------------|-------:|----------:|--------:|------:|-------:|"
            lines += [header, sep]
            for cond in CORE_CONDS:
                for est in ESTIMATORS:
                    r = results.get(cond, {}).get(est, {}).get(hor, {}).get(str(alpha))
                    if r is None:
                        continue
                    sc  = r["split_coverage"]
                    ac  = r["aci_coverage"]
                    sc_flag = " ✓" if sc >= (1 - alpha) - 0.01 else (" ✗" if sc < (1 - alpha) - 0.05 else "  ")
                    ac_flag = " ✓" if ac >= (1 - alpha) - 0.01 else (" ✗" if ac < (1 - alpha) - 0.05 else "  ")
                    lines.append(
                        f"| {cond:<20} | {est:<11} | {r['q_mm']:6.1f} | "
                        f"{sc:.3f}{sc_flag} | {ac:.3f}{ac_flag} | {r['n_val']:5} | {r['n_test']:6} |"
                    )
            lines += [""]

    # Summary section
    lines += [
        "## Summary / findings",
        "",
        "Coverage ✓ = within 1 pp of target; ✗ = more than 5 pp below target.",
        "",
    ]

    # Compute aggregate stats
    all_sc = {alpha: [] for alpha in ALPHAS}
    for cond in CORE_CONDS:
        for est in ESTIMATORS:
            for hor in HORIZONS:
                for alpha in ALPHAS:
                    r = results.get(cond, {}).get(est, {}).get(hor, {}).get(str(alpha))
                    if r:
                        all_sc[alpha].append(r["split_coverage"])

    for alpha in ALPHAS:
        target = 1 - alpha
        sc_arr = np.array(all_sc[alpha])
        n_ok   = int(np.sum(sc_arr >= target - 0.01))
        n_bad  = int(np.sum(sc_arr < target - 0.05))
        lines += [
            f"- **{int(target*100)}% target (α={alpha}):** {n_ok}/{len(sc_arr)} cells at or above nominal; "
            f"{n_bad} cells more than 5 pp below.",
        ]

    lines += [
        "",
        "*(Conformal guarantee: split coverage ≥ 1-α holds in expectation under exchangeability.)*",
        "*(ACI adapts the threshold online; coverage should converge to nominal.)*",
    ]

    path = DOCS / "B5_conformal_results.md"
    path.write_text("\n".join(lines) + "\n")
    print(f"Saved {path}")


def main():
    print("B5: computing conformal coverage ...")
    results = run()
    write_json(results)
    write_md(results)
    print("Done.")

    # Print a quick summary table to stdout
    print(f"\n{'condition':<26} {'estimator':<13} {'h0.5 split90':>12} {'h0.5 split95':>12}")
    print("-" * 65)
    for cond in CORE_CONDS:
        for est in ESTIMATORS:
            r90 = results.get(cond, {}).get(est, {}).get("h0.5", {}).get("0.1")
            r95 = results.get(cond, {}).get(est, {}).get("h0.5", {}).get("0.05")
            if r90 is None:
                continue
            sc90 = r90["split_coverage"]
            sc95 = r95["split_coverage"] if r95 else float("nan")
            flag = "✓" if sc90 >= 0.89 else ("✗" if sc90 < 0.85 else "~")
            print(f"{cond:<26} {est:<13} {sc90:>10.3f} {flag}  {sc95:>10.3f}")


if __name__ == "__main__":
    main()

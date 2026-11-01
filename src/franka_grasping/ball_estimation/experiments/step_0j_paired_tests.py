#!/usr/bin/env python3
"""Step 0.8c (research audit A3): paired comparisons of the Phase-0 baselines, from the saved per-frame errors
(outputs/baselines_v3_errors.pkl.gz, written by step_0d).  Paired = same launches, same noise seeds.
For each comparison A vs B: launch-mean difference (A - B, negative = A better) with 95 % bootstrap interval,
Wilcoxon signed-rank p-value, fraction of launches where A is better.  Reported on TEST (dev) and VAL.
Output: outputs/paired_tests_v3.json
Usage: python3 experiments/step_0j_paired_tests.py"""
import sys, json, gzip, pickle
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from estimators.stats import paired

D = pickle.load(gzip.open(root / "outputs" / "baselines_v3_errors.pkl.gz", "rb"))
E = D["errors"]
EKF, FIT, ORA = "ekf19b-none+stale", "ekf19b_fit-none+stale", "ekf19b_oracle-none+stale"

# (label, condition A, estimator A, condition B, estimator B)
COMPARISONS = []
for c in E:
    COMPARISONS += [(f"{c}: ekf19b_fit vs ekf19b (train-fitted c,E vs midpoints)", c, FIT, c, EKF),
                    (f"{c}: oracle vs ekf19b (gap)", c, ORA, c, EKF),
                    (f"{c}: ekf19b-hybrid vs ekf19b+stale (gate)", c, "ekf19b-hybrid", c, EKF),
                    (f"{c}: ekf19b+stale vs cvkf", c, EKF, c, "cvkf")]
for a, b in (("plane", "noise_real"), ("all_plane", "all_real"), ("noise", "ideal"), ("noise_real", "noise"),
             ("dropout_silent_noisy", "noise")):
    if a in E and b in E:
        COMPARISONS += [(f"condition {a} vs {b}: ekf19b+stale", a, EKF, b, EKF),
                        (f"condition {a} vs {b}: oracle", a, ORA, b, ORA)]


def main():
    out = {}
    for split in ("test", "val"):
        ids = D["split"][split]
        print(f"\n===== {split.upper()}{' (dev)' if split == 'test' else ''}: {len(ids)} launches; "
              f"mean diff A-B in mm [95% CI], Wilcoxon p, A better in % of launches =====")
        for key in ("h0.5", "now_pos"):
            print(f"\n  -- {key}")
            for label, ca, ea, cb, eb in COMPARISONS:
                r = paired([E[ca][i][ea] for i in ids], [E[cb][i][eb] for i in ids], key)
                out.setdefault(split, {}).setdefault(key, {})[label] = r
                if "mean_diff" not in r:
                    continue
                print(f"  {label:68s} {1000 * r['mean_diff']:+7.1f} [{1000 * r['mean_diff_ci'][0]:+6.1f},"
                      f"{1000 * r['mean_diff_ci'][1]:+6.1f}]  p {r['wilcoxon_p']:.1e}  {100 * r['a_better_frac']:3.0f}%  (n {r['n']})")
    (root / "outputs" / "paired_tests_v3.json").write_text(json.dumps(out, indent=1))
    print("\nsaved outputs/paired_tests_v3.json")


if __name__ == "__main__":
    main()

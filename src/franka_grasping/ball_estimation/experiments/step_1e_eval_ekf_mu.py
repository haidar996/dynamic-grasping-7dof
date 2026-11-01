#!/usr/bin/env python3
"""Step 1.2e: evaluate ekf-mu against the baselines on VAL and TEST (dev), matched physics, every sensor condition.
Estimators (all with StaleGuard, no gate):
  ekf19b         c, E = generator midpoints            (noise: val grid, family "ekf")
  ekf19b_fit     c, E fitted on train (step_0k)        (noise: val grid, family "ekf_fit")
  ekfmu_grid     mu in the state, val grid (step_1d)
  ekfmu_nll      mu in the state, 7 numbers learned by maximum likelihood on train (step_1c nll)  <- paper 03
  ekfmu_h05      same 7 numbers learned on train by minimising h0.5 (step_1c h05)                 <- ablation
  oracle         true mu, E (family "oracle")
Reported (estimators/stats.py): per-launch RMS -> launch mean / median with 95 % bootstrap CI; paired tests vs ekf19b_fit;
mu recovery |mu_hat - mu| at fixed times after the first frame (moving balls); consistency: mean NIS (expected 2) and
mean NEES of (x, y, vx, vy) (expected 4); run time per update.
Output: outputs/ekf_mu_eval.json, outputs/ekf_mu_eval_errors.pkl.gz
Usage: python3 experiments/step_1e_eval_ekf_mu.py [conds|all_conds] [workers=3]"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import sys, json, gzip, pickle, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import mujoco

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from arena_sim import Arena
from sensing.camera import Pinhole
from dataset import CONDITIONS, apply_condition
from estimators.filters import make
from estimators.metrics import evaluate
from estimators.stats import summary, paired

RAW = root / "outputs" / "raw_60hz_h3_f49_sq480_with_arm"
SPLIT = json.loads((root / "split_500.json").read_text())
GEO = Arena(mujoco.MjModel.from_xml_path(str(root / "scenes" / "arena.xml"))).estimator_geometry()
cam = Pinhole(3.0, 49.0, 480, 480)
OUT = root / "outputs"
TUNED = json.loads((OUT / "ekf_tuning_val.json").read_text())["best"]


def _load(name):
    f = OUT / name
    return json.loads(f.read_text()) if f.exists() else {}


MU_PARAMS = {"ekfmu_grid": _load("ekf_mu_grid_val.json"), "ekfmu_nll": _load("ekf_mu_learned_nll.json"),
             "ekfmu_h05": _load("ekf_mu_learned_h05.json")}
BASELINES = {"ekf19b": ("ekf19b-none+stale", "ekf"), "ekf19b_fit": ("ekf19b_fit-none+stale", "ekf_fit"),
             "oracle": ("ekf19b_oracle-none+stale", "oracle")}
KEYS = ["now_pos", "now_vel", "h0.2", "h0.5", "h1.0"]
MU_TIMES = (0.1, 0.25, 0.5, 1.0, 2.0)


def estimators(cond):
    """label -> (make() name, kwargs) for the estimators that have parameters for this condition."""
    out = {}
    for lab, (name, fam) in BASELINES.items():
        t = TUNED[fam][cond]
        out[lab] = (name, {k: t[k] for k in ("sigma", "q", "c", "E") if k in t})
    for lab, d in MU_PARAMS.items():
        if cond in d:
            out[lab] = (lab + "+stale", dict(d[cond]["params"]))
    return out


def diagnostics(est, rec, every=6, warmup=0.1, v_min=0.05):
    """Second pass: NIS, NEES (x, y, vx, vy at the filter time), mu-hat at fixed times, time per update."""
    est.reset()
    tp, pp, vp = rec["t_phys"], rec["pos"], rec["vel"]
    nees, mu_hat, first, t_upd = [], {}, None, 0.0
    for k in range(len(rec["t_capture"])):
        t0 = time.perf_counter()
        est.update(rec["t_capture"][k], rec["xy"][k], rec["valid"][k])
        t_upd += time.perf_counter() - t0
        if not est.ready:
            continue
        if first is None:
            first = rec["t_capture"][k]
        tt = est.t
        tv = np.array([np.interp(tt, tp, vp[:, 0]), np.interp(tt, tp, vp[:, 1])])
        moving = np.hypot(*tv) >= v_min
        if hasattr(est, "mu"):
            for T in MU_TIMES:
                if T not in mu_hat and rec["t_capture"][k] - first >= T and moving:
                    mu_hat[T] = est.mu
        if k % every or rec["t_capture"][k] - first < warmup or not moving:
            continue
        truth = np.array([np.interp(tt, tp, pp[:, 0]), np.interp(tt, tp, pp[:, 1]), *tv])
        e = est.s[:4] - truth
        nees.append(float(e @ np.linalg.solve(est.P[:4, :4], e)))
    inner = est.est if hasattr(est, "est") else est
    return dict(nis=np.array(inner.nis), nees=np.array(nees), mu_hat=mu_hat,
                ms_per_update=1000 * t_upd / len(rec["t_capture"]))


def task(args):
    i, cond = args
    rec = apply_condition(cam, np.load(RAW / f"launch_{i:04d}.npz"), CONDITIONS[cond], 1000 + i)
    out = {}
    for lab, (name, kw) in estimators(cond).items():
        est = make(name, rec, **kw, **GEO)
        out[lab] = evaluate(est, rec)
        out[lab]["diag"] = diagnostics(make(name, rec, **kw, **GEO), rec)
    return i, out, float(rec["mu"])


def report(res, ids, mu_true, labs, title):
    print(f"\n  [{title}: {len(ids)} launches] per-launch RMS (mm; now_vel m/s): launch-mean [95% CI] / launch-median")
    print(f"  {'estimator':12s} " + " ".join(f"{k:>24s}" for k in KEYS) + f" {'NIS':>6s} {'NEES':>6s} {'ms/upd':>6s}")
    tab = {}
    for lab in labs:
        items = [res[i][lab] for i in ids]
        tab[lab] = {k: summary(items, k) for k in KEYS}
        nis = np.concatenate([o["diag"]["nis"] for o in items]); nees = np.concatenate([o["diag"]["nees"] for o in items])
        tab[lab]["consistency"] = dict(nis_mean=float(nis.mean()), nees_mean=float(nees.mean()),
                                       nis_median=float(np.median(nis)), nees_median=float(np.median(nees)))
        tab[lab]["ms_per_update"] = float(np.mean([o["diag"]["ms_per_update"] for o in items]))

        def cell(k):
            s = tab[lab][k]; f = 1 if k == "now_vel" else 1000; d = 2 if k == "now_vel" else 0
            return f"{s['mean'] * f:.{d}f} [{s['mean_ci'][0] * f:.{d}f},{s['mean_ci'][1] * f:.{d}f}]/{s['median'] * f:.{d}f}"
        print(f"  {lab:12s} " + " ".join(f"{cell(k):>24s}" for k in KEYS) +
              f" {nis.mean():6.2f} {nees.mean():6.1f} {tab[lab]['ms_per_update']:6.2f}")
    print("  paired vs ekf19b_fit, h0.5 (A - B, mm, negative = A better):")
    for lab in labs:
        if lab == "ekf19b_fit":
            continue
        r = paired([res[i][lab] for i in ids], [res[i]["ekf19b_fit"] for i in ids], "h0.5")
        tab[lab]["paired_vs_ekf19b_fit_h05"] = r
        print(f"    {lab:12s} {1000 * r['mean_diff']:+7.1f} [{1000 * r['mean_diff_ci'][0]:+6.1f},{1000 * r['mean_diff_ci'][1]:+6.1f}]"
              f"  Wilcoxon p {r['wilcoxon_p']:.1e}  better in {100 * r['a_better_frac']:3.0f}% of launches")
    print("  mu recovery: median |mu_hat - mu| (m/s^2) at time after the first frame (moving balls only); prior 0.85 -> "
          f"{np.median([abs(0.85 - mu_true[i]) for i in ids]):.2f}")
    for lab in labs:
        if not lab.startswith("ekfmu"):
            continue
        row = {}
        for T in MU_TIMES:
            e = [abs(res[i][lab]["diag"]["mu_hat"][T] - mu_true[i]) for i in ids if T in res[i][lab]["diag"]["mu_hat"]]
            row[T] = (float(np.median(e)) if e else float("nan"), len(e))
        tab[lab]["mu_recovery"] = {str(T): v for T, v in row.items()}
        print(f"    {lab:12s} " + "  ".join(f"{T:4.2f} s: {m:.3f} (n {n})" for T, (m, n) in row.items()))
    return tab


def main():
    conds = sys.argv[1].split(",") if len(sys.argv) > 1 and sys.argv[1] != "all_conds" else list(CONDITIONS)
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    val, test = SPLIT["val"], SPLIT["test"]
    ids = sorted(val + test)
    summ, errors = {}, {}
    with Pool(workers) as pool:
        for cond in conds:
            t0 = time.time()
            out = pool.map(task, [(i, cond) for i in ids], chunksize=1)
            res = {i: o for i, o, _ in out}; mu_true = {i: m for i, _, m in out}
            labs = list(estimators(cond))
            print(f"\n{'=' * 140}\nCondition {cond}  ({time.time() - t0:.0f} s)  estimators: {', '.join(labs)}\n{'=' * 140}")
            summ[cond] = dict(test=report(res, test, mu_true, labs, "TEST (dev)"), val=report(res, val, mu_true, labs, "VAL"))
            errors[cond] = {i: {lab: {k: res[i][lab][k] for k in KEYS} for lab in labs} for i in ids}
    (OUT / "ekf_mu_eval.json").write_text(json.dumps(summ, indent=1, default=float))
    with gzip.open(OUT / "ekf_mu_eval_errors.pkl.gz", "wb") as f:
        pickle.dump(dict(split=dict(val=val, test=test), errors=errors), f)
    print("\nsaved outputs/ekf_mu_eval.json, ekf_mu_eval_errors.pkl.gz")


if __name__ == "__main__":
    main()

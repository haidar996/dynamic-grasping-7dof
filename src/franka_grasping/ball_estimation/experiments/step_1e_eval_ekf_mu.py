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
Output: outputs/ekf_mu_eval.json, outputs/ekf_mu_eval_errors.pkl.gz (merged: only the conditions run now are replaced)
Usage: python3 experiments/step_1e_eval_ekf_mu.py [conds|all_conds] [workers=3]

Mismatch physics (audit A1, design docs/A1_ekfmu_mismatch_eval_design.md, 2026-11-10): PHYSICS=mismatch ARM=transfer|retrain
  transfer: every parameter as learned / tuned on MATCHED data, applied unchanged    -> outputs/ekf_mu_eval_mismatch_transfer.*
  retrain : every parameter re-learned / re-tuned on MISMATCH data (*_mismatch.json) -> outputs/ekf_mu_eval_mismatch_retrain.*
  The oracle (true mu, E) does not exist there; it is replaced by "hindsight" = ekf19b_fit with the per-launch best constant
  pair of Step A (outputs/mismatch_hindsight.json, "ekf_pair", picked on ideal) and the MATCHED "ekf" noise of the
  condition: exactly Step A's "best mu & E" bound, in both arms (a fixed reference).  mu recovery is replaced by the
  correlation of mu_hat with the true rolling resistance a0 (descriptive), and "gap closed" = (ekf19b_fit - X) /
  (ekf19b_fit - hindsight) on launch-mean h0.5 is added."""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import sys, json, gzip, pickle, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import mujoco
from scipy.stats import pearsonr, spearmanr

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from arena_sim import Arena
from sensing.camera import Pinhole
from dataset import CONDITIONS, apply_condition
from estimators.filters import make
from estimators.metrics import evaluate
from estimators.stats import summary, paired

import physics_cfg as phys
RAW = phys.RAW
ARM = os.environ.get("ARM", "")
assert (ARM in ("transfer", "retrain")) if phys.MISMATCH else ARM in ("", "matched"), \
    "PHYSICS=mismatch needs ARM=transfer|retrain; ARM is not used on matched physics"
PSFX = "_mismatch" if ARM == "retrain" else ""            # which learned / tuned files the parameters come from
SPLIT = json.loads((root / "split_500.json").read_text())
GEO = Arena(mujoco.MjModel.from_xml_path(str(root / "scenes" / "arena.xml"))).estimator_geometry()
cam = Pinhole(3.0, 49.0, 480, 480)
OUT = root / "outputs"
TUNED_MATCHED = json.loads((OUT / "ekf_tuning_val.json").read_text())["best"]
TUNED = json.loads(phys.out_path("ekf_tuning_val.json", PSFX).read_text())["best"]
HS = ({int(k): v["ekf_pair"] for k, v in json.loads((OUT / "mismatch_hindsight.json").read_text())["launches"].items()}
      if phys.MISMATCH else {})


def _load(name):
    f = OUT / name
    return json.loads(f.read_text()) if f.exists() else {}


MU_PARAMS = {"ekfmu_grid": _load(f"ekf_mu_grid_val{PSFX}.json"), "ekfmu_nll": _load(f"ekf_mu_learned_nll{PSFX}.json"),
             "ekfmu_h05": _load(f"ekf_mu_learned_h05{PSFX}.json")}
BASELINES = {"ekf19b": ("ekf19b-none+stale", "ekf"), "ekf19b_fit": ("ekf19b_fit-none+stale", "ekf_fit"),
             "oracle": ("ekf19b_oracle-none+stale", "oracle")}
if phys.MISMATCH:
    del BASELINES["oracle"]
KEYS = ["now_pos", "now_vel", "h0.2", "h0.5", "h1.0"]
MU_TIMES = (0.1, 0.25, 0.5, 1.0, 2.0)


def estimators(cond):
    """label -> (make() name, kwargs) for the estimators that have parameters for this condition."""
    out = {}
    for lab, (name, fam) in BASELINES.items():
        assert cond in TUNED[fam], f"no {fam} noise tuning for {cond} ({phys.PHYSICS}, ARM={ARM or 'matched'})"
        t = TUNED[fam][cond]
        out[lab] = (name, {k: t[k] for k in ("sigma", "q", "c", "E") if k in t})
    if phys.MISMATCH:                                     # c, E filled in per launch from HS (task)
        t = TUNED_MATCHED["ekf"][cond]
        out["hindsight"] = ("ekf19b_fit-none+stale", dict(sigma=t["sigma"], q=t["q"]))
    for lab, d in MU_PARAMS.items():
        if cond in d:
            out[lab] = (lab + "+stale", dict(d[cond]["params"]))
    w = OUT / f"knet_{cond}{PSFX}.pt"                       # P7 learned gain: only when trained for this physics / arm
    if w.exists():
        t = TUNED["ekf_fit"][cond]
        out["knet"] = ("knet", dict(weights=str(w), c=t["c"], E=t["E"]))
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
        if getattr(est, "P", None) is None:               # knet has no covariance: no NEES
            continue
        truth = np.array([np.interp(tt, tp, pp[:, 0]), np.interp(tt, tp, pp[:, 1]), *tv])
        e = est.s[:4] - truth
        nees.append(float(e @ np.linalg.solve(est.P[:4, :4], e)))
    inner = est.est if hasattr(est, "est") else est
    return dict(nis=np.array(inner.nis), nees=np.array(nees), mu_hat=mu_hat,
                ms_per_update=1000 * t_upd / len(rec["t_capture"]))


def task(args):
    i, cond = args
    raw = np.load(RAW / f"launch_{i:04d}.npz")
    rec = apply_condition(cam, raw, CONDITIONS[cond], 1000 + i)
    out = {}
    for lab, (name, kw) in estimators(cond).items():
        if lab == "hindsight":
            kw = dict(kw, c=HS[i]["mu"], E=HS[i]["E"])
        est = make(name, rec, **kw, **GEO)
        out[lab] = evaluate(est, rec)
        out[lab]["diag"] = diagnostics(make(name, rec, **kw, **GEO), rec)
    return i, out, float(raw["phys_a0"] if phys.MISMATCH else rec["mu"])     # mismatch: true a0 (no true mu)


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
    if phys.MISMATCH:
        ref = "hindsight"
        print("  gap closed = (ekf19b_fit - X) / (ekf19b_fit - hindsight), launch-mean h0.5:")
        for lab in labs:
            if lab in ("ekf19b_fit", ref):
                continue
            f, x, h = (tab[k]["h0.5"]["mean"] for k in ("ekf19b_fit", lab, ref))
            tab[lab]["gap_closed_h05"] = float((f - x) / (f - h)) if f != h else float("nan")
            print(f"    {lab:12s} {tab[lab]['gap_closed_h05']:+.2f}")
        print("  mu_hat vs true rolling resistance a0 (no true mu on mismatch data; descriptive): Pearson r / Spearman rho (n)")
        for lab in labs:
            if not lab.startswith("ekfmu"):
                continue
            row = {}
            for T in MU_TIMES:
                pr = [(res[i][lab]["diag"]["mu_hat"][T], mu_true[i]) for i in ids if T in res[i][lab]["diag"]["mu_hat"]]
                if len(pr) > 3:
                    a, b = np.array(pr).T
                    row[T] = (float(pearsonr(a, b)[0]), float(spearmanr(a, b)[0]), len(pr))
                else:
                    row[T] = (float("nan"), float("nan"), len(pr))
            tab[lab]["mu_hat_vs_a0"] = {str(T): v for T, v in row.items()}
            print(f"    {lab:12s} " + "  ".join(f"{T:4.2f} s: {r:+.2f}/{rho:+.2f} ({n})" for T, (r, rho, n) in row.items()))
        return tab
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
    quick = "--quick" in sys.argv                        # smoke test: 5 val + 5 test launches, *_quick output files
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    conds = argv[0].split(",") if argv and argv[0] != "all_conds" else list(CONDITIONS)
    workers = int(argv[1]) if len(argv) > 1 else 3
    val, test = SPLIT["val"], SPLIT["test"]
    if quick:
        val, test = val[::15], test[::15]
    ids = sorted(val + test)
    # 2026-11-08: MERGE into existing outputs (run per condition / section): only the conditions run now are replaced
    sfx = (f"_mismatch_{ARM}" if phys.MISMATCH else "") + ("_quick" if quick else "")
    jpath, ppath = phys.out_path("ekf_mu_eval.json", sfx), phys.out_path("ekf_mu_eval_errors.pkl.gz", sfx)
    summ = json.loads(jpath.read_text()) if jpath.exists() else {}
    errors = pickle.load(gzip.open(ppath, "rb"))["errors"] if ppath.exists() else {}
    with Pool(workers) as pool:
        for cond in conds:
            missing = [lab for lab, d in MU_PARAMS.items() if cond not in d]
            if missing:
                print(f"WARNING {cond}: no parameters yet for {missing} -> evaluated without them", flush=True)
            t0 = time.time()
            out = pool.map(task, [(i, cond) for i in ids], chunksize=1)
            res = {i: o for i, o, _ in out}; mu_true = {i: m for i, _, m in out}
            labs = list(estimators(cond))
            print(f"\n{'=' * 140}\nCondition {cond}  [{phys.PHYSICS}{', ARM=' + ARM if ARM else ''}]  ({time.time() - t0:.0f} s)  estimators: {', '.join(labs)}\n{'=' * 140}")
            summ[cond] = dict(test=report(res, test, mu_true, labs, "TEST (dev)"), val=report(res, val, mu_true, labs, "VAL"),
                              evaluated=time.strftime("%Y-%m-%d %H:%M"), missing=missing,
                              **(dict(physics=phys.PHYSICS, arm=ARM) if phys.MISMATCH else {}))
            errors[cond] = {i: {lab: {k: res[i][lab][k] for k in KEYS} for lab in labs} for i in ids}
            jpath.write_text(json.dumps(summ, indent=1, default=float))          # saved after every condition
            with gzip.open(ppath, "wb") as f:
                pickle.dump(dict(split=dict(val=val, test=test), errors=errors), f)
            print(f"saved {cond} -> {jpath.name}, {ppath.name} (conditions in file: {list(summ)})", flush=True)


if __name__ == "__main__":
    main()

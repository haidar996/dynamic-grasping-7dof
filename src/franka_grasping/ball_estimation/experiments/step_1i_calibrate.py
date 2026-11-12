#!/usr/bin/env python3
"""P3 (plan 2026-11-11): post-hoc covariance calibration of ekf-mu by one scalar inflation alpha per condition and variant.
The h05-trained filter forecasts best but its covariance is too small (NIS 0.1-0.7, expected 2); the NLL-trained one has NIS ~ 2
but NEES >> 4.  A planner needs an honest covariance, so: scale the filter covariance P by alpha (P' = alpha P; the mean and
gains are untouched, this is a reporting / downstream-use correction).  alpha is chosen on VAL only, then applied on TEST (dev):
  alpha_nis : solves mean NIS(alpha) = 2 on val,   NIS(alpha) = y^T (alpha H P^- H^T + R)^-1 y      <- needs no ground truth (deployable)
  alpha_nees: mean NEES(alpha) = 4 on val,         NEES(alpha) = NEES / alpha                      <- needs truth (simulation reference)
Same sampling as step_1e.diagnostics (NEES every 6th frame, warm-up 0.1 s, moving ball; NIS at every accepted measurement).
Usage: [PHYSICS=mismatch ARM=transfer|retrain] python3 experiments/step_1i_calibrate.py [conds] [workers=3] [--quick]
Output: outputs/ekf_mu_calibration[_mismatch_<arm>][_quick].json (merged per condition)."""
import os, sys, json, time, importlib.util
from multiprocessing import Pool
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
spec = importlib.util.spec_from_file_location("s1e", root / "experiments" / "step_1e_eval_ekf_mu.py")
E = importlib.util.module_from_spec(spec); sys.modules["s1e"] = E; spec.loader.exec_module(E)
from estimators.ekf_mu import EKFMu

CORE = ["ideal", "noise", "all_real", "all_plane", "dropout_silent_noisy"]
EVERY, WARMUP, V_MIN = 6, 0.1, 0.05


def run_one(rec, kw):
    """EKFMu with StaleGuard logic inline; returns per-update (y, HPH^T) and per-sample NEES."""
    est = EKFMu(**kw, **E.GEO)
    tp, pp, vp = rec["t_phys"], rec["pos"], rec["vel"]
    ys, hph, nees, last, first = [], [], [], None, None
    for k in range(len(rec["t_capture"])):
        t, xy, valid = rec["t_capture"][k], rec["xy"][k], bool(rec["valid"][k])
        if valid:                                          # StaleGuard: exact repeat -> no measurement
            z = np.asarray(xy, float)
            if last is not None and np.array_equal(z, last):
                valid = False
            last = z
        if est.ready and t > est.t:
            est._predict(t - est.t); est.t = t
            if valid:
                ys.append(np.asarray(xy, float) - est.s[:2]); hph.append(est.P[:2, :2].copy())
        est.update(t, xy, valid)
        if not est.ready:
            continue
        if first is None:
            first = t
        tv = np.array([np.interp(est.t, tp, vp[:, 0]), np.interp(est.t, tp, vp[:, 1])])
        if k % EVERY or t - first < WARMUP or np.hypot(*tv) < V_MIN:
            continue
        e = est.s[:4] - np.array([np.interp(est.t, tp, pp[:, 0]), np.interp(est.t, tp, pp[:, 1]), *tv])
        nees.append(float(e @ np.linalg.solve(est.P[:4, :4], e)))
    return dict(y=np.array(ys).reshape(-1, 2), hph=np.array(hph).reshape(-1, 2, 2), R=float(est.R), nees=np.array(nees))


def task(args):
    i, cond = args
    rec = E.apply_condition(E.cam, np.load(E.RAW / f"launch_{i:04d}.npz"), E.CONDITIONS[cond], 1000 + i)
    return i, {lab: run_one(rec, kw) for lab, (name, kw) in E.estimators(cond).items() if lab.startswith("ekfmu")}


def nis_mean(items, alpha):
    y = np.concatenate([d["y"] for d in items]); hph = np.concatenate([d["hph"] for d in items])
    R = items[0]["R"]
    S = alpha * hph + R * np.eye(2)
    return float(np.mean(np.einsum("ni,nij,nj->n", y, np.linalg.inv(S), y)))


def solve_alpha(items, target=2.0):
    lo, hi = -8.0, 10.0                                    # log alpha; NIS decreases with alpha
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if nis_mean(items, np.exp(mid)) > target:
            lo = mid
        else:
            hi = mid
    return float(np.exp(0.5 * (lo + hi)))


def main():
    quick = "--quick" in sys.argv
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    conds = argv[0].split(",") if argv and argv[0] != "core" else CORE
    workers = int(argv[1]) if len(argv) > 1 else 3
    val, test = E.SPLIT["val"], E.SPLIT["test"]
    if quick:
        val, test = val[::15], test[::15]
    sfx = (f"_mismatch_{E.ARM}" if E.phys.MISMATCH else "") + ("_quick" if quick else "")
    out_path = E.phys.out_path("ekf_mu_calibration.json", sfx)
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    with Pool(workers) as pool:
        for cond in conds:
            t0 = time.time()
            res = dict(pool.map(task, [(i, cond) for i in sorted(val + test)], chunksize=1))
            labs = list(next(iter(res.values())))
            print(f"\n== {cond} [{E.phys.PHYSICS}{', ARM=' + E.ARM if E.ARM else ''}] ({time.time() - t0:.0f} s): alpha chosen on VAL, applied on TEST (dev)")
            print(f"  {'variant':11s} {'alpha_nis':>9s} {'alpha_nees':>10s} | test NIS before -> nis-cal / nees-cal | test NEES before -> nis-cal / nees-cal | val NEES before")
            row = {}
            for lab in labs:
                iv = [res[i][lab] for i in val]; it = [res[i][lab] for i in test]
                a_nis = solve_alpha(iv)
                nees_v = np.concatenate([d["nees"] for d in iv]).mean()
                a_nees = float(nees_v / 4.0)
                nees_t = np.concatenate([d["nees"] for d in it]).mean()
                r = dict(alpha_nis=a_nis, alpha_nees=a_nees,
                         val=dict(nis_before=nis_mean(iv, 1.0), nees_before=float(nees_v)),
                         test=dict(nis_before=nis_mean(it, 1.0), nis_cal_nis=nis_mean(it, a_nis), nis_cal_nees=nis_mean(it, a_nees),
                                   nees_before=float(nees_t), nees_cal_nis=float(nees_t / a_nis), nees_cal_nees=float(nees_t / a_nees)),
                         n_val_updates=int(sum(len(d["y"]) for d in iv)), n_test_nees=int(sum(len(d["nees"]) for d in it)))
                row[lab] = r; t = r["test"]
                print(f"  {lab:11s} {a_nis:9.3g} {a_nees:10.3g} | {t['nis_before']:6.2f} -> {t['nis_cal_nis']:5.2f} / {t['nis_cal_nees']:5.2f}"
                      f"          | {t['nees_before']:7.1f} -> {t['nees_cal_nis']:6.1f} / {t['nees_cal_nees']:5.1f}        | {nees_v:6.1f}", flush=True)
            out[cond] = dict(variants=row, physics=E.phys.PHYSICS, arm=E.ARM, split=dict(val=len(val), test=len(test)),
                             when=time.strftime("%Y-%m-%d %H:%M"))
            out_path.write_text(json.dumps(out, indent=1))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()

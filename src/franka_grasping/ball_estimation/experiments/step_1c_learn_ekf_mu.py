#!/usr/bin/env python3
"""Step 1.2c: learn the 7 numbers of ekf-mu on the TRAIN launches, per sensor condition (paper 03, Eq. 7).
  psi = (sigma, q, q_mu, mu0, s_l, sigma_v0, E): measurement noise, white-acceleration noise, random walk of log mu,
        prior mean and std of log mu, prior velocity std, restitution.  Optimised in log space (E directly), with bounds.
  criterion "nll": mean negative log-likelihood of the innovations per update (main method, paper 03)
  criterion "h05": launch-mean h0.5 RMS (ablation: train directly on the forecast we report)
Optimiser: scipy L-BFGS-B; gradient by central differences, the 15 evaluations of one gradient run in a pool of workers
(each holds the condition's 350 launches; estimators/ekf_mu_batch.py, 2-3 s per evaluation).  One start per condition
(from the val-tuned ekf19b noise; N_STARTS = 2 adds a low-process-noise start) + early stop (TOL, PATIENCE below).
First version (2 starts, no early stop, 71 min for nll/ideal) kept: outputs/ekf_mu_learned_nll_v1_2starts.json.
Afterwards the learned filter is scored on VAL (NLL and launch-mean h0.5) — val is not used for learning.
Output: outputs/ekf_mu_learned_<criterion>.json
Usage: python3 experiments/step_1c_learn_ekf_mu.py nll|h05 [conds|all_conds] [workers=3] [--quick] [--redo]
Time budget: env RUN_DEADLINE (unix time): a condition starts only if it is expected to finish before it (estimate = 1.1 x
mean minutes of the conditions already learned); otherwise exit code 3, nothing lost, rerun resumes."""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import sys, json, math, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import mujoco
from scipy.optimize import minimize

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from arena_sim import Arena
from sensing.camera import Pinhole
from dataset import CONDITIONS, apply_condition
from estimators import ekf_mu_batch as eb
from estimators.stats import per_launch

RAW = root / "outputs" / "raw_60hz_h3_f49_sq480_with_arm"
SPLIT = json.loads((root / "split_500.json").read_text())
GEO = Arena(mujoco.MjModel.from_xml_path(str(root / "scenes" / "arena.xml"))).estimator_geometry()
# start values = current val tuning (2026-11-07 fix: was ekf_tuning_val_v2_phase0.json = pre-audit, no plane/all_plane/
# dropout_silent_noisy -> KeyError on those conditions, outdated sigma/q on the others)
BASE = json.loads((root / "outputs" / "ekf_tuning_val.json").read_text())["best"]["ekf"]
cam = Pinhole(3.0, 49.0, 480, 480)
NAMES = ["sigma", "q", "q_mu", "mu0", "s_l", "sigma_v0", "E"]
LOG = [True, True, True, True, True, True, False]
BOUNDS = [(1e-4, 0.1), (1e-4, 100.0), (1e-9, 1.0), (0.1, 2.0), (0.01, 3.0), (0.05, 10.0), (0.3, 0.99)]
SIGMA_E = 0.13                                     # extra velocity variance at a hit, as ekf19b (not learned)
EPS = {"nll": 1e-3, "h05": 2e-2}                   # finite-difference step in theta space (h05 is less smooth)
DEADLINE = float(os.environ.get("RUN_DEADLINE", "0"))  # unix time; 0 = no budget.  Set by run_overnight.sh --hours H
EST_MIN = {"nll": 15.0, "h05": 22.0}               # minutes per condition before any is finished (estimate)
# Early stop (2026-11-08): the first full run (nll, ideal) reached its final objective within ~20-40 gradient evaluations
# (change < 0.01 afterwards) and then spent 60-100 more in failing line searches (noisy central differences ->
# ABNORMAL_TERMINATION_IN_LNSRCH): 71 min for one condition.  Rule: stop when the best objective improved by less than
# TOL over the last PATIENCE gradient evaluations; the best point seen is returned.
#   nll: TOL 1e-3 (mean NLL per update, values ~ -20 .. -27);  h05: TOL 1e-5 m (= 0.01 mm of launch-mean h0.5)
TOL = {"nll": 1e-3, "h05": 1e-5}
PATIENCE = 10
N_STARTS = 1          # 2026-11-08: one start (the two ended within 0.06 on nll/ideal); the 2nd stays defined in learn()
_w = {}


def to_theta(p):
    return np.array([math.log(p[n]) if lg else p[n] for n, lg in zip(NAMES, LOG)])


def to_params(th):
    p = {n: (math.exp(v) if lg else float(v)) for n, v, lg in zip(NAMES, th, LOG)}
    p["sigma_E"] = SIGMA_E
    return p


def theta_bounds():
    return [(math.log(a), math.log(b)) if lg else (a, b) for (a, b), lg in zip(BOUNDS, LOG)]


def load(cond, ids):
    return eb.prepare([apply_condition(cam, np.load(RAW / f"launch_{i:04d}.npz"), CONDITIONS[cond], 1000 + i)
                       for i in ids])


def objective(data, th, crit):
    p = to_params(th)
    nll, n, errs = eb.run(data, p, GEO["L"], GEO["obs"], horizon=0.5 if crit == "h05" else None)
    if crit == "nll":
        v = float(nll.sum() / max(n.sum(), 1))
    else:
        v = float(np.nanmean(per_launch([{"h": e} for e in errs], "h")))
    return v if np.isfinite(v) else 1e10


def _init(cond, ids):
    _w["data"] = load(cond, ids)


def _eval(args):
    th, crit = args
    return objective(_w["data"], np.asarray(th), crit)


class _EarlyStop(Exception):
    pass


def learn(cond, crit, workers, ids, maxiter):
    bnd = theta_bounds()
    lo, hi = np.array([b[0] for b in bnd]), np.array([b[1] for b in bnd])
    eps = EPS[crit]
    b = BASE[cond]
    starts = {"from ekf19b tuning": dict(sigma=b["sigma"], q=b["q"], q_mu=1e-3, mu0=0.85, s_l=0.5, sigma_v0=3.0, E=0.725),
              "low process noise": dict(sigma=b["sigma"], q=0.01, q_mu=1e-4, mu0=0.85, s_l=0.7, sigma_v0=3.0, E=0.725)}
    starts = dict(list(starts.items())[:N_STARTS])
    results = {}
    with Pool(workers, initializer=_init, initargs=(cond, ids)) as pool:
        for sname, p0 in starts.items():
            hist, t0 = [], time.time()
            best_pt = {"f": np.inf, "th": None}

            def fg(th):
                pts = [th]
                for i in range(len(th)):
                    for s in (+1, -1):
                        q = th.copy(); q[i] = np.clip(q[i] + s * eps, lo[i], hi[i]); pts.append(q)
                vals = pool.map(_eval, [(list(q), crit) for q in pts])
                g = np.array([(vals[1 + 2 * i] - vals[2 + 2 * i]) / max(pts[1 + 2 * i][i] - pts[2 + 2 * i][i], 1e-12)
                              for i in range(len(th))])
                hist.append(vals[0])
                if vals[0] < best_pt["f"]:
                    best_pt["f"], best_pt["th"] = vals[0], np.array(th, float)
                print(f"    [{cond} | {crit} | {sname}] eval {len(hist):3d}: objective {vals[0]:.5f}  "
                      + " ".join(f"{n}={v:.4g}" for n, v in to_params(th).items() if n != "sigma_E"), flush=True)
                if len(hist) > PATIENCE and min(hist[:-PATIENCE]) - min(hist) < TOL[crit]:
                    raise _EarlyStop()
                return vals[0], g

            th0 = np.clip(to_theta(p0), lo, hi)
            try:
                r = minimize(fg, th0, jac=True, method="L-BFGS-B", bounds=bnd, options=dict(maxiter=maxiter))
                msg, nit, ok = str(r.message), int(r.nit), bool(r.success)
            except _EarlyStop:
                msg, nit, ok = (f"EARLY STOP: best improved < {TOL[crit]:g} over the last {PATIENCE} gradient "
                                f"evaluations", -1, True)
            x, fx = best_pt["th"], float(best_pt["f"])    # best point seen (= r.x unless a line search ended worse)
            p = to_params(x)
            at_bound = [n for n, v, a, z in zip(NAMES, x, lo, hi) if min(v - a, z - v) < 1e-6]
            results[sname] = dict(params=p, objective=fx, start=p0, success=ok, message=msg, n_iter=nit,
                                  n_grad=len(hist), history=hist, at_bound=at_bound, minutes=(time.time() - t0) / 60)
            print(f"  -> {sname}: objective {fx:.5f} after {len(hist)} gradient evaluations "
                  f"({results[sname]['minutes']:.1f} min), {msg}" + (f"  AT BOUND: {at_bound}" if at_bound else ""),
                  flush=True)
    best = min(results, key=lambda k: results[k]["objective"])
    return best, results


def main():
    crit = sys.argv[1]
    assert crit in ("nll", "h05"), "criterion: nll or h05"
    args = [a for a in sys.argv[2:] if not a.startswith("--")]
    conds = args[0].split(",") if args and args[0] != "all_conds" else list(CONDITIONS)
    workers = int(args[1]) if len(args) > 1 else 3
    quick = "--quick" in sys.argv
    ids = SPLIT["train"][::10] if quick else SPLIT["train"]
    maxiter = 3 if quick else 100
    out_path = root / "outputs" / f"ekf_mu_learned_{crit}{'_quick' if quick else ''}.json"
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    print(f"learning ekf-mu ({crit}) on {len(ids)} train launches, conditions {conds}, {workers} workers", flush=True)
    for cond in conds:
        if cond in out and "--redo" not in sys.argv:                # resume after a crash / shutdown
            print(f"{cond}: already learned (use --redo to learn again)", flush=True)
            continue
        if DEADLINE:                                                   # time budget (run_overnight.sh --hours H)
            done_min = [v["minutes"] for v in out.values() if "minutes" in v]
            est = 1.1 * (float(np.mean(done_min)) if done_min else EST_MIN[crit])
            if time.time() + 60 * est > DEADLINE:
                left = (DEADLINE - time.time()) / 60
                print(f"BUDGET: {cond} needs ~{est:.0f} min, {left:.0f} min left -> stop here; rerun to resume "
                      f"(learned so far: {[c for c in conds if c in out]})", flush=True)
                sys.exit(3)
        t0 = time.time()
        best, results = learn(cond, crit, workers, ids, maxiter)
        p = results[best]["params"]
        val = load(cond, SPLIT["val"][::5] if quick else SPLIT["val"])
        th = to_theta(p)
        score = dict(val_nll=objective(val, th, "nll"), val_h05_launch_mean_mm=1000 * objective(val, th, "h05"))
        out[cond] = dict(best_start=best, params=p, starts=results, val=score, minutes=(time.time() - t0) / 60)
        print(f"== {cond}: best start '{best}', train objective {results[best]['objective']:.5f}; VAL: NLL "
              f"{score['val_nll']:.4f}, launch-mean h0.5 {score['val_h05_launch_mean_mm']:.1f} mm\n", flush=True)
        out_path.write_text(json.dumps(out, indent=1))                 # saved after every condition (resumable)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()

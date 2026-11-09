"""Pilot: can NLL learning escape the stuck point on `noise`?  (2026-11-09; TOL 1e-4 alone stopped at -21.80748, while the parameters
learned on `all` give -21.86183 on the same train launches.)
Run from ball_estimation/: python3 -u experiments/step_1h_nll_restart_pilot.py outputs/pilot_nll_noise.json
Same objective and finite differences as step_1c, plus:
  - two starts: the original "from ekf19b tuning" and the "low process noise" start defined in step_1c.learn();
  - restart on stall: a round of L-BFGS ends when the best NLL improved < TOL over PATIENCE gradient evals, or L-BFGS ends by itself; the next
    round restarts from the best point with fresh memory.  A start ends after 2 rounds in a row gain < TOL, or at MAXG gradient evals.
Pass condition (fixed before running): best train NLL <= -21.86.
Writes only the JSON path given as argv[1] (refuses to overwrite).  Scores on val only: the test (dev) launches are not touched."""
import os, sys, json, time, importlib.util
from multiprocessing import Pool
import numpy as np
from scipy.optimize import minimize

root = os.getcwd(); sys.path.insert(0, root)
spec = importlib.util.spec_from_file_location("s1c", "experiments/step_1c_learn_ekf_mu.py")
m = importlib.util.module_from_spec(spec); sys.modules["s1c"] = m; spec.loader.exec_module(m)
OUT = sys.argv[1]
if os.path.exists(OUT):
    sys.exit(f"{OUT} exists: not overwriting")
COND, CRIT, TOL, PAT, MAXG, TARGET = "noise", "nll", 1e-4, 10, 80, -21.86
b = m.BASE[COND]
starts = {"from ekf19b tuning": dict(sigma=b["sigma"], q=b["q"], q_mu=1e-3, mu0=0.85, s_l=0.5, sigma_v0=3.0, E=0.725),
          "low process noise": dict(sigma=b["sigma"], q=0.01, q_mu=1e-4, mu0=0.85, s_l=0.7, sigma_v0=3.0, E=0.725)}
val = m.load(COND, m.SPLIT["val"])
bnd = m.theta_bounds(); lo = np.array([x[0] for x in bnd]); hi = np.array([x[1] for x in bnd]); eps = m.EPS[CRIT]


class RoundEnd(Exception):
    pass


res = {}
with Pool(3, initializer=m._init, initargs=(COND, m.SPLIT["train"])) as pool:
    for name, p0 in starts.items():
        hist, rounds, best, t0 = [], [], {"f": np.inf, "th": np.clip(m.to_theta(p0), lo, hi)}, time.time()
        small = 0
        while small < 2 and len(hist) < MAXG:
            n0, f0 = len(hist), best["f"]

            def fg(th):
                pts = [th]
                for i in range(len(th)):
                    for s in (+1, -1):
                        q = th.copy(); q[i] = np.clip(q[i] + s * eps, lo[i], hi[i]); pts.append(q)
                v = pool.map(m._eval, [(list(q), CRIT) for q in pts])
                g = np.array([(v[1 + 2 * i] - v[2 + 2 * i]) / max(pts[1 + 2 * i][i] - pts[2 + 2 * i][i], 1e-12)
                              for i in range(len(th))])
                hist.append(v[0])
                if v[0] < best["f"]:
                    best.update(f=v[0], th=np.array(th, float))
                print(f"[{name} | round {len(rounds) + 1}] grad {len(hist):3d}: NLL {v[0]:.5f} best {best['f']:.5f} "
                      + " ".join(f"{k}={x:.4g}" for k, x in m.to_params(th).items() if k != "sigma_E"), flush=True)
                h = hist[n0:]
                if len(hist) >= MAXG or (len(h) > PAT and min(h[:-PAT]) - min(h) < TOL):
                    raise RoundEnd()
                return v[0], g

            try:
                minimize(fg, best["th"].copy(), jac=True, method="L-BFGS-B", bounds=bnd, options=dict(maxiter=100))
            except RoundEnd:
                pass
            gain = f0 - best["f"] if np.isfinite(f0) else np.inf
            rounds.append(dict(n_grad=len(hist) - n0, best=best["f"], gain=gain))
            print(f"-- {name}: round {len(rounds)} ended, {len(hist) - n0} grads, best {best['f']:.5f}, gain {gain:.5f}", flush=True)
            small = small + 1 if gain < TOL else 0
        h05 = 1e3 * m.objective(val, best["th"], "h05")
        res[name] = dict(params=m.to_params(best["th"]), train_nll=best["f"], val_h05_mm=h05, n_grad=len(hist), rounds=rounds,
                         minutes=(time.time() - t0) / 60, history=hist, reached_target=bool(best["f"] <= TARGET), hit_cap=len(hist) >= MAXG)
        print(f"== {name}: train NLL {best['f']:.5f} (target {TARGET}: {'PASS' if best['f'] <= TARGET else 'NOT reached'}), val h0.5 {h05:.1f} mm, "
              f"{len(hist)} grads, {len(rounds)} rounds, {res[name]['minutes']:.1f} min", flush=True)
        json.dump(res, open(OUT, "w"), indent=1)

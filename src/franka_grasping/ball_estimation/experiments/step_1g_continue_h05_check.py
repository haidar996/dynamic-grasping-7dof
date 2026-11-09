"""Continuation test: did early stop cut h05 learning short on noise_real?  Companion of step_1f (same test for NLL).
Run from ball_estimation/: python3 -u experiments/step_1g_continue_h05_check.py outputs/continue_h05_noise_real.json
Same procedure as step_1c h05 (train launches, same finite-difference step), but a tighter stop: gain < 1e-6 m over 10 gradient
evals (published rule: 1e-5 m), or 40 gradient evals.  Starts:
  A = the published noise_real h05 parameters; B = the h05 parameters learned on all_real.
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
COND, CRIT, TOL, PAT, MAXG = "noise_real", "h05", 1e-6, 10, 40
pub = json.load(open("outputs/ekf_mu_learned_h05.json"))
starts = {"A from published noise_real": pub[COND]["params"], "B from all_real": pub["all_real"]["params"]}
val = m.load(COND, m.SPLIT["val"])
bnd = m.theta_bounds(); lo = np.array([b[0] for b in bnd]); hi = np.array([b[1] for b in bnd]); eps = m.EPS[CRIT]


class Stop(Exception):
    pass


res = {}
with Pool(3, initializer=m._init, initargs=(COND, m.SPLIT["train"])) as pool:
    for name, p0 in starts.items():
        hist, best, t0 = [], {"f": np.inf, "th": None}, time.time()

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
            print(f"[{name}] grad {len(hist):3d}: h05 {1e3 * v[0]:.4f} mm "
                  + " ".join(f"{k}={x:.4g}" for k, x in m.to_params(th).items() if k != "sigma_E"), flush=True)
            if len(hist) >= MAXG or (len(hist) > PAT and min(hist[:-PAT]) - min(hist) < TOL):
                raise Stop()
            return v[0], g

        try:
            minimize(fg, np.clip(m.to_theta(p0), lo, hi), jac=True, method="L-BFGS-B", bounds=bnd, options=dict(maxiter=100))
        except Stop:
            pass
        p = m.to_params(best["th"]); h = 1e3 * m.objective(val, best["th"], CRIT)
        res[name] = dict(params=p, train_h05_m=best["f"], val_h05_mm=h, n_grad=len(hist),
                         minutes=(time.time() - t0) / 60, history=hist)
        print(f"== {name}: train h0.5 {1e3 * best['f']:.3f} mm, val h0.5 {h:.1f} mm, {len(hist)} grads, "
              f"{res[name]['minutes']:.1f} min", flush=True)
        json.dump(res, open(OUT, "w"), indent=1)

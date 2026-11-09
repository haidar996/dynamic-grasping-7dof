"""Continuation test: did early stop cut NLL learning short on noise_real?  Writes only the JSON path given as argv[1].
Run from ball_estimation/: python3 -u experiments/step_1f_continue_nll_check.py outputs/continue_nll_noise_real.json
Same procedure as step_1c, tighter stop (gain < 1e-4 over 10 gradient evals, max 40).  Starts:
  A = the learned noise_real parameters; B = the all_real parameters (low q)."""
import os, sys, json, time, importlib.util
from multiprocessing import Pool
import numpy as np
from scipy.optimize import minimize
root = os.getcwd(); sys.path.insert(0, root)
spec = importlib.util.spec_from_file_location("s1c", "experiments/step_1c_learn_ekf_mu.py")
m = importlib.util.module_from_spec(spec); sys.modules["s1c"] = m; spec.loader.exec_module(m)
OUT = sys.argv[1]; TOL, PAT, MAXG = 1e-4, 10, 40
nll = json.load(open("outputs/ekf_mu_learned_nll.json"))
starts = {"A from learned noise_real": nll["noise_real"]["params"], "B from all_real": nll["all_real"]["params"]}
val = m.load("noise_real", m.SPLIT["val"])
bnd = m.theta_bounds(); lo = np.array([b[0] for b in bnd]); hi = np.array([b[1] for b in bnd]); eps = m.EPS["nll"]
class Stop(Exception): pass
res = {}
with Pool(3, initializer=m._init, initargs=("noise_real", m.SPLIT["train"])) as pool:
    for name, p0 in starts.items():
        hist, best, t0 = [], {"f": np.inf, "th": None}, time.time()
        def fg(th):
            pts = [th]
            for i in range(len(th)):
                for s in (+1, -1):
                    q = th.copy(); q[i] = np.clip(q[i] + s * eps, lo[i], hi[i]); pts.append(q)
            v = pool.map(m._eval, [(list(q), "nll") for q in pts])
            g = np.array([(v[1 + 2 * i] - v[2 + 2 * i]) / max(pts[1 + 2 * i][i] - pts[2 + 2 * i][i], 1e-12) for i in range(len(th))])
            hist.append(v[0])
            if v[0] < best["f"]: best.update(f=v[0], th=np.array(th, float))
            print(f"[{name}] grad {len(hist):3d}: NLL {v[0]:.5f} " + " ".join(f"{k}={x:.4g}" for k, x in m.to_params(th).items() if k != "sigma_E"), flush=True)
            if len(hist) >= MAXG or (len(hist) > PAT and min(hist[:-PAT]) - min(hist) < TOL): raise Stop()
            return v[0], g
        try: minimize(fg, np.clip(m.to_theta(p0), lo, hi), jac=True, method="L-BFGS-B", bounds=bnd, options=dict(maxiter=100))
        except Stop: pass
        p = m.to_params(best["th"]); h = 1e3 * m.objective(val, best["th"], "h05")
        res[name] = dict(params=p, train_nll=best["f"], val_h05_mm=h, n_grad=len(hist), minutes=(time.time() - t0) / 60, history=hist)
        print(f"== {name}: train NLL {best['f']:.5f}, val h0.5 {h:.1f} mm, {len(hist)} grads, {res[name]['minutes']:.1f} min", flush=True)
        json.dump(res, open(OUT, "w"), indent=1)

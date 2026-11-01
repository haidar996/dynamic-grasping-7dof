#!/usr/bin/env python3
"""Step 1.2a tests for estimators/ekf_mu.py (no heavy compute).
  T1 the mean of propagate_jac is bit-identical to estimators.model.propagate (random states incl. walls, arm box,
     inside the box, stopping)
  T2 the Jacobian matches central finite differences, per case: free rolling / friction stop / wall bounce / arm bounce /
     push-out (states whose case changes within +-eps are skipped: the map is not differentiable there)
  T3 on real launches (matched physics, ideal): mu-hat converges towards the true mu; NLL is finite
Usage: python3 experiments/step_1b_test_ekf_mu.py"""
import sys, json, math
from pathlib import Path
import numpy as np
import mujoco

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from arena_sim import Arena
from sensing.camera import Pinhole
from dataset import CONDITIONS, apply_condition
from estimators.model import propagate
from estimators.ekf_mu import propagate_jac, EKFMu
from estimators.filters import StaleGuard

GEO = Arena(mujoco.MjModel.from_xml_path(str(root / "scenes" / "arena.xml"))).estimator_geometry()
L, OBS = GEO["L"], GEO["obs"]
E, H = 0.7, 0.005
fails = []


def case(s):
    """Which branch the step takes (to skip finite differences across a branch change)."""
    x, y, vx, vy, l = s
    mu = math.exp(l); sp = math.hypot(vx, vy)
    stop = sp > 0 and mu * H >= sp
    xn, yn, vxn, vyn = propagate(x, y, vx, vy, H, mu, E, L, OBS)
    v1 = (0.0, 0.0) if stop else ((vx - mu * H / sp * vx, vy - mu * H / sp * vy) if sp > 0 else (vx, vy))
    flips = tuple(np.sign(a) != np.sign(b) and b != 0 for a, b in zip((vxn, vyn), v1))
    cx, cy, hx, hy = OBS
    inside = abs(x - cx) < hx and abs(y - cy) < hy
    wall = abs(x + v1[0] * H) > L or abs(y + v1[1] * H) > L
    return (stop, flips, inside, wall, np.sign(vxn), np.sign(vyn))


def random_states(rng, n):
    cx, cy, hx, hy = OBS
    out = []
    for k in range(n):
        kind = k % 5
        v = rng.uniform(0.02, 5.0) * np.array([math.cos(a := rng.uniform(0, 2 * math.pi)), math.sin(a)])
        l = math.log(rng.uniform(0.2, 1.5))
        if kind == 0:                                   # anywhere
            p = rng.uniform(-L, L, 2)
        elif kind == 1:                                 # just before a wall, moving into it
            ax = rng.integers(2); sd = rng.choice([-1, 1]); p = rng.uniform(-L, L, 2)
            p[ax] = sd * (L - rng.uniform(0, 0.02)); v[ax] = sd * abs(v[ax])
        elif kind == 2:                                 # just before the arm box, moving into it
            ax = rng.integers(2); sd = rng.choice([-1, 1]); hw = (hx, hy); c = (cx, cy)
            p = np.array([cx + rng.uniform(-hx, hx), cy + rng.uniform(-hy, hy)])
            p[ax] = c[ax] + sd * (hw[ax] + rng.uniform(0, 0.02)); v[ax] = -sd * abs(v[ax])
        elif kind == 3:                                 # inside the box (noisy state): push-out
            p = np.array([cx + rng.uniform(-hx, hx), cy + rng.uniform(-hy, hy)])
        else:                                           # slow: friction stop
            p = rng.uniform(-1, 1, 2); v = v / np.linalg.norm(v) * rng.uniform(0, 0.01)
        out.append((float(p[0]), float(p[1]), float(v[0]), float(v[1]), l))
    return out


def t1_t2():
    rng = np.random.default_rng(0)
    states = random_states(rng, 5000)
    bad_mean = 0
    for s in states:
        (x, y, vx, vy, l), _ = propagate_jac(s, H, E, L, OBS)
        if (x, y, vx, vy) != propagate(*s[:4], H, math.exp(s[4]), E, L, OBS):
            bad_mean += 1
    print(f"T1 mean identical to model.propagate: {len(states) - bad_mean}/{len(states)}")
    if bad_mean:
        fails.append("T1")
    eps = 1e-7
    names = ["free", "stop", "wall bounce", "arm bounce", "push-out"]
    err, cnt, skipped = {k: 0.0 for k in names}, {k: 0 for k in names}, 0
    for s in states:
        c0 = case(s)
        stop, flips, inside, wall = c0[:4]
        kind = "push-out" if inside else "stop" if stop else ("wall bounce" if wall and any(flips) else
                                                               "arm bounce" if any(flips) else "free")
        _, F = propagate_jac(s, H, E, L, OBS)
        Fd = np.zeros((5, 5)); ok = True
        for j in range(5):
            sp_, sm_ = list(s), list(s)
            sp_[j] += eps; sm_[j] -= eps
            if case(tuple(sp_)) != c0 or case(tuple(sm_)) != c0:
                ok = False; break
            Fd[:, j] = (np.array(propagate_jac(tuple(sp_), H, E, L, OBS)[0]) -
                        np.array(propagate_jac(tuple(sm_), H, E, L, OBS)[0])) / (2 * eps)
        if not ok:
            skipped += 1; continue
        cnt[kind] += 1
        err[kind] = max(err[kind], float(np.max(np.abs(F - Fd))))
    print(f"T2 Jacobian vs central differences (max |F - F_fd|), {skipped} states skipped at a branch change:")
    for k in names:
        ok = err[k] < 1e-6 and cnt[k] > 20
        print(f"   [{'PASS' if ok else 'FAIL'}] {k:12s} n {cnt[k]:5d}  max error {err[k]:.2e}")
        if not ok:
            fails.append(f"T2 {k}")


def t3():
    cam = Pinhole(3.0, 49.0, 480, 480)
    split = json.loads((root / "split_500.json").read_text())
    print("T3 mu convergence (matched physics, ideal, train launches, default settings sigma 5 mm q 2):")
    rows = []
    for i in split["train"][::25]:
        rec = apply_condition(cam, np.load(root / f"outputs/raw_60hz_h3_f49_sq480_with_arm/launch_{i:04d}.npz"),
                              CONDITIONS["ideal"], 1000 + i)
        f = StaleGuard(EKFMu(L=L, obs=OBS, E=0.725, sigma=0.005, q=2.0, q_mu=1e-3, s_l=0.7))
        mus = {}
        for k in range(len(rec["t_capture"])):
            f.update(rec["t_capture"][k], rec["xy"][k], rec["valid"][k])
            if f.ready:
                for T in (0.25, 0.5, 1.0):
                    if T not in mus and rec["t_capture"][k] >= T:
                        mus[T] = f.mu
        v0 = float(np.linalg.norm(rec["v0"]))
        t_stop = float(rec["t_phys"][np.argmax(np.linalg.norm(rec["vel"], axis=1) == 0)]) if np.any(
            np.linalg.norm(rec["vel"], axis=1) == 0) else 10.0
        rows.append((i, v0, rec["mu"], mus, t_stop, f.est.nll / max(f.est.n_upd, 1)))
        print(f"   launch {i:3d} v0 {v0:4.1f} m/s, stops at {t_stop:4.1f} s, true mu {rec['mu']:.2f} | mu-hat at 0.25 s "
              f"{mus.get(0.25, float('nan')):.2f}, 0.5 s {mus.get(0.5, float('nan')):.2f}, 1 s {mus.get(1.0, float('nan')):.2f}"
              f" | NLL/update {rows[-1][5]:.2f}")
    if not all(np.isfinite(r[5]) for r in rows):
        fails.append("T3 NLL")


if __name__ == "__main__":
    t1_t2()
    t3()
    print("\nALL TESTS PASS" if not fails else f"\nFAILED: {fails}")

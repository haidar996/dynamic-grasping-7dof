#!/usr/bin/env python3
"""Step A1 check (2026-11-01): the mismatch physics (arena_sim.MismatchArena) against closed-form results, then the
500 paired launches without rendering (truth only) compared with the matched physics.
  1 slide -> roll: a pure-sliding launch starts rolling at 5/7 v0 after t = 2 v0 / (7 mu_s g)
  2 rolling: deceleration a0 (1 + k v)
  3 tilt: a rolling ball on a tilted floor with a0 = 0 accelerates at 5/7 g sin(alpha)
  4 wall impact (no spin, no friction, beta = 0): normal speed * e, tangential unchanged; with beta: e = e0 - beta |v_n|
  5 a rolling ball after a head-on wall impact slides, then rolls at (5 e - 2 (1 - f)) / 7 of the impact speed
    (f = fraction of the top spin removed by the wall friction impulse; f = 0 for mu_w = 0)
  6 energy never increases at an impact; the ball never ends a step inside the arm box or outside the walls
Usage: python3 experiments/step_0l_check_mismatch_physics.py [workers=3]"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import mujoco

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from arena_sim import MismatchArena, MismatchPhysics, Arena, G
from launches import sample_launches, with_mismatch, run_launch

XML = str(root / "scenes" / "arena.xml")
MODEL = mujoco.MjModel.from_xml_path(XML)
ARENA = MismatchArena(MODEL)
H = MODEL.opt.timestep
fails = []


def check(name, got, want, tol):
    ok = abs(got - want) <= tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got {got:.5f}, expected {want:.5f} (tol {tol})")
    if not ok:
        fails.append(name)


def phys(**kw):
    base = dict(mu_s=0.25, a0=0.0, k=0.0, slip0=1.0, e_wall=0.8, e_arm=0.8, beta=0.0, mu_w=0.0, tilt=(0.0, 0.0))
    return MismatchPhysics(**{**base, **kw})


def simulate(p0, v0, ph, T):
    data = mujoco.MjData(MODEL)
    ARENA.reset(data, np.array(p0, float), np.array(v0, float), ph)
    out = []
    for _ in range(int(round(T / H))):
        hits = ARENA.step_mismatch(data, ph)
        out.append((data.time, data.qpos[:2].copy(), data.qvel[:2].copy(), ARENA.omega.copy(), hits))
    return out


def unit_tests():
    print("closed-form checks")
    # 1 slide -> roll
    ph = phys(mu_s=0.25)
    tr = simulate((-1.0, 0.8), (2.0, 0.0), ph, 1.0)
    t_roll = 2 * 2.0 / (7 * 0.25 * G)
    slip = lambda s: np.hypot(s[2][0] - ARENA.radius * s[3][1], s[2][1] + ARENA.radius * s[3][0])
    k = next(i for i, s in enumerate(tr) if slip(s) < 1e-6)
    check("1 rolling speed = 5/7 v0", tr[k][2][0], 2.0 * 5 / 7, 1e-3)
    check("1 time to roll = 2 v0 / (7 mu_s g)", tr[k][0], t_roll, 2 * H)
    # 2 rolling deceleration with speed dependence
    ph = phys(a0=0.3, k=0.1, slip0=0.0)
    tr = simulate((-1.0, 0.8), (1.5, 0.0), ph, 0.5)
    a = (tr[-1][2][0] - tr[-101][2][0]) / (100 * H)
    v_mid = 0.5 * (tr[-1][2][0] + tr[-101][2][0])
    check("2 rolling deceleration a0 (1 + k v)", -a, 0.3 * (1 + 0.1 * v_mid), 1e-3)
    # 3 tilt
    alpha = np.radians(0.15)
    ph = phys(slip0=0.0, tilt=(G * np.sin(alpha), 0.0))
    tr = simulate((-1.0, 0.8), (0.5, 0.0), ph, 0.5)
    check("3 tilt: 5/7 g sin(alpha)", (tr[-1][2][0] - tr[0][2][0]) / (tr[-1][0] - tr[0][0]), 5 / 7 * G * np.sin(alpha), 1e-4)
    # 4 wall impact, no spin effects (pure sliding launch from right before the wall, mu_s tiny)
    for beta in (0.0, 0.04):
        ph = phys(mu_s=1e-9, e_wall=0.8, beta=beta)
        tr = simulate((1.10, 0.0), (3.0, 1.0), ph, 0.1)
        v_after = next(s[2] for s in tr if s[4])
        check(f"4 wall normal restitution (beta {beta})", -v_after[0] / 3.0, 0.8 - beta * 3.0, 1e-3)
        check(f"4 wall tangential unchanged (beta {beta})", v_after[1], 1.0, 1e-3)
    # 5 rolling ball, head-on, then slides and rolls again
    for mu_w in (0.0, 0.14):
        ph = phys(mu_s=0.25, slip0=0.0, e_wall=0.8, mu_w=mu_w)
        tr = simulate((1.0, 0.0), (1.0, 0.0), ph, 1.5)
        i_hit = next(i for i, s in enumerate(tr) if s[4])
        v_in = tr[i_hit - 1][2][0]
        om_before, om_after = tr[i_hit - 1][3][1], tr[i_hit][3][1]
        f = 1 - om_after / om_before
        if mu_w == 0:
            check("5 no wall friction: spin kept", f, 0.0, 1e-9)
        else:   # friction removes mu_w (1+e) v_in * (5/2) / r of spin, as long as the slip is not stopped
            check("5 wall friction removes (5/2) mu_w (1+e) of the top spin", f, min(1.0, 2.5 * mu_w * 1.8), 1e-3)
        check(f"5 final rolling speed (mu_w {mu_w})", tr[-1][2][0] / v_in, -(5 * 0.8 - 2 * (1 - f)) / 7, 2e-3)


def energy_task(args):
    i, launch = args
    m = mujoco.MjModel.from_xml_path(XML)
    ar = MismatchArena(m)
    data = mujoco.MjData(m)
    ar.reset(data, launch.p0, launch.v0, launch.phys)
    r = ar.radius
    bad_energy = bad_pos = 0
    n_hits = n_arm = 0
    t_stop = None
    for _ in range(int(10.0 / m.opt.timestep)):
        v0 = data.qvel[:2].copy(); w0 = ar.omega.copy()
        e_before = v0 @ v0 + 0.4 * r * r * (w0 @ w0)
        hits = ar.step_mismatch(data, launch.phys)
        v = data.qvel[:2]
        e_after = v @ v + 0.4 * r * r * (ar.omega @ ar.omega)
        if hits:
            n_hits += len(hits); n_arm += sum(h[2] == 1 for h in hits)
            if e_after > e_before * (1 + 1e-9) + 1e-12:
                bad_energy += 1
        p = data.qpos[:2]
        if ar.inside_obstacle(p, -1e-9) or np.any(np.abs(p) > ar.Lx - r + 1e-9):
            bad_pos += 1
        if t_stop is None and not np.any(v):
            t_stop = data.time
    return i, bad_energy, bad_pos, n_hits, n_arm, t_stop if t_stop is not None else 10.0


def matched_task(args):
    i, launch = args
    m = mujoco.MjModel.from_xml_path(XML)
    tr = run_launch(m, Arena(m), launch, T=10.0)
    moving = np.linalg.norm(tr["vel"], axis=1) > 0
    t_stop = tr["t"][np.argmin(moving)] if not moving.all() else 10.0
    return i, len(tr["bounces"]), int((tr["bounces"][:, 3] == 1).sum()), t_stop


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    unit_tests()
    launches = sample_launches(np.random.default_rng(42), Arena(MODEL), k=10)
    mis = with_mismatch(launches, np.random.default_rng(43))
    print("\n500 paired launches, truth only (10 s each)")
    with Pool(workers) as p:
        rm = sorted(p.map(energy_task, list(enumerate(mis)), chunksize=8))
        rc = sorted(p.map(matched_task, list(enumerate(launches)), chunksize=8))
    be = sum(r[1] for r in rm); bp = sum(r[2] for r in rm)
    print(f"  impacts that gain energy: {be}   steps ending inside the arm / outside the walls: {bp}")
    if be or bp:
        fails.append("6 energy / position")
    for name, hits, arm, ts in (("matched ", [r[1] for r in rc], [r[2] for r in rc], [r[3] for r in rc]),
                                ("mismatch", [r[3] for r in rm], [r[4] for r in rm], [r[5] for r in rm])):
        ts = np.array(ts)
        print(f"  {name}: bounces / launch mean {np.mean(hits):5.1f} (median {np.median(hits):3.0f}), launches hitting the arm "
              f"{np.mean(np.array(arm) > 0) * 100:4.1f} %, moving time median {np.median(ts):4.1f} s, still moving at 10 s "
              f"{np.mean(ts >= 10.0) * 100:4.1f} %")
    print("\nALL CHECKS PASS" if not fails else f"\nFAILED: {fails}")


if __name__ == "__main__":
    main()

"""Random launches and ground-truth logging for the bare arena."""
from dataclasses import dataclass
import numpy as np
import mujoco

# launch speeds: 0.1, 0.2, ..., 5.0 m/s (50 values), K launches per speed
SPEEDS = np.round(np.arange(1, 51) * 0.1, 1)


@dataclass
class LaunchRanges:
    mu: tuple = (0.2, 1.5)         # sliding deceleration, m/s^2 (equal ends = fixed value)
    e: tuple = (0.5, 0.95)         # wall restitution (equal ends = fixed value)
    margin: float = 0.1            # start at least this far inside the wall limit (m)


@dataclass
class Launch:
    p0: np.ndarray                 # start position (m)
    v0: np.ndarray                 # start velocity (m/s)
    mu: float
    e: float
    phys: object = None            # MismatchPhysics (arena_sim) -> "mismatch" physics; None -> mu, e physics


@dataclass
class MismatchRanges:
    """Per-launch ranges of the mismatch physics (docs/A1_mismatch_physics_design.md, user decisions 2026-11-01)."""
    mu_s: tuple = (0.15, 0.40)
    a0: tuple = (0.10, 0.80)       # m/s^2, realistic rolling resistance (c_rr ~ 0.01-0.08)
    k: tuple = (0.0, 0.1)          # s/m
    slip0: tuple = (0.0, 1.0)
    e_wall: tuple = (0.55, 0.95)
    e_arm: tuple = (0.40, 0.90)
    beta: tuple = (0.0, 0.04)      # s/m
    mu_w: tuple = (0.10, 0.30)
    tilt_deg: tuple = (0.0, 0.15)


def with_mismatch(launches, rng, r=MismatchRanges()):
    """Same launches (p0, v0 unchanged: paired with the matched data) + mismatch physics drawn from a separate stream."""
    from arena_sim import MismatchPhysics, G
    out = []
    for L in launches:
        u = lambda a: float(rng.uniform(*a))
        alpha, phi = np.radians(u(r.tilt_deg)), u((0.0, 2 * np.pi))
        ph = MismatchPhysics(mu_s=u(r.mu_s), a0=u(r.a0), k=u(r.k), slip0=u(r.slip0), e_wall=u(r.e_wall),
                             e_arm=u(r.e_arm), beta=u(r.beta), mu_w=u(r.mu_w),
                             tilt=(G * np.sin(alpha) * np.cos(phi), G * np.sin(alpha) * np.sin(phi)))
        out.append(Launch(L.p0, L.v0, float("nan"), float("nan"), ph))
    return out


def sample_launch(rng, arena, speed, r=LaunchRanges()):
    lx, ly = arena.Lx - arena.radius - r.margin, arena.Ly - arena.radius - r.margin
    p0 = rng.uniform([-lx, -ly], [lx, ly])
    while arena.inside_obstacle(p0, r.margin):          # never start in or touching the arm base
        p0 = rng.uniform([-lx, -ly], [lx, ly])
    angle = rng.uniform(0, 2 * np.pi)
    v0 = speed * np.array([np.cos(angle), np.sin(angle)])
    return Launch(p0, v0, rng.uniform(*r.mu), rng.uniform(*r.e))


def sample_launches(rng, arena, k=10, speeds=SPEEDS, r=LaunchRanges()):
    """k launches for every speed in the grid (len(speeds) * k launches)."""
    return [sample_launch(rng, arena, s, r) for s in speeds for _ in range(k)]


def run_launch(model, arena, launch, T=10.0):
    """Simulate one launch; return ground-truth arrays at every physics step."""
    data = mujoco.MjData(model)
    arena.reset(data, launch.p0, launch.v0)
    n = int(T / model.opt.timestep)
    t = np.zeros(n); pos = np.zeros((n, 2)); vel = np.zeros((n, 2))
    acc = np.zeros((n, 2)); bounces = []
    for i in range(n):
        v_prev = data.qvel[:2].copy()
        hits = arena.step(data, launch.mu, launch.e)
        t[i], pos[i], vel[i] = data.time, data.qpos[:2], data.qvel[:2]
        acc[i] = (vel[i] - v_prev) / model.opt.timestep     # includes bounce impulses
        bounces += [(data.time, *h) for h in hits]
    return dict(t=t, pos=pos, vel=vel, acc=acc,
                bounces=np.array(bounces).reshape(-1, 4),
                mu=launch.mu, e=launch.e, p0=launch.p0, v0=launch.v0)

"""Planar ball physics for the arena: friction + bounces off the walls and the (static) arm base.

The scene XML holds only geometry.  Everything physical that matters to the
estimators (friction deceleration MU, restitution E) lives here, so each
launch can use its own values and the estimators can refer to them.
"""
import numpy as np
import mujoco

WALL, ARM = 0, 1                                     # bounce sources


class Arena:
    def __init__(self, model):
        self.model = model
        self.ball_body = model.body("ball").id
        bx, by = model.joint("bx"), model.joint("by")
        assert (bx.qposadr[0], by.qposadr[0], bx.dofadr[0], by.dofadr[0]) == (0, 1, 0, 1), \
            "ball joints must be qpos/qvel[0:2]; is the arm still jointed?"
        self.mass = model.body_mass[self.ball_body]
        self.radius = model.geom("ball_geom").size[0]
        # inner faces of the walls, read from the XML (no duplicated numbers)
        gx, gy = model.geom("wall_px"), model.geom("wall_py")
        self.Lx = gx.pos[0] - gx.size[0]
        self.Ly = gy.pos[1] - gy.size[1]
        # arm base box (optional): centre and half sizes grown by the ball radius
        try:
            g = model.geom("arm_base")
            self.obs_c = g.pos[:2].copy()
            self.obs_h = g.size[:2] + self.radius
        except KeyError:
            self.obs_c = self.obs_h = None

    def estimator_geometry(self):
        """Scene geometry for the estimators: wall limit L and arm box for the ball centre."""
        assert abs(self.Lx - self.Ly) < 1e-9, "estimators assume a square arena"
        obs = None if self.obs_c is None else (*map(float, self.obs_c), *map(float, self.obs_h))
        return dict(L=float(self.Lx - self.radius), obs=obs)

    def inside_obstacle(self, p, margin=0.0):
        return self.obs_c is not None and np.all(np.abs(p - self.obs_c) < self.obs_h + margin)

    def reset(self, data, p0, v0):
        mujoco.mj_resetData(self.model, data)
        data.qpos[:2] = p0
        data.qvel[:2] = v0
        mujoco.mj_forward(self.model, data)

    def step(self, data, mu, e, v_stop=1e-3):
        """Advance one time step.
        mu : sliding deceleration (m/s^2), friction opposite to velocity
        e  : restitution in [0, 1], applied to the normal velocity (walls and arm)
        """
        v = data.qvel[:2]
        s = np.linalg.norm(v)
        if s > max(v_stop, mu * self.model.opt.timestep):   # else friction would reverse v: stop exactly
            data.qfrc_applied[:2] = -self.mass * mu * v / s
        else:
            data.qfrc_applied[:2] = 0.0
            data.qvel[:2] = 0.0                      # exact stop, no residual drift
        mujoco.mj_step(self.model, data)
        return self._bounce(data, e)          # list of (axis, side, source)

    def _bounce(self, data, e):
        hits = []
        for axis, L in ((0, self.Lx), (1, self.Ly)):
            limit = L - self.radius                  # ball centre may not pass this
            p, v = data.qpos[axis], data.qvel[axis]
            if p > limit and v > 0:
                data.qpos[axis] = limit
                data.qvel[axis] = -e * v
                hits.append((axis, +1, WALL))
            elif p < -limit and v < 0:
                data.qpos[axis] = -limit
                data.qvel[axis] = -e * v
                hits.append((axis, -1, WALL))
        if self.inside_obstacle(data.qpos[:2]):
            d = data.qpos[:2] - self.obs_c
            axis = int(np.argmin(self.obs_h - np.abs(d)))   # face with the smallest penetration
            side = 1 if d[axis] >= 0 else -1
            data.qpos[axis] = self.obs_c[axis] + side * self.obs_h[axis]
            if data.qvel[axis] * side < 0:
                data.qvel[axis] *= -e
            hits.append((axis, side, ARM))
        return hits


# ---------------------------------------------------------------------------------------------------------------------
# "Mismatch" physics (research audit A1, 2026-11-01; design: docs/A1_mismatch_physics_design.md).
# Physics the estimators do NOT contain: slide -> roll with spin, speed-dependent rolling resistance, launch spin,
# speed-dependent restitution (different for walls and arm), Coulomb friction at the wall/arm (changes tangential speed
# and spin, ball slides again after a bounce), exact mirror bounce inside the step, floor tilt.
# The ball stays planar (2 slide joints, rendering unchanged); the spin omega (3-vector) is integrated here.
from dataclasses import dataclass

G = 9.81
INERTIA = 0.4                                         # I / (m r^2): solid sphere


@dataclass
class MismatchPhysics:
    mu_s: float              # floor sliding friction coefficient (deceleration mu_s * g while sliding)
    a0: float                # rolling resistance deceleration at zero speed (m/s^2)
    k: float                 # speed dependence (s/m): a_roll = a0 * (1 + k |v|)
    slip0: float             # launch slip ratio: 0 = already rolling, 1 = pure sliding
    e_wall: float            # restitution at low impact speed, walls
    e_arm: float             # restitution at low impact speed, arm base
    beta: float              # restitution drop with normal impact speed (s/m): e = e0 - beta |v_n|, clipped
    mu_w: float              # friction coefficient ball - wall/arm
    tilt: tuple              # floor-tilt acceleration (ax, ay) = g sin(alpha) (cos phi, sin phi), m/s^2


class MismatchArena(Arena):
    E_MIN, E_MAX = 0.3, 0.98

    def reset(self, data, p0, v0, phys=None):
        super().reset(data, p0, v0)
        self.omega = np.zeros(3)
        if phys is not None:                          # rolling spin for the (1 - slip0) part of the launch velocity
            self.omega[:2] = (1.0 - phys.slip0) * np.array([-v0[1], v0[0]]) / self.radius

    def _slip(self, v):
        """Velocity of the floor contact point: u = v + omega x (-r z) = (vx - r wy, vy + r wx)."""
        return np.array([v[0] - self.radius * self.omega[1], v[1] + self.radius * self.omega[0]])

    def _roll_spin(self, v):
        self.omega[0], self.omega[1] = -v[1] / self.radius, v[0] / self.radius

    def step_mismatch(self, data, ph, eps=1e-6):
        """One time step: floor friction (semi-implicit Euler, like MuJoCo's integrator) then exact bounces."""
        h, r = self.model.opt.timestep, self.radius
        v = data.qvel[:2].copy()
        tilt = np.asarray(ph.tilt, float)
        u = self._slip(v)
        su = np.linalg.norm(u)
        if su > eps:                                  # sliding: friction mu_s g against the slip
            uh = u / su
            du = 3.5 * ph.mu_s * G * h                # slip decay this step: (1 + 1/INERTIA) mu_s g h
            if du >= su:                              # slip ends inside the step: impulse that cancels it exactly
                v = v - u / 3.5 + tilt * h
                self._roll_spin(v)
            else:
                v = v + (-ph.mu_s * G * uh + tilt) * h
                self.omega[:2] += (ph.mu_s * G * h / (INERTIA * r)) * np.array([-uh[1], uh[0]])
        else:                                         # rolling: rolling resistance + 5/7 of the tilt
            sp = np.linalg.norm(v)
            a_roll = ph.a0 * (1.0 + ph.k * sp)
            g_eff = tilt / (1.0 + INERTIA)
            if sp <= a_roll * h and np.linalg.norm(g_eff) < ph.a0:
                v = np.zeros(2)                       # stops and stays (tilt smaller than rolling resistance)
            else:
                v = v + (g_eff - (a_roll * v / sp if sp > 0 else 0.0)) * h
            self._roll_spin(v)
        data.qfrc_applied[:2] = 0.0
        data.qvel[:2] = v
        p_old = data.qpos[:2].copy()
        mujoco.mj_step(self.model, data)              # no forces: qpos += v h
        return self._bounce_mismatch(data, p_old, ph)

    def _impact(self, v, n, e0, ph):
        """Velocity and spin after an impact on a surface with inward normal n (2D unit vector), contact at the
        ball's centre height.  Normal: speed-dependent restitution.  Tangential plane (horizontal t, vertical z):
        Coulomb impulse against the contact slip, capped at mu_w * J_n and at the value that stops the slip."""
        r, vn = self.radius, float(v @ n)
        if vn >= 0:
            return v
        e = float(np.clip(e0 - ph.beta * abs(vn), self.E_MIN, self.E_MAX))
        jn = (1.0 + e) * abs(vn)                       # normal impulse per unit mass
        v = v - (1.0 + e) * vn * n
        w = np.array([-n[0], -n[1], 0.0])              # centre -> contact point direction (towards the wall)
        t = np.array([-n[1], n[0], 0.0])
        vc = np.array([v[0], v[1], 0.0]) + np.cross(self.omega, r * w)
        s = np.array([vc @ t, vc[2]])                  # slip in the wall plane: (horizontal tangent, vertical)
        ss = np.linalg.norm(s)
        if ss > 1e-9:
            sh = s / ss
            k_inv = sh[0] ** 2 * (1.0 + 1.0 / INERTIA) + sh[1] ** 2 * (1.0 / INERTIA)   # vertical: floor holds translation
            lam = min(ph.mu_w * jn, ss / k_inv)
            J = -lam * (sh[0] * t + sh[1] * np.array([0.0, 0.0, 1.0]))   # impulse per unit mass
            v = v + (J @ t) * t[:2]                    # the vertical part only changes the spin
            self.omega += np.cross(r * w, J) / (INERTIA * r * r)
        return v

    def _bounce_mismatch(self, data, p_old, ph):
        hits = []
        p, v = data.qpos[:2].copy(), data.qvel[:2].copy()
        for axis, L in ((0, self.Lx), (1, self.Ly)):
            limit = L - self.radius
            for side in (1, -1):
                over = side * p[axis] - limit
                if over > 0 and side * v[axis] > 0:
                    n = np.zeros(2); n[axis] = -side
                    tau = over / abs(v[axis])          # time since the impact
                    p = p - v * tau                    # back to the impact point ...
                    v = self._impact(v, n, ph.e_wall, ph)
                    p = p + v * tau                    # ... and forward with the new velocity
                    hits.append((axis, side, WALL))
        if self.obs_c is not None:
            hit = self._box_entry(p_old, p)
            if hit is not None:
                axis, side, frac = hit
                n = np.zeros(2); n[axis] = side
                tau = (1.0 - frac) * self.model.opt.timestep
                p = p - v * tau
                p[axis] = self.obs_c[axis] + side * self.obs_h[axis]   # exactly on the face
                v = self._impact(v, n, ph.e_arm, ph)
                p = p + v * tau
                hits.append((axis, side, ARM))
            elif self.inside_obstacle(p):              # safety net (should not happen): push out, no energy gain
                d = p - self.obs_c
                axis = int(np.argmin(self.obs_h - np.abs(d))); side = 1 if d[axis] >= 0 else -1
                p[axis] = self.obs_c[axis] + side * self.obs_h[axis]
                if v[axis] * side < 0:
                    v[axis] = 0.0
                hits.append((axis, side, ARM))
        data.qpos[:2], data.qvel[:2] = p, v
        return hits

    def _box_entry(self, p0, p1):
        """Slab test: (axis, outward side of the face hit, fraction of the step at entry) or None."""
        t_in, t_out, axis = 0.0, 1.0, -1
        for a in (0, 1):
            d = p1[a] - p0[a]
            lo, hi = self.obs_c[a] - self.obs_h[a], self.obs_c[a] + self.obs_h[a]
            if d == 0.0:
                if not (lo < p0[a] < hi):
                    return None
                continue
            t1, t2 = (lo - p0[a]) / d, (hi - p0[a]) / d
            if t1 > t2:
                t1, t2 = t2, t1
            if t1 > t_in:
                t_in, axis = t1, a
            t_out = min(t_out, t2)
            if t_in >= t_out:
                return None
        if axis < 0:
            return None
        side = 1 if p0[axis] >= self.obs_c[axis] else -1
        return axis, side, t_in

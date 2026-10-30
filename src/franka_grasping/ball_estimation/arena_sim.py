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

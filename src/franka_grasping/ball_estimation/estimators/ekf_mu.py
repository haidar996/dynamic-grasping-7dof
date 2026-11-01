"""ekf-mu (Phase 1.2): the friction is part of the EKF state, as the drag/Magnus constants in paper 03 (Achterhold et al.).

State z = (x, y, vx, vy, l) with mu = exp(l) > 0 (friction deceleration, m/s^2).  Same motion model as ekf19b
(estimators/model.py: friction, walls, arm box), but the covariance is propagated through the full Jacobian, INCLUDING
the bounces (the mirror map has slope -E on the hit axis: research audit B5; for this discrete model that is exact and
contains the impact-timing effect of paper 03 S.4).  The filter predicts at every frame clock tick, also when the frame
carries no measurement, so all launches of a dataset share the same time grid (needed by the batch version used for
learning, estimators/ekf_mu_batch.py).  It accumulates the negative log-likelihood of its innovations (paper 03, Eq. 7)
for learning the noise levels and priors.  Design: docs/phase1_step12_ekf_mu_plan.md."""
import math
import numpy as np
from .model import forecast

NX = 5


def propagate_jac(s, h, E, L, obs=None):
    """One sub-step of the mean state s = (x, y, vx, vy, l) and its Jacobian F = d s_new / d s (5x5).
    The mean follows estimators.model.propagate (c = exp(l)) operation by operation (checked by experiments/
    step_1b_test_ekf_mu.py); each row of F is carried along: friction, move, wall mirror (slope -E), arm box."""
    x, y, vx, vy, l = s
    mu = math.exp(l)
    ex, ey = np.eye(NX)[0], np.eye(NX)[1]
    jvx, jvy = np.eye(NX)[2], np.eye(NX)[3]
    sp = math.hypot(vx, vy)
    if sp > 0.0:
        if mu * h >= sp:                                  # friction stops the ball inside the sub-step
            vx = vy = 0.0
            jvx, jvy = np.zeros(NX), np.zeros(NX)
        else:
            f = mu * h / sp
            vh = (vx / sp, vy / sp)
            Fv = np.eye(2) - f * (np.eye(2) - np.outer(vh, vh))
            jvx = np.array([0.0, 0.0, Fv[0, 0], Fv[0, 1], -mu * h * vh[0]])
            jvy = np.array([0.0, 0.0, Fv[1, 0], Fv[1, 1], -mu * h * vh[1]])
            vx -= f * vx
            vy -= f * vy
    xo, yo = x, y
    x += vx * h
    y += vy * h
    jx, jy = ex + h * jvx, ey + h * jvy
    if x > L:
        x = L - E * (x - L); vx = -E * vx; jx, jvx = -E * jx, -E * jvx
    elif x < -L:
        x = -L + E * (-L - x); vx = -E * vx; jx, jvx = -E * jx, -E * jvx
    if y > L:
        y = L - E * (y - L); vy = -E * vy; jy, jvy = -E * jy, -E * jvy
    elif y < -L:
        y = -L + E * (-L - y); vy = -E * vy; jy, jvy = -E * jy, -E * jvy
    if obs is not None:
        x, y, vx, vy, jx, jy, jvx, jvy = _box_bounce_jac(xo, yo, x, y, vx, vy, E, obs, jx, jy, jvx, jvy)
    return (x, y, vx, vy, l), np.array([jx, jy, jvx, jvy, np.eye(NX)[4]])


def _box_bounce_jac(xo, yo, x, y, vx, vy, E, obs, jx, jy, jvx, jvy):
    """estimators.model._box_bounce with the Jacobian rows carried along."""
    cx, cy, hx, hy = obs
    if abs(xo - cx) < hx and abs(yo - cy) < hy:           # started inside: push out (position clamped onto the face)
        px, py = hx - abs(x - cx), hy - abs(y - cy)
        if px <= py:
            side = 1.0 if x >= cx else -1.0
            x = cx + side * hx; jx = np.zeros(NX)
            if vx * side < 0:
                vx = -E * vx; jvx = -E * jvx
        else:
            side = 1.0 if y >= cy else -1.0
            y = cy + side * hy; jy = np.zeros(NX)
            if vy * side < 0:
                vy = -E * vy; jvy = -E * jvy
        return x, y, vx, vy, jx, jy, jvx, jvy
    t_in, t_out, axis = 0.0, 1.0, -1
    for a, (p0, p1, c, hw) in enumerate(((xo, x, cx, hx), (yo, y, cy, hy))):
        d = p1 - p0
        if d == 0.0:
            if abs(p0 - c) >= hw:
                return x, y, vx, vy, jx, jy, jvx, jvy
            continue
        t1, t2 = (c - hw - p0) / d, (c + hw - p0) / d
        if t1 > t2:
            t1, t2 = t2, t1
        if t1 > t_in:
            t_in, axis = t1, a
        t_out = min(t_out, t2)
        if t_in >= t_out:
            return x, y, vx, vy, jx, jy, jvx, jvy
    if axis < 0:
        return x, y, vx, vy, jx, jy, jvx, jvy
    if axis == 0:
        face = cx + (hx if xo >= cx else -hx)
        x = face - E * (x - face); vx = -E * vx; jx, jvx = -E * jx, -E * jvx
    else:
        face = cy + (hy if yo >= cy else -hy)
        y = face - E * (y - face); vy = -E * vy; jy, jvy = -E * jy, -E * jvy
    return x, y, vx, vy, jx, jy, jvx, jvy


class EKFMu:
    """Same interface as EKF19: reset(), update(t, xy, valid), ready, t, s, forecast(dt); plus nll / n_upd."""
    def __init__(self, L, obs=None, E=0.725, sigma=0.005, q=0.5, q_mu=1e-4, mu0=0.85, s_l=0.5, sigma_v0=3.0,
                 sigma_E=0.13, max_h=0.005):
        self.L, self.obs, self.E, self.q, self.q_mu = L, obs, E, q, q_mu
        self.R = sigma ** 2
        self.P0 = np.diag([sigma ** 2, sigma ** 2, sigma_v0 ** 2, sigma_v0 ** 2, s_l ** 2])
        self.l0 = math.log(mu0)
        self.sigma_E, self.max_h = sigma_E, max_h      # sigma_E: extra velocity variance at a hit (as ekf19b)
        self.reset()

    def reset(self):
        self.ready, self.t, self.s, self.P = False, None, None, None
        self.nll, self.n_upd = 0.0, 0
        self.nis = []                                     # normalised innovation squared per update (consistency check)

    @property
    def mu(self):
        return math.exp(self.s[4])

    def Q(self, h):
        Q = np.zeros((NX, NX))
        for i in (0, 1):
            Q[i, i] = self.q * h ** 3 / 3; Q[i, i + 2] = Q[i + 2, i] = self.q * h ** 2 / 2
            Q[i + 2, i + 2] = self.q * h
        Q[4, 4] = self.q_mu * h                           # random walk of l = log mu
        return Q

    def _predict(self, dt):
        n = max(1, math.ceil(dt / self.max_h - 1e-9)); h = dt / n
        Q = self.Q(h)
        s, P = tuple(self.s), self.P
        for _ in range(n):
            s_new, F = propagate_jac(s, h, self.E, self.L, self.obs)
            P = F @ P @ F.T + Q
            if self.sigma_E > 0:
                for a in (0, 1):
                    if s_new[2 + a] * s[2 + a] < 0:
                        P[2 + a, 2 + a] += (self.sigma_E * abs(s[2 + a])) ** 2
            s = s_new
        self.s, self.P = np.array(s), P

    def update(self, t, xy, valid):
        if not self.ready:
            if not valid:
                return
            z = np.asarray(xy, float)
            self.s = np.array([z[0], z[1], 0.0, 0.0, self.l0]); self.P = self.P0.copy()
            self.t, self.ready = t, True
            return
        if t > self.t:                                    # predict at every frame tick, measurement or not
            self._predict(t - self.t)
            self.t = t
        if not valid:
            return
        z = np.asarray(xy, float)
        y = z - self.s[:2]
        S = self.P[:2, :2] + self.R * np.eye(2)
        Si = np.linalg.inv(S)
        m = float(y @ Si @ y)
        self.nll += math.log(np.linalg.det(S)) + m
        self.n_upd += 1
        self.nis.append(m)
        K = self.P[:, :2] @ Si
        self.s = self.s + K @ y
        IKH = np.eye(NX); IKH[:, :2] -= K
        self.P = IKH @ self.P @ IKH.T + self.R * K @ K.T  # Joseph form: stays symmetric positive definite
        self.P = 0.5 * (self.P + self.P.T)

    def forecast(self, dt):
        return forecast(self.s[:4], dt, self.mu, self.E, self.L, self.obs)

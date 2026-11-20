"""B3: ekf-imm (2-mode IMM: slide / roll) and the ekf-2ph ablation.  Design: docs/B3_imm_ekf_design.md (frozen model).
State per mode x = (x, y, vx, vy); mu is NOT in the state.  Mode j has a constant Coulomb deceleration c_j (slide mu_s,
roll mu_r) and process noise q_j.  Same sub-step (5 ms), walls, arm box, restitution E, sigma_E, first-frame init as
EKFMu / ekf19b.  Interface of EKFMu: reset(), update(t, xy, valid), ready, t, s, P, forecast(dt), nis, nll, n_upd, mu.
Existing modules are reused unchanged (propagate_jac of ekf_mu.py with a fixed c: l = log c, 5th row/column dropped)."""
import math
import numpy as np
from .model import forecast, propagate
from .ekf_mu import propagate_jac


def _predict_mode(s, P, dt, c, q, E, L, obs, sigma_E, max_h):
    """EKF predict of one mode with fixed deceleration c over dt (sub-steps <= max_h), as EKFMu._predict without mu."""
    n = max(1, math.ceil(dt / max_h - 1e-9)); h = dt / n
    Q = np.zeros((4, 4))
    for i in (0, 1):
        Q[i, i] = q * h ** 3 / 3; Q[i, i + 2] = Q[i + 2, i] = q * h ** 2 / 2; Q[i + 2, i + 2] = q * h
    st = (s[0], s[1], s[2], s[3], math.log(c))
    for _ in range(n):
        st_new, F5 = propagate_jac(st, h, E, L, obs)
        F = F5[:4, :4]
        P = F @ P @ F.T + Q
        if sigma_E > 0:
            for a in (0, 1):
                if st_new[2 + a] * st[2 + a] < 0:
                    P[2 + a, 2 + a] += (sigma_E * abs(st[2 + a])) ** 2
        st = st_new
    return np.array(st[:4]), P


def _kf_update(s, P, z, R):
    """Joseph-form update; returns new s, P, innovation y, S."""
    y = z - s[:2]
    S = P[:2, :2] + R * np.eye(2)
    Si = np.linalg.inv(S)
    K = P[:, :2] @ Si
    IKH = np.eye(4); IKH[:, :2] -= K
    Pn = IKH @ P @ IKH.T + R * K @ K.T
    return s + K @ y, 0.5 * (Pn + Pn.T), y, S, Si


class EKFIMM:
    """modes: 0 = slide (mu_s, q_s), 1 = roll (mu_r, q_r).  Generator G = [[-a, a], [b, -b]] (1/s), Pi(dt) = expm(G dt)
    (closed form).  forecast_rule: 'fmix' (m-weighted deceleration, primary) | 'fml' (most probable mode)."""
    def __init__(self, L, obs=None, E=0.725, sigma=0.005, mu_s=1.0, mu_r=0.5, q_s=0.5, q_r=0.5, a=2.0, b=0.05,
                 p_s0=0.9, sigma_v0=3.0, sigma_E=0.13, max_h=0.005, m_min=1e-4, forecast_rule="fmix"):
        assert forecast_rule in ("fmix", "fml")
        self.L, self.obs, self.E = L, obs, E
        self.R = sigma ** 2
        self.c = np.array([mu_s, mu_r], float); self.q = np.array([q_s, q_r], float)
        self.a, self.b, self.p_s0 = a, b, p_s0
        self.P0 = np.diag([sigma ** 2, sigma ** 2, sigma_v0 ** 2, sigma_v0 ** 2])
        self.sigma_E, self.max_h, self.m_min, self.rule = sigma_E, max_h, m_min, forecast_rule
        self.reset()

    def reset(self):
        self.ready, self.t, self.s, self.P = False, None, None, None
        self.nll, self.n_upd = 0.0, 0
        self.nis = []
        self.m = np.array([self.p_s0, 1.0 - self.p_s0])
        self.xs = self.Ps = None

    @property
    def mode_prob(self):
        return self.m

    @property
    def mu(self):                                          # effective deceleration (descriptive; keeps the mu-recovery path alive)
        return float(self.m @ self.c)

    def Pi(self, dt):
        a, b = self.a, self.b
        s = a + b
        if s <= 0:
            return np.eye(2)
        e = math.exp(-s * dt)
        pss = (b + a * e) / s; prs = b * (1 - e) / s
        return np.array([[pss, 1 - pss], [prs, 1 - prs]])

    def _combine(self, w):
        x = sum(w[j] * self.xs[j] for j in range(2))
        P = sum(w[j] * (self.Ps[j] + np.outer(self.xs[j] - x, self.xs[j] - x)) for j in range(2))
        return x, P

    def _mix_predict(self, dt):
        """IMM steps 1-3: mixing probabilities, mixed initial conditions, per-mode predict.  Returns cbar."""
        Pi = self.Pi(dt)
        cbar = Pi.T @ self.m                               # cbar_j = sum_i Pi_ij m_i
        mu_ij = (Pi * self.m[:, None]) / np.maximum(cbar[None, :], 1e-300)   # mu_{i|j}
        xs, Ps = [], []
        for j in range(2):
            x0 = sum(mu_ij[i, j] * self.xs[i] for i in range(2))
            P0 = sum(mu_ij[i, j] * (self.Ps[i] + np.outer(self.xs[i] - x0, self.xs[i] - x0)) for i in range(2))
            x, P = _predict_mode(x0, P0, dt, self.c[j], self.q[j], self.E, self.L, self.obs, self.sigma_E, self.max_h)
            xs.append(x); Ps.append(P)
        self.xs, self.Ps = xs, Ps
        return cbar

    def update(self, t, xy, valid):
        if not self.ready:
            if not valid:
                return
            z = np.asarray(xy, float)
            x0 = np.array([z[0], z[1], 0.0, 0.0])
            self.xs = [x0.copy(), x0.copy()]; self.Ps = [self.P0.copy(), self.P0.copy()]
            self.m = np.array([self.p_s0, 1.0 - self.p_s0])
            self.s, self.P = x0, self.P0.copy()
            self.t, self.ready = t, True
            return
        if t > self.t:
            cbar = self._mix_predict(t - self.t)
            self.m = cbar
            self.t = t
            self.s, self.P = self._combine(self.m)
        if not valid:
            return
        z = np.asarray(xy, float)
        prior_s, prior_P = self.s, self.P                  # combined prior, for NIS
        ll = np.empty(2)
        for j in range(2):
            xj, Pj, y, S, Si = _kf_update(self.xs[j], self.Ps[j], z, self.R)
            ll[j] = -0.5 * (float(y @ Si @ y) + math.log(np.linalg.det(S)) + 2 * math.log(2 * math.pi))
            self.xs[j], self.Ps[j] = xj, Pj
        lw = np.log(np.maximum(self.m, 1e-300)) + ll
        mx = lw.max()
        lse = mx + math.log(np.exp(lw - mx).sum())
        self.nll -= lse
        m = np.exp(lw - lse)
        m = np.maximum(m, self.m_min); self.m = m / m.sum()
        y = z - prior_s[:2]; S = prior_P[:2, :2] + self.R * np.eye(2)
        self.nis.append(float(y @ np.linalg.solve(S, y)))
        self.n_upd += 1
        self.s, self.P = self._combine(self.m)

    def forecast(self, dt):
        c = float(self.m @ self.c) if self.rule == "fmix" else float(self.c[int(np.argmax(self.m))])
        return forecast(self.s, dt, c, self.E, self.L, self.obs)


class EKF2PH:
    """Ablation (variant (i)): one EKF whose deceleration switches slide -> roll by a deterministic rule.
    rule 'speed': slide while estimated speed > v_sw; 'time': slide while t - t_first < t_sw."""
    def __init__(self, L, obs=None, E=0.725, sigma=0.005, mu_s=1.0, mu_r=0.5, q=0.5, rule="speed", v_sw=0.3, t_sw=0.5,
                 sigma_v0=3.0, sigma_E=0.13, max_h=0.005):
        assert rule in ("speed", "time")
        self.L, self.obs, self.E, self.q = L, obs, E, q
        self.R = sigma ** 2
        self.mu_s, self.mu_r, self.rule, self.v_sw, self.t_sw = mu_s, mu_r, rule, v_sw, t_sw
        self.P0 = np.diag([sigma ** 2, sigma ** 2, sigma_v0 ** 2, sigma_v0 ** 2])
        self.sigma_E, self.max_h = sigma_E, max_h
        self.reset()

    def reset(self):
        self.ready, self.t, self.s, self.P, self.t0 = False, None, None, None, None
        self.nll, self.n_upd = 0.0, 0
        self.nis = []

    def _c(self, speed, t):
        slide = speed > self.v_sw if self.rule == "speed" else (t - self.t0) < self.t_sw
        return self.mu_s if slide else self.mu_r

    @property
    def mu(self):
        return self._c(math.hypot(self.s[2], self.s[3]), self.t)

    def _predict(self, dt):
        n = max(1, math.ceil(dt / self.max_h - 1e-9)); h = dt / n
        s, P, t = self.s, self.P, self.t
        for _ in range(n):                                  # c chosen per sub-step by the rule
            c = self._c(math.hypot(s[2], s[3]), t)
            s, P = _predict_mode(s, P, h, c, self.q, self.E, self.L, self.obs, self.sigma_E, self.max_h)
            t += h
        self.s, self.P = s, P

    def update(self, t, xy, valid):
        if not self.ready:
            if not valid:
                return
            z = np.asarray(xy, float)
            self.s = np.array([z[0], z[1], 0.0, 0.0]); self.P = self.P0.copy()
            self.t, self.t0, self.ready = t, t, True
            return
        if t > self.t:
            self._predict(t - self.t)
            self.t = t
        if not valid:
            return
        z = np.asarray(xy, float)
        self.s, self.P, y, S, Si = _kf_update(self.s, self.P, z, self.R)
        m = float(y @ Si @ y)
        self.nll += math.log(np.linalg.det(S)) + m
        self.nis.append(m); self.n_upd += 1

    def forecast(self, dt):
        if dt <= 0:
            return tuple(self.s)
        n = max(1, math.ceil(dt / 0.01)); h = dt / n
        x, y, vx, vy = self.s; t = self.t
        for _ in range(n):
            x, y, vx, vy = propagate(x, y, vx, vy, h, self._c(math.hypot(vx, vy), t), self.E, self.L, self.obs)
            t += h
        return x, y, vx, vy

"""Three baseline estimators with one interface:
    est.reset();  est.update(t_capture, xy, valid);  est.ready;  est.t (time of the state);
    est.forecast(dt) -> (x, y, vx, vy) after dt seconds."""
import math
import numpy as np
from scipy.stats import chi2
from .model import propagate, forecast


class FiniteDifference:
    """Position = latest measurement; velocity = difference of the last two valid measurements."""
    def __init__(self, c=0.0, E=0.0, L=1e9):
        self.reset()

    def reset(self):
        self.ready, self.t, self.s, self._last = False, None, None, None

    def update(self, t, xy, valid):
        if not valid:
            return
        v = (0.0, 0.0)
        if self._last is not None and t > self._last[0]:
            d = (xy - self._last[1]) / (t - self._last[0])
            v = (d[0], d[1])
        self._last = (t, np.array(xy, float))
        self.s, self.t, self.ready = (xy[0], xy[1], v[0], v[1]), t, True

    def forecast(self, dt):
        x, y, vx, vy = self.s
        return x + vx * dt, y + vy * dt, vx, vy            # straight line


class EKF19:
    """EKF after Liu et al. 2021: friction along v/|v|, wall reflection, gate on the innovation.
    Extension: reflection off the arm base box `obs` too (same rule as the walls).
    c = 0 and no walls gives the plain constant-velocity Kalman filter (policy 'none').
    L and obs have no default on purpose: they come from the scene (Arena.estimator_geometry())."""
    def __init__(self, L, obs=None, c=0.85, E=0.725, q=0.5, sigma=0.005, sigma_v0=3.0,
                 gate_p=0.99, policy="hybrid", k_reset=3, max_h=0.005, sigma_E=0.0):
        self.c, self.E, self.L, self.obs, self.q = c, E, L, obs, q
        self.sigma_E = sigma_E                  # 0 = paper (P unchanged at a bounce); >0 = ekf19b
        self.R = sigma ** 2
        self.P0 = np.diag([sigma ** 2, sigma ** 2, sigma_v0 ** 2, sigma_v0 ** 2])
        self.gate = chi2.ppf(gate_p, 2)
        self.policy, self.k_reset, self.max_h = policy, k_reset, max_h   # none | skip | reset | hybrid
        self.reset()

    def reset(self):
        self.ready, self.t, self.s, self.P, self.fails = False, None, None, None, 0
        self.n_reset = self.n_gate = 0

    # ---- prediction: mean through the model, covariance through the Jacobian -----------
    def _predict(self, dt):
        n = max(1, math.ceil(dt / self.max_h)); h = dt / n
        Q = np.zeros((4, 4))
        for i in (0, 1):
            Q[i, i] = self.q * h ** 3 / 3; Q[i, i + 2] = Q[i + 2, i] = self.q * h ** 2 / 2
            Q[i + 2, i + 2] = self.q * h
        s = tuple(self.s); P = self.P
        for _ in range(n):
            F = np.eye(4); F[0, 2] = F[1, 3] = h
            sp = math.hypot(s[2], s[3])
            if self.c > 0 and sp > self.c * h:
                vh = np.array([s[2], s[3]]) / sp
                F[2:, 2:] = np.eye(2) - (self.c * h / sp) * (np.eye(2) - np.outer(vh, vh))
            P = F @ P @ F.T + Q                    # paper: a bounce does not change P
            s_new = propagate(*s, h, self.c, self.E, self.L, self.obs)
            if self.sigma_E > 0:                   # ekf19b: E is uncertain -> extra velocity variance at a hit
                for a in (0, 1):                   # hit (wall or arm) = velocity component changed sign
                    if s_new[2 + a] * s[2 + a] < 0:
                        P[2 + a, 2 + a] += (self.sigma_E * abs(s[2 + a])) ** 2
            s = s_new
        self.s, self.P = np.array(s), P

    def _reset_to(self, t, z):
        self.s = np.array([z[0], z[1], 0.0, 0.0]); self.P = np.eye(4)      # paper: P = identity
        self.t, self.fails = t, 0; self.n_reset += 1

    def update(self, t, xy, valid):
        if not valid:
            return
        z = np.asarray(xy, float)
        if not self.ready:
            self.s = np.array([z[0], z[1], 0.0, 0.0]); self.P = self.P0.copy()
            self.t, self.ready = t, True
            return
        if t > self.t:
            self._predict(t - self.t)
        self.t = t
        y = z - self.s[:2]
        S = self.P[:2, :2] + self.R * np.eye(2)
        if self.policy != "none" and y @ np.linalg.solve(S, y) > self.gate:
            self.n_gate += 1; self.fails += 1
            if self.policy == "reset" or (self.policy == "hybrid" and self.fails >= self.k_reset):
                self._reset_to(t, z)
            return                                  # otherwise: skip this measurement
        self.fails = 0
        K = self.P[:, :2] @ np.linalg.inv(S)
        self.s = self.s + K @ y
        self.P = (np.eye(4) - K @ np.eye(4)[:2]) @ self.P
        self.P = 0.5 * (self.P + self.P.T)

    def forecast(self, dt):
        return forecast(self.s, dt, self.c, self.E, self.L, self.obs)


class StaleGuard:
    """Wraps any estimator: a measurement bit-identical to the previous one is treated as 'no measurement'.
    Assumption (documented): a frozen camera repeats the exact same values, while a real noisy
    measurement never repeats exactly. In a noise-free simulation a ball at rest also repeats; that is harmless."""
    def __init__(self, est):
        self.est, self.n_stale, self._last = est, 0, None

    def reset(self):
        self.est.reset(); self.n_stale, self._last = 0, None

    def update(self, t, xy, valid):
        if valid:
            z = np.asarray(xy, float)
            if self._last is not None and np.array_equal(z, self._last):
                valid = False; self.n_stale += 1
            self._last = z
        self.est.update(t, xy, valid)

    def __getattr__(self, name):                 # ready, t, s, forecast, n_gate, ... come from the wrapped estimator
        return getattr(self.est, name)


def make(name, rec=None, **kw):
    """fd | cvkf | ekf19[-none|-skip|-reset|-hybrid] | ekf19b[...] | ekf19_oracle[...] | ekf19b_oracle[...]  (oracle needs rec['mu'], rec['e']).
    The EKFs need the scene geometry in kw: L=..., obs=... (Arena.estimator_geometry())."""
    if name.endswith("+stale"):                            # e.g. "ekf19b-none+stale"
        return StaleGuard(make(name[:-len("+stale")], rec, **kw))
    if name == "fd":
        return FiniteDifference()
    if name == "cvkf":                                     # no model: ignores walls and arm
        kw = {k: v for k, v in kw.items() if k not in ("L", "obs")}
        return EKF19(L=1e9, c=0.0, E=0.0, policy="none", **kw)
    base, _, pol = name.partition("-")
    if base == "ekf19":
        return EKF19(policy=pol or "hybrid", **kw)
    if base == "ekf19b":                                   # + extra uncertainty at wall bounces
        return EKF19(policy=pol or "hybrid", sigma_E=0.13, **kw)
    if base == "ekf19_oracle":
        return EKF19(c=rec["mu"], E=rec["e"], policy=pol or "hybrid", **kw)
    if base == "ekf19b_oracle":
        return EKF19(c=rec["mu"], E=rec["e"], policy=pol or "hybrid", sigma_E=0.13, **kw)
    raise ValueError(name)

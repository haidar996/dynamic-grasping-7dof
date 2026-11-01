"""Batch version of ekf-mu (estimators/ekf_mu.py) for LEARNING: all launches of a dataset run in lockstep in numpy.

Every launch has the same frame times (60 Hz clock of the simulator), and EKFMu predicts at every frame tick, so one
loop over the frames with arrays of B launches does exactly what B separate EKFMu + StaleGuard runs do (checked by
experiments/step_1b_test_ekf_mu_batch.py).  Masks handle launches that are not yet initialised, frames without a
measurement, stale (bit-identical) measurements, friction stops, wall / arm bounces.
Outputs: per-launch negative log-likelihood of the innovations (paper 03, Eq. 7) and, optionally, the h-second forecast
errors exactly as estimators.metrics.evaluate computes them."""
import math
import numpy as np

NX = 5


def _propagate(S, h, E, L, obs, jac=True):
    """Vectorised estimators.ekf_mu.propagate_jac.  S: (B, 5).  Returns S_new (B, 5) and F (B, 5, 5) (or None)."""
    B = S.shape[0]
    x, y, vx, vy, l = (S[:, i].copy() for i in range(5))
    mu = np.exp(l)
    I = np.eye(NX)
    jx = np.tile(I[0], (B, 1)); jy = np.tile(I[1], (B, 1))
    jvx = np.tile(I[2], (B, 1)); jvy = np.tile(I[3], (B, 1))
    sp = np.hypot(vx, vy)
    stop = (sp > 0) & (mu * h >= sp)
    mov = (sp > 0) & ~stop
    spm = np.where(mov, sp, 1.0)
    f = np.where(mov, mu * h / spm, 0.0)
    if jac:
        hx_, hy_ = vx / spm, vy / spm
        fx = np.stack([np.zeros(B), np.zeros(B), 1 - f * (1 - hx_ ** 2), f * hx_ * hy_, -mu * h * hx_], 1)
        fy = np.stack([np.zeros(B), np.zeros(B), f * hx_ * hy_, 1 - f * (1 - hy_ ** 2), -mu * h * hy_], 1)
        jvx = np.where(mov[:, None], fx, np.where(stop[:, None], 0.0, jvx))
        jvy = np.where(mov[:, None], fy, np.where(stop[:, None], 0.0, jvy))
    vx = np.where(mov, vx - f * vx, np.where(stop, 0.0, vx))
    vy = np.where(mov, vy - f * vy, np.where(stop, 0.0, vy))
    xo, yo = x.copy(), y.copy()
    x = x + vx * h
    y = y + vy * h
    if jac:
        jx = jx + h * jvx; jy = jy + h * jvy
    for p, v, jp, jv in ((x, vx, jx, jvx), (y, vy, jy, jvy)):     # walls (in place)
        hi, lo = p > L, p < -L
        hit = hi | lo
        p[:] = np.where(hi, L - E * (p - L), np.where(lo, -L + E * (-L - p), p))
        v[:] = np.where(hit, -E * v, v)
        if jac:
            jp[:] = np.where(hit[:, None], -E * jp, jp); jv[:] = np.where(hit[:, None], -E * jv, jv)
    if obs is not None:
        cx, cy, hx, hy = obs
        inside = (np.abs(xo - cx) < hx) & (np.abs(yo - cy) < hy)
        # started inside: push out through the face with the smallest penetration
        px, py = hx - np.abs(x - cx), hy - np.abs(y - cy)
        ax0 = inside & (px <= py); ax1 = inside & ~(px <= py)
        sx = np.where(x >= cx, 1.0, -1.0); sy = np.where(y >= cy, 1.0, -1.0)
        fx0 = ax0 & (vx * sx < 0); fy1 = ax1 & (vy * sy < 0)
        x = np.where(ax0, cx + sx * hx, x); y = np.where(ax1, cy + sy * hy, y)
        vx = np.where(fx0, -E * vx, vx); vy = np.where(fy1, -E * vy, vy)
        if jac:
            jx = np.where(ax0[:, None], 0.0, jx); jy = np.where(ax1[:, None], 0.0, jy)
            jvx = np.where(fx0[:, None], -E * jvx, jvx); jvy = np.where(fy1[:, None], -E * jvy, jvy)
        # otherwise: swept slab test on the segment (xo, yo) -> (x, y)
        t_in = np.zeros(B); t_out = np.ones(B); axis = -np.ones(B, int); miss = inside.copy()
        for a, (p0, p1, c, hw) in enumerate(((xo, x, cx, hx), (yo, y, cy, hy))):
            d = p1 - p0
            dz = d == 0.0
            miss |= dz & (np.abs(p0 - c) >= hw)
            ds = np.where(dz, 1.0, d)
            t1, t2 = (c - hw - p0) / ds, (c + hw - p0) / ds
            t1, t2 = np.minimum(t1, t2), np.maximum(t1, t2)
            t1 = np.where(dz, -np.inf, t1); t2 = np.where(dz, np.inf, t2)
            upd = t1 > t_in
            axis = np.where(upd, a, axis); t_in = np.where(upd, t1, t_in)
            t_out = np.minimum(t_out, t2)
            miss |= t_in >= t_out
        hit = ~miss & (axis >= 0)
        h0, h1 = hit & (axis == 0), hit & (axis == 1)
        facex = cx + np.where(xo >= cx, hx, -hx); facey = cy + np.where(yo >= cy, hy, -hy)
        x = np.where(h0, facex - E * (x - facex), x); vx = np.where(h0, -E * vx, vx)
        y = np.where(h1, facey - E * (y - facey), y); vy = np.where(h1, -E * vy, vy)
        if jac:
            jx = np.where(h0[:, None], -E * jx, jx); jvx = np.where(h0[:, None], -E * jvx, jvx)
            jy = np.where(h1[:, None], -E * jy, jy); jvy = np.where(h1[:, None], -E * jvy, jvy)
    S_new = np.stack([x, y, vx, vy, l], 1)
    F = np.stack([jx, jy, jvx, jvy, np.tile(I[4], (B, 1))], 1) if jac else None
    return S_new, F


def forecast(S, dt, E, L, obs, max_h=0.01):
    """Vectorised estimators.model.forecast of the mean (position, velocity) dt seconds ahead (same dt for all)."""
    if dt <= 0:
        return S[:, :4].copy()
    n = max(1, math.ceil(dt / max_h)); h = dt / n
    for _ in range(n):
        S, _ = _propagate(S, h, E, L, obs, jac=False)
    return S[:, :4]


def run(data, p, L, obs, max_h=0.005, horizon=None, every=6, warmup=0.1):
    """data: dict t (T,), Z (B, T, 2), V (B, T) bool, and for horizon errors: delay, target (B, T, 2) = true position at
    t + delay + horizon, ok (B, T) = frame may be evaluated (true speed >= v_min at arrival, inside the recording).
    p: dict sigma, q, q_mu, mu0, s_l, sigma_v0, E, sigma_E.
    Returns nll (B,), n_upd (B,), and if horizon: list over launches of the forecast-error arrays."""
    t, Z, V = data["t"], data["Z"], data["V"]
    B, T = V.shape
    E, R = p["E"], p["sigma"] ** 2
    P0 = np.diag([R, R, p["sigma_v0"] ** 2, p["sigma_v0"] ** 2, p["s_l"] ** 2])
    S = np.zeros((B, NX)); P = np.tile(P0, (B, 1, 1))
    ready = np.zeros(B, bool); first_t = np.full(B, np.nan)
    last = np.full((B, 2), np.nan)
    nll = np.zeros(B); n_upd = np.zeros(B, int)
    errs = [[] for _ in range(B)] if horizon is not None else None
    I = np.eye(NX)
    for k in range(T):
        z, val = Z[:, k], V[:, k].copy()
        # StaleGuard: a valid measurement bit-identical to the previous valid one counts as missing
        same = val & np.all(z == last, axis=1)
        last = np.where(val[:, None], z, last)
        val &= ~same
        if k > 0 and ready.any():                         # predict (all ready launches share the same dt)
            dt = t[k] - t[k - 1]
            n = max(1, math.ceil(dt / max_h - 1e-9)); h = dt / n
            Q = np.zeros((NX, NX))
            for i in (0, 1):
                Q[i, i] = p["q"] * h ** 3 / 3; Q[i, i + 2] = Q[i + 2, i] = p["q"] * h ** 2 / 2; Q[i + 2, i + 2] = p["q"] * h
            Q[4, 4] = p["q_mu"] * h
            for _ in range(n):
                Sn, F = _propagate(S, h, E, L, obs)
                Pn = F @ P @ F.transpose(0, 2, 1) + Q
                if p["sigma_E"] > 0:
                    for a in (2, 3):
                        flip = Sn[:, a] * S[:, a] < 0
                        Pn[:, a, a] += np.where(flip, (p["sigma_E"] * np.abs(S[:, a])) ** 2, 0.0)
                S = np.where(ready[:, None], Sn, S); P = np.where(ready[:, None, None], Pn, P)
        upd = val & ready
        init = val & ~ready
        if upd.any():
            y = z - S[:, :2]
            a_, b_, d_ = P[:, 0, 0] + R, P[:, 0, 1], P[:, 1, 1] + R
            det = a_ * d_ - b_ * b_
            Si = np.stack([np.stack([d_, -b_], 1), np.stack([-b_, a_], 1)], 1) / det[:, None, None]
            m = np.einsum("bi,bij,bj->b", y, Si, y)
            nll += np.where(upd, np.log(np.where(upd, det, 1.0)) + m, 0.0)
            n_upd += upd
            K = P[:, :, :2] @ Si
            Sn = S + np.einsum("bij,bj->bi", K, np.where(upd[:, None], y, 0.0))
            IKH = np.tile(I, (B, 1, 1)); IKH[:, :, :2] -= K
            Pn = IKH @ P @ IKH.transpose(0, 2, 1) + R * K @ K.transpose(0, 2, 1)
            Pn = 0.5 * (Pn + Pn.transpose(0, 2, 1))
            S = np.where(upd[:, None], Sn, S); P = np.where(upd[:, None, None], Pn, P)
        if init.any():
            S[init] = np.column_stack([z[init], np.zeros((init.sum(), 2)), np.full(init.sum(), math.log(p["mu0"]))])
            P[init] = P0
            ready |= init
            first_t[init] = t[k]
        if horizon is not None and k % every == 0:
            ev = ready & (t[k] - first_t >= warmup) & data["ok"][:, k]
            if ev.any():
                # same float expression as metrics.evaluate (t_arrival + h - est.t): the number of forecast sub-steps
                # ceil(dt / 0.01) is sensitive to the last bit of dt
                fc = forecast(S[ev], data["ta"][k] + horizon - t[k], E, L, obs)
                e = np.hypot(fc[:, 0] - data["target"][ev, k, 0], fc[:, 1] - data["target"][ev, k, 1])
                for b, ee in zip(np.flatnonzero(ev), e):
                    errs[b].append(ee)
    if errs is not None:
        errs = [np.array(e) for e in errs]
    return nll, n_upd, errs


def prepare(recs, horizon=0.5, v_min=0.05):
    """Stack launch records (dataset.apply_condition) into the arrays run() needs.  Targets follow
    estimators.metrics.evaluate: true position at arrival + horizon, frame evaluated only if the true speed at
    arrival >= v_min and arrival + horizon is inside the recording."""
    t = recs[0]["t_capture"]
    assert all(np.array_equal(r["t_capture"], t) for r in recs), "launches must share the frame clock"
    delay = float(recs[0]["t_arrival"][0] - t[0])
    Z = np.stack([np.nan_to_num(r["xy"], nan=0.0) for r in recs]); V = np.stack([r["valid"] for r in recs])
    target = np.zeros(Z.shape); ok = np.zeros(V.shape, bool)
    for b, r in enumerate(recs):
        tp, pp, vp = r["t_phys"], r["pos"], r["vel"]
        ta = r["t_arrival"]
        sp = np.hypot(np.interp(ta, tp, vp[:, 0]), np.interp(ta, tp, vp[:, 1]))
        ok[b] = (sp >= v_min) & (ta + horizon <= tp[-1])
        target[b, :, 0] = np.interp(ta + horizon, tp, pp[:, 0]); target[b, :, 1] = np.interp(ta + horizon, tp, pp[:, 1])
    return dict(t=t, Z=Z, V=V, delay=delay, ta=recs[0]["t_arrival"], target=target, ok=ok)

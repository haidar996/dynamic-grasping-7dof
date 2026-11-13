"""P7: learned Kalman gain (KalmanNet-style, paper 04) on the ekf19b motion model.  Design: docs/P7_learned_gain_design.md.

Prediction = the physics model of estimators/model.py (friction c, walls, arm box; here re-implemented in torch, vectorised over
launches, checked against the float version by experiments/step_2a_train_knet.py --check).  No covariance: at every accepted
measurement a GRU maps KalmanNet-style features to the 4x2 gain K, and s += K (z - Hs).  Missing / stale frames: predict only.
Two entry points:
  run_batch(net, data, c, E, L, obs, ...)  all launches in lockstep on the 60 Hz grid -> differentiable loss (training, validation)
  KNet(weights, c, E, L, obs)              one-launch estimator with the common interface (reset / update / forecast) for step_1e
"""
import math
import numpy as np
import torch
import torch.nn as nn
from .model import forecast as forecast_np

SCALE_POS, SCALE_VEL, SCALE_Y, SCALE_DT = 0.1, 1.0, 0.05, 0.1
N_FEAT, HIDDEN = 14, 32
K0 = torch.tensor([[0.6, 0.0], [0.0, 0.6], [6.0, 0.0], [0.0, 6.0]])     # start as a fixed-gain filter (then learned)


# ----------------------------------------------------------------------------------------------------- torch physics model
def propagate_t(S, h, c, E, L, obs):
    """estimators.model.propagate for a (B, 4) tensor; piecewise smooth, autograd-friendly.  Returns S_new, bounce flag (B,)."""
    x, y, vx, vy = S.unbind(1)
    sp = torch.hypot(vx, vy)
    stop = (sp > 0) & (c * h >= sp)
    mov = (sp > 0) & ~stop
    spm = torch.where(mov, sp, torch.ones_like(sp))
    f = torch.where(mov, c * h / spm, torch.zeros_like(sp))
    vx = torch.where(stop, torch.zeros_like(vx), vx - f * vx)
    vy = torch.where(stop, torch.zeros_like(vy), vy - f * vy)
    xo, yo = x, y
    x = x + vx * h
    y = y + vy * h
    vx0, vy0 = vx, vy
    over, under = x > L, x < -L
    x = torch.where(over, L - E * (x - L), torch.where(under, -L + E * (-L - x), x))
    vx = torch.where(over | under, -E * vx, vx)
    over, under = y > L, y < -L
    y = torch.where(over, L - E * (y - L), torch.where(under, -L + E * (-L - y), y))
    vy = torch.where(over | under, -E * vy, vy)
    if obs is not None:
        cx, cy, hx, hy = obs
        inside = (torch.abs(xo - cx) < hx) & (torch.abs(yo - cy) < hy)
        # started inside: push out through the face with the smallest penetration
        px, py = hx - torch.abs(x - cx), hy - torch.abs(y - cy)
        sx = torch.where(x >= cx, 1.0, -1.0); sy = torch.where(y >= cy, 1.0, -1.0)
        xi, vxi = cx + sx * hx, torch.where(vx * sx < 0, -E * vx, vx)
        yi, vyi = cy + sy * hy, torch.where(vy * sy < 0, -E * vy, vy)
        usex = px <= py
        x_in = torch.where(usex, xi, x); vx_in = torch.where(usex, vxi, vx)
        y_in = torch.where(usex, y, yi); vy_in = torch.where(usex, vy, vyi)
        # swept slab test on the segment (xo, yo) -> (x, y)
        dx, dy = x - xo, y - yo
        zx, zy = dx == 0, dy == 0
        dxs, dys = torch.where(zx, torch.ones_like(dx), dx), torch.where(zy, torch.ones_like(dy), dy)
        ax1, ax2 = (cx - hx - xo) / dxs, (cx + hx - xo) / dxs
        ay1, ay2 = (cy - hy - yo) / dys, (cy + hy - yo) / dys
        t1x, t2x = torch.minimum(ax1, ax2), torch.maximum(ax1, ax2)
        t1y, t2y = torch.minimum(ay1, ay2), torch.maximum(ay1, ay2)
        inf = torch.full_like(dx, float("inf"))
        t1x = torch.where(zx, -inf, t1x); t2x = torch.where(zx, inf, t2x)
        t1y = torch.where(zy, -inf, t1y); t2y = torch.where(zy, inf, t2y)
        miss = (zx & (torch.abs(xo - cx) >= hx)) | (zy & (torch.abs(yo - cy) >= hy))
        t_in = torch.clamp(torch.maximum(t1x, t1y), min=0.0)
        t_out = torch.clamp(torch.minimum(t2x, t2y), max=1.0)
        axis_y = (t1y > t1x) & (t1y > 0)
        axis_x = (~axis_y) & (t1x > 0)
        hit = (~inside) & (~miss) & (t_in < t_out) & (axis_x | axis_y)
        fxf = cx + torch.where(xo >= cx, hx, -hx); fyf = cy + torch.where(yo >= cy, hy, -hy)
        xb, vxb = fxf - E * (x - fxf), -E * vx
        yb, vyb = fyf - E * (y - fyf), -E * vy
        hx_, hy_ = hit & axis_x, hit & axis_y
        x = torch.where(inside, x_in, torch.where(hx_, xb, x)); vx = torch.where(inside, vx_in, torch.where(hx_, vxb, vx))
        y = torch.where(inside, y_in, torch.where(hy_, yb, y)); vy = torch.where(inside, vy_in, torch.where(hy_, vyb, vy))
    bounce = (vx * vx0 < 0) | (vy * vy0 < 0)
    return torch.stack([x, y, vx, vy], 1), bounce


def forecast_t(S, dt, c, E, L, obs, max_h=0.01):
    """estimators.model.forecast for a (B, 4) tensor, same dt for all rows."""
    if dt <= 0:
        return S
    n = max(1, math.ceil(dt / max_h)); h = dt / n
    for _ in range(n):
        S, _ = propagate_t(S, h, c, E, L, obs)
    return S


# ----------------------------------------------------------------------------------------------------- the gain network
class GainNet(nn.Module):
    def __init__(self, hidden=HIDDEN):
        super().__init__()
        self.gru = nn.GRUCell(N_FEAT, hidden)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 8))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)   # K = K0 at the start

    def forward(self, feat, hdn):
        hdn = self.gru(feat, hdn)
        K = K0.to(feat) + self.head(hdn).view(-1, 4, 2)
        return K, hdn


def features(y, ds_pred, ds_upd, y_prev, dt_valid, bounce):
    return torch.cat([y / SCALE_Y, ds_pred[:, :2] / SCALE_POS, ds_pred[:, 2:] / SCALE_VEL, ds_upd[:, :2] / SCALE_POS,
                      ds_upd[:, 2:] / SCALE_VEL, y_prev / SCALE_Y, dt_valid[:, None] / SCALE_DT, bounce[:, None].float()], 1)


# ----------------------------------------------------------------------------------------------------- batch run (training)
def run_batch(net, data, c, E, L, obs, horizon=0.5, every=6, warmup=0.1, max_h=0.005, w_now=0.1):
    """All launches in lockstep.  data (torch tensors): t (T,), ta (T,), Z (B,T,2), V (B,T) bool, target (B,T,2) true position at
    arrival + horizon, target_now (B,T,2) true position at arrival, ok (B,T).  Returns loss (scalar), per-launch RMS h (B,), now (B,)."""
    t, ta, Z, V, ok = data["t"], data["ta"], data["Z"], data["V"], data["ok"]
    B, T = V.shape
    dev = Z.device
    S = torch.zeros(B, 4, device=dev); hdn = torch.zeros(B, net.gru.hidden_size, device=dev)
    ready = torch.zeros(B, dtype=torch.bool, device=dev); first_t = torch.full((B,), float("nan"), device=dev)
    last = torch.full((B, 2), float("nan"), device=dev)
    S_prev_upd = torch.zeros(B, 4, device=dev); ds_upd = torch.zeros(B, 4, device=dev); y_prev = torch.zeros(B, 2, device=dev)
    t_valid = torch.zeros(B, device=dev)
    sse_h = torch.zeros(B, device=dev); n_h = torch.zeros(B, device=dev); sse_n = torch.zeros(B, device=dev); n_n = torch.zeros(B, device=dev)
    for k in range(T):
        z, val = Z[:, k], V[:, k].clone()
        same = val & torch.all(z == last, dim=1)
        last = torch.where(val[:, None], z, last)
        val = val & ~same
        bounce = torch.zeros(B, dtype=torch.bool, device=dev)
        S_before = S
        if k > 0:
            dt = float(t[k] - t[k - 1])
            n = max(1, math.ceil(dt / max_h - 1e-9)); h = dt / n
            Sn = S
            for _ in range(n):
                Sn, bf = propagate_t(Sn, h, c, E, L, obs)
                bounce = bounce | bf
            S = torch.where(ready[:, None], Sn, S)
        upd = val & ready
        init = val & ~ready
        if bool(upd.any()):
            y = z - S[:, :2]
            feat = features(y, S - S_prev_upd, ds_upd, y_prev, float(t[k]) - t_valid, bounce)
            K, hn = net(feat, hdn)
            Sn = S + torch.einsum("bij,bj->bi", K, y)
            ds_upd = torch.where(upd[:, None], Sn - S, ds_upd)
            S = torch.where(upd[:, None], Sn, S)
            hdn = torch.where(upd[:, None], hn, hdn)
            y_prev = torch.where(upd[:, None], y, y_prev)
            S_prev_upd = torch.where(upd[:, None], S, S_prev_upd)
            t_valid = torch.where(upd, torch.full_like(t_valid, float(t[k])), t_valid)
        if bool(init.any()):
            S = torch.where(init[:, None], torch.cat([z, torch.zeros(B, 2, device=dev)], 1), S)
            S_prev_upd = torch.where(init[:, None], S, S_prev_upd)
            ready = ready | init
            first_t = torch.where(init, torch.full_like(first_t, float(t[k])), first_t)
            t_valid = torch.where(init, torch.full_like(t_valid, float(t[k])), t_valid)
        if k % every == 0:
            ev = ready & (float(t[k]) - first_t >= warmup) & ok[:, k]
            if bool(ev.any()):
                d0 = float(ta[k] - t[k])
                Snow = forecast_t(S, d0, c, E, L, obs)
                Sh = forecast_t(Snow, horizon, c, E, L, obs)
                en = torch.hypot(Snow[:, 0] - data["target_now"][:, k, 0], Snow[:, 1] - data["target_now"][:, k, 1])
                eh = torch.hypot(Sh[:, 0] - data["target"][:, k, 0], Sh[:, 1] - data["target"][:, k, 1])
                evf = ev.float()
                sse_n = sse_n + evf * en ** 2; n_n = n_n + evf
                sse_h = sse_h + evf * eh ** 2; n_h = n_h + evf
    has = n_h > 0
    rms_h = torch.sqrt(sse_h[has] / n_h[has]); rms_n = torch.sqrt(sse_n[has] / n_n[has].clamp(min=1))
    loss = rms_h.mean() + w_now * rms_n.mean()
    return loss, rms_h.detach(), rms_n.detach()


def prepare_t(recs, horizon=0.5, v_min=0.05):
    """Like estimators.ekf_mu_batch.prepare, plus target_now; returns torch tensors (float64 for the physics)."""
    t = recs[0]["t_capture"]
    assert all(np.array_equal(r["t_capture"], t) for r in recs)
    ta = recs[0]["t_arrival"]
    Z = np.stack([np.nan_to_num(r["xy"], nan=0.0) for r in recs]); V = np.stack([r["valid"] for r in recs])
    target = np.zeros(Z.shape); target_now = np.zeros(Z.shape); ok = np.zeros(V.shape, bool)
    for b, r in enumerate(recs):
        tp, pp, vp = r["t_phys"], r["pos"], r["vel"]
        sp = np.hypot(np.interp(ta, tp, vp[:, 0]), np.interp(ta, tp, vp[:, 1]))
        ok[b] = (sp >= v_min) & (ta + horizon <= tp[-1])
        for j in (0, 1):
            target[b, :, j] = np.interp(ta + horizon, tp, pp[:, j]); target_now[b, :, j] = np.interp(ta, tp, pp[:, j])
    f = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float64)
    return dict(t=f(t), ta=f(ta), Z=f(Z), V=torch.as_tensor(V), target=f(target), target_now=f(target_now), ok=torch.as_tensor(ok))


# ----------------------------------------------------------------------------------------------------- one-launch estimator
class KNet:
    """Common interface (reset / update / forecast / ready / t / s).  Same step semantics as run_batch (tick-wise prediction,
    StaleGuard inline), so no '+stale' wrapper is needed."""
    def __init__(self, weights, c, E, L, obs=None, max_h=0.005, **_):
        self.c, self.E, self.L, self.obs, self.max_h = c, E, L, obs, max_h
        self.net = GainNet().double(); self.net.load_state_dict(torch.load(weights, map_location="cpu")); self.net.eval()
        self.nis = []
        self.reset()

    def reset(self):
        self.ready, self.t, self.s = False, None, None
        self._S = None; self._h = torch.zeros(1, HIDDEN, dtype=torch.float64)
        self._last = None; self._S_prev = None; self._ds_upd = torch.zeros(1, 4, dtype=torch.float64)
        self._y_prev = torch.zeros(1, 2, dtype=torch.float64); self._t_valid = 0.0

    @torch.no_grad()
    def update(self, t, xy, valid):
        if valid:
            z = np.asarray(xy, float)
            if self._last is not None and np.array_equal(z, self._last):
                valid = False
            self._last = z
        bounce = torch.zeros(1, dtype=torch.bool)
        if self.ready and t > self.t:
            dt = t - self.t; n = max(1, math.ceil(dt / self.max_h - 1e-9)); h = dt / n
            for _ in range(n):
                self._S, bf = propagate_t(self._S, h, self.c, self.E, self.L, self.obs); bounce = bounce | bf
            self.t = t
        if not valid:
            self.t = t if self.ready else self.t
            self.s = self._S[0].numpy() if self.ready else None
            return
        zt = torch.as_tensor(np.asarray(xy, float), dtype=torch.float64)[None]
        if not self.ready:
            self._S = torch.cat([zt, torch.zeros(1, 2, dtype=torch.float64)], 1); self._S_prev = self._S.clone()
            self.t, self.ready, self._t_valid = t, True, t
            self.s = self._S[0].numpy()
            return
        y = zt - self._S[:, :2]
        feat = features(y, self._S - self._S_prev, self._ds_upd, self._y_prev, torch.tensor([t - self._t_valid], dtype=torch.float64), bounce)
        K, self._h = self.net(feat, self._h)
        Sn = self._S + torch.einsum("bij,bj->bi", K, y)
        self._ds_upd = Sn - self._S; self._S = Sn; self._S_prev = Sn.clone(); self._y_prev = y; self._t_valid = t
        self.s = self._S[0].numpy()

    def forecast(self, dt):
        return forecast_np(tuple(self.s), dt, self.c, self.E, self.L, self.obs)

"""Motion model shared by the filters: friction opposite to the velocity + reflection off the walls
and the (static) arm base box.  Plain floats (no numpy) so that long forecasts are fast."""
import math


def propagate(x, y, vx, vy, h, c, E, L, obs=None):
    """Advance the mean state by one sub-step h.
    c: friction deceleration (m/s^2), E: restitution, L: wall limit for the ball centre,
    obs: (cx, cy, hx, hy) box the ball centre may not enter (arm base grown by the radius), or None."""
    sp = math.hypot(vx, vy)
    if sp > 0.0:
        if c * h >= sp:
            vx = vy = 0.0                       # friction would stop the ball inside this sub-step
        else:
            f = c * h / sp
            vx -= f * vx
            vy -= f * vy
    xo, yo = x, y
    x += vx * h
    y += vy * h
    if x > L:
        x = L - E * (x - L); vx = -E * vx       # mirror the overshoot, keep E of the speed
    elif x < -L:
        x = -L + E * (-L - x); vx = -E * vx
    if y > L:
        y = L - E * (y - L); vy = -E * vy
    elif y < -L:
        y = -L + E * (-L - y); vy = -E * vy
    if obs is not None:
        x, y, vx, vy = _box_bounce(xo, yo, x, y, vx, vy, E, obs)
    return x, y, vx, vy


def _box_bounce(xo, yo, x, y, vx, vy, E, obs):
    """Reflect off the box if the step (xo, yo) -> (x, y) crosses it.  Swept test, so a long sub-step
    can neither pick the wrong face nor jump over a corner."""
    cx, cy, hx, hy = obs
    if abs(xo - cx) < hx and abs(yo - cy) < hy:          # started inside (e.g. a noisy state): push out
        px, py = hx - abs(x - cx), hy - abs(y - cy)        # through the face with the smallest penetration
        if px <= py:
            side = 1.0 if x >= cx else -1.0
            x = cx + side * hx
            if vx * side < 0:
                vx = -E * vx
        else:
            side = 1.0 if y >= cy else -1.0
            y = cy + side * hy
            if vy * side < 0:
                vy = -E * vy
        return x, y, vx, vy
    t_in, t_out, axis = 0.0, 1.0, -1                     # slab method on the segment, fraction t in [0, 1]
    for a, (p0, p1, c, hw) in enumerate(((xo, x, cx, hx), (yo, y, cy, hy))):
        d = p1 - p0
        if d == 0.0:
            if abs(p0 - c) >= hw:
                return x, y, vx, vy
            continue
        t1, t2 = (c - hw - p0) / d, (c + hw - p0) / d
        if t1 > t2:
            t1, t2 = t2, t1
        if t1 > t_in:
            t_in, axis = t1, a
        t_out = min(t_out, t2)
        if t_in >= t_out:
            return x, y, vx, vy                          # the segment misses the box
    if axis < 0:
        return x, y, vx, vy
    if axis == 0:                                        # mirror the overshoot behind the face, keep E of the speed
        face = cx + (hx if xo >= cx else -hx)
        x = face - E * (x - face); vx = -E * vx
    else:
        face = cy + (hy if yo >= cy else -hy)
        y = face - E * (y - face); vy = -E * vy
    return x, y, vx, vy


def forecast(s, dt, c, E, L, obs=None, max_h=0.01):
    """Mean state s = (x, y, vx, vy) after dt seconds."""
    if dt <= 0:
        return tuple(s)
    n = max(1, math.ceil(dt / max_h))
    h = dt / n
    x, y, vx, vy = s
    for _ in range(n):
        x, y, vx, vy = propagate(x, y, vx, vy, h, c, E, L, obs)
    return x, y, vx, vy

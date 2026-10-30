"""Realistic sensor on top of the ideal DepthCamera:
depth noise, depth quantisation, centroid jitter, frame rate, delay, dropout.
Every effect is off by default (value 0)."""
from dataclasses import dataclass
import numpy as np
import mujoco


@dataclass
class SensorParams:
    fps: float = 60.0              # frames per second
    delay: float = 0.0             # s; a frame captured at t arrives at t + delay
    k_depth: float = 0.0           # depth noise sigma_Z = k_depth * Z^2   (0 = off)
    quant_step: float = 0.0        # disparity step in pixels, e.g. 0.125  (0 = off)
    baseline: float = 0.075        # m, used only by the quantisation
    jitter_px: float = 0.0         # std of the pixel-centroid jitter       (0 = off)
    loss_rate: float = 0.0         # long-run fraction of lost frames       (0 = off)
    mean_gap: float = 6.0          # mean length of one loss, in frames
    loss_mode: str = "flagged"     # "flagged" or "silent_hold"
    seed: int = 0


class Sensor:
    def __init__(self, camera, params=SensorParams()):
        self.cam, self.p = camera, params
        # separate generators: the SAME gaps can be reused whatever the noise settings are
        self.rng_loss = np.random.default_rng([params.seed, 0])
        self.rng_noise = np.random.default_rng([params.seed, 1])
        self.lost = False          # state of the two-state loss chain
        self.last = None           # last delivered position (for silent hold)

    # ---- the effects ------------------------------------------------------
    def corrupt_depth(self, z):
        p = self.p
        if p.k_depth > 0:                                   # noise first ...
            z = z + self.rng_noise.normal(0.0, p.k_depth * np.asarray(z) ** 2)
        if p.quant_step > 0:                                # ... then rounding
            fB = self.cam.f * p.baseline
            z = fB / (np.round(fB / z / p.quant_step) * p.quant_step)
        return z

    def _step_loss(self):
        """One step of the seen/lost Markov chain; returns True if this frame is lost."""
        r, g = self.p.loss_rate, self.p.mean_gap
        if r <= 0:
            return False
        q = 1.0 / g                       # P(lost -> seen)
        pl = r * q / (1.0 - r)            # P(seen -> lost), gives long-run loss rate r
        if self.lost:
            self.lost = not (self.rng_loss.random() < q)
        else:
            self.lost = self.rng_loss.random() < pl
        return self.lost

    # ---- one frame ----------------------------------------------------------
    def observe(self, data):
        return self.apply(lambda: self.cam.measure_parts(data))

    def apply(self, get_parts):
        """Returns (xy or None, valid, was_lost).  get_parts() -> (u, v, z) or None.
        valid=False means 'no measurement' (flagged).  was_lost is the hidden label."""
        lost = self._step_loss()
        if lost and (self.p.loss_mode == "flagged" or self.last is None):
            return None, False, True
        if lost:                                            # silent hold: stale value, looks valid
            return self.last.copy(), True, True
        parts = get_parts()
        if parts is None:                                   # detector found nothing
            return None, False, False
        u, v, z = parts
        if self.p.jitter_px > 0:
            u, v = np.array([u, v]) + self.rng_noise.normal(0.0, self.p.jitter_px, 2)
        xy = self.cam.back_project(u, v, self.corrupt_depth(z))
        self.last = xy
        return xy, True, False


def record_launch(model, arena, launch, sensor, T=10.0):
    """Simulate one launch; capture frames at the sensor's frame rate.
    Returns arrays: t_capture, t_arrival, xy (NaN if no measurement), valid,
    was_lost (hidden label), true_xy, true_v."""
    data = mujoco.MjData(model)
    arena.reset(data, launch.p0, launch.v0)
    period = 1.0 / sensor.p.fps
    next_t = 0.0
    keys = ("t_capture", "xy", "valid", "was_lost", "true_xy", "true_v")
    out = {k: [] for k in keys}
    for _ in range(int(T / model.opt.timestep)):
        arena.step(data, launch.mu, launch.e)
        if data.time >= next_t:
            mujoco.mj_forward(model, data)       # refresh geometry after step + bounce
            xy, valid, lost = sensor.observe(data)
            out["t_capture"].append(data.time)
            out["xy"].append(np.full(2, np.nan) if xy is None else xy)
            out["valid"].append(valid)
            out["was_lost"].append(lost)
            out["true_xy"].append(data.qpos[:2].copy())
            out["true_v"].append(data.qvel[:2].copy())
            next_t += period
    r = {k: np.array(v) for k, v in out.items()}
    r["t_arrival"] = r["t_capture"] + sensor.p.delay
    return r

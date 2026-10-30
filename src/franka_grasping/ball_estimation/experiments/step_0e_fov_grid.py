#!/usr/bin/env python3
"""Step 0.2: FOV check with a static ball grid (480x480 image, camera 3 m, arm scene).
Ball placed every 5 cm over the reachable area, same points + same noise draws for every FOV (paired).
Per FOV: detection, whole-ball-in-image margin, ball size, ideal / noisy position error, arena fill.
Rule (agreed): smallest FOV keeping the arena + 5 cm margin in view (analytic 48.3 -> 49 deg); this test verifies it.
Usage: python3 experiments/step_0e_fov_grid.py [--workers 3]"""
import argparse, json, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

H, W, R = 3.0, 480, 480
FOVS = list(range(40, 71))
CHOSEN = 49
STEP = 0.05
OUT = root / "outputs" / "fov_grid"
_w = {}


def grid_points():
    import mujoco
    from arena_sim import Arena
    arena = Arena(mujoco.MjModel.from_xml_path(str(root / "scenes" / "arena.xml")))
    lim = arena.Lx - arena.radius
    g = np.arange(-lim, lim + 1e-9, STEP)
    pts = np.array([(x, y) for x in g for y in g])
    return pts[[not arena.inside_obstacle(p) for p in pts]], arena


def _init():
    import mujoco
    from sensing.camera import DepthCamera
    m = mujoco.MjModel.from_xml_path(str(root / "scenes" / "arena.xml"))
    _w.update(m=m, d=mujoco.MjData(m), cam=DepthCamera(m, H, 50.0, width=W, rows=R))


def _run(args):
    import mujoco
    from dataset import CONDITIONS
    from sensing.sensor import Sensor, SensorParams
    fov, pts = args
    m, d, cam = _w["m"], _w["d"], _w["cam"]
    cam.configure(H, float(fov))
    n = len(pts)
    det = np.zeros(n, bool); margin = np.full(n, -1.0); npx = np.zeros(n)
    e_ideal = np.full(n, np.nan); e_noise = np.full(n, np.nan)
    noise = CONDITIONS["noise"]
    for i, p in enumerate(pts):
        d.qpos[:2] = p
        mujoco.mj_forward(m, d)
        rgb, depth = cam.render(d)
        g = (rgb[..., 1] > rgb[..., 0] + 40) & (rgb[..., 1] > rgb[..., 2] + 40)
        npx[i] = g.sum()
        if npx[i] < 3:
            continue
        v, u = np.nonzero(g)
        det[i] = True
        margin[i] = min(u.min(), W - 1 - u.max(), v.min(), R - 1 - v.max())
        uc, vc = u.mean(), v.mean()
        parts = (uc, vc, depth[int(round(vc)), int(round(uc))])
        e_ideal[i] = np.linalg.norm(cam.back_project(*parts) - p)
        s = Sensor(cam, SensorParams(**{**noise.__dict__, "seed": i}))      # same draw per point for every FOV
        xy, _, _ = s.apply(lambda: parts)
        e_noise[i] = np.linalg.norm(xy - p)
    return fov, dict(det=det, margin=margin, npx=npx, e_ideal=e_ideal, e_noise=e_noise, f=cam.f)


def arena_fill(fov):
    f = (R / 2) / np.tan(np.radians(fov) / 2)
    u, v = np.meshgrid(np.arange(W), np.arange(R))
    x, y = (u - (W - 1) / 2) * H / f, (v - (R - 1) / 2) * H / f
    return 100 * np.mean((np.abs(x) <= 1.25) & (np.abs(y) <= 1.25))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pts, arena = grid_points()
    print(f"{len(pts)} grid points, FOVs {FOVS[0]}..{FOVS[-1]}, {W}x{R}, {a.workers} workers", flush=True)
    t0, res = time.time(), {}
    with Pool(a.workers, initializer=_init) as pool:
        for fov, r in pool.imap_unordered(_run, [(f, pts) for f in FOVS]):
            res[fov] = r
            print(f"  FOV {fov} done ({len(res)}/{len(FOVS)}, {(time.time() - t0) / 60:.1f} min)", flush=True)

    seen_wide = res[FOVS[-1]]["det"]                          # not seen even at 70 deg = hidden by the arm
    corner = (np.abs(pts[:, 0]) > 1.1) & (np.abs(pts[:, 1]) > 1.1)
    rows = []
    print(f"\narm-hidden points (not seen at {FOVS[-1]}°): {(~seen_wide).sum()}")
    print(f"{'FOV':>4} {'f px':>6} {'mm/px':>6} {'seen%':>7} {'cut':>5} {'minMarg':>8} {'ball px':>8} "
          f"{'ideal med':>9} {'noise med':>9} {'noise p95':>9} {'corner p95':>10} {'fill%':>6}  pass")
    for fov in FOVS:
        r = res[fov]; vis = seen_wide
        ok_det = r["det"][vis]
        cut = int(np.sum(r["margin"][vis] < 1))                 # ball missing or touching the image border
        row = dict(fov=fov, f=r["f"], mm_px=1000 * (H - 0.03) / r["f"], seen=100 * ok_det.mean(), cut=cut,
                   min_margin=float(r["margin"][vis].min()), ball_px=float(np.median(r["npx"][vis & r["det"]])),
                   ideal_med=1000 * np.nanmedian(r["e_ideal"][vis]), noise_med=1000 * np.nanmedian(r["e_noise"][vis]),
                   noise_p95=1000 * np.nanpercentile(r["e_noise"][vis], 95),
                   corner_p95=1000 * np.nanpercentile(r["e_noise"][vis & corner], 95) if (vis & corner & r["det"]).any() else np.nan,
                   fill=arena_fill(fov), passed=cut == 0)
        rows.append(row)
        print(f"{fov:4d} {row['f']:6.0f} {row['mm_px']:6.2f} {row['seen']:7.2f} {cut:5d} {row['min_margin']:8.0f} "
              f"{row['ball_px']:8.0f} {row['ideal_med']:9.2f} {row['noise_med']:9.2f} {row['noise_p95']:9.2f} "
              f"{row['corner_p95']:10.2f} {row['fill']:6.1f}  {'PASS' if row['passed'] else 'fail'}"
              + ("   <- chosen" if fov == CHOSEN else ""))
    first = next((r["fov"] for r in rows if r["passed"]), None)
    print(f"\nsmallest FOV with the whole ball in view at every non-hidden point: {first}°  (chosen {CHOSEN}°)")
    (OUT / "table.json").write_text(json.dumps(rows, indent=1, default=float))
    np.savez_compressed(OUT / "raw.npz", pts=pts, **{f"{k}_{f}": v for f, r in res.items() for k, v in r.items()})
    plot(rows, res, pts, seen_wide)
    print(f"saved: {OUT}/table.json, raw.npz, fov_grid.png")


def plot(rows, res, pts, seen_wide):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f = [r["fov"] for r in rows]
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    ax[0, 0].plot(f, [r["noise_med"] for r in rows], label="noise median")
    ax[0, 0].plot(f, [r["noise_p95"] for r in rows], label="noise p95")
    ax[0, 0].plot(f, [r["ideal_med"] for r in rows], label="ideal median")
    ax[0, 0].set_ylabel("position error (mm)")
    ax[0, 1].plot(f, [r["fill"] for r in rows]); ax[0, 1].set_ylabel("arena fill of image (%)")
    ax[1, 0].plot(f, [r["cut"] for r in rows], "o-"); ax[1, 0].set_ylabel("points with ball cut / not seen")
    for a in (ax[0, 0], ax[0, 1], ax[1, 0]):
        a.axvline(CHOSEN, color="k", ls="--", lw=0.8); a.set_xlabel("FOV (deg)"); a.grid(alpha=0.3)
    ax[0, 0].legend()
    r = res[CHOSEN]
    sc = ax[1, 1].scatter(pts[:, 0], pts[:, 1], c=1000 * r["e_noise"], s=6, cmap="viridis", vmax=8)
    ax[1, 1].plot(*pts[~seen_wide].T, "rx", ms=3, label="hidden by arm")
    ax[1, 1].set_aspect("equal"); ax[1, 1].legend(loc="upper right", fontsize=8)
    ax[1, 1].set_title(f"noisy position error at {CHOSEN}° (mm)"); fig.colorbar(sc, ax=ax[1, 1])
    fig.tight_layout(); fig.savefig(OUT / "fov_grid.png", dpi=110)


if __name__ == "__main__":
    main()

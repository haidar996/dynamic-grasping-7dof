"""Step B check: 50 speeds x K=10 random launches, statistics, plots, reproducibility."""
import sys; from pathlib import Path
import numpy as np, mujoco
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
root = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(root))
from arena_sim import Arena
from launches import SPEEDS, sample_launches, run_launch

K = 10
model = mujoco.MjModel.from_xml_path(str(root / "scenes" / "arena.xml"))
arena = Arena(model)

def make(seed, k):
    rng = np.random.default_rng(seed)
    return [run_launch(model, arena, l) for l in sample_launches(rng, arena, k)]

runs = make(seed=0, k=K)
speed = [np.linalg.norm(r["vel"], axis=1) for r in runs]
v0 = np.array([np.linalg.norm(r["v0"]) for r in runs])
lim = arena.Lx - arena.radius
print("launches                 :", len(runs), "(%d speeds x %d)" % (len(SPEEDS), K))
print("max |pos| over all runs  :", max(np.abs(r["pos"]).max() for r in runs), " limit", lim)
gain = max(s.max() - v for s, v in zip(speed, v0))
print("max speed above launch   : %.2e m/s  (should be ~0: no energy gain)" % gain)
nb = np.array([len(r["bounces"]) for r in runs])
print("bounces per run          : mean %.1f, max %d" % (nb.mean(), nb.max()))
print("runs still moving at T   :", sum(s[-1] > 0 for s in speed), "of", len(runs))
print("mu range seen            : %.2f .. %.2f" % (min(r["mu"] for r in runs), max(r["mu"] for r in runs)))
print("E  range seen            : %.2f .. %.2f" % (min(r["e"] for r in runs), max(r["e"] for r in runs)))
again = make(seed=0, k=1)
same = all(np.array_equal(a["pos"], b["pos"]) for a, b in zip(runs[:1], again[:1]))
print("same seed -> identical   :", same, "(first launch)")

fig, ax = plt.subplots(1, 3, figsize=(16, 5))
for r in runs[::40]:
    ax[0].plot(*r["pos"].T, lw=0.8)
ax[0].set_aspect("equal"); ax[0].set_title("launches (every 40th)"); ax[0].grid()
ax[1].scatter(v0, nb, s=8); ax[1].set_xlabel("launch speed (m/s)")
ax[1].set_ylabel("wall bounces in 10 s"); ax[1].set_title("bounces vs speed"); ax[1].grid()
ax[2].scatter([r["mu"] for r in runs], [r["e"] for r in runs], s=8, c=v0)
ax[2].set_xlabel("mu (m/s^2)"); ax[2].set_ylabel("E"); ax[2].set_title("sampled (mu, E)"); ax[2].grid()
fig.savefig(root / "experiments" / "step_b_dataset.png", dpi=110)
print("saved plot")

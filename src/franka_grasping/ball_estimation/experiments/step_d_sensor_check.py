"""Step D check: (1) switches off = ideal, (2) depth-noise law, (3) dropout, (4) delay + frame rate."""
import sys; from pathlib import Path
import numpy as np, mujoco
root = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(root))
from arena_sim import Arena
from launches import Launch
from sensing.camera import DepthCamera
from sensing.sensor import Sensor, SensorParams, record_launch

model = mujoco.MjModel.from_xml_path(str(root / "scenes" / "arena.xml"))
arena, cam = Arena(model), DepthCamera(model)
launch = Launch(np.array([-0.5, -0.5]), np.array([2.4, 1.6]), 0.5, 0.8)

# 1. every switch off  ==  the ideal camera
class Probe(Sensor):
    ideal = []
    def observe(self, data):
        self.ideal.append(self.cam.measure(data))
        return super().observe(data)
probe = Probe(cam, SensorParams())
rec = record_launch(model, arena, launch, probe, T=3.0)
diff = np.abs(rec["xy"] - np.array(probe.ideal)).max()
print(f"1. switches off: max |sensor - ideal| = {diff:.1e} m over {len(rec['xy'])} frames")

# 2. depth noise: std should follow k * Z^2
k = 8.2e-4
print("2. depth noise (k = %.1e):  Z   measured std   expected k*Z^2   mean offset" % k)
for Z in (1.5, 3.0, 4.0):
    s = Sensor(cam, SensorParams(k_depth=k, seed=1))
    z = s.corrupt_depth(np.full(100000, Z))
    print(f"   {Z:4.1f} m    {z.std()*1000:6.2f} mm     {k*Z*Z*1000:6.2f} mm     {(z.mean()-Z)*1000:+.3f} mm")

# 3. dropout: rate, mean gap, reproducibility, and the two modes
print("3. dropout:  target rate/gap  ->  measured rate / mean gap (frames)")
for r, g in ((0.02, 1), (0.05, 6), (0.10, 15)):
    s = Sensor(cam, SensorParams(loss_rate=r, mean_gap=g, seed=3))
    seq = np.array([s._step_loss() for _ in range(200000)])
    edges = np.flatnonzero(np.diff(np.r_[0, seq.astype(int), 0]))
    gaps = edges[1::2] - edges[0::2]
    print(f"   {r:.2f} / {g:2d}  ->  {seq.mean():.3f} / {gaps.mean():.1f}")
a = Sensor(cam, SensorParams(loss_rate=0.05, mean_gap=6, seed=3))
b = Sensor(cam, SensorParams(loss_rate=0.05, mean_gap=6, seed=3, k_depth=k))   # other noise, same seed
same = all(a._step_loss() == b._step_loss() for _ in range(5000))
print("   same seed -> same gaps (even with different noise settings):", same)
for mode in ("flagged", "silent_hold"):
    rec = record_launch(model, arena, launch,
                        Sensor(cam, SensorParams(loss_rate=0.3, mean_gap=5, loss_mode=mode, seed=2)), T=3.0)
    lost = np.flatnonzero(rec["was_lost"])
    if mode == "flagged":
        print(f"   flagged: lost frames {len(lost)}, all invalid:", not rec["valid"][lost].any())
    else:
        held = all(np.array_equal(rec["xy"][i], rec["xy"][i - 1]) for i in lost if i > 0 and rec["valid"][i - 1])
        print(f"   silent_hold: lost frames {len(lost)}, all look valid:", rec["valid"][lost].all(),
              ", repeat the previous position:", held)

# 4. delay and frame rate
rec = record_launch(model, arena, launch, Sensor(cam, SensorParams(fps=60, delay=0.04)), T=3.0)
spacing = np.diff(rec["t_capture"])
print(f"4. frames in 3 s: {len(spacing)+1} (expected 180);  spacing mean {spacing.mean()*1000:.2f} ms"
      f" (ideal 16.67), max deviation {np.abs(spacing - 1/60).max()*1000:.2f} ms (physics step 2 ms)")
print("   arrival - capture = 0.04 s exactly:", np.allclose(rec["t_arrival"] - rec["t_capture"], 0.04))
cam.close()

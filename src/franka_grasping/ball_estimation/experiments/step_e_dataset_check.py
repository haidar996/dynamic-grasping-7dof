"""Step E check: raw recording + offline conditions (4 launches, 3 s each)."""
import sys, shutil; from pathlib import Path
import numpy as np, mujoco
root = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(root))
from arena_sim import Arena
from launches import sample_launches
from sensing.camera import DepthCamera
from sensing.sensor import Sensor, SensorParams, record_launch
from dataset import CONDITIONS, generate_raw, apply_condition, load_condition

model = mujoco.MjModel.from_xml_path(str(root / "scenes" / "arena.xml"))
arena, cam = Arena(model), DepthCamera(model, 3.0, 60.0)
launches = sample_launches(np.random.default_rng(0), arena, 1)[9::12][:4]   # 4 launches, different speeds
OUT = root / "outputs" / "_check_raw"
shutil.rmtree(OUT, ignore_errors=True)
generate_raw(model, arena, cam, launches, OUT, T=3.0)
files = sorted(OUT.glob("launch_*.npz"))
print(f"files: {len(files)}, sizes (kB): {[f.stat().st_size // 1024 for f in files]}")

# 1. ideal condition == direct sensor with all switches off
raw0 = np.load(files[0])
a = apply_condition(cam, raw0, CONDITIONS["ideal"], seed=0)
b = record_launch(model, arena, launches[0], Sensor(cam, SensorParams()), T=3.0)
print("1. ideal offline == direct sensor:", np.allclose(a["xy"], b["xy"], equal_nan=True),
      " frames", len(a["xy"]), len(b["xy"]))

# 2. pairing: same loss gaps across conditions that share loss parameters
sets = {n: load_condition(cam, OUT, CONDITIONS[n]) for n in ("dropout_flagged", "dropout_silent", "all")}
ok = all(np.array_equal(sets["dropout_flagged"][i]["was_lost"], sets[n][i]["was_lost"])
         for n in ("dropout_silent", "all") for i in range(len(files)))
print("2. same gaps in dropout_flagged / dropout_silent / all:", ok,
      " lost frames per launch:", [int(r["was_lost"].sum()) for r in sets["dropout_flagged"]])

# 3. determinism
x = apply_condition(cam, raw0, CONDITIONS["all"], seed=5)
y = apply_condition(cam, raw0, CONDITIONS["all"], seed=5)
print("3. deterministic:", all(np.array_equal(x[k], y[k], equal_nan=True) for k in ("xy", "valid", "was_lost")))

# 4. resume: a second run changes nothing
before = [f.stat().st_mtime_ns for f in files]
generate_raw(model, arena, cam, launches, OUT, T=3.0)
print("4. second generate_raw leaves files untouched:", before == [f.stat().st_mtime_ns for f in files])

# 5. camera mismatch is refused
cam.configure(2.0, 60.0)
try:
    apply_condition(cam, raw0, CONDITIONS["ideal"], seed=0); print("5. mismatch refused: False")
except AssertionError:
    print("5. mismatch refused: True")
cam.close()

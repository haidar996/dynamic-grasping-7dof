#!/usr/bin/env python3
"""P7: train the learned Kalman gain (estimators/knet.py) per sensor condition on the TRAIN launches, select on VAL.
Loss = launch-mean h0.5 RMS + 0.1 x launch-mean now_pos RMS (the forecast criterion of ekf-mu-h05).  Adam 1e-3, mini-batches of
32 launches, whole-launch BPTT, up to EPOCHS epochs, early stop on val h0.5 (patience 10).  c, E of the prediction model = the
train-fitted constants of ekf19b_fit for this physics (outputs/ekf_tuning_val[_mismatch].json, family ekf_fit).
Usage: [PHYSICS=mismatch] python3 experiments/step_2a_train_knet.py <cond[,cond]> [--seed S] [--quick] [--check]
Outputs: outputs/knet_<cond>[_mismatch][_s<seed>].pt (best val weights) + outputs/knet_train[_mismatch].json (history, merged).
--check: compare the torch physics with estimators.model on random states (no training)."""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "4")
import sys, json, time, math
from pathlib import Path
import numpy as np
import torch
import mujoco

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
import physics_cfg as phys
from arena_sim import Arena
from sensing.camera import Pinhole
from dataset import CONDITIONS, apply_condition
from estimators import knet
from estimators.model import propagate

SPLIT = json.loads((root / "split_500.json").read_text())
GEO = Arena(mujoco.MjModel.from_xml_path(str(root / "scenes" / "arena.xml"))).estimator_geometry()
cam = Pinhole(3.0, 49.0, 480, 480)
TUNED = json.loads(phys.out_path("ekf_tuning_val.json").read_text())["best"]["ekf_fit"]
EPOCHS, PATIENCE, BATCH, LR = 60, 10, 32, 1e-3
torch.set_num_threads(4)
torch.set_default_dtype(torch.float64)


def load(cond, ids):
    return knet.prepare_t([apply_condition(cam, np.load(phys.RAW / f"launch_{i:04d}.npz"), CONDITIONS[cond], 1000 + i) for i in ids])


def subset(d, idx):
    return {k: (v[idx] if v.dim() >= 2 and v.shape[0] == len(d["V"]) else v) for k, v in d.items()}


def check_physics(n=20000, seed=0):
    """Torch propagate vs the float model on random states near walls / the box (incl. steps that bounce)."""
    rng = np.random.default_rng(seed)
    c, E, L, obs = 0.8, 0.7, GEO["L"], GEO["obs"]
    S = np.column_stack([rng.uniform(-1.3, 1.3, n), rng.uniform(-1.3, 1.3, n), rng.normal(0, 3, n), rng.normal(0, 3, n)])
    S[: n // 4, 0] = obs[0] + rng.uniform(-obs[2] - 0.05, obs[2] + 0.05, n // 4)    # many near / inside the box
    S[: n // 4, 1] = obs[1] + rng.uniform(-obs[3] - 0.05, obs[3] + 0.05, n // 4)
    h = 0.005
    ref = np.array([propagate(*s, h, c, E, L, obs) for s in S])
    out, _ = knet.propagate_t(torch.as_tensor(S), h, c, E, L, obs)
    d = np.abs(out.numpy() - ref).max(1)
    print(f"physics check: {n} states, max |diff| {d.max():.2e}, states differing > 1e-9: {(d > 1e-9).sum()}")
    return d.max() < 1e-9


def train(cond, seed, quick):
    torch.manual_seed(seed); np.random.seed(seed)
    tr_ids = SPLIT["train"][::35] if quick else SPLIT["train"]
    va_ids = SPLIT["val"][::15] if quick else SPLIT["val"]
    t0 = time.time()
    tr, va = load(cond, tr_ids), load(cond, va_ids)
    c, E = TUNED[cond]["c"], TUNED[cond]["E"]
    print(f"[{cond} | {phys.PHYSICS} | seed {seed}] {len(tr_ids)} train / {len(va_ids)} val launches, c {c} E {E}, loaded in {time.time() - t0:.0f} s", flush=True)
    net = knet.GainNet()
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    with torch.no_grad():
        _, rh, _ = knet.run_batch(net, va, c, E, GEO["L"], GEO["obs"])
    best = dict(val_h05_mm=1000 * float(rh.mean()), epoch=0, state={k: v.clone() for k, v in net.state_dict().items()})
    print(f"  epoch  0: val h0.5 {best['val_h05_mm']:.1f} mm (fixed gain K0)", flush=True)
    hist, bad = [best["val_h05_mm"]], 0
    epochs = 2 if quick else EPOCHS
    B = len(tr_ids)
    for ep in range(1, epochs + 1):
        te = time.time(); perm = torch.randperm(B); losses = []
        for i in range(0, B, BATCH):
            idx = perm[i:i + BATCH]
            loss, _, _ = knet.run_batch(net, subset(tr, idx), c, E, GEO["L"], GEO["obs"])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); losses.append(float(loss))
        with torch.no_grad():
            _, rh, rn = knet.run_batch(net, va, c, E, GEO["L"], GEO["obs"])
        vh = 1000 * float(rh.mean()); hist.append(vh)
        improved = vh < best["val_h05_mm"] - 1e-3
        if improved:
            best = dict(val_h05_mm=vh, epoch=ep, state={k: v.clone() for k, v in net.state_dict().items()}); bad = 0
        else:
            bad += 1
        print(f"  epoch {ep:2d}: train loss {1000 * np.mean(losses):.1f} mm  val h0.5 {vh:.1f} mm  now {1000 * float(rn.mean()):.1f}"
              f"  ({time.time() - te:.0f} s){'  *' if improved else ''}", flush=True)
        if bad >= PATIENCE:
            print(f"  early stop: no val gain for {PATIENCE} epochs", flush=True); break
    sfx = phys.SUFFIX + (f"_s{seed}" if seed else "") + ("_quick" if quick else "")
    wpath = root / "outputs" / f"knet_{cond}{sfx}.pt"
    torch.save(best["state"], wpath)
    return dict(cond=cond, physics=phys.PHYSICS, seed=seed, c=c, E=E, best_epoch=best["epoch"], val_h05_mm=best["val_h05_mm"],
                history_val_h05_mm=hist, epochs_run=len(hist) - 1, minutes=(time.time() - t0) / 60, weights=str(wpath.name),
                n_params=sum(p.numel() for p in net.parameters()))


def main():
    if "--check" in sys.argv:
        ok = check_physics(); sys.exit(0 if ok else 1)
    quick = "--quick" in sys.argv
    seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 0
    argv = [a for a in sys.argv[1:] if not a.startswith("--") and not (sys.argv[sys.argv.index(a) - 1] == "--seed")]
    conds = argv[0].split(",")
    jpath = phys.out_path("knet_train.json", phys.SUFFIX + ("_quick" if quick else ""))
    out = json.loads(jpath.read_text()) if jpath.exists() else {}
    for cond in conds:
        r = train(cond, seed, quick)
        out[f"{cond}" + (f"_s{seed}" if seed else "")] = r
        jpath.write_text(json.dumps(out, indent=1))
        print(f"== {cond}: best epoch {r['best_epoch']}, val h0.5 {r['val_h05_mm']:.1f} mm, {r['minutes']:.1f} min, saved {r['weights']}\n", flush=True)


if __name__ == "__main__":
    main()

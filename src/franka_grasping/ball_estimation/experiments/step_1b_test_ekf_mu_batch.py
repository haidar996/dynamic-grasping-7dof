#!/usr/bin/env python3
"""Step 1.2b test: the batch ekf-mu (estimators/ekf_mu_batch.py) gives the same numbers as separate EKFMu + StaleGuard
runs evaluated with estimators.metrics.evaluate: per-launch NLL, number of updates, h0.5 forecast errors.
Several sensor conditions (dropout, stale frames, delay, noise) and random parameter sets; then the run time for the
350 train launches (what one likelihood evaluation costs during learning).
Usage: python3 experiments/step_1b_test_ekf_mu_batch.py"""
import sys, json, time
from pathlib import Path
import numpy as np
import mujoco

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from arena_sim import Arena
from sensing.camera import Pinhole
from dataset import CONDITIONS, apply_condition
from estimators.ekf_mu import EKFMu
from estimators.filters import StaleGuard
from estimators.metrics import evaluate
from estimators import ekf_mu_batch as eb

GEO = Arena(mujoco.MjModel.from_xml_path(str(root / "scenes" / "arena.xml"))).estimator_geometry()
cam = Pinhole(3.0, 49.0, 480, 480)
SPLIT = json.loads((root / "split_500.json").read_text())
RAW = root / "outputs" / "raw_60hz_h3_f49_sq480_with_arm"


def recs(cond, ids):
    return [apply_condition(cam, np.load(RAW / f"launch_{i:04d}.npz"), CONDITIONS[cond], 1000 + i) for i in ids]


def single(rec, p):
    f = StaleGuard(EKFMu(L=GEO["L"], obs=GEO["obs"], **p))
    for k in range(len(rec["t_capture"])):
        f.update(rec["t_capture"][k], rec["xy"][k], rec["valid"][k])
    nll, n = f.est.nll, f.est.n_upd
    return nll, n, evaluate(f, rec)["h0.5"]


def main():
    rng = np.random.default_rng(1)
    ids = SPLIT["train"][::12]                    # 30 launches over all speeds
    worst = dict(nll=0.0, h05=0.0); bad_n = 0
    for cond in ("ideal", "dropout_silent", "dropout_silent_noisy", "delay", "all_real", "all_plane"):
        rs = recs(cond, ids)
        data = eb.prepare(rs)
        for trial in range(2):
            p = dict(sigma=float(rng.uniform(0.003, 0.02)), q=float(10 ** rng.uniform(-2, 0.5)),
                     q_mu=float(10 ** rng.uniform(-5, -1)), mu0=float(rng.uniform(0.4, 1.2)), s_l=float(rng.uniform(0.2, 1)),
                     sigma_v0=float(rng.uniform(0.5, 4)), E=float(rng.uniform(0.5, 0.95)), sigma_E=0.13)
            nll_b, n_b, err_b = eb.run(data, p, GEO["L"], GEO["obs"], horizon=0.5)
            for b, r in enumerate(rs):
                nll_s, n_s, err_s = single(r, p)
                bad_n += int(n_s != n_b[b] or len(err_s) != len(err_b[b]))
                worst["nll"] = max(worst["nll"], abs(nll_s - nll_b[b]) / max(1.0, abs(nll_s)))
                if len(err_s) == len(err_b[b]) and len(err_s):
                    worst["h05"] = max(worst["h05"], float(np.max(np.abs(err_s - err_b[b]))))
        print(f"  {cond:22s} checked (2 parameter sets x {len(ids)} launches)", flush=True)
    ok = bad_n == 0 and worst["nll"] < 1e-8 and worst["h05"] < 1e-9
    print(f"[{'PASS' if ok else 'FAIL'}] count mismatches {bad_n}; max relative NLL difference {worst['nll']:.1e}; "
          f"max h0.5 difference {worst['h05']:.1e} m")
    rs = recs("all_real", SPLIT["train"])
    data = eb.prepare(rs)
    p = dict(sigma=0.012, q=0.1, q_mu=1e-3, mu0=0.85, s_l=0.5, sigma_v0=3.0, E=0.725, sigma_E=0.13)
    t0 = time.time(); eb.run(data, p, GEO["L"], GEO["obs"]); t_nll = time.time() - t0
    t0 = time.time(); eb.run(data, p, GEO["L"], GEO["obs"], horizon=0.5); t_h = time.time() - t0
    print(f"run time, {len(rs)} train launches: NLL only {t_nll:.1f} s, NLL + h0.5 forecasts {t_h:.1f} s")
    print("\nALL TESTS PASS" if ok else "\nFAILED")


if __name__ == "__main__":
    main()

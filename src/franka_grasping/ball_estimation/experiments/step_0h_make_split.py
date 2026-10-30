#!/usr/bin/env python3
"""Step 0.5: fixed train / val / test split of the 500 launches (350 / 75 / 75), stratified by launch speed.
Each of the 50 speeds has 10 launches: 7 -> train, the other 3 -> val/test as 2+1 or 1+2 (alternating over speeds).
Written once to split_500.json; every later phase loads it (train on train, tune on val, report on test only).
Usage: python3 experiments/step_0h_make_split.py [--force]"""
import json, sys
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parents[1]
RAW = root / "outputs" / "raw_60hz_h3_f49_sq480_with_arm"
OUT = root / "split_500.json"
SEED = 2026


def main():
    if OUT.exists() and "--force" not in sys.argv:
        sys.exit(f"{OUT.name} exists; it must stay fixed (use --force only if you really mean to change it)")
    files = sorted(RAW.glob("launch_????.npz"))
    assert len(files) == 500, len(files)
    info = []
    for i, f in enumerate(files):
        r = np.load(f)
        b = r["bounces"]
        info.append(dict(i=i, speed=round(float(np.hypot(*r["v0"])), 1), mu=float(r["mu"]), e=float(r["e"]),
                         arm=int((b[:, 3] == 1).sum()), wall=int((b[:, 3] == 0).sum()),
                         hidden=int(np.isnan(r["uvz"][:, 0]).sum())))
    rng = np.random.default_rng(SEED)
    split = {"train": [], "val": [], "test": []}
    for n, s in enumerate(sorted({d["speed"] for d in info})):
        idx = rng.permutation([d["i"] for d in info if d["speed"] == s])
        assert len(idx) == 10, (s, len(idx))
        nv = 2 if n % 2 == 0 else 1
        split["train"] += idx[:7].tolist(); split["val"] += idx[7:7 + nv].tolist(); split["test"] += idx[7 + nv:].tolist()
    for k in split:
        split[k] = sorted(int(i) for i in split[k])
    assert sorted(sum(split.values(), [])) == list(range(500))

    print(f"{'set':6s} {'n':>4s} {'speeds':>7s} {'speed mean':>10s} {'mu mean':>8s} {'e mean':>7s} {'arm-hit launches':>17s} "
          f"{'wall hits/launch':>17s} {'>100 hidden frames':>19s}")
    for k, ids in split.items():
        d = [info[i] for i in ids]
        print(f"{k:6s} {len(d):4d} {len({x['speed'] for x in d}):7d} {np.mean([x['speed'] for x in d]):10.2f} "
              f"{np.mean([x['mu'] for x in d]):8.2f} {np.mean([x['e'] for x in d]):7.3f} "
              f"{sum(x['arm'] > 0 for x in d):10d} ({100 * np.mean([x['arm'] > 0 for x in d]):4.1f}%) "
              f"{np.mean([x['wall'] for x in d]):17.2f} {sum(x['hidden'] > 100 for x in d):19d}")
    print("launch 303 (exact-corner arm hit) is in:", [k for k, v in split.items() if 303 in v][0])
    OUT.write_text(json.dumps(dict(seed=SEED, data=RAW.name, rule="per speed: 7 train, 2+1 / 1+2 val+test alternating",
                                   **split), indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Step A (research audit A1, "inverse crime"): the Phase-0 baselines on the MISMATCH-physics data
(outputs/raw_60hz_h3_f49_sq480_with_arm_mismatch: rolling resistance a0(1+k|v|), launch slip, floor tilt, wall/arm
restitution + wall friction; the estimators' model is the constant-deceleration + mirror-bounce model of the matched data).
Same 500 launches (same p0, v0) and same noise seeds (1000 + i) as the matched data -> every comparison is paired by launch.

Stage 1  hindsight best constants (replaces the oracle, which has no true mu / E on this data):
         per launch, grid mu 0.05-2.00 (step 0.05) x E 0.05-1.00 (step 0.05, + 0.725), condition ideal, criterion = that
         launch's h0.5 RMS.  Done twice: with the oracle family's val-tuned noise (filter ekf19b_oracle-none+stale) and
         with the ekf family's noise (filter of the mu-vs-E check, step_1a).  Ties (e.g. E for a launch without a bounce)
         -> pair closest to nominal (0.85, 0.725).  Chosen ON the evaluated launches = an upper bound for ANY constant-
         parameter filter of this family, not a method.  Picked once on ideal, reused in every condition.
Stage 2  all 11 conditions on VAL + TEST (dev): the 13 estimators of step_0d (oracle names = hindsight pair) with the
         MATCHED val-tuned (sigma, q) and train-fitted (c, E): nothing is re-tuned on mismatch data.
         mu-first check (as step_1a) on ideal + all_real: nominal / best mu only (E = 0.725) / best E only (mu = 0.85) /
         best mu & E, ekf-family noise.
Stage 3  paired tests: mismatch vs matched (same estimator, same launch, same seeds), hindsight gap (oracle vs ekf19b+stale)
         on both data sets, fitted vs midpoints, mu-first vs nominal.  Reported on VAL and on TEST (dev) separately.
Outputs: outputs/mismatch_hindsight.json, mismatch_baselines.json, mismatch_errors.pkl.gz, mismatch_paired.json
Usage:   python3 experiments/step_0m_mismatch_baselines.py [workers=3] [--quick N]   (quick: N val + N test, 2 conditions,
         outputs *_quick.*)"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import sys, time, json, gzip, pickle
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import mujoco

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from arena_sim import Arena
from sensing.camera import Pinhole
from dataset import CONDITIONS, apply_condition
from estimators.filters import make, EKF19, StaleGuard
from estimators.metrics import evaluate
from estimators.stats import summary, paired, per_launch

RAW = root / "outputs" / "raw_60hz_h3_f49_sq480_with_arm_mismatch"
GEO = Arena(mujoco.MjModel.from_xml_path(str(root / "scenes" / "arena.xml"))).estimator_geometry()
SPLIT = json.loads((root / "split_500.json").read_text())
TUNED = json.loads((root / "outputs" / "ekf_tuning_val.json").read_text())["best"]
MATCHED = root / "outputs" / "baselines_v3_errors.pkl.gz"
cam = Pinhole(3.0, 49.0, 480, 480)

# 2026-11-07: widened (first run: 0.05-1.50 x 0.30-1.00; 15/147 oracle picks at E = 0.30, 5 mu-only picks at mu = 1.5).
# E = 1.0 is the physical limit (no energy gain at a wall) -> not counted as an edge.  Cells of the first run are
# reused from outputs/mismatch_hindsight_narrowgrid.json (same launches, same filters, deterministic).
MU_GRID = np.round(np.arange(0.05, 2.0001, 0.05), 3)
E_GRID = np.round(np.sort(np.append(np.arange(0.05, 1.0001, 0.05), 0.725)), 3)
OLD_FILE = root / "outputs" / "mismatch_hindsight_narrowgrid.json"
OLD = {}                                   # (launch, family) -> {(mu, E): value}; filled in main, inherited by workers
NOM = (0.85, 0.725)
ESTIMATORS = ["fd", "fd+stale", "cvkf", "ekf19-none", "ekf19-hybrid",
              "ekf19b-none", "ekf19b-none+stale", "ekf19b-hybrid", "ekf19b_fit-none+stale",
              "ekf19_oracle-none", "ekf19_oracle-hybrid", "ekf19b_oracle-none",
              "ekf19b_oracle-none+stale"]
MU_FIRST = ["nominal mu & E", "best mu only", "best E only", "best mu & E"]
MU_FIRST_CONDS = ("ideal", "all_real")
KEYS = ["now_pos", "now_vel", "h0.2", "h0.5", "h1.0"]
EKF, FIT, ORA = "ekf19b-none+stale", "ekf19b_fit-none+stale", "ekf19b_oracle-none+stale"


def family(name):
    if name.startswith("fd"):
        return None
    if name.startswith("cvkf"):
        return "cvkf"
    if "_oracle" in name:
        return "oracle"
    if name.startswith("ekf19b_fit"):
        return "ekf_fit"
    return "ekf"


def noise_kw(name, cond):
    fam = family(name)
    return {} if fam is None else {k: v for k, v in TUNED[fam][cond].items() if k in ("sigma", "q", "c", "E")}


def load(i, cond):
    return apply_condition(cam, np.load(RAW / f"launch_{i:04d}.npz"), CONDITIONS[cond], 1000 + i)


def mu_first_filter(c, E, cond):
    """Exactly the filter of step_1a: EKF19, policy none, sigma_E 0.13, ekf-family noise, stale guard."""
    t = TUNED["ekf"][cond]
    return StaleGuard(EKF19(c=c, E=E, sigma=t["sigma"], q=t["q"], policy="none", sigma_E=0.13, **GEO))


# ---------------- stage 1: hindsight grid ----------------
def grid_task(i):
    rec = load(i, "ideal")
    t = TUNED["oracle"]["ideal"]
    G = {"oracle": np.full((len(MU_GRID), len(E_GRID)), np.nan), "ekf": np.full((len(MU_GRID), len(E_GRID)), np.nan)}
    oo, oe = OLD.get((i, "oracle"), {}), OLD.get((i, "ekf"), {})
    for a, c in enumerate(MU_GRID):
        for b, E in enumerate(E_GRID):
            key = (float(c), float(E))
            if key in oo and key in oe:
                G["oracle"][a, b], G["ekf"][a, b] = oo[key], oe[key]
                continue
            r2 = dict(rec, mu=float(c), e=float(E))
            o = evaluate(make(ORA, r2, sigma=t["sigma"], q=t["q"], **GEO), r2)["h0.5"]
            e = evaluate(mu_first_filter(float(c), float(E), "ideal"), rec)["h0.5"]
            G["oracle"][a, b] = np.sqrt(np.mean(o ** 2)) if len(o) else np.nan
            G["ekf"][a, b] = np.sqrt(np.mean(e ** 2)) if len(e) else np.nan
    return i, G


def pick(score, mus, Es):
    """argmin over candidate (mu, E) pairs; ties (within 1 µm) -> closest to nominal (scaled by grid span)."""
    s = np.asarray(score, float)
    if np.all(np.isnan(s)):
        return dict(mu=NOM[0], E=NOM[1], h05=None, no_frames=True, edge=False)
    best = np.nanmin(s)
    cand = np.where(s <= best + 1e-6)[0]
    d = [((mus[k] - NOM[0]) / 1.45) ** 2 + ((Es[k] - NOM[1]) / 0.7) ** 2 for k in cand]
    k = int(cand[int(np.argmin(d))])
    tie = len(cand) > 1
    edge = (not tie) and (mus[k] in (MU_GRID[0], MU_GRID[-1]) or Es[k] == E_GRID[0])
    return dict(mu=float(mus[k]), E=float(Es[k]), h05=float(s[k]), n_ties=int(len(cand)), no_frames=False,
                edge=bool(edge))


def choose(G):
    M, Ee = np.meshgrid(MU_GRID, E_GRID, indexing="ij")
    ib = int(np.where(E_GRID == NOM[1])[0][0]); ia = int(np.where(MU_GRID == NOM[0])[0][0])
    return {"oracle_pair": pick(G["oracle"].ravel(), M.ravel(), Ee.ravel()),
            "ekf_pair": pick(G["ekf"].ravel(), M.ravel(), Ee.ravel()),
            "ekf_mu_only": pick(G["ekf"][:, ib], MU_GRID, np.full(len(MU_GRID), NOM[1])),
            "ekf_E_only": pick(G["ekf"][ia, :], np.full(len(E_GRID), NOM[0]), E_GRID),
            "ekf_nominal_h05": None if np.isnan(G["ekf"][ia, ib]) else float(G["ekf"][ia, ib])}


# ---------------- stage 2: all conditions ----------------
HS = {}   # filled in main, inherited by the workers (fork)


def eval_task(a):
    i, cond = a
    rec = load(i, cond)
    h = HS[i]
    r2 = dict(rec, mu=h["oracle_pair"]["mu"], e=h["oracle_pair"]["E"])
    out = {name: evaluate(make(name, r2, **noise_kw(name, cond), **GEO), r2) for name in ESTIMATORS}
    if cond in MU_FIRST_CONDS:
        V = {"nominal mu & E": NOM, "best mu only": (h["ekf_mu_only"]["mu"], NOM[1]),
             "best E only": (NOM[0], h["ekf_E_only"]["E"]), "best mu & E": (h["ekf_pair"]["mu"], h["ekf_pair"]["E"])}
        out.update({k: evaluate(mu_first_filter(c, E, cond), rec) for k, (c, E) in V.items()})
    return i, out


def table(res, idx, title, names):
    print(f"\n  [{title}: {len(idx)} launches]  per-launch RMS -> launch-mean [95% CI] | launch-median | pooled-frame RMS"
          f"   (mm; now_vel m/s)")
    print(f"  {'Estimator':26s} " + " ".join(f"{k:>30s}" for k in ("now_pos", "now_vel", "h0.5", "h1.0")))
    out = {}
    for name in names:
        items = [res[i][name] for i in idx]
        out[name] = {k: summary(items, k) for k in KEYS}

        def cell(k):
            s, f = out[name][k], (1 if k == "now_vel" else 1000)
            if "mean" not in s:
                return "-"
            d = 2 if k == "now_vel" else 0
            return (f"{s['mean'] * f:.{d}f} [{s['mean_ci'][0] * f:.{d}f},{s['mean_ci'][1] * f:.{d}f}]"
                    f" {s['median'] * f:.{d}f} {s['pooled_rms'] * f:.{d}f}")
        print(f"  {name:26s} " + " ".join(f"{cell(k):>30s}" for k in ("now_pos", "now_vel", "h0.5", "h1.0")))
    return out


def fmt(r):
    if "mean_diff" not in r:
        return f"(n {r['n']})"
    return (f"{1000 * r['mean_diff']:+7.1f} [{1000 * r['mean_diff_ci'][0]:+6.1f},{1000 * r['mean_diff_ci'][1]:+6.1f}]"
            f"  p {r['wilcoxon_p']:.1e}  {100 * r['a_better_frac']:3.0f}%  (n {r['n']})")


def main():
    argv = sys.argv[1:]
    quick = 0
    if "--quick" in argv:
        k = argv.index("--quick"); quick = int(argv[k + 1]); del argv[k:k + 2]
    workers = int(argv[0]) if argv else 3
    val, test = SPLIT["val"], SPLIT["test"]
    if quick:
        val, test = sorted(val)[:quick], sorted(test)[:quick]
    idx = sorted(val + test)
    conds = ["ideal", "all_real"] if quick else list(CONDITIONS)
    sfx = "_quick" if quick else ""
    print(f"Step A: baselines on mismatch physics | data {RAW.name} | {len(val)} val + {len(test)} test (dev) launches"
          f" | {len(conds)} conditions | workers {workers}")
    print(f"Hindsight grid: mu {MU_GRID[0]}-{MU_GRID[-1]} ({len(MU_GRID)}) x E {E_GRID[0]}-{E_GRID[-1]} ({len(E_GRID)})"
          f" = {len(MU_GRID) * len(E_GRID)} pairs x 2 noise families, condition ideal, criterion per-launch h0.5 RMS")
    r0 = np.load(RAW / f"launch_{idx[0]:04d}.npz")
    assert "phys_a0" in r0.files and np.isnan(r0["mu"]), "not the mismatch data set"

    # stage 1
    if OLD_FILE.exists() and not quick:
        o = json.loads(OLD_FILE.read_text())
        for li, g in o["grids"].items():
            for fam, rows in g.items():
                OLD[(int(li), fam)] = {(float(m), float(e)): (np.nan if rows[a][b] is None else rows[a][b])
                                       for a, m in enumerate(o["mu_grid"]) for b, e in enumerate(o["E_grid"])}
        print(f"reusing {len(o['mu_grid']) * len(o['E_grid'])} cells per launch from {OLD_FILE.name}")
    t0 = time.time()
    with Pool(workers) as pool:
        grids = dict(pool.map(grid_task, idx, chunksize=1))
    for i in idx:
        HS[i] = choose(grids[i])
        r = np.load(RAW / f"launch_{i:04d}.npz")
        HS[i]["phys"] = {k[5:]: np.asarray(r[k]).tolist() for k in r.files if k.startswith("phys_")}
    print(f"\n[stage 1] hindsight grid done in {time.time() - t0:.0f} s")
    for key in ("oracle_pair", "ekf_pair", "ekf_mu_only", "ekf_E_only"):
        P = [HS[i][key] for i in idx if not HS[i][key]["no_frames"]]
        mus, Es = np.array([p["mu"] for p in P]), np.array([p["E"] for p in P])
        print(f"  {key:12s} n {len(P)}  mu median {np.median(mus):.2f} [{np.percentile(mus, 5):.2f}, "
              f"{np.percentile(mus, 95):.2f}]  E median {np.median(Es):.3f} [{np.percentile(Es, 5):.3f}, "
              f"{np.percentile(Es, 95):.3f}]  on grid edge: {sum(p['edge'] for p in P)}  ties: "
              f"{sum(p['n_ties'] > 1 for p in P)}  no frames: {sum(HS[i][key]['no_frames'] for i in idx)}")
    def corr(key, fld, phys):
        ok = [i for i in idx if not HS[i][key]["no_frames"] and HS[i][key]["n_ties"] == 1]
        x = np.array([HS[i]["phys"][phys] for i in ok]); y = np.array([HS[i][key][fld] for i in ok])
        return (f"{np.corrcoef(x, y)[0, 1]:.2f} (n {len(ok)})" if len(ok) > 2 and x.std() > 0 and y.std() > 0
                else f"n/a (n {len(ok)})")
    print(f"  sanity (launches with frames, no ties): corr(best mu only, true a0) = {corr('ekf_mu_only', 'mu', 'a0')};"
          f" corr(best E only, true e_wall) = {corr('ekf_E_only', 'E', 'e_wall')}")
    json_out = dict(mu_grid=MU_GRID.tolist(), E_grid=E_GRID.tolist(), nominal=NOM,
                    launches={str(i): HS[i] for i in idx},
                    grids={str(i): {k: np.where(np.isnan(g), None, g).tolist() for k, g in grids[i].items()} for i in idx})
    (root / "outputs" / f"mismatch_hindsight{sfx}.json").write_text(json.dumps(json_out))

    # stage 2
    summary_out, errors = {}, {}
    with Pool(workers) as pool:
        for cond in conds:
            t0 = time.time()
            res = dict(pool.map(eval_task, [(i, cond) for i in idx], chunksize=1))
            errors[cond] = res
            names = ESTIMATORS + (MU_FIRST if cond in MU_FIRST_CONDS else [])
            print(f"\n{'=' * 150}\n[stage 2] Condition: {cond} | MATCHED val-tuned noise, oracle = hindsight pair | "
                  f"{time.time() - t0:.0f} s\n{'=' * 150}")
            summary_out[cond] = dict(test=table(res, test, "TEST (dev)", names), val=table(res, val, "VAL", names))
            print(flush=True)
    (root / "outputs" / f"mismatch_baselines{sfx}.json").write_text(json.dumps(summary_out, indent=1))
    with gzip.open(root / "outputs" / f"mismatch_errors{sfx}.pkl.gz", "wb") as f:
        pickle.dump(dict(split=dict(val=val, test=test), estimators=ESTIMATORS, mu_first=MU_FIRST, errors=errors), f)

    # stage 3
    M = pickle.load(gzip.open(MATCHED, "rb"))["errors"]
    out = {}
    for split, ids in (("val", val), ("test", test)):
        print(f"\n===== [stage 3] {split.upper()}{' (dev)' if split == 'test' else ''}: {len(ids)} launches; "
              f"mean diff A-B in mm [95% CI], Wilcoxon p, A better in % of launches =====")
        for key in ("h0.5", "now_pos"):
            print(f"\n  -- {key}")
            for cond in conds:
                X = errors[cond]
                comps = [(f"{cond}: mismatch vs matched, {n}", [X[i][n] for i in ids], [M[cond][i][n] for i in ids])
                         for n in ("fd", "cvkf", EKF, FIT, ORA)]
                comps += [(f"{cond}: hindsight gap MISMATCH (oracle - ekf19b+stale)",
                           [X[i][ORA] for i in ids], [X[i][EKF] for i in ids]),
                          (f"{cond}: true-constant gap MATCHED (oracle - ekf19b+stale)",
                           [M[cond][i][ORA] for i in ids], [M[cond][i][EKF] for i in ids]),
                          (f"{cond}: MISMATCH ekf19b_fit vs ekf19b", [X[i][FIT] for i in ids], [X[i][EKF] for i in ids]),
                          (f"{cond}: MISMATCH ekf19b-hybrid vs ekf19b+stale",
                           [X[i]["ekf19b-hybrid"] for i in ids], [X[i][EKF] for i in ids])]
                if cond in MU_FIRST_CONDS:
                    comps += [(f"{cond}: MISMATCH {k} vs nominal", [X[i][k] for i in ids],
                               [X[i]["nominal mu & E"] for i in ids]) for k in MU_FIRST[1:]]
                    comps += [(f"{cond}: MISMATCH best mu only vs best E only", [X[i]["best mu only"] for i in ids],
                               [X[i]["best E only"] for i in ids])]
                for label, A, B in comps:
                    r = paired(A, B, key)
                    out.setdefault(split, {}).setdefault(key, {})[label] = r
                    print(f"  {label:72s} {fmt(r)}")
    (root / "outputs" / f"mismatch_paired{sfx}.json").write_text(json.dumps(out, indent=1))
    print(f"\nSaved: mismatch_hindsight{sfx}.json, mismatch_baselines{sfx}.json, mismatch_errors{sfx}.pkl.gz,"
          f" mismatch_paired{sfx}.json")


if __name__ == "__main__":
    main()

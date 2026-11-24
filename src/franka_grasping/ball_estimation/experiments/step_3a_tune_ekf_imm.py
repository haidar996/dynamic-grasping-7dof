#!/usr/bin/env python3
"""B3 tuning of ekf-imm (design docs/B3_imm_ekf_design.md section 7).  VAL split only, never test.
Objective = launch-mean h0.5 RMS (mm), per-launch RMS over moving frames (every 6th, warm-up 0.1 s) = estimators.metrics.evaluate,
exactly the step_1c/1d criterion.  Fixed: sigma, E from outputs/ekf_tuning_val[_mismatch].json "ekf_fit" entry; sigma_v0=3, sigma_E=0.13, m_min=1e-4.
Tuned (7): mu_s, r (mu_r = r*mu_s, r in [0.02,1]), q_s, q_r, a, b, p_s0.   --shared-q: q_s = q_r (6 tuned numbers: mu_s, r, q, a, b, p_s0).  Gradient-free: stage 1 Latin hypercube (120 pts, seed 0), stage 2 (40 evals) = 8 rounds x the 5
best stage-1 points, one greedy coordinate move per start per round (coordinate = (round+start) mod d, sign from rng(seed, idx), step 0.1 in unit cube).
Deterministic given the history, so the evaluation log (saved after EVERY evaluation) makes the run resumable: done indices are skipped.
Files per (cond, physics[, sharedq]):  outputs/ekf_imm_tune_<cond>[_mismatch][_sharedq].json (log), .log (progress),
and when all evals are done  outputs/ekf_imm_<cond>[_mismatch].json  (shared-q: outputs/ekf_imm_sharedq_<cond>[_mismatch].json)
with {"params": {...}} as read by step_3b_eval_ekf_imm.py.
Usage: python3 experiments/step_3a_tune_ekf_imm.py [conds=ideal,noise,all_real] --physics matched|mismatch [--workers N<=4] [--shared-q]
       [--max-evals N] [--quick] [--dry-run] [--nice 10] [--seed 0]
(mismatch = ARM retrain tuning: matched-transfer arm needs no tuning.)"""
import os, sys, json, time, argparse
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("conds", nargs="?", default="ideal,noise,all_real")
ap.add_argument("--physics", choices=("matched", "mismatch"), default="matched")
ap.add_argument("--workers", type=int, default=1)
ap.add_argument("--shared-q", action="store_true", help="ablation: tie q_r = q_s (6 tuned parameters instead of 7); files get _sharedq suffix")
ap.add_argument("--max-evals", type=int, default=None, help="stop after this many NEW evaluations (per invocation, over all sets)")
ap.add_argument("--quick", action="store_true", help="val[::15] only (plumbing), files get _quick suffix")
ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--nice", type=int, default=10)
ap.add_argument("--seed", type=int, default=0)
ARGS = ap.parse_args()
assert 1 <= ARGS.workers <= 4, "workers must be 1..4"
os.environ["PHYSICS"] = ARGS.physics
if ARGS.physics == "mismatch":
    os.environ["ARM"] = "retrain"                       # step_1e then reads the mismatch-val tuning (sigma, E, c)
if not ARGS.dry_run:
    try:
        os.nice(ARGS.nice)
    except OSError:
        pass

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root)); sys.path.insert(0, str(root / "experiments"))
import physics_cfg as phys
OUT = root / "outputs"
SFX = ("_mismatch" if phys.MISMATCH else "") + ("_sharedq" if ARGS.shared_q else "") + ("_quick" if ARGS.quick else "")
NSTAGE1, NROUND, NSTART = 120, 8, 5
NTOT = NSTAGE1 + NROUND * NSTART
SIGMA_V0, SIGMA_E = 3.0, 0.13
NAMES = ["mu_s", "r", "q_s", "q_r", "a", "b", "p_s0"]
# (lo, hi, log-scale).  Ranges widened 2026-11-20 after the search-range audit (evidence, all from the val/mismatch data):
#  mu_s: 38% of mismatch launches have sliding decel mu_s*g > 3.0 (observed 1.48-3.92 m/s2 -> mu_s up to ~5); c_fit = 0.75 on matched.
#  r = mu_r/mu_s: mismatch a0/(mu_s g) 5-95% = 0.04-0.36, min 0.026 -> lo 0.02 (linear scale kept; hi 1 keeps mu_r <= mu_s).
#  q_s, q_r: chosen q were 0.03 (matched noise), 0.003/0.01 (mismatch), 20 (mismatch ideal) -> [0.001, 100] log.
#  a: implied 1/slide-time median ~10, 95% ~210 -> hi 300.   b, p_s0: audit found them adequate, unchanged.
BOUNDS = {"mu_s": (0.2, 5.0, True), "r": (0.02, 1.0, False), "q_s": (0.001, 100.0, True), "q_r": (0.001, 100.0, True),
          "a": (0.5, 300.0, True), "b": (0.005, 1.0, True), "p_s0": (0.5, 0.99, False)}
DIMS = [n for n in NAMES if not (ARGS.shared_q and n == "q_r")]


def tune_file(cond):
    return OUT / f"ekf_imm_tune_{cond}{SFX}.json"


def final_file(cond):
    return OUT / (f"ekf_imm_sharedq_{cond}.json" if ARGS.shared_q else f"ekf_imm_{cond}.json") if not phys.MISMATCH else \
        OUT / (f"ekf_imm_sharedq_{cond}_mismatch.json" if ARGS.shared_q else f"ekf_imm_{cond}_mismatch.json")


def to_params(u):
    p = {}
    for n, x in zip(DIMS, u):
        lo, hi, lg = BOUNDS[n]
        p[n] = float(np.exp(np.log(lo) + x * (np.log(hi) - np.log(lo))) if lg else lo + x * (hi - lo))
    if ARGS.shared_q:
        p["q_r"] = p["q_s"]
    p["mu_r"] = p["mu_s"] * p.pop("r")
    return p


def lhs(n, d, seed):
    rng = np.random.default_rng(seed)
    return np.stack([(rng.permutation(n) + rng.random(n)) / n for _ in range(d)], axis=1)


def propose(idx, hist, u1):
    """Deterministic parameter vector (unit cube) for evaluation index idx given completed history {idx: value}."""
    if idx < NSTAGE1:
        return u1[idx]
    rnd, j = divmod(idx - NSTAGE1, NSTART)
    starts = sorted(range(NSTAGE1), key=lambda i: hist[i])[:NSTART]
    best_i, best_v = starts[j], hist[starts[j]]
    cur = u1[best_i]
    for r in range(rnd):                                # replay accepted moves of this start
        ci = NSTAGE1 + r * NSTART + j
        if hist[ci] < best_v:
            best_v, cur = hist[ci], CAND[ci]
    d = len(DIMS)
    coord = (rnd + j) % d
    sign = 1.0 if np.random.default_rng([ARGS.seed, idx]).random() < 0.5 else -1.0
    new = cur.copy()
    new[coord] = float(np.clip(new[coord] + sign * 0.1, 0.0, 1.0))
    return new


CAND = {}                                               # idx -> unit vector actually evaluated (filled from the log)

# ---- evaluation (worker side) -------------------------------------------------------------------------------------
_W = {}


def _init(cond):
    import step_1e_eval_ekf_mu as S
    from estimators.filters import StaleGuard
    from estimators.ekf_imm import EKFIMM
    from estimators.metrics import evaluate
    from dataset import CONDITIONS, apply_condition
    t = S.TUNED["ekf_fit"][cond]
    val = S.SPLIT["val"][::15] if ARGS.quick else S.SPLIT["val"]
    recs = [apply_condition(S.cam, np.load(S.RAW / f"launch_{i:04d}.npz"), CONDITIONS[cond], 1000 + i) for i in val]
    _W.update(S=S, SG=StaleGuard, K=EKFIMM, ev=evaluate, recs=recs, geo=S.GEO, fixed=dict(sigma=t["sigma"], E=t["E"]))


def objective(p):
    """launch-mean h0.5 RMS in mm over the val launches."""
    kw = dict(_W["fixed"], **p, sigma_v0=SIGMA_V0, sigma_E=SIGMA_E)
    vals = []
    for rec in _W["recs"]:
        e = _W["ev"](_W["SG"](_W["K"](**kw, **_W["geo"])), rec)["h0.5"]
        if len(e):
            vals.append(float(np.sqrt(np.mean(e ** 2))))
    return 1000.0 * float(np.mean(vals))


def _job(args):
    idx, p = args
    t0 = time.time()
    return idx, objective(p), time.time() - t0


def _pool_init(cond):
    os.nice(ARGS.nice) if ARGS.nice else None
    _init(cond)


# ---- driver ---------------------------------------------------------------------------------------------------------
def cfit(cond):
    f = phys.out_path("ekf_tuning_val.json", "_mismatch" if phys.MISMATCH else "")
    return json.loads(f.read_text())["best"]["ekf_fit"][cond]


def run_set(cond, budget):
    """Returns number of new evaluations done."""
    u1 = lhs(NSTAGE1, len(DIMS), ARGS.seed)
    tf, lf = tune_file(cond), OUT / f"ekf_imm_tune_{cond}{SFX}.log"
    log = json.loads(tf.read_text()) if tf.exists() else dict(cond=cond, physics=ARGS.physics, shared_q=ARGS.shared_q, quick=ARGS.quick,
                                                               seed=ARGS.seed, dims=DIMS, bounds=BOUNDS, evals={})
    ev = log["evals"]
    CAND.clear()
    hist = {int(k): v["val_h05_mm"] for k, v in ev.items()}
    for k, v in ev.items():
        CAND[int(k)] = np.array(v["u"])

    def say(s):
        line = f"[{time.strftime('%H:%M:%S')}] {cond}/{phys.PHYSICS}{'/sharedq' if ARGS.shared_q else ''}: {s}"
        print(line, flush=True)
        with open(lf, "a") as f:
            f.write(line + "\n")

    done_new = 0
    pool = None
    if ARGS.workers > 1:
        from multiprocessing import Pool
        pool = Pool(ARGS.workers, initializer=_pool_init, initargs=(cond,))
    else:
        _init(cond)
    say(f"start: {len(hist)}/{NTOT} done, workers={ARGS.workers}")
    try:
        # stage 1: all indices independent; stage 2: round by round (5 independent per round)
        groups = [list(range(i, min(i + max(ARGS.workers, 1), NSTAGE1))) for i in range(0, NSTAGE1, max(ARGS.workers, 1))]
        groups += [list(range(NSTAGE1 + r * NSTART, NSTAGE1 + (r + 1) * NSTART)) for r in range(NROUND)]
        for g in groups:
            todo = [i for i in g if i not in hist]
            if budget is not None:
                todo = todo[:max(0, budget - done_new)]
            if not todo:
                if budget is not None and done_new >= budget:
                    break
                continue
            if todo[0] >= NSTAGE1 and any(i not in hist for i in range(NSTAGE1)):
                break
            jobs = []
            for i in todo:
                u = propose(i, hist, u1)
                CAND[i] = u
                jobs.append((i, to_params(u)))
            it = pool.imap_unordered(_job, jobs) if pool else map(_job_local, jobs)
            for i, val, dt in it:
                hist[i] = val
                ev[str(i)] = dict(u=[float(x) for x in CAND[i]], params=to_params(CAND[i]), val_h05_mm=val, secs=dt)
                tf.write_text(json.dumps(log, indent=1))            # save after every evaluation
                done_new += 1
                best = min(hist.values())
                say(f"eval {i} ({len(hist)}/{NTOT}) val h0.5 = {val:.2f} mm ({dt:.0f} s), best so far {best:.2f} mm")
            if budget is not None and done_new >= budget:
                break
    finally:
        if pool:
            pool.close(); pool.join()
    if len(hist) == NTOT:
        order = sorted(hist, key=lambda i: hist[i])
        b = order[0]
        p = to_params(CAND[b])
        t = cfit(cond)
        p.update(sigma=t["sigma"], E=t["E"], sigma_v0=SIGMA_V0, sigma_E=SIGMA_E)
        spread = hist[order[4]] - hist[order[0]]
        res = dict(params=p, val_h05_mm=hist[b], fifth_best_val_h05_mm=hist[order[4]], spread_5th_mm=spread,
                   flat_optimum=bool(spread < 1.0), n_evals=NTOT, seed=ARGS.seed, tuned_on="val", physics=ARGS.physics,
                   shared_q=ARGS.shared_q, quick=ARGS.quick, tuned=time.strftime("%Y-%m-%d %H:%M"))
        ff = final_file(cond)
        if ARGS.quick:
            ff = ff.with_name(ff.stem + "_quick.json")
        ff.write_text(json.dumps(res, indent=1))
        say(f"COMPLETE best val h0.5 {hist[b]:.2f} mm, 5th best {hist[order[4]]:.2f} mm (spread {spread:.2f}, "
            f"{'flat/stable' if spread < 1.0 else 'not flat'}) -> {ff.name}")
    return done_new


def _job_local(a):
    return _job(a)


def main():
    conds = ARGS.conds.split(",")
    if ARGS.dry_run:
        print(f"DRY RUN (nothing evaluated).  physics={ARGS.physics} shared_q={ARGS.shared_q} quick={ARGS.quick} workers={ARGS.workers} seed={ARGS.seed}")
        print(f"search dims ({len(DIMS)}): {DIMS}  [mu_r = r*mu_s{'; q_r = q_s' if ARGS.shared_q else ''}]")
        for n in DIMS:
            lo, hi, lg = BOUNDS[n]
            print(f"  {n:5s} [{lo}, {hi}] {'log' if lg else 'linear'}")
        print(f"evaluations per set: stage 1 = {NSTAGE1} LHS, stage 2 = {NROUND} rounds x {NSTART} starts = {NROUND * NSTART}; total {NTOT}")
        print(f"objective: val launch-mean h0.5 RMS (mm){' on val[::15]' if ARGS.quick else ''}; fixed sigma, E from "
              f"{phys.out_path('ekf_tuning_val.json', '_mismatch' if phys.MISMATCH else '').name} ekf_fit; sigma_v0={SIGMA_V0}, sigma_E={SIGMA_E}")
        for c in conds:
            try:
                t = cfit(c)
                fx = f"sigma={t['sigma']} E={t['E']} c_fit={t['c']}"
            except Exception as e:
                fx = f"(tuning file entry unreadable: {e})"
            print(f"set {c}: {fx}\n   log   {tune_file(c)}\n   final {final_file(c).with_name(final_file(c).stem + ('_quick' if ARGS.quick else '') + '.json')}")
        u = lhs(NSTAGE1, len(DIMS), ARGS.seed)
        print("first LHS point ->", {k: round(v, 4) for k, v in to_params(u[0]).items()})
        return
    budget = ARGS.max_evals
    for c in conds:
        n = run_set(c, budget)
        if budget is not None:
            budget -= n
            if budget <= 0:
                break


if __name__ == "__main__":
    main()

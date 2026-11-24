#!/usr/bin/env python3
"""B3 eval: ekf-imm (label ekfimm) and ekf-2ph (ekf2ph) next to the step_1e references, same interfaces / launches / seeds.
Reuses step_1e_eval_ekf_mu.py unchanged (imported as a module: estimators(), task pieces, diagnostics, report), so
PHYSICS=mismatch ARM=transfer|retrain behave exactly as there.  Parameter files (written by the tuning script, not yet existing):
  outputs/ekf_imm_<cond>[_mismatch].json   {"params": {mu_s, mu_r, q_s, q_r, a, b, p_s0, sigma, E, sigma_v0, sigma_E}}  (ekfimm)
  outputs/ekf_2ph_<cond>[_mismatch].json   {"params": {mu_s, mu_r, q, rule, v_sw|t_sw, sigma, E, ...}}                    (ekf2ph)
--smoke-defaults: when a file is missing use UNTUNED defaults (mu_s = 1.3 c_fit, mu_r = 0.5 c_fit, q = fit q, a=2, b=0.05,
p_s0=0.9; 2ph: speed rule v_sw=0.3); numbers are a speed / plumbing check only, never a result.
Reference set kept small: ekf19b_fit, ekfmu_grid, ekfmu_h05 (+ hindsight on mismatch).  Output: outputs/ekf_imm_eval*.json (+ _errors.pkl.gz),
separate from the ekf_mu_eval files.   Usage: python3 experiments/step_3b_eval_ekf_imm.py [conds] [workers=1] [--quick] [--smoke-defaults] [--nlaunch N]"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import sys, json, gzip, pickle, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root)); sys.path.insert(0, str(root / "experiments"))
import step_1e_eval_ekf_mu as S
from estimators.filters import StaleGuard
from estimators.ekf_imm import EKFIMM, EKF2PH
from estimators.metrics import evaluate
from dataset import CONDITIONS, apply_condition
import physics_cfg as phys

OUT, PSFX, SMOKE = S.OUT, S.PSFX, "--smoke-defaults" in sys.argv
KEEP = ("ekf19b_fit", "ekfmu_grid", "ekfmu_h05", "hindsight")
GEO = S.GEO


def imm_params(cond, kind):
    f = OUT / f"ekf_{kind}_{cond}{PSFX}.json"
    if f.exists():
        return json.loads(f.read_text())["params"], False
    if not SMOKE:
        return None, False
    t = S.TUNED["ekf_fit"][cond]
    p = dict(sigma=t["sigma"], E=t["E"], mu_s=1.3 * t["c"], mu_r=0.5 * t["c"], sigma_v0=3.0, sigma_E=0.13)
    p.update(dict(q_s=t["q"], q_r=t["q"], a=2.0, b=0.05, p_s0=0.9) if kind == "imm" else dict(q=t["q"], rule="speed", v_sw=0.3))
    return p, True


def estimators(cond):
    out = {k: v for k, v in S.estimators(cond).items() if k in KEEP}
    for kind, lab, cls in (("imm", "ekfimm", EKFIMM), ("2ph", "ekf2ph", EKF2PH)):
        p, untuned = imm_params(cond, kind)
        if p is not None:
            out[lab] = (cls, p)
    return out


def build(name, kw, rec):
    if isinstance(name, str):
        return S.make(name, rec, **kw, **GEO)
    return StaleGuard(name(**kw, **GEO))


def task(args):
    i, cond = args
    raw = np.load(S.RAW / f"launch_{i:04d}.npz")
    rec = apply_condition(S.cam, raw, CONDITIONS[cond], 1000 + i)
    out = {}
    for lab, (name, kw) in estimators(cond).items():
        if lab == "hindsight":
            kw = dict(kw, c=S.HS[i]["mu"], E=S.HS[i]["E"])
        out[lab] = evaluate(build(name, kw, rec), rec)
        out[lab]["diag"] = S.diagnostics(build(name, kw, rec), rec)
    return i, out, float(raw["phys_a0"] if phys.MISMATCH else rec["mu"])


def main():
    quick = "--quick" in sys.argv
    nl = int(sys.argv[sys.argv.index("--nlaunch") + 1]) if "--nlaunch" in sys.argv else None
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    if nl is not None:
        pos = [a for a in pos if a != str(nl)]
    conds = pos[0].split(",") if pos and pos[0] != "all_conds" else list(CONDITIONS)
    workers = int(pos[1]) if len(pos) > 1 else 1
    val, test = S.SPLIT["val"], S.SPLIT["test"]
    if quick:
        val, test = val[::15], test[::15]
    if nl:
        val, test = val[:nl], test[:nl]
    ids = sorted(val + test)
    sfx = (f"_mismatch_{S.ARM}" if phys.MISMATCH else "") + ("_quick" if quick or nl else "") + ("_smoke" if SMOKE else "")
    jpath, ppath = phys.out_path("ekf_imm_eval.json", sfx), phys.out_path("ekf_imm_eval_errors.pkl.gz", sfx)
    summ = json.loads(jpath.read_text()) if jpath.exists() else {}
    errors = pickle.load(gzip.open(ppath, "rb"))["errors"] if ppath.exists() else {}
    with Pool(workers) as pool:
        for cond in conds:
            t0 = time.time()
            out = pool.map(task, [(i, cond) for i in ids], chunksize=1)
            res = {i: o for i, o, _ in out}; mu_true = {i: m for i, _, m in out}
            labs = list(estimators(cond))
            print(f"\n{'=' * 120}\nCondition {cond} [{phys.PHYSICS}{', ARM=' + S.ARM if S.ARM else ''}] ({time.time() - t0:.0f} s) "
                  f"estimators: {', '.join(labs)}{'  SMOKE DEFAULTS (untuned)' if SMOKE else ''}\n{'=' * 120}")
            tab = dict(test=S.report(res, test, mu_true, labs, "TEST (dev)"), val=S.report(res, val, mu_true, labs, "VAL"),
                       evaluated=time.strftime("%Y-%m-%d %H:%M"), smoke_defaults=SMOKE)
            for lab in ("ekfimm",):                      # mode diagnostic: m_roll = (mu_s - c_eff) / (mu_s - mu_r) at fixed times
                if lab in labs:
                    p = estimators(cond)[lab][1]
                    denom = p["mu_s"] - p["mu_r"]
                    if abs(denom) < 1e-9:                  # degenerate tuning: mu_s == mu_r, m_roll undefined
                        tab["mean_m_roll"] = {str(T): float("nan") for T in S.MU_TIMES}
                        print("  mean m_roll: undefined (mu_s == mu_r in tuned params)")
                    else:
                        mr = {str(T): float(np.mean([(p["mu_s"] - res[i][lab]["diag"]["mu_hat"][T]) / denom
                                                     for i in ids if T in res[i][lab]["diag"]["mu_hat"]] or [np.nan]))
                              for T in S.MU_TIMES}
                        tab["mean_m_roll"] = mr
                        print("  mean m_roll at time since first frame:", mr)
            print("  true slide->roll time per launch is NOT stored in the raw files (only phys_slip0): switch-time correlation skipped")
            summ[cond] = tab
            errors[cond] = {i: {lab: {k: res[i][lab][k] for k in S.KEYS} for lab in labs} for i in ids}
            jpath.write_text(json.dumps(summ, indent=1, default=float))
            with gzip.open(ppath, "wb") as f:
                pickle.dump(dict(split=dict(val=val, test=test), errors=errors), f)
            print(f"saved {cond} -> {jpath.name}", flush=True)


if __name__ == "__main__":
    main()

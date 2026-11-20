"""Unit tests for estimators/ekf_imm.py (B3).  Run: python3 tests/test_ekf_imm.py   (pure numpy, < 10 s)."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from estimators.ekf_imm import EKFIMM, EKF2PH
from estimators.ekf_mu import EKFMu
from estimators.model import propagate

L, OBS, E, C, SIG = 0.9, (0.5, 0.0, 0.12, 0.12), 0.725, 0.75, 0.002


def track(n=120, dt=1 / 60, seed=0):
    rng = np.random.default_rng(seed)
    s = (-0.6, 0.3, 1.9, -0.8); out = []
    for k in range(n):
        for _ in range(4):
            s = propagate(*s, dt / 4, C, E, L, OBS)
        out.append((k * dt, np.array(s[:2]) + SIG * rng.standard_normal(2), rng.random() > 0.1))
    return out


def run(est, tr):
    for t, z, v in tr:
        est.update(t, z, v)
    return est


def test_identical_modes_equal_single_ekf():
    tr = track()
    imm = run(EKFIMM(L, OBS, E, SIG, mu_s=C, mu_r=C, q_s=0.5, q_r=0.5, a=2.0, b=0.3, p_s0=0.7), tr)
    ref = run(EKFMu(L, OBS, E, SIG, q=0.5, q_mu=0.0, mu0=C, s_l=1e-9), tr)
    assert np.allclose(imm.s, ref.s[:4], atol=1e-9), (imm.s, ref.s)
    assert np.allclose(imm.P, ref.P[:4, :4], atol=1e-9)
    assert np.allclose(imm.forecast(0.7), ref.forecast(0.7), atol=1e-9)


def test_probabilities_sum_to_one_and_floor():
    imm = EKFIMM(L, OBS, E, SIG, mu_s=1.2, mu_r=0.3, a=3.0, b=0.1)
    for t, z, v in track():
        imm.update(t, z, v)
        if imm.ready:
            assert abs(imm.m.sum() - 1) < 1e-12 and imm.m.min() >= 1e-4 / 1.001
    assert np.allclose(imm.Pi(0.0), np.eye(2)) and np.allclose(imm.Pi(0.02).sum(1), 1)


def test_static_weights_when_no_switching():
    """a = b = 0: Pi = I, so the weights change only through the likelihood (static multiple-model estimator)."""
    imm = EKFIMM(L, OBS, E, SIG, mu_s=1.2, mu_r=0.3, a=0.0, b=0.0)
    assert np.allclose(imm.Pi(0.5), np.eye(2))
    tr = track(); run(imm, tr[:1])
    assert np.allclose(imm.m, [0.9, 0.1])
    run(imm, tr[1:3])
    assert imm.m.sum() > 0.999


def test_2ph_degenerates_to_fixed_c():
    tr = track()
    for kw in (dict(rule="speed", v_sw=-1.0), dict(rule="time", t_sw=1e9)):          # always slide -> mu_s
        p = run(EKF2PH(L, OBS, E, SIG, mu_s=C, mu_r=0.1, q=0.5, **kw), tr)
        ref = run(EKFMu(L, OBS, E, SIG, q=0.5, q_mu=0.0, mu0=C, s_l=1e-9), tr)
        assert np.allclose(p.s, ref.s[:4], atol=1e-9) and np.allclose(p.P, ref.P[:4, :4], atol=1e-9)
        assert np.allclose(p.forecast(0.7), ref.forecast(0.7), atol=1e-9)


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("ok", n)

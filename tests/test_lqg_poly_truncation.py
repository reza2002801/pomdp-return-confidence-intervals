"""Truncation-mass correction for continuous LQG polynomial TailLP bounds."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from returnbounds import continuous as gm  # noqa: E402
from returnbounds import polybound as pb  # noqa: E402


def _toy_tau(d=8):
    """Finite discrete law on [-1, 1] (finite-model path analogue)."""
    y = np.array([-0.8, -0.2, 0.1, 0.5, 0.9])
    w = np.array([0.1, 0.25, 0.3, 0.2, 0.15])
    moments = np.array([float(np.sum(w * y ** k)) for k in range(d + 1)])
    return pb.cheb_moments(moments)


def _random_form(seed=0, scale=1e-4):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(4, 4))
    J = (A.T @ A) * scale
    b = rng.normal(size=4) * np.sqrt(scale)
    d0 = 1.0
    return J, b, d0, rng


def _sample_G(J, b, d0, n, rng):
    z = rng.normal(size=(n, J.shape[0]))
    return np.einsum("ni,ij,nj->n", z, J, z) + 2.0 * (z @ b) + d0


def _uncorrected_poly_bounds(J, b, d0, degree, alpha, support_mult):
    """Pre-correction continuous poly path (eps_out=tau_d=0, moments only to order d)."""
    kappa = gm.cumulants_quadratic_form(J, b, d0, degree)
    mean = float(kappa[1])
    std = float(np.sqrt(max(kappa[2], 0.0)))
    Lsup = 0.0
    Usup = max(mean + support_mult * std, Lsup + 1e-9)
    c = 0.5 * (Lsup + Usup)
    h = 0.5 * (Usup - Lsup)
    ky = np.zeros(degree + 1)
    ky[1] = (kappa[1] - c) / h
    for n in range(2, degree + 1):
        ky[n] = kappa[n] / (h ** n)
    tau = pb.cheb_moments(gm.raw_moments_from_cumulants(ky))
    y_lo, y_hi, diag = pb.poly_interval_t(tau, degree, alpha)
    return (
        float(np.clip(c + h * y_lo, Lsup, Usup)),
        float(np.clip(c + h * y_hi, Lsup, Usup)),
        Lsup,
        Usup,
        diag,
    )


def test_poly_interval_t_default_matches_finite_regression():
    """Default kwargs (eps_out=tau_d=0) stay bit-identical to the finite-model path."""
    d, alpha = 8, 0.025
    tau = _toy_tau(d)
    lo, hi, diag = pb.poly_interval_t(tau, d, alpha)
    lo2, hi2, diag2 = pb.poly_interval_t(tau, d, alpha, eps_out=0.0, tau_d=0.0)
    assert lo == lo2 and hi == hi2
    assert diag["eps_out"] == 0.0 and diag["max_tail_correction"] == 0.0
    # Frozen regression values for this toy law / degree / alpha.
    assert lo == pytest.approx(-0.8846226501464844, abs=0.0)
    assert hi == pytest.approx(1.0, abs=0.0)
    assert diag2["n_inconsistent"] == 0


def test_cumulants_match_mc_and_matrix_powers():
    J, b, d0, rng = _random_form(seed=0, scale=1e-4)
    kappa = gm.cumulants_quadratic_form(J, b, d0, 4)
    mu = gm.raw_moments_from_cumulants(kappa)

    n = 1_000_000
    G = _sample_G(J, b, d0, n, rng)
    for k in range(1, 5):
        emp = float(np.mean(G ** k))
        rel = abs(emp - mu[k]) / max(abs(mu[k]), 1e-12)
        assert rel < 1e-2, (k, emp, mu[k], rel)

    k_eig = gm.cumulants_quadratic_form(J, b, d0, 32)
    k_mat = gm.cumulants_quadratic_form_matrix_powers(J, b, d0, 32)
    rel = np.abs(k_eig - k_mat) / np.maximum(np.abs(k_mat), 1e-30)
    assert float(np.max(rel[1:])) < 1e-10


def test_eps_plus_dominates_mc_right_tail():
    J, b, d0, rng = _random_form(seed=0, scale=1e-4)
    mean, var = gm.moments_from_form(J, b, d0)
    std = float(np.sqrt(max(var, 0.0)))
    u = mean + 3.0 * std
    l = 0.0
    eps_plus, eps_minus, _, _ = gm.truncation_mass(J, b, d0, l, u)
    G = _sample_G(J, b, d0, 1_000_000, rng)
    emp = float(np.mean(G > u))
    assert eps_plus >= emp - 1e-12, (eps_plus, emp)
    assert eps_minus == 0.0


def test_tail_terms_dominate_mc():
    J, b, d0, rng = _random_form(seed=0, scale=1e-4)
    mean, var = gm.moments_from_form(J, b, d0)
    std = float(np.sqrt(max(var, 0.0)))
    u = mean + 3.0 * std
    c, h = 0.5 * u, 0.5 * u
    d = 8
    eps, tau_d, _, _, _ = gm.tail_terms(J, b, d0, c, h, d)
    G = _sample_G(J, b, d0, 400_000, rng)
    Y = (G - c) / h
    emp_eps = float(np.mean(Y > 1.0))
    emp_tau = float(np.mean((Y ** d) * (Y > 1.0)))
    assert eps >= emp_eps - 1e-12, (eps, emp_eps)
    assert tau_d >= emp_tau - 1e-12, (tau_d, emp_tau)


def test_validity_under_heavy_truncation():
    """support_mult=2: corrected levels must cover at alpha (3 SE); report uncorrected."""
    J, b, d0, rng = _random_form(seed=0, scale=1e-4)
    alpha = 0.025
    n = 1_000_000
    G = _sample_G(J, b, d0, n, rng)

    lo, hi, Lsup, Usup, diag = gm.poly_bounds_from_form(
        J, b, d0, degree=8, alpha=alpha, support_mult=2.0
    )
    ulo, uhi, _, _, udiag = _uncorrected_poly_bounds(
        J, b, d0, degree=8, alpha=alpha, support_mult=2.0
    )

    se = float(np.sqrt(alpha * (1.0 - alpha) / n))
    tol = alpha + 3.0 * se
    frac_above = float(np.mean(G > hi))
    frac_below = float(np.mean(G < lo))
    unc_above = float(np.mean(G > uhi))
    unc_below = float(np.mean(G < ulo))

    test_validity_under_heavy_truncation.uncorrected = {
        "L": ulo,
        "U": uhi,
        "frac_above": unc_above,
        "frac_below": unc_below,
        "n_inconsistent": udiag.get("n_inconsistent"),
    }
    test_validity_under_heavy_truncation.corrected = {
        "L": lo,
        "U": hi,
        "frac_above": frac_above,
        "frac_below": frac_below,
        "eps_out": diag["eps_out"],
        "tau_d": diag.get("tau_d"),
        "max_tail_correction": diag["max_tail_correction"],
        "support_L": Lsup,
        "support_U": Usup,
    }

    assert Lsup == 0.0
    # Upper AND lower Monte Carlo validity under heavy truncation (3 SE).
    assert frac_above <= tol, (frac_above, tol, hi, Usup)
    assert frac_below <= tol, (
        "lower-limit MC validity failed",
        frac_below,
        tol,
        lo,
        Lsup,
    )
    assert diag["eps_out"] > 1e-3  # truncation mass not negligible
    # Lower-tail off-support Chebyshev term must be active under heavy truncation.
    assert float(diag.get("max_tail_correction_lower", 0.0)) > 0.0
    assert float(diag.get("max_off_lower", 0.0)) > 0.0


def test_eps_out_guard_returns_inf():
    """When B_d(1) > alpha and eps_out > alpha, upper endpoint is +inf."""
    d, alpha = 8, 0.025
    tau = _toy_tau(d)
    lo, hi, diag = pb.poly_interval_t(tau, d, alpha, eps_out=0.1, tau_d=1e-40)
    assert np.isfinite(lo)
    assert not np.isfinite(hi) and hi > 0
    assert diag["n_fallback_upper"] == 1
    assert diag["n_guard_upper"] == 1
    # Finite path unchanged when eps_out=tau_d=0 even if endpoint hits the wall.
    lo0, hi0, diag0 = pb.poly_interval_t(tau, d, alpha, eps_out=0.0, tau_d=0.0)
    assert hi0 == pytest.approx(1.0, abs=0.0)
    assert diag0["n_guard_upper"] == 0


def test_correction_negligible_at_support_mult_40():
    J, b, d0, _ = _random_form(seed=0, scale=1e-4)
    alpha, degree = 0.025, 8
    lo, hi, _, _, diag = gm.poly_bounds_from_form(
        J, b, d0, degree=degree, alpha=alpha, support_mult=40.0
    )
    ulo, uhi, _, _, _ = _uncorrected_poly_bounds(
        J, b, d0, degree=degree, alpha=alpha, support_mult=40.0
    )
    scale = max(abs(hi), abs(uhi), abs(lo), abs(ulo), 1e-12)
    assert abs(hi - uhi) / scale < 1e-6
    assert abs(lo - ulo) / scale < 1e-6
    assert diag["eps_out"] < 1e-6 or diag["max_tail_correction"] / scale < 1e-6


def test_corrupted_moments_flagged_with_tiny_eps():
    """Deliberately bad moments of order <= d must trip consistency with tiny eps/tau."""
    d = 8
    tau = _toy_tau(d)
    lp_ok = pb.TailLP(d, eps_out=1e-40, tau_d=1e-40)
    _ = lp_ok.bound(tau, 0.5)
    assert lp_ok.n_inconsistent == 0

    tau_bad = tau.copy()
    tau_bad[1:] += 0.5
    lp = pb.TailLP(d, eps_out=1e-40, tau_d=1e-40)
    flagged = False
    for beta in np.linspace(-0.9, 1.0, 25):
        val = lp.bound(tau_bad, float(beta))
        if lp.n_inconsistent > 0:
            flagged = True
            assert val == 1.0
            assert lp.n_inconsistent_dot > 0
            break
    if not flagged:
        # Force a clearly impossible moment sequence.
        tau_bad2 = tau.copy()
        tau_bad2[:] = 0.0
        tau_bad2[0] = 1.0
        tau_bad2[1] = 10.0  # |E[Y]| >> 1
        lp2 = pb.TailLP(d, eps_out=1e-40, tau_d=1e-40)
        for beta in np.linspace(-0.5, 1.0, 12):
            val = lp2.bound(tau_bad2, float(beta))
            if lp2.n_inconsistent > 0:
                flagged = True
                assert val == 1.0
                break
    assert flagged

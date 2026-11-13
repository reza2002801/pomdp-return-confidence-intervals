"""Validity of TailLP / poly_interval_t against exact discrete distributions on [-1, 1]."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from returnbounds import polybound as pb  # noqa: E402


def _random_atoms(rng, n_atoms):
    y = rng.uniform(-1.0, 1.0, size=n_atoms)
    w = rng.dirichlet(np.ones(n_atoms))
    return y, w


def _monomial_moments(y, w, d):
    return np.array([float(np.sum(w * y ** k)) for k in range(d + 1)])


def test_tail_lp_dominates_true_tail():
    betas = np.linspace(-0.9, 1.0, 15)
    degrees = (2, 4, 8, 16)
    for seed in range(10):
        rng = np.random.default_rng(1000 + seed)
        n_atoms = int(rng.integers(5, 31))
        y, w = _random_atoms(rng, n_atoms)
        for d in degrees:
            moments = _monomial_moments(y, w, d)
            tau = pb.cheb_moments(moments)
            lp = pb.TailLP(d)
            for beta in betas:
                bound = lp.bound(tau, float(beta))
                true = float(np.sum(w[y >= beta]))
                assert bound >= true - 1e-12, (seed, d, beta, bound, true)


def test_poly_interval_coverage_and_consistency():
    alpha = 0.025
    for seed in range(10):
        rng = np.random.default_rng(2000 + seed)
        n_atoms = int(rng.integers(5, 31))
        y, w = _random_atoms(rng, n_atoms)
        for d in (2, 4, 8, 16):
            moments = _monomial_moments(y, w, d)
            tau = pb.cheb_moments(moments)
            lo, hi, diag = pb.poly_interval_t(tau, d, alpha)
            assert diag["n_inconsistent"] == 0, (seed, d, diag)
            assert float(np.sum(w[y > hi])) <= alpha + 1e-12
            assert float(np.sum(w[y < lo])) <= alpha + 1e-12

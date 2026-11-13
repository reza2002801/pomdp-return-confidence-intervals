"""Product chain + mu0 equivalence for finite POMDPs."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from returnbounds import mgf  # noqa: E402
from returnbounds.build_quantized import FiniteModel, simulate  # noqa: E402
from returnbounds.moments import raw_moments, support_interval  # noqa: E402

from experiments.finite_pomdps.policies import (  # noqa: E402
    make_policies,
    tiger_hand,
)
from experiments.finite_pomdps.pomdp_parse import parse_pomdp  # noqa: E402
from experiments.finite_pomdps.product_model import build_product  # noqa: E402
from experiments.finite_pomdps.shopping5 import build_shopping5  # noqa: E402

TIGER = ROOT / "experiments" / "finite_pomdps" / "pomdps" / "tiger.95.POMDP"
NETWORK = ROOT / "experiments" / "finite_pomdps" / "pomdps" / "network.POMDP"


def _tiny_chain():
    """Two-state chain with a single start state k=0."""
    src = np.array([0, 1], dtype=np.int64)
    dst = np.array([1, 0], dtype=np.int64)
    prob = np.array([1.0, 1.0])
    cost = np.array([1.0, 2.0])
    z = np.zeros(2)
    return src, dst, prob, cost, z


def test_x0_equals_mu0_ek():
    src, dst, prob, cost, z = _tiny_chain()
    fm_x0 = FiniteModel(
        n_states=2, src=src, dst=dst, prob=prob, cost=cost,
        p_state_clip=z.copy(), p_memory_clip=z.copy(), p_saturation=z.copy(),
        x0=0, n_raw_edges=2, build_seconds=0.0,
        state_keys=np.arange(2), dims=(2,), pilot={}, n_s=2, n_u_actions=1, mu0=None,
    )
    mu0 = np.array([1.0, 0.0])
    fm_mu = FiniteModel(
        n_states=2, src=src.copy(), dst=dst.copy(), prob=prob.copy(), cost=cost.copy(),
        p_state_clip=z.copy(), p_memory_clip=z.copy(), p_saturation=z.copy(),
        x0=0, n_raw_edges=2, build_seconds=0.0,
        state_keys=np.arange(2), dims=(2,), pilot={}, n_s=2, n_u_actions=1, mu0=mu0,
    )
    T, degree = 5, 4
    op0, opm = mgf.TiltedOperator(fm_x0), mgf.TiltedOperator(fm_mu)
    reach0 = mgf.reachable_sets(op0, fm_x0.x0, T)
    reachm = mgf.reachable_sets(opm, fm_mu.x0, T)
    assert all(np.array_equal(a, b) for a, b in zip(reach0, reachm))
    lams = np.array([-0.5, 0.0, 0.5])
    lm0, v0 = mgf.log_mgf(op0, lams, T, fm_x0.x0, reach0)
    lmm, vm = mgf.log_mgf(opm, lams, T, fm_mu.x0, reachm)
    assert np.allclose(lm0, lmm)
    assert np.array_equal(v0, vm)
    m0, v0m, _ = mgf.exact_moments(op0, fm_x0.x0, T)
    mm, vmm, _ = mgf.exact_moments(opm, fm_mu.x0, T)
    assert np.allclose(m0, mm) and np.allclose(v0m, vmm)
    assert np.allclose(raw_moments(fm_x0, T, degree), raw_moments(fm_mu, T, degree))
    L0, U0 = support_interval(fm_x0, T)
    Lm, Um = support_interval(fm_mu, T)
    assert np.allclose(L0, Lm) and np.allclose(U0, Um)
    C0 = simulate(fm_x0, 2000, T, seed=1)
    Cm = simulate(fm_mu, 2000, T, seed=1)
    assert np.allclose(C0, Cm)


def test_tiger_product_valid_and_hand_policy():
    pomdp = parse_pomdp(TIGER, name="tiger")
    ctrl = tiger_hand(pomdp)
    fm = build_product(pomdp, ctrl, log=lambda *a, **k: None)
    fm.validate()
    assert fm.mu0 is not None
    assert abs(fm.mu0.sum() - 1.0) < 1e-12
    assert fm.n_states > 0 and fm.n_edges > 0
    # row sums already checked by validate; mean at T=5 finite
    mu = float(raw_moments(fm, 5, 1)[5, 1])
    assert np.isfinite(mu)


@pytest.mark.skipif(
    os.environ.get("RUN_HEAVY", "") != "1",
    reason="allocates ~11 GiB (Shopping-5 reward tensor); set RUN_HEAVY=1 to run",
)
def test_network_and_shopping_policies_build():
    net = parse_pomdp(NETWORK, name="network")
    pols = make_policies(net, lambda p, c: build_product(p, c, log=lambda *a, **k: None))
    assert set(pols) == {"A", "B", "A_rand"}
    for c in pols.values():
        fm = build_product(net, c, log=lambda *a, **k: None)
        fm.validate()

    shop = build_shopping5()
    assert shop.reward_depends_only_on_sa()
    # Only build hand policy product (full make_policies for shopping is cheap: no brute)
    from experiments.finite_pomdps.policies import shopping_hand

    fm = build_product(shop, shopping_hand(shop), log=lambda *a, **k: None)
    fm.validate()
    assert fm.n_states < 625 * 650 * 26  # reachable subset

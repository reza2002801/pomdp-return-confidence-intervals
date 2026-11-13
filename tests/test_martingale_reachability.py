"""
Reachability must start from the full support of mu0, not only fm.x0.

Two disconnected components:
  A = {0, 1} — small rewards; contains x0 = 0; mu0 mass 0.7
  B = {2, 3} — large increments; mu0 mass 0.3
      2 -> 2 (p=0.9, r=0), 2 -> 3 (p=0.1, r=50), 3 -> 2 (p=1.0, r=-20)

Reference K / sigma^2 / mean are computed by direct path enumeration over the
full mu0 support (no call to reachable_sets). Expected at T=5: K=45, sigma^2=225.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from returnbounds.build_quantized import FiniteModel  # noqa: E402
from returnbounds import martingale as mg  # noqa: E402


def _make_fm(src, dst, prob, cost, mu0, x0=0):
    n = int(max(int(src.max()), int(dst.max())) + 1)
    z = np.zeros(n)
    return FiniteModel(
        n_states=n,
        src=np.asarray(src, dtype=np.int64),
        dst=np.asarray(dst, dtype=np.int64),
        prob=np.asarray(prob, dtype=float),
        cost=np.asarray(cost, dtype=float),
        p_state_clip=z.copy(),
        p_memory_clip=z.copy(),
        p_saturation=z.copy(),
        x0=int(x0),
        n_raw_edges=len(src),
        build_seconds=0.0,
        state_keys=np.arange(n),
        dims=(n,),
        pilot={},
        n_s=n,
        n_u_actions=1,
        mu0=np.asarray(mu0, dtype=float),
    )


def disconnected_ab():
    # Component A: {0,1} small rewards
    # Component B: {2,3} as specified
    src = np.array([0, 0, 1, 2, 2, 3], dtype=np.int64)
    dst = np.array([0, 1, 0, 2, 3, 2], dtype=np.int64)
    prob = np.array([0.5, 0.5, 1.0, 0.9, 0.1, 1.0], dtype=float)
    cost = np.array([1.0, -1.0, 0.0, 0.0, 50.0, -20.0], dtype=float)
    mu0 = np.array([0.7, 0.0, 0.3, 0.0], dtype=float)
    return _make_fm(src, dst, prob, cost, mu0, x0=0)


def _component_a_only():
    """Same as disconnected_ab but only component A (for K_A)."""
    src = np.array([0, 0, 1], dtype=np.int64)
    dst = np.array([0, 1, 0], dtype=np.int64)
    prob = np.array([0.5, 0.5, 1.0], dtype=float)
    cost = np.array([1.0, -1.0, 0.0], dtype=float)
    mu0 = np.array([1.0, 0.0], dtype=float)
    return _make_fm(src, dst, prob, cost, mu0, x0=0)


def _enumerate_reference(fm, t):
    """
    Brute-force K, sigma2, mean over the full support of mu0 without reachable_sets.

    Reachability is the set of states that appear on any positive-probability path
    of length k under mu0.
    """
    adj = [[] for _ in range(fm.n_states)]
    for s, d, p, c in zip(fm.src, fm.dst, fm.prob, fm.cost):
        adj[int(s)].append((int(d), float(p), float(c)))

    mu0 = fm.init_dist()
    V = mg.value_functions(fm, t)
    mean = float(mu0 @ V[0])

    # states reachable at time k via path enumeration
    reach = [np.zeros(fm.n_states, dtype=bool) for _ in range(t + 1)]
    for x0 in range(fm.n_states):
        if mu0[x0] <= 0:
            continue
        # BFS / expand frontier with positive probability
        frontier = {x0}
        reach[0][x0] = True
        for k in range(t):
            nxt = set()
            for x in frontier:
                for d, pe, _c in adj[x]:
                    if pe > 0:
                        nxt.add(d)
                        reach[k + 1][d] = True
            frontier = nxt

    c_steps = []
    s2_steps = []
    point_mass = bool(np.count_nonzero(mu0 > 0) <= 1)
    if not point_mass:
        D0 = V[0] - mean
        support = mu0 > 0
        c_steps.append(float(np.max(np.abs(D0[support]))))
        s2_steps.append(float(mu0 @ (D0 ** 2)))

    for k in range(t):
        reach_k = reach[k]
        from_reach = reach_k[fm.src]
        if not from_reach.any():
            c_steps.append(0.0)
            s2_steps.append(0.0)
            continue
        D = fm.cost + V[k + 1, fm.dst] - V[k, fm.src]
        c_k = float(np.max(np.abs(D[from_reach])))
        w = np.zeros(fm.n_states)
        np.add.at(w, fm.src[from_reach], fm.prob[from_reach] * (D[from_reach] ** 2))
        s2_k = float(np.max(w[reach_k])) if reach_k.any() else 0.0
        c_steps.append(c_k)
        s2_steps.append(s2_k)

    K = float(max(c_steps)) if c_steps else 0.0
    sigma2 = float(max(s2_steps)) if s2_steps else 0.0
    return K, sigma2, mean, reach


def test_full_mu0_reachability_dominates_x0_only():
    fm = disconnected_ab()
    t = 5
    K_ref, s2_ref, mean_ref, _ = _enumerate_reference(fm, t)

    # Component A alone has much smaller K
    fm_a = _component_a_only()
    K_A, _, _, _ = _enumerate_reference(fm_a, t)
    assert K_ref > 5.0 * K_A, (K_ref, K_A)
    assert abs(K_ref - 45.0) < 1e-10
    assert abs(s2_ref - 225.0) < 1e-10

    # martingale_stats with default reach (full init_dist)
    st = mg.martingale_stats(fm, t)
    assert abs(st["K"] - K_ref) < 1e-10
    assert abs(st["sigma2"] - s2_ref) < 1e-10
    assert abs(st["mean"] - mean_ref) < 1e-10

    # bands at T=t must match the same K / sigma at horizon t
    bands = mg.bands(fm, t, delta=0.025)
    assert abs(bands["K"][t] - K_ref) < 1e-10
    assert abs(bands["sigma"][t] ** 2 - s2_ref) < 1e-10
    assert abs(bands["mean"][t] - mean_ref) < 1e-10

    # Bypassing the mu0 branch (start from e_{x0} only) excludes B's transitions,
    # so the per-step transition ranges miss the dominating K=45 (initial-step
    # |V-mean| can still be moderate because mean mixes A and B under full mu0).
    from returnbounds import mgf as mgf_mod
    op = mgf_mod.TiltedOperator(fm)
    x0_vec = mgf_mod.initial_vector(op.n, fm.x0)
    reach_x0 = mgf_mod.reachable_sets(op, x0_vec, t)
    st_x0 = mg.martingale_stats(fm, t, reach_x0)
    # transition-only max range (drop optional initial step)
    c_trans = st_x0["c_k"][1:] if not st_x0["point_mass"] else st_x0["c_k"]
    assert float(np.max(c_trans)) < K_ref / 5.0
    assert abs(float(np.max(c_trans)) - K_A) < 1e-10

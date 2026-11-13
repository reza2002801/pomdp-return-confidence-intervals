"""Brute-force checks for finite-horizon Azuma / Freedman Doob martingale."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from returnbounds import mgf  # noqa: E402
from returnbounds.build_quantized import FiniteModel  # noqa: E402
from returnbounds import martingale as mg  # noqa: E402


def _make_fm(src, dst, prob, cost, mu0, x0=0):
    n = int(max(src.max(), dst.max()) + 1)
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


def toy_chain_random_mu0():
    """3-state chain with parallel edges and a non-degenerate mu0."""
    # 0 --{0.7/1, 0.3/5}--> 1 --{0.4/0, 0.6/8}--> 2 --{1.0/2}--> 2
    # also self-loop from 0 with small mass via second outgoing? keep simple.
    src = np.array([0, 0, 1, 1, 2], dtype=np.int64)
    dst = np.array([1, 1, 2, 2, 2], dtype=np.int64)
    prob = np.array([0.7, 0.3, 0.4, 0.6, 1.0], dtype=float)
    cost = np.array([1.0, 5.0, 0.0, 8.0, 2.0], dtype=float)
    mu0 = np.array([0.55, 0.45, 0.0], dtype=float)
    return _make_fm(src, dst, prob, cost, mu0, x0=0)


def toy_chain_deterministic():
    fm = toy_chain_random_mu0()
    mu0 = np.zeros(fm.n_states)
    mu0[0] = 1.0
    return _make_fm(fm.src, fm.dst, fm.prob, fm.cost, mu0, x0=0)


def enumerate_paths(fm, t):
    """All paths of length t under mu0; returns list of (prob, G, states[0..t])."""
    adj = [[] for _ in range(fm.n_states)]
    for s, d, p, c in zip(fm.src, fm.dst, fm.prob, fm.cost):
        adj[int(s)].append((int(d), float(p), float(c)))
    mu0 = fm.init_dist()
    paths = []
    for x0 in range(fm.n_states):
        if mu0[x0] <= 0:
            continue
        cur = [(mu0[x0], 0.0, [x0])]
        for _ in range(t):
            nxt = []
            for p, g, st in cur:
                x = st[-1]
                for d, pe, c in adj[x]:
                    nxt.append((p * pe, g + c, st + [d]))
            cur = nxt
        paths.extend(cur)
    return paths


def brute_stats(fm, t, V, reach):
    """Brute-force K, sigma2, c_k, s2_k matching martingale_stats."""
    mu0 = fm.init_dist()
    mean = float(mu0 @ V[0])
    c_steps, s2_steps = [], []
    point_mass = bool(np.count_nonzero(mu0 > 0) <= 1)
    if not point_mass:
        D0 = V[0] - mean
        support = mu0 > 0
        c_steps.append(float(np.max(np.abs(D0[support]))))
        s2_steps.append(float(mu0 @ (D0 ** 2)))
    for k in range(t):
        reach_k = np.asarray(reach[k], dtype=bool)
        from_reach = reach_k[fm.src]
        D = fm.cost + V[k + 1, fm.dst] - V[k, fm.src]
        c_steps.append(float(np.max(np.abs(D[from_reach]))) if from_reach.any() else 0.0)
        w = np.zeros(fm.n_states)
        if from_reach.any():
            np.add.at(w, fm.src[from_reach], fm.prob[from_reach] * (D[from_reach] ** 2))
        s2_steps.append(float(np.max(w[reach_k])) if reach_k.any() else 0.0)
    return (
        float(max(c_steps)),
        float(max(s2_steps)),
        np.asarray(c_steps),
        np.asarray(s2_steps),
        mean,
        point_mass,
    )


def test_mean_and_doob_decomposition():
    fm = toy_chain_random_mu0()
    t = 6
    op = mgf.TiltedOperator(fm)
    reach = mgf.reachable_sets(op, fm.init_dist(), t)
    st = mg.martingale_stats(fm, t, reach)
    V = st["V"]
    mu0 = fm.init_dist()
    assert abs(st["mean"] - float(mu0 @ V[0])) < 1e-12

    paths = enumerate_paths(fm, t)
    # rebuild adj for increments
    for p, G, states in paths:
        # Doob sum: initial + steps
        total = 0.0
        if not st["point_mass"]:
            total += V[0, states[0]] - st["mean"]
        for k in range(t):
            x, xp = states[k], states[k + 1]
            # find matching edge: may be parallel — reconstruct D from costs along path
            # We need the actual edge cost used. Re-enumerate increment from path G pieces.
            # Better: store edge costs while enumerating — redo with costs list.
            pass  # filled below

    # Re-enumerate with per-step costs
    adj = [[] for _ in range(fm.n_states)]
    for s, d, pe, c in zip(fm.src, fm.dst, fm.prob, fm.cost):
        adj[int(s)].append((int(d), float(pe), float(c)))

    def walk(remain, p, g, states, costs):
        if remain == 0:
            # check decomposition
            total_D = 0.0
            if not st["point_mass"]:
                total_D += V[0, states[0]] - st["mean"]
            for k in range(t):
                D = costs[k] + V[k + 1, states[k + 1]] - V[k, states[k]]
                total_D += D
            assert abs(total_D - (g - st["mean"])) < 1e-12, (total_D, g, st["mean"])
            return
        x = states[-1]
        for d, pe, c in adj[x]:
            walk(remain - 1, p * pe, g + c, states + [d], costs + [c])

    for x0 in range(fm.n_states):
        if mu0[x0] <= 0:
            continue
        walk(t, mu0[x0], 0.0, [x0], [])

    # E[D_{k+1} | X_k] == 0 on reachable states
    for k in range(t):
        reach_k = reach[k]
        for x in range(fm.n_states):
            if not reach_k[x]:
                continue
            mask = fm.src == x
            if not mask.any():
                continue
            D = fm.cost[mask] + V[k + 1, fm.dst[mask]] - V[k, x]
            assert abs(float(np.sum(fm.prob[mask] * D))) < 1e-12

    K_b, s2_b, c_b, s2a_b, mean_b, _ = brute_stats(fm, t, V, reach)
    assert abs(st["K"] - K_b) < 1e-12
    assert abs(st["sigma2"] - s2_b) < 1e-12
    assert np.allclose(st["c_k"], c_b)
    assert np.allclose(st["s2_k"], s2a_b)
    assert abs(st["mean"] - mean_b) < 1e-12


def test_bound_validity():
    fm = toy_chain_random_mu0()
    t = 6
    delta = 0.05
    op = mgf.TiltedOperator(fm)
    reach = mgf.reachable_sets(op, fm.init_dist(), t)
    st = mg.martingale_stats(fm, t, reach)
    lev = mg.azuma_freedman(st, delta)

    paths = enumerate_paths(fm, t)
    probs = np.array([p for p, _, _ in paths])
    Gs = np.array([g for _, g, _ in paths])
    assert abs(probs.sum() - 1.0) < 1e-12

    for name, U, L in [
        ("azuma", lev["azuma_U"], lev["azuma_L"]),
        ("azuma2", lev["azuma2_U"], lev["azuma2_L"]),
        ("freedman", lev["freedman_U"], lev["freedman_L"]),
    ]:
        p_above = float(np.sum(probs[Gs > U]))
        p_below = float(np.sum(probs[Gs < L]))
        assert p_above <= delta + 1e-12, (name, p_above, delta, U)
        assert p_below <= delta + 1e-12, (name, p_below, delta, L)


def test_deterministic_start_no_init_step():
    fm = toy_chain_deterministic()
    t = 5
    op = mgf.TiltedOperator(fm)
    reach = mgf.reachable_sets(op, fm.init_dist(), t)
    st = mg.martingale_stats(fm, t, reach)
    assert st["point_mass"]
    assert st["n"] == t
    assert "c_init" not in st


def test_bands_shapes():
    fm = toy_chain_deterministic()
    T = 4
    out = mg.bands(fm, T, delta=0.025)
    for k in ("mean", "azuma_U", "azuma_L", "azuma2_U", "azuma2_L",
              "freedman_U", "freedman_L", "K", "sigma"):
        assert out[k].shape == (T + 1,)
        assert np.isfinite(out[k][1:]).all()

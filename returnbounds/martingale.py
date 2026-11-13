"""
Finite-horizon Azuma / Freedman levels via a Doob martingale on a FiniteModel.

Unlike the discounted infinite-horizon paper (center V(S_0)-γ^T V(S_T)), here the
setting is undiscounted and finite-horizon. Time-varying value functions V_k^{t}
for horizon t make the Doob increments have mean zero, so the CI is centered at
the deterministic mean E[G_t].

All rewards live on edges (``fm.cost``); action/observation randomness is already
folded into the product-chain edge list.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import mgf as mgf_mod


def value_functions(fm, t: int) -> np.ndarray:
    """
    Backward recursion for undiscounted finite-horizon values on the edge list.

    V[t] ≡ 0;  V[k][x] = Σ_{e: x→x'} p_e (c_e + V[k+1][x']),  k = t-1,…,0.

    Returns array of shape (t+1, n_states).
    """
    t = int(t)
    n = fm.n_states
    V = np.zeros((t + 1, n), dtype=float)
    for k in range(t - 1, -1, -1):
        edge_val = fm.prob * (fm.cost + V[k + 1, fm.dst])
        V[k] = np.bincount(fm.src, weights=edge_val, minlength=n)
    return V


def martingale_stats_from_V(fm, t: int, reach, V: np.ndarray) -> dict[str, Any]:
    """
    Per-step Doob ranges/variances for horizon ``t`` given precomputed V.

    For each edge e: x→x' at step k,
        D_e = c_e + V[k+1][x'] − V[k][x].
    Restrict to states reachable at time k (``reach[k]``).

    If ``mu0`` is not a point mass, also include the initial step
        D_0(x) = V[0][x] − E[V[0][X_0]].
    """
    t = int(t)
    mu0 = fm.init_dist()
    mean = float(mu0 @ V[0])

    c_steps: list[float] = []
    s2_steps: list[float] = []

    point_mass = bool(np.count_nonzero(mu0 > 0) <= 1)
    c_init = None
    s2_init = None
    if not point_mass:
        D0 = V[0] - mean
        support = mu0 > 0
        c_init = float(np.max(np.abs(D0[support]))) if support.any() else 0.0
        s2_init = float(mu0 @ (D0 ** 2))
        c_steps.append(c_init)
        s2_steps.append(s2_init)

    for k in range(t):
        reach_k = np.asarray(reach[k], dtype=bool)
        from_reach = reach_k[fm.src]
        if not from_reach.any():
            c_steps.append(0.0)
            s2_steps.append(0.0)
            continue
        D = fm.cost + V[k + 1, fm.dst] - V[k, fm.src]
        c_k = float(np.max(np.abs(D[from_reach])))
        w = np.zeros(fm.n_states, dtype=float)
        src_r = fm.src[from_reach]
        np.add.at(w, src_r, fm.prob[from_reach] * (D[from_reach] ** 2))
        s2_k = float(np.max(w[reach_k])) if reach_k.any() else 0.0
        c_steps.append(c_k)
        s2_steps.append(s2_k)

    c_arr = np.asarray(c_steps, dtype=float)
    s2_arr = np.asarray(s2_steps, dtype=float)
    K = float(np.max(c_arr)) if c_arr.size else 0.0
    sigma2 = float(np.max(s2_arr)) if s2_arr.size else 0.0
    n = int(c_arr.size)

    out: dict[str, Any] = {
        "K": K,
        "sigma2": sigma2,
        "n": n,
        "c_k": c_arr,
        "s2_k": s2_arr,
        "mean": mean,
        "V": V,
        "point_mass": point_mass,
    }
    if c_init is not None:
        out["c_init"] = c_init
        out["s2_init"] = s2_init
    return out


def martingale_stats(fm, t: int, reach=None) -> dict[str, Any]:
    """Per-step Doob ranges/variances for horizon ``t`` (computes V internally).

    If ``reach`` is None, it is computed from the support of ``fm.init_dist()``.
    """
    t = int(t)
    if reach is None:
        reach = mgf_mod.reachable_sets(mgf_mod.TiltedOperator(fm), fm.init_dist(), t)
    return martingale_stats_from_V(fm, t, reach, value_functions(fm, t))


def azuma_freedman(stats: dict[str, Any], delta: float) -> dict[str, float]:
    """
    One-sided-per-tail widths from martingale stats (L = log(1/δ)).

    Azuma (paper form):          w_A  = K √(2 n L)
    Azuma (per-step):            w_A2 = √(2 L Σ c_k²)
    Freedman (paper form):       w_F  = σ √(2 n L + (K L/(3σ))²) + K L / 3
    """
    L = float(np.log(1.0 / delta))
    K = float(stats["K"])
    n = int(stats["n"])
    sigma2 = float(stats["sigma2"])
    sigma = float(np.sqrt(max(sigma2, 0.0)))
    c_k = np.asarray(stats["c_k"], dtype=float)
    mean = float(stats["mean"])

    w_A = K * np.sqrt(2.0 * n * L) if n > 0 else 0.0
    w_A2 = float(np.sqrt(2.0 * L * np.sum(c_k ** 2))) if c_k.size else 0.0
    if sigma <= 0.0:
        w_F = 0.0
    else:
        w_F = float(
            sigma * np.sqrt(2.0 * n * L + (K * L / (3.0 * sigma)) ** 2)
            + (K * L) / 3.0
        )

    return {
        "mean": mean,
        "azuma_w": float(w_A),
        "azuma2_w": float(w_A2),
        "freedman_w": float(w_F),
        "azuma_U": mean + w_A,
        "azuma_L": mean - w_A,
        "azuma2_U": mean + w_A2,
        "azuma2_L": mean - w_A2,
        "freedman_U": mean + w_F,
        "freedman_L": mean - w_F,
        "K": K,
        "sigma": sigma,
        "n": n,
    }


def _value_to_go(fm, T: int) -> np.ndarray:
    """
    W[h] = h-step undiscounted value-to-go (W[0]≡0).

    For horizon t, the time-indexed values are V[k] = W[t−k].
    """
    T = int(T)
    n = fm.n_states
    W = np.zeros((T + 1, n), dtype=float)
    for h in range(T):
        edge_val = fm.prob * (fm.cost + W[h, fm.dst])
        W[h + 1] = np.bincount(fm.src, weights=edge_val, minlength=n)
    return W


def bands(fm, T: int, delta: float) -> dict[str, np.ndarray]:
    """
    Azuma / Freedman levels for every horizon t = 1..T.

    Uses a single value-to-go table W[h]; per-horizon stats still scan edges
    over reachable states (fine for the experiment models).
    """
    T = int(T)
    op = mgf_mod.TiltedOperator(fm)
    # Start from the support of the full initial distribution (explicit vector, not the x0 index),
    # so states reachable only from other initial states are included in K and sigma.
    reach = mgf_mod.reachable_sets(op, fm.init_dist(), T)
    W = _value_to_go(fm, T)

    mean = np.full(T + 1, np.nan)
    azuma_U = np.full(T + 1, np.nan)
    azuma_L = np.full(T + 1, np.nan)
    azuma2_U = np.full(T + 1, np.nan)
    azuma2_L = np.full(T + 1, np.nan)
    freedman_U = np.full(T + 1, np.nan)
    freedman_L = np.full(T + 1, np.nan)
    K = np.full(T + 1, np.nan)
    sigma = np.full(T + 1, np.nan)

    mean[0] = 0.0
    for t in range(1, T + 1):
        # V[k] for horizon t is the (t-k)-step value-to-go
        V = W[t::-1]  # W[t], W[t-1], ..., W[0]  → shape (t+1, n)
        st = martingale_stats_from_V(fm, t, reach, V)
        lev = azuma_freedman(st, delta)
        mean[t] = lev["mean"]
        azuma_U[t], azuma_L[t] = lev["azuma_U"], lev["azuma_L"]
        azuma2_U[t], azuma2_L[t] = lev["azuma2_U"], lev["azuma2_L"]
        freedman_U[t], freedman_L[t] = lev["freedman_U"], lev["freedman_L"]
        K[t], sigma[t] = lev["K"], lev["sigma"]

    return {
        "mean": mean,
        "azuma_U": azuma_U,
        "azuma_L": azuma_L,
        "azuma2_U": azuma2_U,
        "azuma2_L": azuma2_L,
        "freedman_U": freedman_U,
        "freedman_L": freedman_L,
        "K": K,
        "sigma": sigma,
    }

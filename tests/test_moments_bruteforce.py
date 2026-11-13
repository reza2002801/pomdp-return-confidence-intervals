"""Brute-force check of paper moment recursion and exact support on a 4-state chain."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from returnbounds.moments import raw_moments, support_interval  # noqa: E402


@dataclass
class TinyFM:
    n_states: int
    src: np.ndarray
    dst: np.ndarray
    prob: np.ndarray
    cost: np.ndarray
    x0: int = 0


def chain_with_parallel_edges():
    """4-state chain; states 0 and 1 each have two parallel edges with different costs."""
    src = np.array([0, 0, 1, 1, 2, 3], dtype=np.int64)
    dst = np.array([1, 1, 2, 2, 3, 3], dtype=np.int64)
    prob = np.array([0.6, 0.4, 0.5, 0.5, 1.0, 1.0], dtype=float)
    cost = np.array([1.0, 11.0, 0.0, 10.0, 2.0, 0.0], dtype=float)
    return TinyFM(n_states=4, src=src, dst=dst, prob=prob, cost=cost, x0=0)


def enumerate_paths(fm, T):
    adj = [[] for _ in range(fm.n_states)]
    for s, d, p, c in zip(fm.src, fm.dst, fm.prob, fm.cost):
        adj[int(s)].append((int(d), float(p), float(c)))
    # list of (state, log_prob unused, path_prob, G)
    cur = [(int(fm.x0), 1.0, 0.0)]
    by_t = {0: [(1.0, 0.0)]}  # (prob, G)
    for t in range(T):
        nxt = []
        for state, p, g in cur:
            for d, pe, c in adj[state]:
                nxt.append((d, p * pe, g + c))
        cur = nxt
        by_t[t + 1] = [(p, g) for _, p, g in cur]
    return by_t


def test_raw_moments_and_support_bruteforce():
    fm = chain_with_parallel_edges()
    T = 6
    paths = enumerate_paths(fm, T)
    m = raw_moments(fm, T, T)
    L, U = support_interval(fm, T)

    for t in range(T + 1):
        pairs = paths[t]
        probs = np.array([p for p, _ in pairs])
        costs = np.array([g for _, g in pairs])
        assert abs(probs.sum() - 1.0) < 1e-12
        for k in range(T + 1):
            exact = float(np.sum(probs * costs ** k))
            assert abs(m[t, k] - exact) <= 1e-10 * max(1.0, abs(exact)), (t, k, m[t, k], exact)
        pos = probs > 0
        assert L[t] == costs[pos].min()
        assert U[t] == costs[pos].max()

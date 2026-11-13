"""
Shopping-5 POMDP as an in-memory ``POMDP`` (no giant .POMDP text file).

Modelling choices — see also README.md:
- 5×5 grid; state = (agent cell 0..24, item cell 0..24); |S| = 625.
- Actions: LEFT, RIGHT, UP, DOWN, QUERY, BUY.
- Wall bumps: agent stays. Move reward −1.
- QUERY: stay, reward −2; observation reveals the item cell.
- Non-QUERY observations: agent cell only (item hidden).
- BUY: reward +10 if agent==item else −5; agent stays; item re-placed
  uniformly at random (continuing; no termination).
- Start: agent and item i.i.d. uniform on the 25 cells.
"""

from __future__ import annotations

import numpy as np

from .pomdp_parse import POMDP

N = 5
NC = N * N
MOVES = {
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
    "UP": (-1, 0),
    "DOWN": (1, 0),
}


def _rc(k):
    return divmod(int(k), N)


def _move(k, dr, dc):
    r, c = _rc(k)
    nr, nc = r + dr, c + dc
    if 0 <= nr < N and 0 <= nc < N:
        return nr * N + nc
    return int(k)


def _sid(a, i):
    return int(a) * NC + int(i)


def build_shopping5() -> POMDP:
    states = [f"a{a}_i{i}" for a in range(NC) for i in range(NC)]
    actions = ["LEFT", "RIGHT", "UP", "DOWN", "QUERY", "BUY"]
    obs_agent = [f"at{a}" for a in range(NC)]
    obs_query = [f"at{a}_see{i}" for a in range(NC) for i in range(NC)]
    observations = obs_agent + obs_query
    nS, nA, nO = len(states), len(actions), len(observations)
    T = np.zeros((nA, nS, nS))
    O = np.zeros((nA, nS, nO))
    R = np.zeros((nA, nS, nS, nO))

    # moves
    for ai, aname in enumerate(["LEFT", "RIGHT", "UP", "DOWN"]):
        dr, dc = MOVES[aname]
        for a in range(NC):
            a2 = _move(a, dr, dc)
            o = a2  # index into obs_agent
            for i in range(NC):
                s = _sid(a, i)
                sp = _sid(a2, i)
                T[ai, s, sp] = 1.0
                O[ai, sp, o] = 1.0
                R[ai, s, sp, :] = -1.0  # (s,a)-only

    # QUERY
    aq = actions.index("QUERY")
    for a in range(NC):
        for i in range(NC):
            s = _sid(a, i)
            T[aq, s, s] = 1.0
            o = NC + _sid(a, i)  # offset into obs_query
            O[aq, s, o] = 1.0
            R[aq, s, s, :] = -2.0

    # BUY
    ab = actions.index("BUY")
    p_item = 1.0 / NC
    for a in range(NC):
        for i in range(NC):
            s = _sid(a, i)
            rew = 10.0 if a == i else -5.0
            o = a  # agent-only obs after buy
            for i2 in range(NC):
                sp = _sid(a, i2)
                T[ab, s, sp] = p_item
                O[ab, sp, o] = 1.0
                R[ab, s, sp, :] = rew  # depends on s (a==i), not s'/o

    # Fill unused (a, s') observation rows so every O row is stochastic.
    # Default: agent-only obs at the agent cell encoded in s'.
    for ai in range(nA):
        for sp in range(nS):
            if O[ai, sp].sum() <= 0:
                a_cell = sp // NC
                O[ai, sp, a_cell] = 1.0

    start = np.full(nS, 1.0 / nS)
    pomdp = POMDP(
        name="shopping5",
        discount=0.95,
        states=states,
        actions=actions,
        observations=observations,
        T=T,
        O=O,
        R=R,
        start=start,
        source_path="(generated in shopping5.py)",
    )
    pomdp.validate()
    return pomdp

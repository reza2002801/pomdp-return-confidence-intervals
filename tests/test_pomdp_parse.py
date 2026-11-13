"""Cassandra .POMDP parser: Tiger invariants."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.finite_pomdps.pomdp_parse import parse_pomdp  # noqa: E402

TIGER = ROOT / "experiments" / "finite_pomdps" / "pomdps" / "tiger.95.POMDP"


def test_tiger_stochastic_and_rewards():
    p = parse_pomdp(TIGER, name="tiger")
    p.validate()
    assert p.nS == 2 and p.nA == 3 and p.nO == 2
    assert np.allclose(p.start, [0.5, 0.5])
    # transitions: listen = identity; open = uniform
    listen, ol, oright = 0, 1, 2
    assert np.allclose(p.T[listen], np.eye(2))
    assert np.allclose(p.T[ol], 0.5)
    assert np.allclose(p.T[oright], 0.5)
    # observations for listen
    assert np.allclose(p.O[listen], [[0.85, 0.15], [0.15, 0.85]])
    # rewards: listen always -1; open-left: -100 if tiger-left else +10
    assert np.allclose(p.R[listen], -1.0)
    assert np.allclose(p.R[ol, 0], -100.0)  # tiger-left
    assert np.allclose(p.R[ol, 1], 10.0)     # tiger-right
    assert np.allclose(p.R[oright, 0], 10.0)
    assert np.allclose(p.R[oright, 1], -100.0)
    # Tiger rewards depend only on (s, a), not s'/o
    assert p.reward_depends_only_on_sa()


def test_network_loads():
    path = ROOT / "experiments" / "finite_pomdps" / "pomdps" / "network.POMDP"
    p = parse_pomdp(path, name="network")
    p.validate()
    assert p.nS == 7 and p.nA == 4 and p.nO == 2
    assert p.reward_depends_only_on_sa()

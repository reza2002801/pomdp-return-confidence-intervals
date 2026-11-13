"""Sanity checks for the bridge-patrol gridworld."""

from experiments.finite_pomdps.bridge_patrol import (
    BRIDGE,
    GOAL,
    START,
    WATER,
    Z_RISKY,
    Z_SAFE,
    accessible_cells,
    action_for,
    build_bridge_model,
    build_meta,
)


def test_grid_counts_and_path_lengths():
    cells = accessible_cells()
    assert len(cells) == 57
    assert START in cells and GOAL in cells
    assert BRIDGE.issubset(set(cells))
    assert WATER.isdisjoint(set(cells))
    meta = build_meta()
    assert meta.risky_len == 8
    assert meta.safe_len == 14


def test_safe_policy_defined_on_bridge_after_slip():
    meta = build_meta()
    for cell in BRIDGE:
        a = action_for(meta, cell, Z_SAFE)
        assert a in (0, 1, 2, 3)


def test_product_size_and_stochastic():
    fm, meta = build_bridge_model(log=lambda *a, **k: None)
    assert fm.n_states == 232
    fm.validate()
    assert abs(float(fm.mu0.sum()) - 1.0) < 1e-12
    assert int((fm.mu0 > 0).sum()) == 2  # (S, calm) and (S, danger)

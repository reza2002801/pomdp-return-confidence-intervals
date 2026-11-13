"""Finite-model invariants on mass-spring-damper and safety asserts."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from returnbounds.build_quantized import (  # noqa: E402
    _action_code,
    _pack,
    _strides,
    _unpack,
    build,
)
from returnbounds.quantize_setup import setup_from_config  # noqa: E402


CFG_PATH = ROOT / "experiments" / "lqg" / "configs" / "mass_spring_damper.json"


def _load_cfg():
    return json.loads(CFG_PATH.read_text())


def _row_sums(fm):
    n = fm.n_states
    return np.bincount(fm.src, weights=fm.prob, minlength=n)


def _assert_merged_costs_identical(setup, fm):
    """Recompute cost per raw noise branch; merge groups must share one cost."""
    n = setup.n
    dims = fm.dims
    u_dims = tuple(gr.n for gr in setup.U_grids)
    strides = _strides(dims)
    p_vw = (setup.pV[:, None] * setup.pW[None, :]).ravel()
    keys = fm.state_keys
    for sid in range(fm.n_states):
        idx = _unpack(np.array([keys[sid]]), dims)
        s = setup.values_from_idx(idx[:, :n], setup.S_grids)
        z = setup.values_from_idx(idx[:, n:], setup.Z_grids)
        _, iu, u, sat, iz, zclip = setup.control(
            s[:, None, :], z[:, None, :], setup.V[None, :, :])
        is_, sclip = setup.physical(
            s[:, None, None, :], u[:, :, None, :], setup.W[None, None, :, :])
        B, nV, nW = is_.shape[:3]
        assert B == 1
        dst_idx = np.concatenate(
            [is_, np.broadcast_to(iz[:, :, None, :], is_.shape)], axis=-1)
        dst_pack = _pack(dst_idx.reshape(-1, 2 * n), strides)
        iu_b = np.broadcast_to(iu[:, :, None, :], (B, nV, nW, setup.m))
        iu_r = _action_code(iu_b, u_dims).ravel()
        sat_b = np.broadcast_to(sat[:, :, None], (B, nV, nW))
        zclip_b = np.broadcast_to(zclip[:, :, None], (B, nV, nW))
        flag = (sclip.astype(np.int64) * 4
                + zclip_b.astype(np.int64) * 2
                + sat_b.astype(np.int64)).ravel()
        cost = np.broadcast_to(
            setup.cost(s[:, None, :], u)[:, :, None], (B, nV, nW)).ravel()
        cols = np.stack([dst_pack, iu_r, flag], axis=1)
        # within each merge key, costs must be identical
        uniq, inv = np.unique(cols, axis=0, return_inverse=True)
        for j in range(len(uniq)):
            cj = cost[inv == j]
            assert np.allclose(cj, cj[0]), (sid, j, cj)


def test_mass_spring_row_sums_and_merge_costs():
    cfg = _load_cfg()
    setup, _ = setup_from_config(cfg)
    fm = build(setup, max_states=int(cfg.get("max_states", 80000)), log=lambda *a, **k: None)
    rs = _row_sums(fm)
    assert np.allclose(rs, 1.0, atol=1e-12)
    _assert_merged_costs_identical(setup, fm)


def test_h_not_identity_assert():
    cfg = _load_cfg()
    cfg2 = copy.deepcopy(cfg)
    cfg2["observation"] = {"H": [[1.0, 0.0], [0.5, 1.0]]}
    with pytest.raises(AssertionError, match="finite model's observer assumes H = I"):
        setup_from_config(cfg2)

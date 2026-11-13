"""
Build reachable FiniteModel on X=(s, y, z) for a POMDP + finite-state controller.

One edge per (x, a, s', o', z') with the exact reward R[a,s,s',o'] (parallel edges of
different reward stay separate — FiniteModel already allows this).
"""

from __future__ import annotations

import time

import numpy as np

from returnbounds.build_quantized import FiniteModel

from .policies import Controller
from .pomdp_parse import POMDP
from .shopping5 import NC


def _pack3(s, y, z, nY, nZ):
    return (int(s) * nY + int(y)) * nZ + int(z)


def _unpack3(key, nY, nZ):
    z = int(key) % nZ
    t = int(key) // nZ
    y = t % nY
    s = t // nY
    return s, y, z


def _initial_y(pomdp: POMDP, s: int) -> int:
    """Observation available at t=0 before any action.

    Shopping: the agent sees its own cell, so y = agent index.
    Tiger / Network: no free state observation; use a dummy y=0. Hand controllers
    ignore y on the first step when memory is at its reset value (listen / QUERY /
    steady from z0), so the dummy does not change the first action.
    """
    if "shop" in pomdp.name.lower():
        return int(s) // NC
    return 0


def build_product(pomdp: POMDP, ctrl: Controller, log=print) -> FiniteModel:
    """Reachable product chain from the POMDP start and controller init."""
    t0 = time.perf_counter()
    nS, nA, nY, nZ = pomdp.nS, pomdp.nA, pomdp.nO, ctrl.nZ
    mu_z = ctrl.init_z()

    id_of: dict[int, int] = {}
    keys: list[int] = []
    frontier: list[int] = []
    mu0_map: dict[int, float] = {}

    for s in range(nS):
        if pomdp.start[s] <= 0:
            continue
        y0 = _initial_y(pomdp, s)
        for z in range(nZ):
            if mu_z[z] <= 0:
                continue
            k = _pack3(s, y0, z, nY, nZ)
            mass = float(pomdp.start[s] * mu_z[z])
            if k not in id_of:
                id_of[k] = len(keys)
                keys.append(k)
                frontier.append(k)
            mu0_map[k] = mu0_map.get(k, 0.0) + mass

    src_l: list[int] = []
    dst_keys: list[int] = []
    prob_l: list[float] = []
    cost_l: list[float] = []

    # Precompute nonzero transition / observation supports per action
    T_nz = [[np.flatnonzero(pomdp.T[a, s] > 0) for s in range(nS)] for a in range(nA)]
    O_nz = [[np.flatnonzero(pomdp.O[a, sp] > 0) for sp in range(nS)] for a in range(nA)]

    seen_expand: set[int] = set()
    while frontier:
        batch = frontier
        frontier = []
        for key in batch:
            if key in seen_expand:
                continue
            seen_expand.add(key)
            s, y, z = _unpack3(key, nY, nZ)
            src_id = id_of[key]
            for a in range(nA):
                for zp in range(nZ):
                    p_az = ctrl.P[a, zp, z, y]
                    if p_az <= 0:
                        continue
                    for sp in T_nz[a][s]:
                        p_sp = pomdp.T[a, s, sp]
                        for op in O_nz[a][sp]:
                            p_o = pomdp.O[a, sp, op]
                            p = float(p_az * p_sp * p_o)
                            if p <= 0:
                                continue
                            rew = float(pomdp.R[a, s, sp, op])
                            dk = _pack3(int(sp), int(op), zp, nY, nZ)
                            if dk not in id_of:
                                id_of[dk] = len(keys)
                                keys.append(dk)
                                frontier.append(dk)
                            src_l.append(src_id)
                            dst_keys.append(dk)
                            prob_l.append(p)
                            cost_l.append(rew)  # G_t = sum of rewards

    dst_ids = np.array([id_of[k] for k in dst_keys], dtype=np.int64)
    n_states = len(keys)
    mu0 = np.zeros(n_states)
    for k, m in mu0_map.items():
        mu0[id_of[k]] += m
    mu0 = mu0 / mu0.sum()
    x0 = int(np.argmax(mu0))

    zeros = np.zeros(len(src_l))
    fm = FiniteModel(
        n_states=n_states,
        src=np.asarray(src_l, dtype=np.int64),
        dst=dst_ids,
        prob=np.asarray(prob_l, dtype=float),
        cost=np.asarray(cost_l, dtype=float),
        p_state_clip=zeros.copy(),
        p_memory_clip=zeros.copy(),
        p_saturation=zeros.copy(),
        x0=x0,
        n_raw_edges=len(src_l),
        build_seconds=time.perf_counter() - t0,
        state_keys=np.asarray(keys, dtype=np.int64),
        dims=(nS, nY, nZ),
        pilot={
            "pomdp": pomdp.name,
            "policy": ctrl.name,
            "nS": nS,
            "nY": nY,
            "nZ": nZ,
            "reward_only_sa": pomdp.reward_depends_only_on_sa(),
        },
        n_s=nS,
        n_u_actions=nA,
        mu0=mu0,
    )
    fm.validate()
    log(
        f"  product {pomdp.name}/{ctrl.name}: |X|={fm.n_states}, edges={fm.n_edges}, "
        f"build={fm.build_seconds:.2f}s"
    )
    return fm

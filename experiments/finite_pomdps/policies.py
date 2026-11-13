"""
Finite-state controllers pi(a, z' | z, y) for Tiger, Network, Shopping-5.

Each policy exposes nZ, P[a,z',z,y] = Prob(a, z' | z, y), and an initial z / mu_z0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pomdp_parse import POMDP
from .shopping5 import NC, _rc


@dataclass
class Controller:
    name: str
    nZ: int
    z_names: list[str]
    # P[a, z', z, y] = Prob(a, z' | z, y)
    P: np.ndarray
    z0: int = 0
    mu_z0: np.ndarray | None = None

    def init_z(self):
        if self.mu_z0 is not None:
            return np.asarray(self.mu_z0, dtype=float)
        v = np.zeros(self.nZ)
        v[self.z0] = 1.0
        return v


def _eps_greedy(base: Controller, nA: int, eps: float, name: str) -> Controller:
    """Mix base policy with uniform action, keeping the same z'-kernel."""
    P = base.P.copy()
    nZ, nY = base.nZ, P.shape[3]
    out = np.zeros_like(P)
    for z in range(nZ):
        for y in range(nY):
            q = P[:, :, z, y].sum(axis=1)
            if q.sum() <= 0:
                q = np.full(nA, 1.0 / nA)
            else:
                q = q / q.sum()
            q2 = (1.0 - eps) * q + eps / nA
            for a in range(nA):
                kz = P[a, :, z, y]
                if kz.sum() <= 0:
                    kz = np.zeros(nZ)
                    kz[z] = 1.0
                else:
                    kz = kz / kz.sum()
                out[a, :, z, y] = q2[a] * kz
    return Controller(
        name=name, nZ=base.nZ, z_names=base.z_names, P=out, z0=base.z0, mu_z0=base.mu_z0
    )


def tiger_hand(pomdp: POMDP) -> Controller:
    """z = hear-left minus hear-right in {-2..2}, plus post_open.

    Listen until |z|=2, then open the door opposite the tiger and enter post_open.
    From post_open, listen once while ignoring the (uniform) open observation so the
    next count uses a real listen observation only. Start in post_open.
    """
    # z index: 0..4 <-> -2..+2, 5 = post_open
    nZ = 6
    z_names = [str(k) for k in range(-2, 3)] + ["post_open"]
    listen, ol, oright = 0, 1, 2
    post = 5
    nA, nY = pomdp.nA, pomdp.nO
    P = np.zeros((nA, nZ, nZ, nY))
    for y in range(nY):
        # after open (or at start): listen, clear to net=0, ignore junk y
        P[listen, 2, post, y] = 1.0
    for z in range(5):
        net = z - 2
        for y in range(nY):
            if abs(net) < 2:
                net2 = int(np.clip(net + (1 if y == 0 else -1), -2, 2))
                P[listen, net2 + 2, z, y] = 1.0
            else:
                # net=+2 → heard left more → tiger likely left → open-right
                a = oright if net == 2 else ol
                P[a, post, z, y] = 1.0
    return Controller("tiger_hand", nZ, z_names, P, z0=post)


def shopping_hand(pomdp: POMDP) -> Controller:
    """z = remembered item cell, or unknown (NC). QUERY if unknown; else move x then y; BUY on arrival."""
    nZ = NC + 1  # 0..24 item, 25 = unknown
    z_names = [f"item{i}" for i in range(NC)] + ["unknown"]
    nA, nY = pomdp.nA, pomdp.nO
    LEFT, RIGHT, UP, DOWN, QUERY, BUY = range(6)
    P = np.zeros((nA, nZ, nZ, nY))
    unk = NC

    def agent_from_obs(y):
        if y < NC:
            return y
        return (y - NC) // NC

    def item_from_query_obs(y):
        if y < NC:
            return None
        return (y - NC) % NC

    def navigate_or_buy(a_cell, item, z_keep):
        ir, ic = _rc(item)
        ar, ac = _rc(a_cell)
        if (ar, ac) == (ir, ic):
            return BUY, unk
        if ac != ic:
            a = RIGHT if ac < ic else LEFT
        else:
            a = DOWN if ar < ir else UP
        return a, z_keep

    for z in range(nZ):
        for y in range(nY):
            a_cell = agent_from_obs(y)
            if z == unk:
                i_see = item_from_query_obs(y)
                if i_see is not None:
                    # y is a QUERY result: store item and act on it this step.
                    a, z2 = navigate_or_buy(a_cell, i_see, i_see)
                    P[a, z2, z, y] = 1.0
                else:
                    P[QUERY, unk, z, y] = 1.0
            else:
                a, z2 = navigate_or_buy(a_cell, z, z)
                P[a, z2, z, y] = 1.0
    return Controller("shopping_hand", nZ, z_names, P, z0=unk)


def network_hand(pomdp: POMDP) -> Controller:
    """
    z = last two observations (up/down), 4 values UU/UD/DU/DD as 2*y_{t-1}+y_t.

    Rule (documented in README):
    - On observing y, form the pair (y_prev, y) where y_prev is the low bit of z.
    - Both down → reboot; current down → restrict; else steady.
    - unrestrict unused. z' = 2*y_prev + y.
    """
    nZ = 4
    z_names = ["UU", "UD", "DU", "DD"]
    steady, restrict, reboot = 1, 2, 3
    nA, nY = pomdp.nA, pomdp.nO
    P = np.zeros((nA, nZ, nZ, nY))
    for z in range(nZ):
        for y in range(nY):
            y_prev = z % 2
            if y_prev == 1 and y == 1:
                a = reboot
            elif y == 1:
                a = restrict
            else:
                a = steady
            z2 = 2 * y_prev + y
            P[a, z2, z, y] = 1.0
    return Controller("network_hand", nZ, z_names, P, z0=0)


def memoryless_brute(pomdp: POMDP, build_fn, T=100, prefix="ml") -> Controller:
    nA, nY = pomdp.nA, pomdp.nO
    assert nA ** nY <= 100_000
    from returnbounds.moments import raw_moments

    best, best_mu = None, -np.inf
    for code in range(nA ** nY):
        f = [(code // (nA ** y)) % nA for y in range(nY)]
        P = np.zeros((nA, 1, 1, nY))
        for y in range(nY):
            P[f[y], 0, 0, y] = 1.0
        ctrl = Controller(f"{prefix}_{f}", 1, ["*"], P, z0=0)
        fm = build_fn(pomdp, ctrl)
        mu = float(raw_moments(fm, T, 1)[T, 1])
        if mu > best_mu:
            best_mu, best = mu, ctrl
    best.name = f"{prefix}_memoryless"
    return best


def make_policies(pomdp: POMDP, build_fn):
    """Return dict name -> Controller for (A), (B), randomized A."""
    name = pomdp.name.lower()
    if "tiger" in name:
        A = tiger_hand(pomdp)
        B = memoryless_brute(pomdp, build_fn, prefix="tiger")
        Ar = _eps_greedy(A, pomdp.nA, 0.1, "tiger_hand_eps0.1")
    elif "shop" in name:
        A = shopping_hand(pomdp)
        # |A|^|Y| = 6^650 >> 1e5 → (A) with eps=0.1 stands in for memoryless B
        B = _eps_greedy(A, pomdp.nA, 0.1, "shopping_hand_as_B_eps0.1")
        Ar = _eps_greedy(A, pomdp.nA, 0.1, "shopping_hand_eps0.1")
    elif "network" in name:
        A = network_hand(pomdp)
        B = memoryless_brute(pomdp, build_fn, prefix="network")
        Ar = _eps_greedy(A, pomdp.nA, 0.1, "network_hand_eps0.1")
    else:
        raise ValueError(pomdp.name)
    return {"A": A, "B": B, "A_rand": Ar}

"""
Bridge-patrol gridworld POMDP (custom; not a standard benchmark).

Physical state (X, M): known cell X (or fall F) and hidden bridge mode M.
Observation includes X; at the start cell S it also includes a green/red warning.
Agent memory Z ∈ {0=risky, 1=safe} stores the route selected by the last warning at S.

The PDF diagram does not fully specify the safe policy on bridge cells after a slip.
Convention used here: from a bridge cell under Z=safe, take a shortest-path step that
exits toward the nearest non-bridge accessible cell (tie-break below); once off the
bridge, follow the bridge-excluding shortest-path field to G.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from returnbounds.build_quantized import FiniteModel

# Grid
N_ROWS, N_COLS = 7, 11
START = (3, 1)
GOAL = (3, 9)
BRIDGE = {(3, c) for c in range(3, 8)}
WATER = {(r, c) for r in (1, 2, 4, 5) for c in range(3, 8)}

# Actions: tie-break order matches the PDF safe detour (south around the water).
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTIONS = (UP, DOWN, LEFT, RIGHT)
ACTION_NAMES = ("up", "down", "left", "right")
# Prefer Down, then Right, then Up, then Left when several moves cut distance equally.
TIE_BREAK = (DOWN, RIGHT, UP, LEFT)
DELTA = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}

# Dynamics
P_STAY_MODE = 0.95
P_MOVE = 0.92
P_SLIP = 0.08 / 3.0
P_COLLAPSE = {0: 0.015, 1: 0.10}  # calm, danger on bridge
P_GREEN = {0: 0.8, 1: 0.2}  # P(green | mode)

# Costs (current-state); G_t = sum_{k=0}^{t-1} c(X_k)
COST_ORDINARY = 1.0
COST_BRIDGE = 1.5
COST_FALL = 16.0

MODE_CALM, MODE_DANGER = 0, 1
Z_RISKY, Z_SAFE = 0, 1
WARN_GREEN, WARN_RED, WARN_NULL = 0, 1, 2


def _in_grid(r: int, c: int) -> bool:
    return 0 <= r < N_ROWS and 0 <= c < N_COLS


def accessible_cells() -> List[Tuple[int, int]]:
    cells = [
        (r, c)
        for r in range(N_ROWS)
        for c in range(N_COLS)
        if (r, c) not in WATER
    ]
    return cells


def cell_cost(cell: Tuple[int, int] | str) -> float:
    if cell == "F":
        return COST_FALL
    if cell in BRIDGE:
        return COST_BRIDGE
    return COST_ORDINARY


def attempt_move(cell: Tuple[int, int], action: int) -> Tuple[int, int]:
    """Deterministic intended neighbour; water / OOB ⇒ stay."""
    dr, dc = DELTA[action]
    nxt = (cell[0] + dr, cell[1] + dc)
    if (not _in_grid(*nxt)) or nxt in WATER:
        return cell
    return nxt


def _bfs_dist_on(allowed: set[Tuple[int, int]], goal: Tuple[int, int]) -> Dict[Tuple[int, int], int]:
    """Shortest-path distances to ``goal`` moving only inside ``allowed``."""
    from collections import deque

    if goal not in allowed:
        raise ValueError("goal not in allowed set")
    dist = {goal: 0}
    q = deque([goal])
    while q:
        cur = q.popleft()
        for a in ACTIONS:
            dr, dc = DELTA[a]
            nxt = (cur[0] + dr, cur[1] + dc)
            if nxt not in allowed or nxt in dist:
                continue
            dist[nxt] = dist[cur] + 1
            q.append(nxt)
    return dist


def _policy_from_dist(
    dist: Dict[Tuple[int, int], int],
    allowed: set[Tuple[int, int]],
) -> Dict[Tuple[int, int], int]:
    """At each cell, first TIE_BREAK action that steps to a neighbour with dist-1."""
    pi: Dict[Tuple[int, int], int] = {}
    for cell in allowed:
        d = dist.get(cell)
        if d is None or d == 0:
            pi[cell] = RIGHT  # unused at goal; placeholder
            continue
        chosen = None
        for a in TIE_BREAK:
            dr, dc = DELTA[a]
            nxt = (cell[0] + dr, cell[1] + dc)
            if nxt in allowed and dist.get(nxt) == d - 1:
                chosen = a
                break
        if chosen is None:
            # Should not happen on a connected component; stay-oriented fallback.
            chosen = TIE_BREAK[0]
        pi[cell] = chosen
    return pi


@dataclass
class BridgeMeta:
    cells: List[Tuple[int, int]]  # accessible, index 0..56
    cell_of: Dict[Tuple[int, int], int]
    F_idx: int
    risky_pi: Dict[Tuple[int, int], int]
    safe_pi: Dict[Tuple[int, int], int]
    safe_exit_pi: Dict[Tuple[int, int], int]  # on bridge under safe memory
    risky_len: int
    safe_len: int
    n_accessible: int


def build_meta() -> BridgeMeta:
    cells = accessible_cells()
    assert len(cells) == 57, len(cells)
    cell_of = {c: i for i, c in enumerate(cells)}
    F_idx = len(cells)  # 57
    allowed_all = set(cells)
    allowed_safe = allowed_all - BRIDGE

    dist_risky = _bfs_dist_on(allowed_all, GOAL)
    dist_safe = _bfs_dist_on(allowed_safe, GOAL)
    risky_pi = _policy_from_dist(dist_risky, allowed_all)
    safe_pi = _policy_from_dist(dist_safe, allowed_safe)

    # Exit field on bridge: distance to nearest non-bridge cell, then follow exit.
    land = list(allowed_safe)
    from collections import deque

    exit_dist: Dict[Tuple[int, int], int] = {c: 0 for c in land}
    q = deque(land)
    while q:
        cur = q.popleft()
        for a in ACTIONS:
            dr, dc = DELTA[a]
            nxt = (cur[0] + dr, cur[1] + dc)
            if nxt not in allowed_all or nxt in exit_dist:
                continue
            exit_dist[nxt] = exit_dist[cur] + 1
            q.append(nxt)
    safe_exit_pi = _policy_from_dist(exit_dist, allowed_all)

    risky_len = dist_risky[START]
    safe_len = dist_safe[START]
    assert risky_len == 8, risky_len
    assert safe_len == 14, safe_len
    return BridgeMeta(
        cells=cells,
        cell_of=cell_of,
        F_idx=F_idx,
        risky_pi=risky_pi,
        safe_pi=safe_pi,
        safe_exit_pi=safe_exit_pi,
        risky_len=risky_len,
        safe_len=safe_len,
        n_accessible=len(cells),
    )


def action_for(meta: BridgeMeta, cell: Tuple[int, int], z: int) -> int:
    if cell == GOAL:
        return RIGHT  # unused; reset overrides
    if z == Z_RISKY:
        return meta.risky_pi[cell]
    # safe
    if cell in BRIDGE:
        return meta.safe_exit_pi[cell]
    return meta.safe_pi[cell]


def pack_state(x: int, m: int, z: int, nX: int) -> int:
    return (x * 2 + m) * 2 + z


def unpack_state(key: int, nX: int) -> Tuple[int, int, int]:
    z = key % 2
    t = key // 2
    m = t % 2
    x = t // 2
    return x, m, z


def _move_kernel(cell: Tuple[int, int], action: int) -> Dict[Tuple[int, int], float]:
    """Conditional on no collapse: slip kernel over next cells."""
    out: Dict[Tuple[int, int], float] = {}
    for a in ACTIONS:
        p = P_MOVE if a == action else P_SLIP
        nxt = attempt_move(cell, a)
        out[nxt] = out.get(nxt, 0.0) + p
    return out


def _mode_next(m: int) -> List[Tuple[int, float]]:
    return [(m, P_STAY_MODE), (1 - m, 1.0 - P_STAY_MODE)]


def build_bridge_model(log=print) -> Tuple[FiniteModel, BridgeMeta]:
    """
    Reachable FiniteModel on X=(location, mode, memory).

    Warning at S is integrated out inside the outgoing edges, preserving the joint
    dependence between warning → (action, next memory) → next physical state.
    Edge costs are the current-state costs c(X_t).
    """
    import time

    t0 = time.perf_counter()
    meta = build_meta()
    n_loc = meta.n_accessible + 1  # + F
    nX_raw = n_loc * 2 * 2  # 232 before reachability prune
    s_idx = meta.cell_of[START]

    id_of: Dict[int, int] = {}
    keys: List[int] = []
    mu0_map: Dict[int, float] = {}
    frontier: List[int] = []

    def add_state(x, m, z, mass=0.0):
        k = pack_state(x, m, z, n_loc)
        if k not in id_of:
            id_of[k] = len(keys)
            keys.append(k)
            frontier.append(k)
        if mass:
            mu0_map[k] = mu0_map.get(k, 0.0) + mass
        return id_of[k]

    def location_of(x: int):
        if x == meta.F_idx:
            return "F"
        return meta.cells[x]

    # Initial: X=S, M~uniform, Z placeholder 0 (overridden by first warning at S).
    for m in (MODE_CALM, MODE_DANGER):
        add_state(s_idx, m, Z_RISKY, mass=0.5)

    src_l: List[int] = []
    dst_keys: List[int] = []
    prob_l: List[float] = []
    cost_l: List[float] = []

    def emit(src_id, x2, m2, z2, p, cost):
        if p <= 0:
            return
        add_state(x2, m2, z2)
        k = pack_state(x2, m2, z2, n_loc)
        src_l.append(src_id)
        dst_keys.append(k)
        prob_l.append(float(p))
        cost_l.append(float(cost))

    seen: set[int] = set()
    while frontier:
        key = frontier.pop()
        if key in seen:
            continue
        seen.add(key)
        x, m, z = unpack_state(key, n_loc)
        src_id = id_of[key]
        loc = location_of(x)
        cost = cell_cost(loc)

        if loc == GOAL or loc == "F":
            for m2, pm in _mode_next(m):
                emit(src_id, s_idx, m2, z, pm, cost)
            continue

        def apply_action(action: int, z_next: int, weight: float):
            p_col = P_COLLAPSE[m] if loc in BRIDGE else 0.0
            if p_col > 0:
                for m2, pm in _mode_next(m):
                    emit(src_id, meta.F_idx, m2, z_next, weight * p_col * pm, cost)
            p_nc = 1.0 - p_col
            if p_nc <= 0:
                return
            for nxt, p_mv in _move_kernel(loc, action).items():
                x2 = meta.cell_of[nxt]
                for m2, pm in _mode_next(m):
                    emit(src_id, x2, m2, z_next, weight * p_nc * p_mv * pm, cost)

        if loc == START:
            for w, z_next in ((WARN_GREEN, Z_RISKY), (WARN_RED, Z_SAFE)):
                pw = P_GREEN[m] if w == WARN_GREEN else 1.0 - P_GREEN[m]
                apply_action(action_for(meta, loc, z_next), z_next, pw)
        else:
            apply_action(action_for(meta, loc, z), z, 1.0)

    dst_ids = np.array([id_of[k] for k in dst_keys], dtype=np.int64)
    n_states = len(keys)
    mu0 = np.zeros(n_states)
    for k, mass in mu0_map.items():
        mu0[id_of[k]] += mass
    mu0 /= mu0.sum()
    x0 = int(np.argmax(mu0))

    fm = FiniteModel(
        n_states=n_states,
        src=np.asarray(src_l, dtype=np.int64),
        dst=dst_ids,
        prob=np.asarray(prob_l, dtype=float),
        cost=np.asarray(cost_l, dtype=float),
        p_state_clip=np.zeros(n_states),
        p_memory_clip=np.zeros(n_states),
        p_saturation=np.zeros(n_states),
        x0=x0,
        n_raw_edges=len(src_l),
        build_seconds=time.perf_counter() - t0,
        state_keys=np.asarray(keys, dtype=np.int64),
        dims=(n_loc, 2, 2),
        pilot={
            "name": "bridge_patrol",
            "n_accessible": meta.n_accessible,
            "n_locations_with_F": n_loc,
            "n_raw_product": nX_raw,
            "n_reachable": n_states,
            "risky_path_len": meta.risky_len,
            "safe_path_len": meta.safe_len,
            "tie_break": [ACTION_NAMES[a] for a in TIE_BREAK],
            "safe_bridge_convention": (
                "On bridge under Z=safe, follow shortest exit to a non-bridge "
                "accessible cell (same TIE_BREAK); then bridge-excluding SPF to G."
            ),
        },
        n_s=n_loc,
        n_u_actions=4,
        mu0=mu0,
    )
    fm.validate()
    log(
        f"  bridge_patrol FiniteModel: |X|={fm.n_states} (raw {nX_raw}), "
        f"edges={fm.n_edges}, risky={meta.risky_len}, safe={meta.safe_len}, "
        f"build={fm.build_seconds:.2f}s"
    )
    return fm, meta


def exact_cost_distribution(fm: FiniteModel, T: int):
    """
    Exact law of G_t for t=0..T via DP on integer indices of 2 G_t
    (costs are multiples of 1/2).
    """
    # half-cost units
    half_cost = np.rint(2.0 * fm.cost).astype(np.int64)
    if not np.allclose(fm.cost, 0.5 * half_cost):
        raise ValueError("costs are not multiples of 0.5")
    max_half = int(half_cost.max())
    n = fm.n_states
    # mass[state, half] — use list of dicts for sparsity
    cur: List[Dict[int, float]] = [dict() for _ in range(n)]
    mu0 = fm.init_dist()
    for i, p in enumerate(mu0):
        if p > 0:
            cur[i][0] = float(p)

    means = np.zeros(T + 1)
    # store terminal pmf aggregated over states for each t
    pmfs: List[Dict[int, float]] = [{0: 1.0}]

    for t in range(1, T + 1):
        nxt: List[Dict[int, float]] = [dict() for _ in range(n)]
        for e in range(fm.n_edges):
            s, d, p, hc = int(fm.src[e]), int(fm.dst[e]), float(fm.prob[e]), int(half_cost[e])
            if p <= 0 or not cur[s]:
                continue
            bucket = nxt[d]
            for g, mass in cur[s].items():
                gg = g + hc
                bucket[gg] = bucket.get(gg, 0.0) + mass * p
        cur = nxt
        agg: Dict[int, float] = {}
        for st in cur:
            for g, mass in st.items():
                agg[g] = agg.get(g, 0.0) + mass
        # normalize tiny drift
        z = sum(agg.values())
        if abs(z - 1.0) > 1e-9:
            for g in list(agg):
                agg[g] /= z
        pmfs.append(agg)
        means[t] = 0.5 * sum(g * m for g, m in agg.items())

    return {"half_pmf": pmfs, "mean": means}


def pmf_quantile(half_pmf: Dict[int, float], q: float) -> float:
    items = sorted(half_pmf.items())
    acc = 0.0
    for g, p in items:
        acc += p
        if acc >= q:
            return 0.5 * g
    return 0.5 * items[-1][0]


def pmf_tail(half_pmf: Dict[int, float], lo: float, hi: float) -> Tuple[float, float]:
    """P(G < lo), P(G > hi)."""
    below = above = 0.0
    for g, p in half_pmf.items():
        v = 0.5 * g
        if v < lo:
            below += p
        if v > hi:
            above += p
    return below, above

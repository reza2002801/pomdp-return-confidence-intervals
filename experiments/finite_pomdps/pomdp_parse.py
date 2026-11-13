"""
Cassandra ``.POMDP`` file parser (T / O / R lines, wildcards, identity/uniform).

Rewards are stored as a dense tensor R[a, s, s', o] so transition edges keep the
exact (s, a, s', o) reward (never the expectation E[R|s,a]).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class POMDP:
    name: str
    discount: float
    states: list[str]
    actions: list[str]
    observations: list[str]
    T: np.ndarray          # (A, S, S')  P(s'|s,a)
    O: np.ndarray          # (A, S', O)  P(o|s',a)  — Cassandra convention
    R: np.ndarray          # (A, S, S', O)
    start: np.ndarray      # (S,)
    source_path: str = ""

    @property
    def nS(self):
        return len(self.states)

    @property
    def nA(self):
        return len(self.actions)

    @property
    def nO(self):
        return len(self.observations)

    def reward_depends_only_on_sa(self, tol=1e-12) -> bool:
        """True iff every positive-probability (s,a,s',o) shares one reward per (s,a)."""
        for a in range(self.nA):
            for s in range(self.nS):
                vals = []
                for sp in range(self.nS):
                    if self.T[a, s, sp] <= 0:
                        continue
                    for o in range(self.nO):
                        if self.O[a, sp, o] <= 0:
                            continue
                        vals.append(self.R[a, s, sp, o])
                if vals and (max(vals) - min(vals) > tol):
                    return False
        return True

    def validate(self, tol=1e-9):
        for a in range(self.nA):
            rs = self.T[a].sum(axis=1)
            if not np.allclose(rs, 1.0, atol=tol):
                raise AssertionError(f"T rows not stochastic for action {a}: {rs}")
            # O is indexed by next-state
            os = self.O[a].sum(axis=1)
            if not np.allclose(os, 1.0, atol=tol):
                raise AssertionError(f"O rows not stochastic for action {a}: {os}")
        if not np.allclose(self.start.sum(), 1.0, atol=tol):
            raise AssertionError(f"start not a distribution: sum={self.start.sum()}")


def _parse_names(toks):
    """Expand 'states: n' or 'states: name1 name2 ...'."""
    if len(toks) == 1 and toks[0].isdigit():
        n = int(toks[0])
        return [str(i) for i in range(n)]
    return list(toks)


def _index(names, token, allow_star=True):
    if token == "*" and allow_star:
        return None
    if token in names:
        return names.index(token)
    if token.isdigit():
        return int(token)
    raise KeyError(f"unknown name {token!r} in {names}")


def _read_matrix(lines, i, nrows, ncols):
    """Read a dense matrix starting at lines[i]; return matrix and next index."""
    rows = []
    while len(rows) < nrows and i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("#"):
            continue
        # keyword on its own line
        low = line.lower()
        if low in ("identity", "uniform"):
            if nrows != ncols and low == "identity":
                raise ValueError("identity requires square matrix")
            if low == "identity":
                return np.eye(nrows, dtype=float), i
            return np.full((nrows, ncols), 1.0 / ncols, dtype=float), i
        parts = line.split()
        if len(parts) == ncols:
            rows.append([float(x) for x in parts])
        elif len(parts) == 1 and ncols == 1:
            rows.append([float(parts[0])])
        else:
            # single probability for a specific entry was already handled by caller
            raise ValueError(f"expected {ncols} columns, got {line!r}")
    if len(rows) != nrows:
        raise ValueError(f"expected {nrows} rows, got {len(rows)}")
    return np.asarray(rows, dtype=float), i


def parse_pomdp(path: str | Path, name: str | None = None) -> POMDP:
    path = Path(path)
    raw = path.read_text()
    # strip comments
    lines = []
    for line in raw.splitlines():
        if "#" in line:
            line = line[: line.index("#")]
        lines.append(line.rstrip())

    discount = 0.95
    states = actions = observations = None
    start = None
    i = 0
    # first pass: headers
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line:
            continue
        low = line.lower()
        if low.startswith("discount:"):
            discount = float(line.split(":", 1)[1].strip())
        elif low.startswith("values:"):
            continue
        elif low.startswith("states:"):
            rest = line.split(":", 1)[1].strip().split()
            if rest:
                states = _parse_names(rest)
            else:
                # names on following non-empty lines until blank? Cassandra puts on same line usually
                while i < len(lines) and not lines[i].strip():
                    i += 1
                if i < len(lines):
                    states = _parse_names(lines[i].strip().split())
                    i += 1
        elif low.startswith("actions:"):
            rest = line.split(":", 1)[1].strip().split()
            actions = _parse_names(rest) if rest else None
            if not rest:
                while i < len(lines) and not lines[i].strip():
                    i += 1
                actions = _parse_names(lines[i].strip().split()); i += 1
        elif low.startswith("observations:"):
            rest = line.split(":", 1)[1].strip().split()
            observations = _parse_names(rest) if rest else None
            if not rest:
                while i < len(lines) and not lines[i].strip():
                    i += 1
                observations = _parse_names(lines[i].strip().split()); i += 1
        elif low.startswith("start:"):
            rest = line.split(":", 1)[1].strip()
            if rest.lower() == "uniform":
                start = "uniform"
            elif rest:
                start = np.asarray([float(x) for x in rest.split()], dtype=float)
            else:
                # next non-empty line(s)
                while i < len(lines) and not lines[i].strip():
                    i += 1
                tok = lines[i].strip().lower(); i += 1
                if tok == "uniform":
                    start = "uniform"
                else:
                    start = np.asarray([float(x) for x in tok.split()], dtype=float)
        elif low.startswith("t:") or low.startswith("o:") or low.startswith("r:"):
            i -= 1
            break

    if states is None or actions is None or observations is None:
        raise ValueError(f"missing states/actions/observations in {path}")
    nS, nA, nO = len(states), len(actions), len(observations)
    T = np.zeros((nA, nS, nS))
    O = np.zeros((nA, nS, nO))
    R = np.zeros((nA, nS, nS, nO))

    def expand(idx, n):
        return range(n) if idx is None else [idx]

    # second pass from current i
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line:
            continue
        low = line.lower()

        if low.startswith("t:"):
            # forms:
            # T: action
            # T: action : s : s'
            # T: action : s
            body = line.split(":", 1)[1]
            parts = [p.strip() for p in body.split(":")]
            a_tok = parts[0].split()[0]
            a_list = expand(_index(actions, a_tok), nA)

            if len(parts) == 1:
                # full matrix for this action
                # peek next line for identity/uniform or matrix
                M, i = _read_matrix(lines, i, nS, nS)
                for aa in a_list:
                    T[aa] = M
            elif len(parts) == 2:
                s = _index(states, parts[1].split()[0])
                # row vector of length nS, or identity/uniform row?
                # read one line of nS floats or a keyword matrix? Cassandra: next line is nS probs
                while i < len(lines) and not lines[i].strip():
                    i += 1
                row_line = lines[i].strip(); i += 1
                if row_line.lower() == "uniform":
                    row = np.full(nS, 1.0 / nS)
                else:
                    row = np.asarray([float(x) for x in row_line.split()], dtype=float)
                for aa in a_list:
                    for ss in expand(s, nS):
                        T[aa, ss] = row
            elif len(parts) >= 3:
                s = _index(states, parts[1].split()[0])
                sp = _index(states, parts[2].split()[0])
                # probability on same line after last token OR next line
                rem = parts[2].split()[1:] if len(parts[2].split()) > 1 else []
                if len(parts) > 3:
                    rem = parts[3].split() + rem
                if rem:
                    p = float(rem[0])
                else:
                    while i < len(lines) and not lines[i].strip():
                        i += 1
                    p = float(lines[i].strip().split()[0]); i += 1
                for aa in a_list:
                    for ss in expand(s, nS):
                        for ssp in expand(sp, nS):
                            T[aa, ss, ssp] = p
            else:
                raise ValueError(f"bad T line: {line}")

        elif low.startswith("o:"):
            # O: action
            # O: action : s' : o
            # O: action : s'
            body = line.split(":", 1)[1]
            parts = [p.strip() for p in body.split(":")]
            a_tok = parts[0].split()[0]
            a_list = expand(_index(actions, a_tok), nA)
            if len(parts) == 1:
                M, i = _read_matrix(lines, i, nS, nO)
                for aa in a_list:
                    O[aa] = M
            elif len(parts) == 2:
                sp = _index(states, parts[1].split()[0])
                while i < len(lines) and not lines[i].strip():
                    i += 1
                row_line = lines[i].strip(); i += 1
                if row_line.lower() == "uniform":
                    row = np.full(nO, 1.0 / nO)
                else:
                    row = np.asarray([float(x) for x in row_line.split()], dtype=float)
                for aa in a_list:
                    for ssp in expand(sp, nS):
                        O[aa, ssp] = row
            elif len(parts) >= 3:
                sp = _index(states, parts[1].split()[0])
                o = _index(observations, parts[2].split()[0])
                rem = parts[2].split()[1:]
                if len(parts) > 3:
                    rem = parts[3].split() + rem
                if rem:
                    p = float(rem[0])
                else:
                    while i < len(lines) and not lines[i].strip():
                        i += 1
                    p = float(lines[i].strip().split()[0]); i += 1
                for aa in a_list:
                    for ssp in expand(sp, nS):
                        for oo in expand(o, nO):
                            O[aa, ssp, oo] = p
            else:
                raise ValueError(f"bad O line: {line}")

        elif low.startswith("r:"):
            # R: a : s : s' : o [reward]
            body = line.split(":", 1)[1]
            parts = [p.strip() for p in body.split(":")]
            if len(parts) < 4:
                raise ValueError(f"bad R line: {line}")
            a = _index(actions, parts[0].split()[0])
            s = _index(states, parts[1].split()[0])
            sp = _index(states, parts[2].split()[0])
            o_toks = parts[3].split()
            o = _index(observations, o_toks[0])
            if len(o_toks) > 1:
                rew = float(o_toks[1])
            elif len(parts) > 4:
                rew = float(parts[4].split()[0])
            else:
                while i < len(lines) and not lines[i].strip():
                    i += 1
                rew = float(lines[i].strip().split()[0]); i += 1
            for aa in expand(a, nA):
                for ss in expand(s, nS):
                    for ssp in expand(sp, nS):
                        for oo in expand(o, nO):
                            R[aa, ss, ssp, oo] = rew
        else:
            # ignore unknown / leftover header duplicates
            continue

    if start is None or (isinstance(start, str) and start == "uniform"):
        start_v = np.full(nS, 1.0 / nS)
    else:
        start_v = np.asarray(start, dtype=float)
        start_v = start_v / start_v.sum()

    pomdp = POMDP(
        name=name or path.stem,
        discount=discount,
        states=states,
        actions=actions,
        observations=observations,
        T=T,
        O=O,
        R=R,
        start=start_v,
        source_path=str(path),
    )
    pomdp.validate()
    return pomdp

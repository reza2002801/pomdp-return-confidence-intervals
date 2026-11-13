"""
Polynomial moment bound of the paper (eq. moment-tail-bound) for G_t, t = 1..T:

    P(G_t >= b) <= B_{t,d}(b) = min_a sum_k a_k E[G_t^k]
                  s.t. p_a >= 0 on [L_t, U_t],  p_a >= 1 on [b, U_t].

[L_t, U_t] is the exact range of G_t over positive-probability paths. Everything is
solved for Y_t = (G_t - c_t) / h_t in [-1, 1], whose moments come from the moment
recursion (Proposition moments) with rewards shifted and scaled per horizon, and the
polynomial is written in the Chebyshev basis.

The semi-infinite constraints are handled by cutting planes: start from a coarse grid,
add the most violated points of a fine check grid and re-solve; the remaining
violation on the check grid is added to a_0, so the returned value is an upper bound
up to the check-grid resolution. Without further restriction the optimal polynomial
grows to ~1e5 where Y has (almost) no mass and the LP loses precision, so the
polynomials are additionally restricted to p <= P_CAP on [-1, 1]. Restricting the
feasible set keeps every returned value a valid upper bound, possibly a looser one.
"""

from math import comb

import numpy as np
from numpy.polynomial import chebyshev as C
from scipy.optimize import linprog

P_CAP = 1e3
COARSE_GRID = np.unique(np.concatenate([np.cos(np.linspace(0.0, np.pi, 401)), np.linspace(-1, 1, 401)]))
CHECK_GRID = np.unique(np.concatenate([np.cos(np.linspace(0.0, np.pi, 100001)), np.linspace(-1, 1, 100001)]))


def support_range(P_t, r_t, s0):
    """L[t-1], U[t-1] = min / max of G_t over paths with positive probability."""
    T, n, _ = P_t.shape
    lo = np.full(n, np.inf)
    hi = np.full(n, -np.inf)
    lo[s0] = hi[s0] = 0.0
    L, U = np.zeros(T), np.zeros(T)
    for k in range(T):
        reach = np.isfinite(lo)
        L[k], U[k] = (lo + r_t[k])[reach].min(), (hi + r_t[k])[reach].max()
        edge = P_t[k] > 0
        lo = np.where(edge, (lo + r_t[k])[:, None], np.inf).min(axis=0)
        hi = np.where(edge, (hi + r_t[k])[:, None], -np.inf).max(axis=0)
    return L, U


def scaled_moments(P_t, r_t, s0, t, center, scale, d):
    """E[Y^k], k = 0..d, for Y = (G_t - center) / scale, via the moment recursion."""
    n = P_t.shape[1]
    xi = np.zeros((d + 1, n))
    xi[0, s0] = 1.0
    binom = [[comb(k, j) for j in range(k + 1)] for k in range(d + 1)]
    for k in range(t):
        r = (r_t[k] - center / t) / scale
        rp = r[None, :] ** np.arange(d + 1)[:, None]
        new = np.empty_like(xi)
        for m in range(d + 1):
            new[m] = sum(binom[m][j] * rp[j] * xi[m - j] for j in range(m + 1))
        xi = new @ P_t[k]
    return xi.sum(axis=1)


def cheb_moments(moments_y):
    """E[T_k(Y)] from E[Y^k]."""
    d = len(moments_y) - 1
    return np.array([C.cheb2poly(np.eye(d + 1)[k]) @ moments_y[: k + 1] for k in range(d + 1)])


class TailLP:
    """B_d(beta) >= P(Y >= beta) for Y in [-1, 1] with Chebyshev moments tau[0..d].

    Optional truncation correction (continuous LQG): if the true law of Y may put
    mass outside S=[-1,1] on the right (Y>1), pass ``eps_out >= P(Y>1)`` and
    ``tau_d >= E[Y^d 1{Y>1}]`` from the closed-form MGF (no moments above order d).
    With a dominating polynomial q = p_a + c_corr on S,
        P(Y >= beta) <= E[q] + 2^d ||alpha||_1 tau_d + eps_out,
    where alpha is a with alpha_0 shifted by c_corr. With ``eps_out=tau_d=0`` the
    return value is identical to the uncorrected / finite-model bound.
    """

    def __init__(self, d, tol=1e-6, max_rounds=40, ridge=1e-7, eps_out=0.0, tau_d=0.0):
        self.d = d
        self.tol, self.max_rounds = tol, max_rounds
        self.Vc = C.chebvander(CHECK_GRID, d)
        self.coarse = np.searchsorted(CHECK_GRID, COARSE_GRID).clip(0, len(CHECK_GRID) - 1)
        self.warm = self.coarse
        # ridge * (integral of p over [-1, 1]): where Y has (almost) no mass the LP is
        # degenerate and returns wildly oscillating polynomials; this selects a tame one.
        self.integral = ridge * np.array([0.0 if k % 2 else 2.0 / (1.0 - k * k) for k in range(d + 1)])
        self.hmax = float(np.diff(CHECK_GRID).max())
        self.n_inconsistent = 0
        self.n_inconsistent_ep2 = 0  # retained for report compatibility; unused
        self.n_inconsistent_dot = 0
        self.n_ep2_neg = 0
        self.min_ep2_raw = 0.0
        self.max_tol2 = 0.0
        self.max_eps_smooth = 0.0
        self.eps_out = float(eps_out)
        self.tau_d = float(tau_d)
        self.max_tail_correction = 0.0
        self.max_off = 0.0
        self.last_endpoint_fallback = False

    def _reset_diag(self):
        self.n_inconsistent = 0
        self.n_inconsistent_ep2 = 0
        self.n_inconsistent_dot = 0
        self.n_ep2_neg = 0
        self.min_ep2_raw = 0.0
        self.max_tol2 = 0.0
        self.max_eps_smooth = 0.0
        self.max_tail_correction = 0.0
        self.max_off = 0.0
        self.last_endpoint_fallback = False

    def bound(self, tau, beta):
        if beta <= -1.0:
            return 1.0
        if beta > 1.0:
            return 0.0
        ind = (CHECK_GRID >= beta).astype(float)
        j = np.searchsorted(CHECK_GRID, beta)
        active = np.union1d(self.warm, [max(j - 1, 0), min(j, len(CHECK_GRID) - 1)])
        V_beta = C.chebvander(np.array([beta]), self.d)
        free = [(None, None)] * (self.d + 1)
        a = None
        for _ in range(self.max_rounds):
            V = self.Vc[active]
            res = linprog(tau + self.integral, A_ub=np.vstack([-V, -V_beta, V]),
                          b_ub=np.concatenate([-ind[active], [-1.0], np.full(len(active), P_CAP)]),
                          bounds=free, method="highs")
            if not res.success:
                break
            a = res.x
            slack = self.Vc @ a - ind
            if slack.min() >= -self.tol:
                break
            s = np.concatenate([[np.inf], slack, [np.inf]])
            local_min = np.flatnonzero((s[1:-1] <= s[:-2]) & (s[1:-1] <= s[2:]) & (slack < -self.tol))
            violated = np.flatnonzero(slack < -self.tol)
            spread = violated[np.linspace(0, len(violated) - 1, min(200, len(violated))).astype(int)]
            new = np.concatenate([local_min - 1, local_min, local_min + 1, spread]).clip(0, len(slack) - 1)
            active = np.union1d(active, new)
        if a is None:
            return 1.0
        self.warm = np.union1d(self.coarse, active[np.abs(self.Vc[active] @ a - ind[active]) < 1e-3])
        eps = max(0.0, float(-np.min(self.Vc @ a - ind)), float(1.0 - np.asarray(V_beta @ a).ravel()[0]))
        gamma = C.chebder(a, 2)                     # Chebyshev coeffs of p''
        M = float(np.abs(gamma).sum())              # |T_k| <= 1 on [-1, 1]  =>  sup|p''| <= M
        eps_smooth = M / 8.0 * self.hmax ** 2       # linear-interpolation error between check points
        self.max_eps_smooth = max(self.max_eps_smooth, eps_smooth)
        # float64 HiGHS residuals reach ~1e-6 at high degree; only flag clear inconsistency
        dot = float(tau @ a)
        c_corr = 1e-5 + eps + eps_smooth
        trunc = (self.eps_out > 0.0) or (self.tau_d > 0.0)
        if trunc:
            # alpha = a with constant shift c_corr in alpha_0 (q = p_a + c_corr).
            alpha = np.asarray(a, dtype=float).copy()
            alpha[0] = alpha[0] + c_corr
            sum_abs_a = float(np.abs(a).sum())
            sum_abs_alpha = float(np.abs(alpha).sum())
            off = (2.0 ** self.d) * sum_abs_alpha * self.tau_d
            thr = 1e-5 + (2.0 ** self.d) * sum_abs_a * self.tau_d
            if dot < -thr:
                self.n_inconsistent += 1
                self.n_inconsistent_dot += 1
                return 1.0
            add = float(off + self.eps_out)
            self.max_off = max(self.max_off, float(off))
            self.max_tail_correction = max(self.max_tail_correction, add)
            Eq = max(dot, 0.0) + c_corr + add
        else:
            if dot < -1e-5:
                self.n_inconsistent += 1
                self.n_inconsistent_dot += 1
                return 1.0
            Eq = max(dot, 0.0) + c_corr
        return float(min(1.0, Eq))

    def endpoint(self, tau, alpha, start, iters=16):
        """Smallest beta in [start, 1] (up to bisection tolerance) with B_d(beta) <= alpha.

        If ``B_d(1) > alpha``, sets ``last_endpoint_fallback`` and returns the support
        end ``1.0`` (caller may replace with ``+inf`` when ``eps_out > alpha``).
        """
        lo, hi = start, 1.0
        self.warm = self.coarse
        if self.bound(tau, hi) > alpha:
            self.last_endpoint_fallback = True
            return 1.0
        self.last_endpoint_fallback = False
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            if self.bound(tau, mid) <= alpha:
                hi = mid
            else:
                lo = mid
        return hi


def poly_interval_t(tau, d, alpha, lp=None, eps_out=0.0, tau_d=0.0):
    """[lo, hi] in Y units with P(Y > hi) <= alpha and P(Y < lo) <= alpha.

    Also returns diagnostics ``{n_inconsistent, max_eps_smooth, eps_out,
    max_tail_correction_upper/lower, n_fallback_upper, n_guard_upper, ...}``.

    Truncation: upper tail uses ``(eps_out, tau_d)``. Lower tail reflects ``Y -> -Y``;
    ``eps_out`` is zero there (``1{Y <= beta} = 0`` off the right support for
    nonnegative costs), but the off-support Chebyshev term ``2^d ||alpha||_1 tau_d``
    is kept because the polynomial can be negative for ``Y > 1`` / ``Y' < -1``.
    If the upper search falls back to the support end and ``eps_out > alpha``, the
    upper limit is ``+inf`` (fallback guard). With ``eps_out=tau_d=0`` the numerical
    path matches the pre-truncation / finite code bit-identically.
    """
    eps_out = float(eps_out)
    tau_d = float(tau_d)
    tau = np.asarray(tau, dtype=float)[: d + 1]
    if lp is None:
        lp = TailLP(d, eps_out=eps_out, tau_d=tau_d)
    else:
        lp.eps_out = eps_out
        lp.tau_d = tau_d
    lp._reset_diag()

    flip = tau * (-1.0) ** np.arange(d + 1)
    mean_y = tau[1]
    # Upper tail: MGF truncation correction when active.
    hi = lp.endpoint(tau, alpha, mean_y)
    n_fallback_upper = int(lp.last_endpoint_fallback)
    n_guard_upper = 0
    if n_fallback_upper and eps_out > alpha:
        hi = float("inf")
        n_guard_upper = 1
    max_off_upper = float(lp.max_off)
    max_tail_correction_upper = float(lp.max_tail_correction)

    # Lower tail: reflect Y -> -Y. Off-support indicator mass on the right of the
    # flipped variable is P(Y < -1) = 0 for nonnegative costs (eps_out = 0), but
    # keep tau_d so the 2^d ||alpha||_1 tau_d term covers Y' < -1 / Y > 1.
    lp.max_tail_correction = 0.0
    lp.max_off = 0.0
    lp.eps_out = 0.0
    lp.tau_d = tau_d
    lo = -lp.endpoint(flip, alpha, -mean_y)
    n_fallback_lower = int(lp.last_endpoint_fallback)
    max_off_lower = float(lp.max_off)
    max_tail_correction_lower = float(lp.max_tail_correction)
    max_off = max(max_off_upper, max_off_lower)
    max_tail_correction = max(max_tail_correction_upper, max_tail_correction_lower)
    diag = {
        "n_inconsistent": int(lp.n_inconsistent),
        "n_inconsistent_ep2": int(lp.n_inconsistent_ep2),
        "n_inconsistent_dot": int(lp.n_inconsistent_dot),
        "n_ep2_neg": int(lp.n_ep2_neg),
        "min_ep2_raw": float(lp.min_ep2_raw),
        "max_tol2": float(lp.max_tol2),
        "max_eps_smooth": float(lp.max_eps_smooth),
        "eps_out": float(eps_out),
        "tau_d": float(tau_d),
        "max_off": float(max_off),
        "max_off_upper": float(max_off_upper),
        "max_off_lower": float(max_off_lower),
        "max_tail_correction": float(max_tail_correction),
        "max_tail_correction_upper": float(max_tail_correction_upper),
        "max_tail_correction_lower": float(max_tail_correction_lower),
        "n_fallback_upper": int(n_fallback_upper),
        "n_guard_upper": int(n_guard_upper),
        "n_fallback_lower": int(n_fallback_lower),
        "n_guard_lower": 0,
    }
    return lo, hi, diag

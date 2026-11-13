"""
Exact MGF of the accumulated quadratic cost for a continuous-state linear-Gaussian LQG loop.

The stacked vector X_T = (S_0, U_0, ..., S_{T-1}, U_{T-1}) is Gaussian. With a block-diagonal
quadratic form D_T, the cost is a Gaussian quadratic form whose MGF has a closed form.
This module implements that construction for multi-input plants and H not necessarily I.
"""

import numpy as np

from .plant import closed_loop


def _joint(plant, T):
    """Means of X_t=(S_t,Z_t) and Cov(X_t, X_s), plus V_t -> X_{t+1+k} impulse responses."""
    F, Gw, Gv, Hx, Hv = closed_loop(plant)
    n = plant.n
    m = np.zeros((T + 1, 2 * n))
    m[0, :n] = plant.s0
    m[0, n:] = plant.z0
    Sigma = np.zeros((T + 1, 2 * n, 2 * n))
    noise = Gw @ plant.SW @ Gw.T + Gv @ plant.SV @ Gv.T
    for t in range(T):
        m[t + 1] = F @ m[t]
        Sigma[t + 1] = F @ Sigma[t] @ F.T + noise
    cross = np.zeros((T + 1, T + 1, 2 * n, 2 * n))
    for s in range(T + 1):
        cross[s, s] = Sigma[s]
        acc = Sigma[s]
        for t in range(s + 1, T + 1):
            acc = F @ acc
            cross[t, s] = acc
            cross[s, t] = acc.T
    # Cov(V_t, X_{t+1+k}) = SV Gv^T (F^T)^k
    base = plant.SV @ Gv.T
    lag = np.zeros((T, plant.p, 2 * n))
    acc = base.copy()
    for k in range(T):
        lag[k] = acc
        acc = acc @ F.T
    return m, cross, Hx, Hv, lag


def stage_moments(plant, T):
    """Exact E[C_t] and per-step contributions for t = 0..T."""
    m, cross, Hx, Hv, _ = _joint(plant, T)
    n, nu = plant.n, plant.m
    Qc, Rc = plant.Qc, plant.Rc
    mean = np.zeros(T + 1)
    for t in range(T):
        ms = m[t, :n]
        Ss = cross[t, t, :n, :n]
        mu_u = Hx @ m[t]
        Su = Hx @ cross[t, t] @ Hx.T + Hv @ plant.SV @ Hv.T
        mean[t + 1] = mean[t] + float(
            ms @ Qc @ ms + np.trace(Qc @ Ss) + mu_u @ Rc @ mu_u + np.trace(Rc @ Su)
        )
    return mean


def quadratic_form(plant, T):
    """
    Build C_T = zeta' J zeta + 2 b' zeta + d with zeta ~ N(0,I).

    Returns J, b, d, and the factor L such that Cov(X_T) = L L^T (columns for positive
    eigenvalues only), together with the mean vector of X_T.
    """
    m, cross, Hx, Hv, lag = _joint(plant, T)
    n, nu = plant.n, plant.m
    block = n + nu
    d = block * T
    mu = np.zeros(d)
    Cov = np.zeros((d, d))
    for t in range(T):
        mu[block * t:block * t + n] = m[t, :n]
        mu[block * t + n:block * t + block] = Hx @ m[t]
    for t in range(T):
        for s in range(t, T):
            Bts = np.zeros((block, block))
            Bts[:n, :n] = cross[t, s, :n, :n]
            Bts[:n, n:] = cross[t, s, :n, :] @ Hx.T
            Bts[n:, :n] = Hx @ cross[t, s, :, :n]
            Bts[n:, n:] = Hx @ cross[t, s] @ Hx.T
            if t == s:
                Bts[n:, n:] = Bts[n:, n:] + Hv @ plant.SV @ Hv.T
            else:
                cov_vx = lag[s - t - 1]
                # Cov(U_t, S_s) gains Hv Cov(V_t, X_s)[:, :n]
                Bts[n:, :n] = Hx @ cross[t, s, :, :n] + (Hv @ cov_vx)[:, :n]
                Bts[n:, n:] = Hx @ cross[t, s] @ Hx.T + (Hv @ cov_vx) @ Hx.T
                # Cov(S_t, U_s) stays cross[t,s,:n,:] @ Hx.T (V_s independent of S_t)
            Cov[block * t:block * t + block, block * s:block * s + block] = Bts
            if s != t:
                Cov[block * s:block * s + block, block * t:block * t + block] = Bts.T
    Cov = 0.5 * (Cov + Cov.T)
    evals, evecs = np.linalg.eigh(Cov)
    thresh = 1e-10 * max(float(evals.max()), 1.0)
    keep = evals > thresh
    Lfac = evecs[:, keep] * np.sqrt(evals[keep])

    # D is block-diagonal (Qc, Rc, ...)
    # J = L' D L, b = L' D mu, d = mu' D mu
    Dmu = np.zeros(d)
    DL = np.zeros_like(Lfac)
    for t in range(T):
        sl = slice(block * t, block * t + n)
        ul = slice(block * t + n, block * t + block)
        Dmu[sl] = plant.Qc @ mu[sl]
        Dmu[ul] = plant.Rc @ mu[ul]
        DL[sl, :] = plant.Qc @ Lfac[sl, :]
        DL[ul, :] = plant.Rc @ Lfac[ul, :]
    J = Lfac.T @ DL
    b = Lfac.T @ Dmu
    d0 = float(mu @ Dmu)
    return J, b, d0, Lfac, mu, Cov


def moments_from_form(J, b, d0):
    mean = float(np.trace(J) + d0)
    var = float(2.0 * np.trace(J @ J) + 4.0 * b @ b)
    return mean, var


def log_mgf_from_form(J, b, d0, lams):
    """log M(lambda) with domain I - 2 lambda J > 0."""
    lams = np.asarray(lams, dtype=float)
    out = np.full(len(lams), np.nan)
    valid = np.zeros(len(lams), dtype=bool)
    if J.size == 0:
        # deterministic cost
        out[:] = lams * d0
        valid[:] = True
        return out, valid, np.inf
    evals = np.linalg.eigvalsh(J)
    lam_max = 0.5 / evals.max() if evals.max() > 0 else np.inf
    eye = np.eye(J.shape[0])
    # work in the eigenbasis of J for stability
    je, U = np.linalg.eigh(J)
    bt = U.T @ b
    for i, lam in enumerate(lams):
        if lam > 0 and lam >= lam_max - 1e-15:
            continue
        factors = 1.0 - 2.0 * lam * je
        if np.any(factors <= 0):
            continue
        logdet = float(np.sum(np.log(factors)))
        # b' (I - 2 lam J)^{-1} b
        quad = float(np.sum(bt * bt / factors))
        out[i] = lam * d0 - 0.5 * logdet + 2.0 * lam * lam * quad
        valid[i] = True
    return out, valid, float(lam_max)


def chernoff_bounds(J, b, d0, delta=0.025, n_grid=60):
    """
    Pointwise two-sided bounds with failure probability delta on each side.
    Returns L, U, and diagnostic dict.
    """
    evals = np.linalg.eigvalsh(J) if J.size else np.array([0.0])
    lam_max = 0.5 / evals.max() if evals.max() > 0 else 1e3
    # positive grid: (0, lam_max)
    pos = np.geomspace(1e-4, 0.99 * lam_max, n_grid) if np.isfinite(lam_max) else np.geomspace(1e-4, 10.0, n_grid)
    # negative grid: large |lambda| can make I - 2 lambda J fail when J has negative eigenvalues;
    # for PSD J every lambda < 0 is valid.
    neg = -np.geomspace(1e-4, 1e3, n_grid)
    lams = np.concatenate([neg, pos])
    logm, valid, _ = log_mgf_from_form(J, b, d0, lams)
    log_one_over = np.log(1.0 / delta)

    obj_u = np.where(valid & (lams > 0), (logm + log_one_over) / lams, np.inf)
    j_u = int(np.argmin(obj_u))
    U = float(obj_u[j_u])
    # refine around the best positive lambda
    if np.isfinite(U) and lams[j_u] > 0:
        lo = lams[j_u] / 1.5 if j_u > 0 else lams[j_u] / 2
        hi = min(lams[j_u] * 1.5, 0.999 * lam_max) if np.isfinite(lam_max) else lams[j_u] * 1.5
        refine = np.linspace(lo, hi, 40)
        lm, va, _ = log_mgf_from_form(J, b, d0, refine)
        ou = np.where(va, (lm + log_one_over) / refine, np.inf)
        jr = int(np.argmin(ou))
        if ou[jr] < U:
            U, j_u_lam = float(ou[jr]), float(refine[jr])
        else:
            j_u_lam = float(lams[j_u])
    else:
        j_u_lam = float("nan")

    obj_l = np.where(valid & (lams < 0), (logm + log_one_over) / lams, -np.inf)
    j_l = int(np.argmax(obj_l))
    L = float(obj_l[j_l])
    if np.isfinite(L) and lams[j_l] < 0:
        lo = lams[j_l] * 1.5
        hi = lams[j_l] / 1.5
        refine = np.linspace(lo, hi, 40)
        lm, va, _ = log_mgf_from_form(J, b, d0, refine)
        ol = np.where(va, (lm + log_one_over) / refine, -np.inf)
        jr = int(np.argmax(ol))
        if ol[jr] > L:
            L, j_l_lam = float(ol[jr]), float(refine[jr])
        else:
            j_l_lam = float(lams[j_l])
    else:
        j_l_lam = float("nan")

    return L, U, {
        "lambda_upper": j_u_lam,
        "lambda_lower": j_l_lam,
        "lambda_max_domain": float(lam_max),
        "delta": delta,
    }


def analyze_horizon(plant, T, delta=0.025):
    J, b, d0, *_ = quadratic_form(plant, T)
    mean, var = moments_from_form(J, b, d0)
    stage = stage_moments(plant, T)
    L, U, info = chernoff_bounds(J, b, d0, delta=delta)
    # derivative check at 0
    eps = 1e-4
    lm, va, _ = log_mgf_from_form(J, b, d0, np.array([-eps, 0.0, eps]))
    d1 = (lm[2] - lm[0]) / (2 * eps) if va[0] and va[2] else float("nan")
    d2 = (lm[2] - 2 * lm[1] + lm[0]) / eps ** 2 if va.all() else float("nan")
    return {
        "T": T,
        "mean": mean,
        "var": var,
        "stage_mean": float(stage[T]),
        "mean_gap_stage": abs(mean - float(stage[T])),
        "lower": L,
        "upper": U,
        "mgf_d1": d1,
        "mgf_d2": d2,
        "log_M0": float(lm[1]) if va[1] else float("nan"),
        **info,
        "J": J,
        "b": b,
        "d0": d0,
    }


def cumulants_quadratic_form(J, b, d0, degree):
    """Cumulants of C = ζ'Jζ + 2 b'ζ + d0 for ζ ~ N(0, I), orders 1..degree.

    Uses the eigen-decomposition of J: tr(J^n) = sum lam_i^n and
    b' J^{n-2} b = sum btilde_i^2 lam_i^{n-2} with btilde = U' b.
    """
    from math import factorial

    degree = int(degree)
    kappa = np.zeros(degree + 1)
    if J.size == 0:
        kappa[1] = float(d0)
        return kappa
    b = np.asarray(b, dtype=float).ravel()
    lam, U = np.linalg.eigh(np.asarray(J, dtype=float))
    bt2 = (U.T @ b) ** 2
    kappa[1] = float(np.sum(lam) + d0)
    # lam_pow accumulates lam^n; lam_pow_nm2 accumulates lam^{n-2} for the quadratic term.
    lam_pow = lam.copy()          # lam^1 before first n=2 step
    lam_pow_nm2 = np.ones_like(lam)  # lam^0 for n=2
    for n in range(2, degree + 1):
        lam_pow = lam_pow * lam  # now lam^n
        tr = float(np.sum(lam_pow))
        quad = float(np.sum(bt2 * lam_pow_nm2))
        kappa[n] = (2 ** (n - 1)) * factorial(n - 1) * (tr + n * quad)
        lam_pow_nm2 = lam_pow_nm2 * lam  # become lam^{n-1}; next iter needs lam^{(n+1)-2}=lam^{n-1}
    return kappa


def cumulants_quadratic_form_matrix_powers(J, b, d0, degree):
    """Reference cumulants via repeated matrix products (for tests)."""
    from math import factorial

    degree = int(degree)
    kappa = np.zeros(degree + 1)
    if J.size == 0:
        kappa[1] = float(d0)
        return kappa
    b = np.asarray(b, dtype=float)
    J = np.asarray(J, dtype=float)
    kappa[1] = float(np.trace(J) + d0)
    powers = [None, J.copy()]
    for n in range(2, degree + 1):
        powers.append(J @ powers[-1])
        tr = float(np.trace(powers[n]))
        quad = float(b @ b) if n == 2 else float(b @ powers[n - 2] @ b)
        kappa[n] = (2 ** (n - 1)) * factorial(n - 1) * (tr + n * quad)
    return kappa


def raw_moments_from_cumulants(kappa):
    """Raw moments μ'_0..μ'_d from cumulants κ_1..κ_d (κ_0 unused)."""
    from math import comb

    d = len(kappa) - 1
    mu = np.zeros(d + 1)
    mu[0] = 1.0
    for n in range(1, d + 1):
        mu[n] = sum(comb(n - 1, m) * kappa[n - m] * mu[m] for m in range(n))
    return mu


def truncation_mass(J, b, d0, l, u, n_grid=400):
    """
    Chernoff upper bounds on P(G > u) and P(G < l) via the closed-form MGF.

    Returns ``eps_plus``, ``eps_minus``, ``log_eps_plus``, ``log_eps_minus``.
    If ``l <= 0``, the lower off-support mass is taken as 0 (nonnegative costs).
    """
    lams_pos_probe = np.array([1e-8])
    _, _, lam_max = log_mgf_from_form(J, b, d0, lams_pos_probe)
    if not np.isfinite(lam_max) or lam_max <= 0:
        # No positive tilt available; fall back to trivial bound 1 for the right tail.
        eps_plus, log_eps_plus = 1.0, 0.0
    else:
        etas = np.geomspace(1e-6, 0.999 * lam_max, int(n_grid))
        logm, valid, _ = log_mgf_from_form(J, b, d0, etas)
        # log P(G>u) <= -eta u + log M(eta)
        log_bound = np.where(valid, -etas * float(u) + logm, np.inf)
        j = int(np.argmin(log_bound))
        log_eps_plus = float(log_bound[j]) if np.isfinite(log_bound[j]) else 0.0
        eps_plus = float(np.exp(min(log_eps_plus, 0.0)))

    if float(l) <= 0.0:
        eps_minus, log_eps_minus = 0.0, -np.inf
    else:
        # Negative tilts: for PSD J every eta < 0 is in the domain.
        etas_neg = -np.geomspace(1e-6, 1e3, int(n_grid))
        logm, valid, _ = log_mgf_from_form(J, b, d0, etas_neg)
        log_bound = np.where(valid, -etas_neg * float(l) + logm, np.inf)
        j = int(np.argmin(log_bound))
        log_eps_minus = float(log_bound[j]) if np.isfinite(log_bound[j]) else 0.0
        eps_minus = float(np.exp(min(log_eps_minus, 0.0)))

    return eps_plus, eps_minus, log_eps_plus, log_eps_minus


def tail_terms(J, b, d0, c, h, d, n_grid=600):
    """
    MGF bounds on the right-tail mass and truncated moment of Y = (G - c)/h.

    On the grid ``lam = geomspace(1e-3, 0.999 * h * etabar, n_grid)`` with
    ``etabar = 1/(2 λ_max(J))`` and ``log M_Y(lam) = log M_G(lam/h) - lam*c/h``:

      log_eps = min_lam [log M_Y(lam) - lam]                         # >= log P(Y > 1)
      log_tau = min_lam [log M_Y(lam) - (lam - th) + d log(d/(e th))]
                with th = min(d, lam)                                  # >= log E[Y^d 1{Y>1}]

    Returns ``eps = exp(log_eps)``, ``tau = exp(log_tau)`` (computed in logs).
    """
    d = int(d)
    c = float(c)
    h = float(h)
    if d < 0 or h <= 0.0:
        return 0.0, 0.0, -np.inf, -np.inf, {"lam_hi": 0.0, "etabar": 0.0}

    if J.size == 0:
        etabar = np.inf
    else:
        lam_max_J = float(np.linalg.eigvalsh(np.asarray(J, dtype=float)).max())
        etabar = (0.5 / lam_max_J) if lam_max_J > 0.0 else np.inf

    lam_hi = 0.999 * h * etabar
    if (not np.isfinite(lam_hi)) or lam_hi <= 1e-3:
        # No usable positive tilt for Y; return trivial unit bounds.
        return 1.0, 1.0, 0.0, 0.0, {"lam_hi": float(lam_hi), "etabar": float(etabar)}

    lams = np.geomspace(1e-3, float(lam_hi), int(n_grid))
    etas = lams / h
    logm_G, valid, _ = log_mgf_from_form(J, b, d0, etas)
    logm_Y = np.full(len(lams), np.nan)
    logm_Y[valid] = logm_G[valid] - lams[valid] * (c / h)

    obj_eps = np.where(valid, logm_Y - lams, np.inf)
    log_eps = float(np.min(obj_eps)) if np.any(valid) else 0.0

    th = np.minimum(float(d), lams)
    th = np.maximum(th, 1e-300)
    obj_tau = np.where(
        valid,
        logm_Y - (lams - th) + float(d) * np.log(float(d) / (np.e * th)),
        np.inf,
    )
    log_tau = float(np.min(obj_tau)) if np.any(valid) else 0.0

    # P <= 1; truncated moment may exceed 1.
    eps = float(np.exp(min(log_eps, 0.0))) if np.isfinite(log_eps) else 1.0
    if not np.isfinite(log_tau):
        tau = float("inf")
    else:
        tau = float(np.exp(min(log_tau, 700.0)))  # overflow guard

    return eps, tau, log_eps, log_tau, {
        "lam_hi": float(lam_hi),
        "etabar": float(etabar),
        "n_valid": int(np.sum(valid)),
    }


def poly_bounds_from_form(J, b, d0, degree=16, alpha=0.025, support_mult=40.0):
    """
    Degree-d polynomial TailLP on the continuous quadratic-form cost.

    The LP is solved on the truncated support ``S = [0, mean + k σ]``
    (``k = support_mult``). Right-tail mass ``eps >= P(Y > 1)`` and
    ``tau_d >= E[Y^d 1{Y>1}]`` are bounded from the closed-form MGF (no moments
    above order d). The upper TailLP value is inflated by
    ``2^d ||alpha||_1 tau_d + eps``; the lower (flipped) TailLP keeps the
    ``2^d ||alpha||_1 tau_d`` off-support term with ``eps = 0``. Moments of
    ``Y = (C - c)/h`` are built from scaled eigen-cumulants up to order d only.
    If the upper search falls back and ``eps > alpha``, the upper endpoint is
    ``+inf``.
    """
    from . import polybound as pb

    degree = int(degree)
    kappa = cumulants_quadratic_form(J, b, d0, degree)
    mean = float(kappa[1])
    var = float(max(kappa[2], 0.0))
    std = float(np.sqrt(var))
    Lsup = 0.0
    Usup = max(mean + support_mult * std, Lsup + 1e-9)
    c = 0.5 * (Lsup + Usup)
    h = 0.5 * (Usup - Lsup)

    ky = np.zeros(degree + 1)
    ky[1] = (kappa[1] - c) / h
    for n in range(2, degree + 1):
        ky[n] = kappa[n] / (h ** n)
    tau = pb.cheb_moments(raw_moments_from_cumulants(ky))

    eps_out, tau_d, log_eps, log_tau, tail_diag = tail_terms(J, b, d0, c, h, degree)
    # Keep Chernoff diagnostics for the report (right tail mass; left mass is 0).
    eps_plus, eps_minus, log_eps_plus, log_eps_minus = truncation_mass(J, b, d0, Lsup, Usup)
    log10_eps_out = float(log_eps / np.log(10.0)) if np.isfinite(log_eps) else 0.0

    y_lo, y_hi, diag = pb.poly_interval_t(
        tau, degree, alpha, eps_out=float(eps_out), tau_d=float(tau_d)
    )
    diag = {
        **diag,
        "eps_plus": float(eps_plus),
        "eps_minus": float(eps_minus),
        "log_eps_plus": float(log_eps_plus),
        "log_eps_minus": float(log_eps_minus),
        "log10_eps_out": float(log10_eps_out),
        "log_eps_Y": float(log_eps),
        "log_tau_d": float(log_tau) if np.isfinite(log_tau) else None,
        "tau_d": float(tau_d) if np.isfinite(tau_d) else float("inf"),
        "support_L": float(Lsup),
        "support_U": float(Usup),
        "tail_diag": tail_diag,
    }
    raw_lo = c + h * y_lo
    raw_hi = c + h * y_hi
    lo = float(np.clip(raw_lo, Lsup, Usup)) if np.isfinite(raw_lo) else float(raw_lo)
    hi = float(np.clip(raw_hi, Lsup, Usup)) if np.isfinite(raw_hi) else float(raw_hi)
    return lo, hi, Lsup, Usup, diag

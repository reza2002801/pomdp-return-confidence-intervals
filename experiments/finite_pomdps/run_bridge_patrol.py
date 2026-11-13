"""
Bridge-patrol finite POMDP: MGF, degree-16 poly, Azuma/Freedman, exact 2G_t distribution,
and paper figures. Writes all artifacts under data/runs/finite_pomdps/bridge_patrol/.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from returnbounds import io as save_artifacts
from returnbounds import martingale as mg
from returnbounds import mgf
from returnbounds import moments as paper_moments
from returnbounds import plotting as figures
from returnbounds.build_quantized import simulate as finite_simulate
from returnbounds.plotting import HIST, MGF, POLY, _style
from returnbounds.profiling import StageTimer

from experiments.finite_pomdps.bridge_patrol import (
    ACTION_NAMES,
    BRIDGE,
    GOAL,
    N_COLS,
    N_ROWS,
    START,
    WATER,
    Z_RISKY,
    Z_SAFE,
    action_for,
    build_bridge_model,
    exact_cost_distribution,
    pmf_quantile,
    pmf_tail,
)

OUT_DIR = _ROOT / "data" / "runs" / "finite_pomdps" / "bridge_patrol"
FIG_DIR = _ROOT / "figures"

AZUMA = {"color": "#1f77b4", "ls": ":", "lw": 2.0}
FREED = {"color": "#d62728", "ls": "-.", "lw": 2.2}

DELTA = 0.025
T_MAX = 100
DEGREE = 16
N_TRAJ = 20000
SEED = 20261109


def _dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def _nice_ceil(v):
    v = max(float(v), 1e-9)
    k = np.floor(np.log10(v))
    for m in (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if m * 10 ** k >= v:
            return float(m * 10 ** k)
    return float(v)


def _asinh_axis(ax, axis: str, half_width: float, lim: float):
    """Arcsinh axis with sparse readable ticks (no overlapping ±10 / 0 labels)."""
    width = _nice_ceil(half_width)
    lim = float(max(lim, 1.5 * width))
    # Sparse outer decades only; avoid ±width/2 which crowds 0.
    outer = []
    k0 = int(np.floor(np.log10(width))) + 1
    k1 = int(np.floor(np.log10(lim)))
    for k in range(k0, k1 + 1):
        for m in (1, 3):
            val = m * 10 ** k
            if 1.6 * width <= val <= lim * 1.01:
                outer.append(val)
    if not outer or outer[-1] < 0.9 * lim:
        outer.append(_nice_ceil(lim))
    ticks = sorted({-v for v in outer} | {0.0} | set(outer))
    getattr(ax, f"set_{axis}scale")("asinh", linear_width=width)
    getattr(ax, f"set_{axis}ticks")(ticks)

    def _fmt(v):
        v = float(v)
        if v == 0:
            return "0"
        av = abs(v)
        if av >= 100:
            return f"{v:.0f}"
        if av >= 10:
            return f"{v:.0f}"
        return f"{v:g}"

    getattr(ax, f"set_{axis}ticklabels")([_fmt(v) for v in ticks])
    if axis == "y":
        ax.set_ylim(-lim, lim)
    else:
        ax.set_xlim(-lim, lim)
    ax.minorticks_off()
    ax.tick_params(axis=axis, labelsize=8.5, pad=3)


def draw_grid(meta, out_pdf: Path, out_png: Path):
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    for r in range(N_ROWS):
        for c in range(N_COLS):
            cell = (r, c)
            y = N_ROWS - 1 - r
            if cell in WATER:
                color = "#9ecae1"
            elif cell in BRIDGE:
                color = "#fdae6b"
            elif cell == START:
                color = "#3182bd"
            elif cell == GOAL:
                color = "#31a354"
            else:
                color = "#f7f7f7"
            ax.add_patch(
                Rectangle((c, y), 1, 1, facecolor=color, edgecolor="#444444", linewidth=0.7)
            )
            if cell in WATER:
                ax.text(c + 0.5, y + 0.5, "≈", ha="center", va="center", color="#2171b5", fontsize=9)
    ax.text(START[1] + 0.5, N_ROWS - 1 - START[0] + 0.5, "S", ha="center", va="center",
            color="white", fontsize=12, fontweight="bold")
    ax.text(GOAL[1] + 0.5, N_ROWS - 1 - GOAL[0] + 0.5, "G", ha="center", va="center",
            color="white", fontsize=12, fontweight="bold")

    # Nominal routes (illustrative only; slips can leave these corridors).
    def cells_along(z, steps=20):
        cell = START
        out = [cell]
        for _ in range(steps):
            a = action_for(meta, cell, z)
            dr, dc = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}[a]
            cell = (cell[0] + dr, cell[1] + dc)
            out.append(cell)
            if cell == GOAL:
                break
        return out

    def draw_route(cells, color, lw, label):
        xs = [c + 0.5 for _, c in cells]
        ys = [N_ROWS - 1 - r + 0.5 for r, _ in cells]
        ax.plot(xs, ys, color=color, lw=lw, solid_capstyle="round", label=label, zorder=5)
        ax.annotate(
            "", xy=(xs[-1], ys[-1]), xytext=(xs[-2], ys[-2]),
            arrowprops=dict(arrowstyle="->", color=color, lw=lw), zorder=6,
        )

    draw_route(cells_along(Z_RISKY), "#e6550d", 2.4, f"Risky bridge ({meta.risky_len} steps)")
    draw_route(cells_along(Z_SAFE), "#3182bd", 2.2, f"Safe detour ({meta.safe_len} steps)")

    ax.set_xlim(0, N_COLS)
    ax.set_ylim(0, N_ROWS)
    ax.set_aspect("equal")
    ax.set_xticks(range(N_COLS + 1))
    ax.set_yticks(range(N_ROWS + 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(
        "Bridge-patrol gridworld (custom): known position, hidden bridge mode.\n"
        "Drawn routes are nominal shortest paths; slips are unrestricted.",
        fontsize=10,
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False, fontsize=9)
    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_asinh_compare(C, mean, mgf_L, mgf_U, poly_L, poly_U, bands, out_stem: Path):
    """
    Side-by-side traj + hist on arcsinh axes so Azuma fits without flattening the
    bulk. Sparse ticks, shared legend above the panels, high DPI.
    """
    T = mean.shape[0] - 1
    t = np.arange(T + 1)
    rng = np.random.default_rng(0)
    n_draw = min(350, C.shape[0])
    idx = rng.choice(C.shape[0], size=n_draw, replace=False)
    centered = C[idx] - mean
    q_lo = np.quantile(C, 0.025, axis=0) - mean
    q_hi = np.quantile(C, 0.975, axis=0) - mean

    mL, mU = mgf_L - mean, mgf_U - mean
    pL, pU = poly_L - mean, poly_U - mean
    fL, fU = bands["freedman_L"] - mean, bands["freedman_U"] - mean
    aL, aU = bands["azuma_L"] - mean, bands["azuma_U"] - mean

    series = [
        ("MGF", mL, mU, MGF, 6),
        ("polynomial (d=16)", pL, pU, POLY, 5),
        ("Freedman", fL, fU, FREED, 4),
        ("Azuma", aL, aU, AZUMA, 3),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.78, bottom=0.12, wspace=0.22)

    ax = axes[0]
    ax.plot(t, centered.T, color="#4c566a", alpha=0.55, lw=0.35, rasterized=True, zorder=1)
    h_mean, = ax.plot(t, np.zeros_like(t), color="#2e3440", lw=1.8, zorder=7)
    h_emp, = ax.plot(t, q_lo, color="#2e3440", ls=":", lw=1.5, zorder=7)
    ax.plot(t, q_hi, color="#2e3440", ls=":", lw=1.5, zorder=7)
    handles = [h_mean, h_emp]
    labels = ["exact mean", "empirical 2.5%/97.5%"]
    for lab, lo, hi, sty, zo in series:
        h, = ax.plot(t[1:], hi[1:], color=sty["color"], ls=sty["ls"], lw=sty["lw"], zorder=zo)
        ax.plot(t[1:], lo[1:], color=sty["color"], ls=sty["ls"], lw=sty["lw"], zorder=zo)
        handles.append(h)
        labels.append(lab)

    az_lim = 1.05 * float(np.nanmax(np.abs(np.concatenate([aL[1:], aU[1:]]))))
    lin_w = float(np.nanmax(np.abs(np.concatenate([mL[1:], mU[1:], pL[1:], pU[1:]]))))
    _asinh_axis(ax, "y", lin_w, az_lim)
    ax.set_xlim(0, T)
    ax.set_xlabel(r"time step $t$", fontsize=10)
    ax.set_ylabel(r"$G_t-\mathbb{E}[G_t]$", fontsize=10)
    ax.set_title("Centred trajectories", fontsize=11, pad=6)
    _style(ax)

    ax = axes[1]
    x = C[:, T] - mean[T]
    ax.hist(x, bins=55, weights=np.full(len(x), 1.0 / len(x)), zorder=1, **HIST)
    # Poly under MGF so nearly coincident lower endpoints stay distinguishable.
    vlines = [
        (0.0, "#2e3440", "-", 1.8, 0.88, 4),
        (float(np.quantile(x, 0.025)), "#2e3440", ":", 1.5, 0.88, 4),
        (float(np.quantile(x, 0.975)), "#2e3440", ":", 1.5, 0.88, 4),
        (float(pL[T]), POLY["color"], POLY["ls"], POLY["lw"], 0.80, 5),
        (float(pU[T]), POLY["color"], POLY["ls"], POLY["lw"], 0.80, 5),
        (float(mL[T]), MGF["color"], MGF["ls"], MGF["lw"], 0.94, 6),
        (float(mU[T]), MGF["color"], MGF["ls"], MGF["lw"], 0.94, 6),
        (float(fL[T]), FREED["color"], FREED["ls"], FREED["lw"], 0.88, 3),
        (float(fU[T]), FREED["color"], FREED["ls"], FREED["lw"], 0.88, 3),
        (float(aL[T]), AZUMA["color"], AZUMA["ls"], AZUMA["lw"], 0.88, 2),
        (float(aU[T]), AZUMA["color"], AZUMA["ls"], AZUMA["lw"], 0.88, 2),
    ]
    for v, c, ls, lw, ymax, zo in vlines:
        ax.axvline(v, ymax=ymax, color=c, ls=ls, lw=lw, zorder=zo)
    hist_lim = 1.05 * max(abs(float(aL[T])), abs(float(aU[T])))
    _asinh_axis(ax, "x", max(abs(float(mL[T])), abs(float(mU[T])), abs(float(pU[T]))), hist_lim)
    ax.set_ylim(0, ax.get_ylim()[1] / 0.82)
    ax.set_xlabel(rf"$G_{{{T}}}-\mathbb{{E}}[G_{{{T}}}]$", fontsize=10)
    ax.set_ylabel("fraction of trajectories", fontsize=10)
    ax.set_title(rf"Terminal histogram ($T={T}$)", fontsize=11, pad=6)
    _style(ax)

    fig.legend(
        handles, labels,
        loc="upper center", bbox_to_anchor=(0.5, 0.99),
        ncol=3, frameon=False, fontsize=9.5,
        handlelength=2.6, columnspacing=1.4, labelspacing=0.6,
    )
    fig.suptitle("Bridge-patrol: MGF / poly vs Freedman / Azuma", fontsize=12, y=1.04)

    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(str(out_stem) + ".pdf"), bbox_inches="tight", dpi=450)
    fig.savefig(Path(str(out_stem) + ".png"), dpi=450, bbox_inches="tight")
    plt.close(fig)


def validate_against_pdf(fm, log=print):
    """Separate δ=0.05 run for PDF numerical comparison (not used in the paper)."""
    T = 100
    delta = 0.05
    op = mgf.TiltedOperator(fm)
    mean, var, _ = mgf.exact_moments(op, fm.x0, T)
    reach = mgf.reachable_sets(op, fm.x0, T)
    lam_cfg = {"min": 0.01, "max": 100.0, "points_per_decade": 8, "max_expansions": 2}
    lams, log_m, valid, expansions, _ = mgf.bounds_with_expansion(
        op, fm.x0, T, [T], [delta], lam_cfg, reach, log=lambda *a, **k: None
    )
    upper = np.full(T + 1, np.nan)
    for t in range(1, T + 1):
        upper[t], _ = mgf.chernoff(log_m[t], valid[t], lams, delta)
    lower, *_ = mgf.lower_bounds(
        op, fm.x0, T, reach, lams, delta, [T], per_decade=8, max_expansions=2
    )
    dist = exact_cost_distribution(fm, T)
    pmf = dist["half_pmf"][T]
    q05, q95 = pmf_quantile(pmf, 0.05), pmf_quantile(pmf, 0.95)
    report = {
        "delta": delta,
        "mean": float(mean[T]),
        "pdf_mean": 121.74,
        "mean_gap": float(mean[T] - 121.74),
        "centered_q05_q95": [float(q05 - mean[T]), float(q95 - mean[T])],
        "pdf_centered_percentiles": [-19.24, 27.26],
        "centered_chernoff": [float(lower[T] - mean[T]), float(upper[T] - mean[T])],
        "pdf_centered_chernoff": [-21.03, 47.71],
        "lambda_expansions": expansions,
    }
    log(
        f"  PDF validation (δ=0.05): mean={report['mean']:.4f} (pdf 121.74, gap {report['mean_gap']:.4f}); "
        f"cent.q={report['centered_q05_q95']}; cent.Chernoff={report['centered_chernoff']}"
    )
    return report


def run(log=print):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    timer = StageTimer()

    with timer.stage("build"):
        fm, meta = build_bridge_model(log=log)
    save_artifacts.save_finite_model(fm, OUT_DIR / "finite_model.npz")
    _dump(OUT_DIR / "finite_pilot.json", fm.pilot)
    draw_grid(meta, FIG_DIR / "bridge_patrol_grid.pdf", FIG_DIR / "bridge_patrol_grid.png")
    draw_grid(meta, OUT_DIR / "bridge_patrol_grid.pdf", OUT_DIR / "bridge_patrol_grid.png")

    with timer.stage("pdf_validation"):
        pdf_val = validate_against_pdf(fm, log=log)
    _dump(OUT_DIR / "pdf_validation_delta05.json", pdf_val)

    T, delta, degree = T_MAX, DELTA, DEGREE
    horizons = [25, 50, 100]

    with timer.stage("mgf"):
        op = mgf.TiltedOperator(fm)
        reach = mgf.reachable_sets(op, fm.x0, T)
        mgf.check_stochastic(op, reach, T)
        lam_cfg = {"min": 0.01, "max": 100.0, "points_per_decade": 8, "max_expansions": 2}
        lams, log_m, valid, expansions, mgf_seconds = mgf.bounds_with_expansion(
            op, fm.x0, T, horizons, [delta], lam_cfg, reach, log=log
        )
        upper = np.full(T + 1, np.nan)
        for t in range(1, T + 1):
            upper[t], _ = mgf.chernoff(log_m[t], valid[t], lams, delta)
        lower, lams_neg, log_m_neg, valid_neg = mgf.lower_bounds(
            op, fm.x0, T, reach, lams, delta, horizons, per_decade=8, max_expansions=2
        )
        mean, var, occ = mgf.exact_moments(op, fm.x0, T)

    with timer.stage("poly"):
        poly_lower, poly_upper, Lsup, Usup, moments_0to2, poly_diagnostics = (
            paper_moments.poly_intervals(
                fm, T, degree, alpha=delta, times=list(range(1, T + 1)), log=log
            )
        )

    with timer.stage("exact_dist"):
        dist = exact_cost_distribution(fm, T)
        pmf_T = dist["half_pmf"][T]
        exact_q025 = pmf_quantile(pmf_T, 0.025)
        exact_q975 = pmf_quantile(pmf_T, 0.975)
        exact_below_mgf, exact_above_mgf = pmf_tail(pmf_T, lower[T], upper[T])
        exact_below_poly, exact_above_poly = pmf_tail(pmf_T, poly_lower[T], poly_upper[T])

    with timer.stage("simulate"):
        C = finite_simulate(fm, N_TRAJ, T, SEED)

    with timer.stage("martingale"):
        bands = mg.bands(fm, T, delta)

    gT = C[:, T]
    summary = {
        "method": "bridge_patrol (X,M,Z) MGF + poly + Azuma/Freedman + exact 2G distribution",
        "name": "bridge_patrol",
        "T": T,
        "delta": delta,
        "finite_states": fm.n_states,
        "finite_edges": fm.n_edges,
        "poly_degree": degree,
        "exact_mean": float(mean[T]),
        "exact_var": float(var[T]),
        "exact_std": float(np.sqrt(max(var[T], 0.0))),
        "dist_mean": float(dist["mean"][T]),
        "sim_mean": float(gT.mean()),
        "support_L": float(Lsup[T]),
        "support_U": float(Usup[T]),
        "lower": float(lower[T]),
        "upper": float(upper[T]),
        "poly_lower": float(poly_lower[T]),
        "poly_upper": float(poly_upper[T]),
        "mgf_width": float(upper[T] - lower[T]),
        "poly_width": float(poly_upper[T] - poly_lower[T]),
        "empirical_q025": float(np.quantile(gT, 0.025)),
        "empirical_q975": float(np.quantile(gT, 0.975)),
        "exact_q025": float(exact_q025),
        "exact_q975": float(exact_q975),
        "frac_below_mgf": float(np.mean(gT < lower[T])),
        "frac_above_mgf": float(np.mean(gT > upper[T])),
        "frac_below_poly": float(np.mean(gT < poly_lower[T])),
        "frac_above_poly": float(np.mean(gT > poly_upper[T])),
        "exact_frac_below_mgf": float(exact_below_mgf),
        "exact_frac_above_mgf": float(exact_above_mgf),
        "exact_frac_below_poly": float(exact_below_poly),
        "exact_frac_above_poly": float(exact_above_poly),
        "poly_diagnostics": poly_diagnostics,
        "mgf_seconds": mgf_seconds,
        "lambda_expansions": expansions,
        "pilot": fm.pilot,
        "pdf_validation_delta05": pdf_val,
        "martingale": {
            "Azuma": {
                "L": float(bands["azuma_L"][T]),
                "U": float(bands["azuma_U"][T]),
                "width": float(bands["azuma_U"][T] - bands["azuma_L"][T]),
                "frac_below": float(np.mean(gT < bands["azuma_L"][T])),
                "frac_above": float(np.mean(gT > bands["azuma_U"][T])),
            },
            "Freedman": {
                "L": float(bands["freedman_L"][T]),
                "U": float(bands["freedman_U"][T]),
                "width": float(bands["freedman_U"][T] - bands["freedman_L"][T]),
                "frac_below": float(np.mean(gT < bands["freedman_L"][T])),
                "frac_above": float(np.mean(gT > bands["freedman_U"][T])),
            },
        },
        "profiling": timer.as_dict(),
        "wall_seconds": timer.wall_seconds(),
    }
    az_w = summary["martingale"]["Azuma"]["width"]
    fr_w = summary["martingale"]["Freedman"]["width"]
    summary["azuma_over_mgf"] = az_w / summary["mgf_width"]
    summary["freedman_over_mgf"] = fr_w / summary["mgf_width"]

    _dump(OUT_DIR / "summary.json", summary)
    _dump(OUT_DIR / "martingale.json", {
        "mean": float(bands["mean"][T]),
        "methods": {
            "Azuma": summary["martingale"]["Azuma"],
            "Freedman": summary["martingale"]["Freedman"],
            "MGF": {
                "L": summary["lower"], "U": summary["upper"], "width": summary["mgf_width"],
                "frac_below": summary["frac_below_mgf"], "frac_above": summary["frac_above_mgf"],
            },
            "polynomial": {
                "L": summary["poly_lower"], "U": summary["poly_upper"], "width": summary["poly_width"],
                "frac_below": summary["frac_below_poly"], "frac_above": summary["frac_above_poly"],
            },
        },
    })

    rows = ["t,mean,var,mgf_L,mgf_U,poly_L,poly_U,support_L,support_U,azuma_L,azuma_U,freedman_L,freedman_U"]
    for t in range(1, T + 1):
        rows.append(",".join(str(x) for x in [
            t, mean[t], var[t], lower[t], upper[t], poly_lower[t], poly_upper[t],
            Lsup[t], Usup[t], bands["azuma_L"][t], bands["azuma_U"][t],
            bands["freedman_L"][t], bands["freedman_U"][t],
        ]))
    (OUT_DIR / "results.csv").write_text("\n".join(rows) + "\n")

    save_artifacts.save_run_arrays(
        OUT_DIR,
        mean=mean, var=var, lower=lower, upper=upper,
        poly_lower=poly_lower, poly_upper=poly_upper,
        Lsup=Lsup, Usup=Usup, moments_0to2=moments_0to2, C=C,
        lams=lams, log_m=log_m, valid=valid,
        lams_neg=lams_neg, log_m_neg=log_m_neg, valid_neg=valid_neg,
    )
    np.savez_compressed(
        OUT_DIR / "martingale_bands.npz",
        azuma_L=bands["azuma_L"], azuma_U=bands["azuma_U"],
        freedman_L=bands["freedman_L"], freedman_U=bands["freedman_U"],
        mean=bands["mean"],
    )

    figures.cost_trajectories(
        C, mean, lower, upper, OUT_DIR / "cost_trajectories_95pct.png",
        "bridge_patrol", n_draw=800, poly_lower=poly_lower, poly_upper=poly_upper,
    )
    figures.cost_trajectories(
        C, mean, lower, upper, OUT_DIR / "cost_trajectories_95pct.pdf",
        "bridge_patrol", n_draw=800, poly_lower=poly_lower, poly_upper=poly_upper,
    )
    figures.histogram(
        gT, mean[T], lower[T], upper[T], T, OUT_DIR / f"histogram_T{T}.png",
        "bridge_patrol", poly_lower=float(poly_lower[T]), poly_upper=float(poly_upper[T]),
    )
    figures.histogram(
        gT, mean[T], lower[T], upper[T], T, OUT_DIR / f"histogram_T{T}.pdf",
        "bridge_patrol", poly_lower=float(poly_lower[T]), poly_upper=float(poly_upper[T]),
    )
    plot_asinh_compare(
        C, mean, lower, upper, poly_lower, poly_upper, bands,
        FIG_DIR / "bridge_azuma_freedman_sinh",
    )
    plot_asinh_compare(
        C, mean, lower, upper, poly_lower, poly_upper, bands,
        OUT_DIR / "bridge_azuma_freedman_sinh",
    )

    # Width bar chart
    mart = json.loads((OUT_DIR / "martingale.json").read_text())
    methods = ["MGF", "polynomial", "Freedman", "Azuma"]
    colors = [MGF["color"], POLY["color"], FREED["color"], AZUMA["color"]]
    widths = [mart["methods"][m]["width"] for m in methods]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.bar(methods, widths, color=colors, edgecolor="#333", linewidth=0.6)
    ax.set_ylabel("interval width at $T=100$")
    ax.set_title("Bridge-patrol: terminal interval widths")
    _style(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "bridge_patrol_widths.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "bridge_patrol_widths.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT_DIR / "bridge_patrol_widths.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "bridge_patrol_widths.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    for name in ("cost_trajectories_95pct.pdf", "cost_trajectories_95pct.png",
                 "histogram_T100.pdf", "histogram_T100.png"):
        shutil.copy(OUT_DIR / name, FIG_DIR / f"bridge_patrol_{name}")

    report = [
        "# Bridge-patrol report",
        "",
        f"- Reachable states: {fm.n_states} / 232 raw",
        f"- Mean@100: {summary['exact_mean']:.4f}",
        f"- MGF: [{summary['lower']:.2f}, {summary['upper']:.2f}] width {summary['mgf_width']:.2f}",
        f"- Poly: [{summary['poly_lower']:.2f}, {summary['poly_upper']:.2f}] width {summary['poly_width']:.2f}",
        f"- Freedman width / MGF: {summary['freedman_over_mgf']:.2f}",
        f"- Azuma width / MGF: {summary['azuma_over_mgf']:.2f}",
        f"- PDF δ=0.05 validation: {json.dumps(pdf_val, indent=2)}",
        f"- Safe bridge convention: {fm.pilot['safe_bridge_convention']}",
        f"- Tie-break: {fm.pilot['tie_break']}",
    ]
    (OUT_DIR / "REPORT.md").write_text("\n".join(report) + "\n")
    log(
        f"  done bridge_patrol: mean={summary['exact_mean']:.3f} "
        f"MGF w={summary['mgf_width']:.2f} poly w={summary['poly_width']:.2f} "
        f"Freedman/MGF={summary['freedman_over_mgf']:.1f}x Azuma/MGF={summary['azuma_over_mgf']:.1f}x"
    )
    return summary


if __name__ == "__main__":
    run()

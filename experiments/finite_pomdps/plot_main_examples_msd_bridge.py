"""Build main-paper MSD | Bridge-patrol paired traj|hist figure."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures"
# MGF drawn after poly / with higher z-order so overlapping lower endpoints stay visible.
MGF = dict(color="#2ca02c", ls=(0, (5, 1.8)), lw=2.35)
POLY = dict(color="#9467bd", ls="-", lw=1.85)
HIST = dict(color="#a3b1c6", edgecolor="#4c566a", linewidth=0.35)
PAD = 0.13  # fractional pad beyond observed range (was 0.08; +5pp)
SAVE_DPI = 450


def _style(ax):
    ax.grid(True, alpha=0.28, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# Shared paper legend wording (main traj|hist figures).
LEGEND_LABELS = [
    "exact mean",
    r"Emp. 2.5% and 97.5%",
    "MGF-based CIs",
    "Moment-based CIs",
]


def _declutter_pair_xticks(ax, axh):
    """Drop traj tick at t=T and hist's leading 0 so seam labels do not collide."""
    T = float(ax.get_xlim()[1])
    ticks = [float(x) for x in ax.get_xticks() if -1e-9 <= float(x) <= T + 1e-9]
    if len(ticks) >= 2 and abs(ticks[-1] - T) <= 1e-6 * max(T, 1.0):
        ax.set_xticks(ticks[:-1])
    hticks = [float(x) for x in axh.get_xticks() if float(x) >= -1e-12]
    if len(hticks) >= 2 and abs(hticks[0]) <= 1e-12:
        axh.set_xticks(hticks[1:])


def _padded_lim(lo, hi, pad=PAD):
    lo, hi = float(lo), float(hi)
    span = max(hi - lo, 1e-9)
    return lo - pad * span, hi + pad * span


def load(kind: str) -> dict:
    if kind == "msd":
        z = np.load(ROOT / "data/runs/continuous_baselines/mass_spring_damper/paper_continuous_arrays.npz")
        return dict(
            mean=np.asarray(z["mean"], float),
            mgf_L=np.asarray(z["lower"], float),
            mgf_U=np.asarray(z["upper"], float),
            poly_L=np.asarray(z["poly_lower"], float),
            poly_U=np.asarray(z["poly_upper"], float),
            C=np.asarray(z["C"], float),
        )
    z = np.load(ROOT / "data/runs/finite_pomdps/bridge_patrol/arrays.npz")
    return dict(
        mean=np.asarray(z["mean"], float),
        mgf_L=np.asarray(z["mgf_lower"], float),
        mgf_U=np.asarray(z["mgf_upper"], float),
        poly_L=np.asarray(z["poly_lower"], float),
        poly_U=np.asarray(z["poly_upper"], float),
        C=np.asarray(z["C"], float),
    )


def draw_pair(fig, inner, env):
    mean, C = env["mean"], env["C"]
    mgf_L, mgf_U, poly_L, poly_U = env["mgf_L"], env["mgf_U"], env["poly_L"], env["poly_U"]
    T = mean.shape[0] - 1
    t = np.arange(T + 1)
    idx = np.linspace(0, C.shape[0] - 1, min(500, C.shape[0])).astype(int)
    centered = C[idx] - mean
    q_lo = np.quantile(C, 0.025, axis=0) - mean
    q_hi = np.quantile(C, 0.975, axis=0) - mean
    mL, mU = mgf_L - mean, mgf_U - mean
    pL, pU = poly_L - mean, poly_U - mean

    ax = fig.add_subplot(inner[0, 0])
    ax.plot(t, centered.T, color="#4c566a", alpha=1.0, lw=0.22, rasterized=True, zorder=1)
    h0, = ax.plot(t, np.zeros_like(t), color="#2e3440", lw=1.6, zorder=3)
    h1, = ax.plot(t, q_lo, color="#2e3440", ls=":", lw=1.35, zorder=4)
    ax.plot(t, q_hi, color="#2e3440", ls=":", lw=1.35, zorder=4)
    # Poly under MGF: at bridge-patrol the lower endpoints nearly coincide.
    h3, = ax.plot(t[1:], pL[1:], color=POLY["color"], ls=POLY["ls"], lw=POLY["lw"], zorder=5)
    ax.plot(t[1:], pU[1:], color=POLY["color"], ls=POLY["ls"], lw=POLY["lw"], zorder=5)
    h2, = ax.plot(t[1:], mL[1:], color=MGF["color"], ls=MGF["ls"], lw=MGF["lw"], zorder=6)
    ax.plot(t[1:], mU[1:], color=MGF["color"], ls=MGF["ls"], lw=MGF["lw"], zorder=6)

    # Zoom to band + trajectory envelope (asymmetric), not a huge symmetric box.
    traj_lo = float(np.nanpercentile(centered[:, 1:], 0.5))
    traj_hi = float(np.nanpercentile(centered[:, 1:], 99.5))
    y_lo = min(traj_lo, float(np.nanmin(mL[1:])), float(np.nanmin(pL[1:])), float(np.nanmin(q_lo[1:])))
    y_hi = max(traj_hi, float(np.nanmax(mU[1:])), float(np.nanmax(pU[1:])), float(np.nanmax(q_hi[1:])))
    ax.set_xlim(0, T)
    ax.set_ylim(*_padded_lim(y_lo, y_hi))
    ax.set_ylabel(r"$G_t-\mathbb{E}[G_t]$", fontsize=9)
    ax.set_xlabel(r"horizon $t$", fontsize=9)
    ax.tick_params(labelsize=7.5, pad=1.5)
    _style(ax)

    axh = fig.add_subplot(inner[0, 1])
    x = C[:, T] - mean[T]
    qT_lo, qT_hi = np.quantile(x, [0.025, 0.975])
    axh.hist(x, bins=55, weights=np.full(len(x), 1 / len(x)), zorder=1, **HIST)
    # When MGF/poly endpoints nearly coincide (bridge lower tail), make MGF
    # a bit taller so the green dash remains readable above the purple line.
    def _vline(v, c, ls, lw, ymax, zo):
        axh.axvline(v, ymax=ymax, color=c, ls=ls, lw=lw, zorder=zo)

    _vline(0.0, "#2e3440", "-", 1.6, 0.82, 4)
    _vline(qT_lo, "#2e3440", ":", 1.35, 0.82, 4)
    _vline(qT_hi, "#2e3440", ":", 1.35, 0.82, 4)
    _vline(pL[T], POLY["color"], POLY["ls"], POLY["lw"], 0.78, 5)
    _vline(pU[T], POLY["color"], POLY["ls"], POLY["lw"], 0.78, 5)
    _vline(mL[T], MGF["color"], MGF["ls"], MGF["lw"], 0.92, 6)
    _vline(mU[T], MGF["color"], MGF["ls"], MGF["lw"], 0.92, 6)
    x_lo = min(float(np.nanpercentile(x, 0.2)), float(mL[T]), float(pL[T]), float(qT_lo))
    x_hi = max(float(np.nanpercentile(x, 99.8)), float(mU[T]), float(pU[T]), float(qT_hi))
    axh.set_xlim(*_padded_lim(x_lo, x_hi))
    axh.set_ylim(0, axh.get_ylim()[1] / 0.80)
    axh.set_title(rf"$T={T}$ hist.", fontsize=9.5, pad=2)
    axh.set_xlabel(rf"$G_{{{T}}}-\mathbb{{E}}[G_{{{T}}}]$", fontsize=8)
    axh.set_ylabel("fraction", fontsize=8)
    axh.tick_params(labelsize=7, pad=1.5)
    axh.yaxis.set_label_coords(-0.18, 0.5)
    _style(axh)
    return ax, [h0, h1, h2, h3]


def main():
    envs = [
        ("(a) Mass-spring-damper", load("msd")),
        ("(b) Bridge-patrol", load("bridge")),
    ]
    fig = plt.figure(figsize=(13.2, 4.35))
    outer = GridSpec(
        1, 2, figure=fig, wspace=0.16,
        left=0.055, right=0.995, top=0.78, bottom=0.14,
    )
    handles = None
    for idx, (title, env) in enumerate(envs):
        inner = GridSpecFromSubplotSpec(
            1, 2, subplot_spec=outer[0, idx],
            wspace=0.28, width_ratios=[1.35, 1.0],
        )
        ax, hs = draw_pair(fig, inner, env)
        ax.set_title(title, fontsize=11, pad=2, loc="left")
        if handles is None:
            handles = hs

    # Compact shared legend above both panels (no side gutter).
    fig.legend(
        handles,
        LEGEND_LABELS,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=4,
        frameon=False,
        fontsize=9.5,
        handlelength=2.4,
        columnspacing=1.6,
    )

    OUT.mkdir(parents=True, exist_ok=True)
    pdf = OUT / "main_examples_msd_bridge.pdf"
    png = OUT / "main_examples_msd_bridge.png"
    fig.savefig(pdf, format="pdf", bbox_inches="tight", dpi=SAVE_DPI)
    fig.savefig(png, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    print("wrote", pdf)
    print("wrote", png)


if __name__ == "__main__":
    main()

"""Main-body curve+rotated-hist figures, split into finite POMDPs and LQGs."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.finite_pomdps.plot_main_examples_msd_bridge import (  # noqa: E402
    HIST,
    LEGEND_LABELS,
    MGF,
    OUT,
    POLY,
    SAVE_DPI,
    _declutter_pair_xticks,
    _padded_lim,
    _style,
)
from experiments.finite_pomdps.plot_main_all_curves import load  # noqa: E402

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

CONT = _ROOT / "data" / "runs" / "continuous_baselines"
FS = _ROOT / "data" / "runs" / "finite_pomdps"

FINITE = [
    ("(a) Bridge-patrol", FS / "bridge_patrol" / "arrays.npz", "fs"),
    ("(b) Tiger", FS / "tiger_tiger_hand" / "arrays.npz", "fs"),
    ("(c) Network", FS / "network_network_hand" / "arrays.npz", "fs"),
    ("(d) Shopping-5", FS / "shopping5_shopping_hand" / "arrays.npz", "fs"),
]
LQG = [
    ("(a) Mass-spring-damper", CONT / "mass_spring_damper" / "paper_continuous_arrays.npz", "cont"),
    ("(b) Coupled masses", CONT / "coupled_masses" / "paper_continuous_arrays.npz", "cont"),
    ("(c) Quarter-car", CONT / "quarter_car" / "paper_continuous_arrays.npz", "cont"),
    ("(d) Thermal", CONT / "two_room_thermal" / "paper_continuous_arrays.npz", "cont"),
    ("(e) Cart-pole", CONT / "cartpole_linear" / "paper_continuous_arrays.npz", "cont"),
]


def draw_setting(fig, outer_spec, env, title):
    mean, C = env["mean"], env["C"]
    mgf_L, mgf_U = env["mgf_L"], env["mgf_U"]
    poly_L, poly_U = env["poly_L"], env["poly_U"]
    T = mean.shape[0] - 1
    t = np.arange(T + 1)
    idx = np.linspace(0, C.shape[0] - 1, min(500, C.shape[0])).astype(int)
    centered = C[idx] - mean
    q_lo = np.quantile(C, 0.025, axis=0) - mean
    q_hi = np.quantile(C, 0.975, axis=0) - mean
    mL, mU = mgf_L - mean, mgf_U - mean
    pL, pU = poly_L - mean, poly_U - mean

    xT = C[:, T] - mean[T]
    qT_lo, qT_hi = np.quantile(xT, [0.025, 0.975])

    traj_lo = float(np.nanpercentile(centered[:, 1:], 0.5))
    traj_hi = float(np.nanpercentile(centered[:, 1:], 99.5))
    y_lo = min(
        traj_lo, float(np.nanmin(mL[1:])), float(np.nanmin(pL[1:])),
        float(np.nanmin(q_lo[1:])), float(np.nanpercentile(xT, 0.2)),
        float(mL[T]), float(pL[T]), float(qT_lo),
    )
    y_hi = max(
        traj_hi, float(np.nanmax(mU[1:])), float(np.nanmax(pU[1:])),
        float(np.nanmax(q_hi[1:])), float(np.nanpercentile(xT, 99.8)),
        float(mU[T]), float(pU[T]), float(qT_hi),
    )
    ylim = _padded_lim(y_lo, y_hi)

    inner = GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer_spec,
        wspace=0.06, width_ratios=[2.4, 1.0],
    )
    ax = fig.add_subplot(inner[0, 0])
    axh = fig.add_subplot(inner[0, 1], sharey=ax)

    ax.plot(t, centered.T, color="#4c566a", alpha=1.0, lw=0.22, rasterized=True, zorder=1)
    h0, = ax.plot(t, np.zeros_like(t), color="#2e3440", lw=1.6, zorder=3)
    h1, = ax.plot(t, q_lo, color="#2e3440", ls=":", lw=1.35, zorder=4)
    ax.plot(t, q_hi, color="#2e3440", ls=":", lw=1.35, zorder=4)
    h3, = ax.plot(t[1:], pL[1:], color=POLY["color"], ls=POLY["ls"], lw=POLY["lw"], zorder=5)
    ax.plot(t[1:], pU[1:], color=POLY["color"], ls=POLY["ls"], lw=POLY["lw"], zorder=5)
    h2, = ax.plot(t[1:], mL[1:], color=MGF["color"], ls=MGF["ls"], lw=MGF["lw"], zorder=6)
    ax.plot(t[1:], mU[1:], color=MGF["color"], ls=MGF["ls"], lw=MGF["lw"], zorder=6)
    ax.set_xlim(0, T)
    ax.set_ylim(*ylim)
    ax.set_xlabel(r"horizon $t$", fontsize=8)
    ax.set_title(title, fontsize=9.5, pad=2, loc="left")
    ax.tick_params(labelsize=7, pad=1.2)
    _style(ax)

    axh.hist(
        xT, bins=55, orientation="horizontal",
        weights=np.full(len(xT), 1 / len(xT)), zorder=1, **HIST,
    )

    def _hline(v, c, ls, lw, xmax, zo):
        axh.axhline(v, xmax=xmax, color=c, ls=ls, lw=lw, zorder=zo)

    _hline(0.0, "#2e3440", "-", 1.6, 0.82, 4)
    _hline(qT_lo, "#2e3440", ":", 1.35, 0.82, 4)
    _hline(qT_hi, "#2e3440", ":", 1.35, 0.82, 4)
    _hline(pL[T], POLY["color"], POLY["ls"], POLY["lw"], 0.78, 5)
    _hline(pU[T], POLY["color"], POLY["ls"], POLY["lw"], 0.78, 5)
    _hline(mL[T], MGF["color"], MGF["ls"], MGF["lw"], 0.92, 6)
    _hline(mU[T], MGF["color"], MGF["ls"], MGF["lw"], 0.92, 6)
    axh.set_ylim(*ylim)
    axh.set_xlim(0, axh.get_xlim()[1] / 0.80)
    axh.set_xlabel(HIST_XLABEL, fontsize=7.5)
    axh.tick_params(labelsize=6.5, pad=1.0)
    axh.tick_params(axis="y", labelleft=False)
    _style(axh)
    _declutter_pair_xticks(ax, axh)
    return [h0, h1, h2, h3]


# Fig 2 (LQG, 2×3) is the height reference. At \includegraphics[width=\textwidth],
# rendered height ∝ fig_h / fig_w, so 2-col figures need a smaller fig_h to match.
_COL_W = 5.3
_ROW_H = 3.05
_LEGEND_PAD = 0.55
_H_REF = _ROW_H * 2 + _LEGEND_PAD  # LQG 2 rows
_W_REF = _COL_W * 3               # LQG 3 cols
# Absolute inches reserved above axes so the shared legend clears panel titles
# (same clearance on short 2-col figs as on the taller LQG fig).
_LEGEND_TOP_IN = 0.62

# Shared legend size/style for all paper traj|hist figures.
LEGEND_KW = dict(
    loc="upper center",
    bbox_to_anchor=(0.5, 0.998),
    frameon=False,
    fontsize=11,
    handlelength=2.4,
    columnspacing=1.6,
    handletextpad=0.6,
)
YLABEL_GT = r"$G_t^\pi-\mathbb{E}[G_t^\pi]$"
HIST_XLABEL = "relative frequency"


def _figsize(ncols: int, nrows: int) -> tuple[float, float]:
    w = _COL_W * ncols
    # Match textwidth-scaled page height to the 2×3 LQG figure.
    h = _H_REF * (w / _W_REF)
    return w, h


def _gs_top(fig_h: float, legend_rows: int = 1) -> float:
    return max(0.72, 1.0 - (_LEGEND_TOP_IN * legend_rows) / fig_h)


def render(panels, ncols, stem, legend_fontsize: float | None = None):
    n = len(panels)
    nrows = (n + ncols - 1) // ncols
    fw, fh = _figsize(ncols, nrows)
    fig = plt.figure(figsize=(fw, fh))
    outer = GridSpec(
        nrows, ncols, figure=fig, wspace=0.24, hspace=0.28,
        left=0.045, right=0.995,
        top=_gs_top(fh) if nrows > 1 else 0.90,
        bottom=0.08 if nrows == 1 else 0.04,
    )
    handles = None
    for idx, (title, path, kind) in enumerate(panels):
        r, c = divmod(idx, ncols)
        env = load(path, kind)
        hs = draw_setting(fig, outer[r, c], env, title)
        if handles is None:
            handles = hs
    for i in range(n):
        if i % ncols == 0:
            fig.axes[2 * i].set_ylabel(YLABEL_GT, fontsize=8.5)

    leg_kw = dict(LEGEND_KW)
    if legend_fontsize is not None:
        leg_kw["fontsize"] = legend_fontsize
    fig.legend(handles, LEGEND_LABELS, ncol=len(LEGEND_LABELS), **leg_kw)
    OUT.mkdir(parents=True, exist_ok=True)
    pdf = OUT / f"{stem}.pdf"
    png = OUT / f"{stem}.png"
    fig.savefig(pdf, format="pdf", bbox_inches="tight", dpi=SAVE_DPI)
    fig.savefig(png, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    print("wrote", pdf)
    print("wrote", png)


def main():
    # Finite: 2 per row. LQG Fig 2: slightly larger shared legend only.
    render(FINITE, ncols=2, stem="main_curve_hist_finite")
    render(LQG, ncols=3, stem="main_curve_hist_lqg", legend_fontsize=14.5)


if __name__ == "__main__":
    main()

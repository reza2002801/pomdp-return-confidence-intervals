"""Four finite POMDPs: traj + rotated hist with MGF/poly/Freedman/Azuma (linear scale)."""
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

from returnbounds import io as save_artifacts
from returnbounds import martingale as mg
from experiments.finite_pomdps.plot_main_examples_msd_bridge import (  # noqa: E402
    HIST,
    MGF,
    OUT,
    POLY,
    SAVE_DPI,
    _declutter_pair_xticks,
    _padded_lim,
    _style,
)

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

FS = _ROOT / "data" / "runs" / "finite_pomdps"
AZUMA = {"color": "#1f77b4", "ls": ":", "lw": 2.0}
FREED = {"color": "#d62728", "ls": "-.", "lw": 2.2}
DELTA = 0.025

PANELS = [
    ("(a) Bridge-patrol", FS / "bridge_patrol"),
    ("(b) Tiger", FS / "tiger_tiger_hand"),
    ("(c) Network", FS / "network_network_hand"),
    ("(d) Shopping-5", FS / "shopping5_shopping_hand"),
]


def load_env(out_dir: Path):
    out_dir = Path(out_dir)
    arr = save_artifacts.load_run_arrays(out_dir)
    mean = np.asarray(arr["mean"], float)
    mgf_L = np.asarray(arr["mgf_lower"], float)
    mgf_U = np.asarray(arr["mgf_upper"], float)
    poly_L = np.asarray(arr["poly_lower"], float)
    poly_U = np.asarray(arr["poly_upper"], float)
    C = np.asarray(arr["C"], float)

    # Always recompute; write cache only as an output (never read as input).
    fm = save_artifacts.load_finite_model(out_dir / "finite_model.npz")
    T = mean.shape[0] - 1
    raw = mg.bands(fm, T, DELTA)
    bands = {
        "azuma_L": np.asarray(raw["azuma_L"], float),
        "azuma_U": np.asarray(raw["azuma_U"], float),
        "freedman_L": np.asarray(raw["freedman_L"], float),
        "freedman_U": np.asarray(raw["freedman_U"], float),
    }
    bands_path = out_dir / "martingale_bands.npz"
    np.savez_compressed(
        bands_path,
        azuma_L=bands["azuma_L"], azuma_U=bands["azuma_U"],
        freedman_L=bands["freedman_L"], freedman_U=bands["freedman_U"],
        mean=np.asarray(raw["mean"], float),
    )
    print("  saved", bands_path)
    return dict(
        mean=mean, mgf_L=mgf_L, mgf_U=mgf_U, poly_L=poly_L, poly_U=poly_U, C=C, **bands
    )


def draw_setting(fig, outer_spec, env, title):
    mean, C = env["mean"], env["C"]
    mgf_L, mgf_U = env["mgf_L"], env["mgf_U"]
    poly_L, poly_U = env["poly_L"], env["poly_U"]
    aL, aU = env["azuma_L"], env["azuma_U"]
    fL, fU = env["freedman_L"], env["freedman_U"]
    T = mean.shape[0] - 1
    t = np.arange(T + 1)
    idx = np.linspace(0, C.shape[0] - 1, min(500, C.shape[0])).astype(int)
    centered = C[idx] - mean
    q_lo = np.quantile(C, 0.025, axis=0) - mean
    q_hi = np.quantile(C, 0.975, axis=0) - mean
    mL, mU = mgf_L - mean, mgf_U - mean
    pL, pU = poly_L - mean, poly_U - mean
    azL, azU = aL - mean, aU - mean
    frL, frU = fL - mean, fU - mean

    xT = C[:, T] - mean[T]
    qT_lo, qT_hi = np.quantile(xT, [0.025, 0.975])

    # Linear scale: include Azuma so all methods are visible (may look wide).
    y_lo = min(
        float(np.nanpercentile(centered[:, 1:], 0.5)),
        float(np.nanmin(mL[1:])), float(np.nanmin(pL[1:])),
        float(np.nanmin(frL[1:])), float(np.nanmin(azL[1:])),
        float(np.nanmin(q_lo[1:])), float(np.nanpercentile(xT, 0.2)),
    )
    y_hi = max(
        float(np.nanpercentile(centered[:, 1:], 99.5)),
        float(np.nanmax(mU[1:])), float(np.nanmax(pU[1:])),
        float(np.nanmax(frU[1:])), float(np.nanmax(azU[1:])),
        float(np.nanmax(q_hi[1:])), float(np.nanpercentile(xT, 99.8)),
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
    h4, = ax.plot(t[1:], frL[1:], color=FREED["color"], ls=FREED["ls"], lw=FREED["lw"], zorder=5)
    ax.plot(t[1:], frU[1:], color=FREED["color"], ls=FREED["ls"], lw=FREED["lw"], zorder=5)
    h5, = ax.plot(t[1:], azL[1:], color=AZUMA["color"], ls=AZUMA["ls"], lw=AZUMA["lw"], zorder=5)
    ax.plot(t[1:], azU[1:], color=AZUMA["color"], ls=AZUMA["ls"], lw=AZUMA["lw"], zorder=5)
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
    _hline(frL[T], FREED["color"], FREED["ls"], FREED["lw"], 0.85, 5)
    _hline(frU[T], FREED["color"], FREED["ls"], FREED["lw"], 0.85, 5)
    _hline(azL[T], AZUMA["color"], AZUMA["ls"], AZUMA["lw"], 0.88, 5)
    _hline(azU[T], AZUMA["color"], AZUMA["ls"], AZUMA["lw"], 0.88, 5)
    axh.set_ylim(*ylim)
    axh.set_xlim(0, axh.get_xlim()[1] / 0.80)
    axh.set_xlabel("relative frequency", fontsize=7.5)
    axh.tick_params(labelsize=6.5, pad=1.0)
    axh.tick_params(axis="y", labelleft=False)
    _style(axh)
    _declutter_pair_xticks(ax, axh)
    return [h0, h1, h2, h3, h4, h5]


def main():
    ncols = 2
    nrows = 2
    # Same textwidth-scaled height / legend style as Fig 2 (LQG).
    from experiments.finite_pomdps.plot_main_curve_hist_pairs import (
        LEGEND_KW,
        YLABEL_GT,
        _figsize,
        _gs_top,
    )

    labels = [
        "exact mean",
        r"Emp. 2.5% and 97.5%",
        "MGF-based CIs",
        "Moment-based CIs",
        "Freedman",
        "Azuma",
    ]
    fw, fh = _figsize(ncols, nrows)
    fig = plt.figure(figsize=(fw, fh))
    outer = GridSpec(
        nrows, ncols, figure=fig, wspace=0.24, hspace=0.28,
        left=0.045, right=0.995, top=_gs_top(fh), bottom=0.04,
    )
    handles = None
    for idx, (title, path) in enumerate(PANELS):
        print(title, flush=True)
        env = load_env(path)
        r, c = divmod(idx, ncols)
        hs = draw_setting(fig, outer[r, c], env, title)
        if handles is None:
            handles = hs
        if c == 0:
            fig.axes[2 * idx].set_ylabel(YLABEL_GT, fontsize=8.5)

    # Same shared legend size as Fig 1/2; slightly tighter columns for 6 entries.
    kw = {**LEGEND_KW, "columnspacing": 1.05}
    fig.legend(handles, labels, ncol=len(labels), **kw)
    OUT.mkdir(parents=True, exist_ok=True)
    # Keep old filename so existing includegraphics can be pointed at the new 4-panel fig.
    pdf = OUT / "finite_azuma_freedman_compare.pdf"
    png = OUT / "finite_azuma_freedman_compare.png"
    fig.savefig(pdf, format="pdf", bbox_inches="tight", dpi=SAVE_DPI)
    fig.savefig(png, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    print("wrote", pdf)
    print("wrote", png)


if __name__ == "__main__":
    main()

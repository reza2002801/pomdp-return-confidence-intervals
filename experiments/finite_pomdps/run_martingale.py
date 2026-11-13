"""
Compute Azuma / Freedman Doob baselines on existing finite-POMDP / LQG runs.

Reuses stored FiniteModel + arrays.npz (MGF / poly / 20k MC trajectories).
Does not rebuild models or recompute MGF / poly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from returnbounds import io as save_artifacts
from returnbounds import martingale as mg
from returnbounds.plotting import HIST, MGF, POLY, _style

DELTA = 0.025
HORIZONS = (25, 50, 100)

AZUMA = {"color": "#1f77b4", "ls": ":", "lw": 2.0}
AZUMA2 = {"color": "#17becf", "ls": (0, (2, 1.2)), "lw": 1.8}
FREED = {"color": "#d62728", "ls": "-.", "lw": 2.2}


def _dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _exceed(C_t, L, U):
    C_t = np.asarray(C_t, dtype=float)
    return float(np.mean(C_t < L)), float(np.mean(C_t > U))


def _method_block(name, L, U, C_t):
    below, above = _exceed(C_t, L, U)
    return {
        "L": float(L),
        "U": float(U),
        "width": float(U - L),
        "frac_below": below,
        "frac_above": above,
    }


def run_one(out_dir: Path, name: str, T: int = 100, delta: float = DELTA, log=print):
    out_dir = Path(out_dir)
    fm = save_artifacts.load_finite_model(out_dir / "finite_model.npz")
    arr = save_artifacts.load_run_arrays(out_dir)
    C = np.asarray(arr["C"], dtype=float)
    mean_stored = np.asarray(arr["mean"], dtype=float)
    mgf_L = np.asarray(arr["mgf_lower"], dtype=float)
    mgf_U = np.asarray(arr["mgf_upper"], dtype=float)
    poly_L = np.asarray(arr["poly_lower"], dtype=float)
    poly_U = np.asarray(arr["poly_upper"], dtype=float)

    log(f"=== martingale on {name}: n={fm.n_states}, e={fm.n_edges}, T={T} ===")
    bands = mg.bands(fm, T, delta)
    # sanity: mean should match stored exact mean
    mean = bands["mean"]
    if np.isfinite(mean_stored[T]):
        err = abs(mean[T] - mean_stored[T])
        log(f"  mean[T] match err={err:.3e}")

    report = {"name": name, "T": T, "delta": delta, "horizons": {}}
    for t in HORIZONS:
        C_t = C[:, t]
        emp_lo, emp_hi = np.quantile(C_t, [0.025, 0.975])
        entry = {
            "mean": float(mean[t]),
            "K": float(bands["K"][t]),
            "sigma": float(bands["sigma"][t]),
            "n": int(t if (fm.mu0 is None or np.count_nonzero(fm.init_dist() > 0) <= 1)
                     else t + 1),
            "empirical": {"q025": float(emp_lo), "q975": float(emp_hi)},
            "methods": {
                "Azuma": _method_block("Azuma", bands["azuma_L"][t], bands["azuma_U"][t], C_t),
                "Azuma_per_step": _method_block(
                    "Azuma_per_step", bands["azuma2_L"][t], bands["azuma2_U"][t], C_t
                ),
                "Freedman": _method_block(
                    "Freedman", bands["freedman_L"][t], bands["freedman_U"][t], C_t
                ),
                "MGF": _method_block("MGF", mgf_L[t], mgf_U[t], C_t),
                "polynomial": _method_block("polynomial", poly_L[t], poly_U[t], C_t),
            },
        }
        from returnbounds import mgf as mgf_mod
        op = mgf_mod.TiltedOperator(fm)
        reach = mgf_mod.reachable_sets(op, fm.init_dist(), t)
        st = mg.martingale_stats(fm, t, reach)
        entry["n"] = int(st["n"])
        entry["K"] = float(st["K"])
        entry["sigma"] = float(np.sqrt(st["sigma2"]))
        report["horizons"][str(t)] = entry
        log(
            f"  t={t}: K={entry['K']:.4g} sigma={entry['sigma']:.4g} n={entry['n']} "
            f"Freedman width={entry['methods']['Freedman']['width']:.4g} "
            f"Azuma={entry['methods']['Azuma']['width']:.4g} "
            f"MGF={entry['methods']['MGF']['width']:.4g}"
        )

    _dump(out_dir / "martingale.json", report)
    _plot_traj(out_dir, name, C, mean, bands, mgf_L, mgf_U, poly_L, poly_U)
    _plot_hist(out_dir, name, C, mean, bands, mgf_L, mgf_U, poly_L, poly_U, T)
    return report


def _plot_traj(out_dir, name, C, mean, bands, mgf_L, mgf_U, poly_L, poly_U):
    n_draw = min(800, C.shape[0])
    idx = np.linspace(0, C.shape[0] - 1, n_draw).astype(int)
    t = np.arange(C.shape[1])
    centered = C[idx] - mean
    q_lo = np.quantile(C, 0.025, axis=0) - mean
    q_hi = np.quantile(C, 0.975, axis=0) - mean

    series = [
        ("MGF", (mgf_L - mean), (mgf_U - mean), MGF),
        ("polynomial (d=16)", (poly_L - mean), (poly_U - mean), POLY),
        ("Freedman", (bands["freedman_L"] - mean), (bands["freedman_U"] - mean), FREED),
        ("Azuma", (bands["azuma_L"] - mean), (bands["azuma_U"] - mean), AZUMA),
    ]

    az_span = float(np.nanmax(np.abs(bands["azuma_U"][1:] - mean[1:])))
    tight_span = max(
        float(np.nanmax(np.abs(mgf_L[1:] - mean[1:]))),
        float(np.nanmax(np.abs(mgf_U[1:] - mean[1:]))),
        float(np.nanmax(np.abs(poly_L[1:] - mean[1:]))),
        float(np.nanmax(np.abs(poly_U[1:] - mean[1:]))),
        float(np.nanmax(np.abs(bands["freedman_U"][1:] - mean[1:]))),
        float(np.nanmax(np.abs(bands["freedman_L"][1:] - mean[1:]))),
        float(np.abs(centered).max()),
    )
    need_zoom = az_span > 2.5 * tight_span

    if need_zoom:
        fig, axes = plt.subplots(2, 1, figsize=(8.0, 8.4), sharex=True)
    else:
        fig, ax0 = plt.subplots(figsize=(8.0, 5.4))
        axes = [ax0]

    def draw(ax, include_azuma: bool, lim: float):
        ax.plot(t, centered.T, color="#4c566a", alpha=1.0, lw=0.25, rasterized=True, zorder=1)
        ax.plot(t, np.zeros_like(t, dtype=float), color="#2e3440", lw=2.0, label="exact mean", zorder=3)
        ax.plot(t, q_lo, color="#2e3440", ls=":", lw=1.8, label="empirical 2.5%", zorder=4)
        ax.plot(t, q_hi, color="#2e3440", ls=(0, (1, 1.2)), lw=1.8, label="empirical 97.5%", zorder=4)
        for lab, lo, hi, sty in series:
            if lab == "Azuma" and not include_azuma:
                continue
            ax.plot(t[1:], lo[1:], color=sty["color"], ls=sty["ls"], lw=sty["lw"], label=lab, zorder=5)
            ax.plot(t[1:], hi[1:], color=sty["color"], ls=sty["ls"], lw=sty["lw"], zorder=5)
        ax.set_xlim(0, t[-1])
        ax.set_ylim(-lim, lim)
        ax.set_ylabel(r"$C_t-\mathbb{E}[C_t]$")
        _style(ax)

    if need_zoom:
        draw(axes[0], True, 1.08 * az_span)
        axes[0].set_title(f"{name}: all bands (incl. Azuma)", pad=8)
        axes[0].legend(loc="upper right", frameon=False, fontsize=8, ncol=2)
        draw(axes[1], False, 1.12 * tight_span)
        axes[1].set_title("zoom without Azuma", pad=8)
        axes[1].legend(loc="upper right", frameon=False, fontsize=8, ncol=2)
        axes[1].set_xlabel(r"time step $t$")
    else:
        draw(axes[0], True, 1.08 * max(az_span, tight_span))
        axes[0].set_title(f"{name}: pointwise 95% bands", pad=28)
        axes[0].legend(
            loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=8
        )
        axes[0].set_xlabel(r"time step $t$")

    fig.tight_layout()
    fig.savefig(out_dir / "comparison_traj.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_hist(out_dir, name, C, mean, bands, mgf_L, mgf_U, poly_L, poly_U, T):
    x = C[:, T] - mean[T]
    q_lo, q_hi = np.quantile(x, [0.025, 0.975])
    lines = [
        ("exact mean", 0.0, "#2e3440", "-", 2.0),
        ("empirical 2.5%/97.5%", q_lo, "#2e3440", ":", 1.8),
        (None, q_hi, "#2e3440", ":", 1.8),
        ("MGF", mgf_L[T] - mean[T], MGF["color"], MGF["ls"], MGF["lw"]),
        (None, mgf_U[T] - mean[T], MGF["color"], MGF["ls"], MGF["lw"]),
        ("polynomial", poly_L[T] - mean[T], POLY["color"], POLY["ls"], POLY["lw"]),
        (None, poly_U[T] - mean[T], POLY["color"], POLY["ls"], POLY["lw"]),
        ("Freedman", bands["freedman_L"][T] - mean[T], FREED["color"], FREED["ls"], FREED["lw"]),
        (None, bands["freedman_U"][T] - mean[T], FREED["color"], FREED["ls"], FREED["lw"]),
        ("Azuma", bands["azuma_L"][T] - mean[T], AZUMA["color"], AZUMA["ls"], AZUMA["lw"]),
        (None, bands["azuma_U"][T] - mean[T], AZUMA["color"], AZUMA["ls"], AZUMA["lw"]),
    ]
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    ax.hist(x, bins=70, weights=np.full(len(x), 1 / len(x)), zorder=1, **HIST)
    top = 0.74
    for lab, v, c, ls, lw in lines:
        ax.axvline(v, ymax=top, color=c, ls=ls, lw=lw, label=lab, zorder=5)
    half = max(abs(float(v)) for _, v, *_ in lines if v is not None)
    half = max(half, float(np.abs(x).max()), 1e-9)
    # if Azuma dwarfs, keep full axis but note — linear y as requested
    ax.set_xlim(-1.12 * half, 1.12 * half)
    ax.set_ylim(0, ax.get_ylim()[1] / (top - 0.03))
    ax.set_xlabel(rf"$C_{{{T}}}-\mathbb{{E}}[C_{{{T}}}]$")
    ax.set_ylabel("fraction of trajectories")
    ax.set_title(f"{name}: T={T} histogram ({len(x):,} traj.)", fontsize=11, pad=8)
    ax.legend(loc="upper right", frameon=False, fontsize=8, handlelength=2.4)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "comparison_hist.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


# Paper finite runs only (hand policies + bridge patrol).
_FS = _ROOT / "data" / "runs" / "finite_pomdps"
DEFAULTS = [
    ("bridge_patrol", _FS / "bridge_patrol"),
    ("tiger", _FS / "tiger_tiger_hand"),
    ("network", _FS / "network_network_hand"),
    ("shopping5", _FS / "shopping5_shopping_hand"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="subset of problem names")
    args = ap.parse_args()
    jobs = DEFAULTS
    if args.only:
        jobs = [(n, p) for n, p in DEFAULTS if n in args.only]
    for name, path in jobs:
        run_one(path, name)


if __name__ == "__main__":
    main()

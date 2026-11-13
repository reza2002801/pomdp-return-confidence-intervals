"""
Polynomial-degree ablation on continuous closed-form LQG plants.

Reuses continuous MC trajectories and closed-form MGF from
data/runs/continuous_baselines/<plant>/. Each degree is routed through
``poly_bounds_from_form`` (truncation-mass correction included).

  PYTHONPATH=. python3 experiments/lqg/ablate_poly_degree_continuous.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from returnbounds import continuous as gm
from returnbounds.plant import build_plant

HERE = Path(__file__).resolve().parent
DATA = _ROOT / "data" / "runs" / "continuous_baselines"
ABL = _ROOT / "data" / "runs" / "ablation_poly_degree"

PLANTS = [
    "mass_spring_damper",
    "coupled_masses",
    "quarter_car",
    "two_room_thermal",
    "cartpole_linear",
]
DISPLAY = {
    "mass_spring_damper": "MSD",
    "coupled_masses": "Coupled",
    "quarter_car": "Quarter-car",
    "two_room_thermal": "Thermal",
    "cartpole_linear": "Cart-pole",
}
DEFAULT_DEGREES = [4, 8, 16, 20, 24]
T_MAX = 100
DELTA = 0.025
MGF_STYLE = {"color": "#2ca02c", "ls": (0, (4, 1.5)), "lw": 2.0}


def _style(ax):
    ax.grid(True, alpha=0.28, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _poly_colors(degrees):
    cmap = plt.get_cmap("plasma")
    xs = np.linspace(0.2, 0.85, max(len(degrees), 1))
    return {d: cmap(x) for d, x in zip(degrees, xs)}


def _collapsed(row, emp_width):
    if not np.isfinite(row["width"]) or row["width"] < 1e-6:
        return True
    if emp_width > 0 and row["width"] < 0.02 * emp_width:
        return True
    if row["frac_below"] + row["frac_above"] > 0.4:
        return True
    return False


def _poly_all_degrees_from_form(J, b, d0, degrees, alpha, support_mult=40.0):
    """Per-degree TailLP via ``poly_bounds_from_form`` (MGF ``tau_d`` correction; no ``tau_ext``)."""
    degrees = [int(d) for d in degrees]
    out = {}
    for d in degrees:
        lo, hi, Lsup, Usup, diag = gm.poly_bounds_from_form(
            J, b, d0, degree=d, alpha=alpha, support_mult=support_mult
        )
        out[d] = (float(lo), float(hi), diag, float(Lsup), float(Usup))
    return out


def _worker(payload):
    t, J, b, d0, degrees, alpha = payload
    return int(t), _poly_all_degrees_from_form(J, b, d0, degrees, alpha)


def run_plant(name: str, degrees: list[int], alpha: float = DELTA, log=print):
    cfg = json.loads((HERE / "configs" / f"{name}.json").read_text())
    plant = build_plant(cfg)
    delta = float(cfg.get("delta", alpha))
    src = DATA / name
    arrays = np.load(src / "paper_continuous_arrays.npz")
    mean = np.asarray(arrays["mean"], float)
    mgf_L = np.asarray(arrays["lower"], float)
    mgf_U = np.asarray(arrays["upper"], float)
    C = np.asarray(arrays["C"], float)
    T = T_MAX
    # Default: every horizon t=1..T so poly_d*.npz stores a full series per degree.
    times = list(range(1, T + 1))

    abl = ABL / name
    abl.mkdir(parents=True, exist_ok=True)
    log("=" * 64)
    log(f"{name}: CONTINUOUS closed-form ablation")
    log(f"  T={T} degrees={degrees} n_times={len(times)} (all horizons)", flush=True)

    # Build quadratic forms once (also refreshes mean/MGF if arrays stale — keep arrays)
    forms = {}
    t0 = time.perf_counter()
    for t in times:
        an = gm.analyze_horizon(plant, t, delta=delta)
        forms[t] = (an["J"], an["b"], an["d0"])
        # sanity: mean should match stored continuous mean
        if abs(an["mean"] - mean[t]) > 1e-6:
            log(f"  WARN t={t}: analyze mean {an['mean']:.6f} vs stored {mean[t]:.6f}")
    log(f"  forms in {time.perf_counter() - t0:.1f}s", flush=True)

    curves = {d: {"lo": np.full(T + 1, np.nan), "hi": np.full(T + 1, np.nan), "times": times} for d in degrees}
    Lsup = np.full(T + 1, np.nan)
    Usup = np.full(T + 1, np.nan)
    diags = {d: {} for d in degrees}

    payloads = [(t, forms[t][0], forms[t][1], forms[t][2], degrees, delta) for t in times]
    import os
    n_workers = max(1, min(len(payloads), (os.cpu_count() or 2) - 1))
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futs = [pool.submit(_worker, p) for p in payloads]
        done = 0
        for fut in as_completed(futs):
            t, per_d = fut.result()
            for d, (lo, hi, diag, Ls, Us) in per_d.items():
                curves[d]["lo"][t] = lo
                curves[d]["hi"][t] = hi
                diags[d][str(t)] = diag
                Lsup[t], Usup[t] = Ls, Us
            done += 1
            if done in (1, len(payloads)) or t in (25, 50, 75, 100):
                d_show = degrees[-1]
                log(
                    f"  t={t}: d={d_show} [{curves[d_show]['lo'][t]:.4f},{curves[d_show]['hi'][t]:.4f}] "
                    f"({done}/{len(payloads)})",
                    flush=True,
                )
    log(f"  poly all degrees in {time.perf_counter() - t0:.1f}s", flush=True)

    # Nesting projection
    n_nest = 0
    for i in range(1, len(degrees)):
        d_prev, d = degrees[i - 1], degrees[i]
        for t in times:
            lo_p, hi_p = curves[d_prev]["lo"][t], curves[d_prev]["hi"][t]
            lo, hi = curves[d]["lo"][t], curves[d]["hi"][t]
            if not np.isfinite(lo_p + hi_p + lo + hi):
                continue
            new_lo, new_hi = max(lo, lo_p), min(hi, hi_p)
            if new_lo > new_hi:
                new_lo, new_hi = lo_p, hi_p
            if abs(new_lo - lo) > 1e-9 or abs(new_hi - hi) > 1e-9:
                n_nest += 1
                curves[d]["lo"][t] = new_lo
                curves[d]["hi"][t] = new_hi
    if n_nest:
        log(f"  nesting repairs: {n_nest}", flush=True)

    emp_w = float(np.quantile(C[:, T], 0.975) - np.quantile(C[:, T], 0.025))
    rows = []
    per_horizon = {}
    csv_lines = [
        "degree,t,poly_L,poly_U,width,mgf_L,mgf_U,exact_mean,n_inconsistent,max_eps_smooth,support_L,support_U"
    ]
    for d in degrees:
        lo, hi = curves[d]["lo"], curves[d]["hi"]
        width = float(hi[T] - lo[T])
        frac_lo = float(np.mean(C[:, T] < lo[T]))
        frac_hi = float(np.mean(C[:, T] > hi[T]))
        tot_incons = int(sum(diags[d][str(t)]["n_inconsistent"] for t in times if str(t) in diags[d]))
        max_eps = float(max((diags[d][str(t)]["max_eps_smooth"] for t in times if str(t) in diags[d]), default=0.0))
        row = {
            "degree": d,
            "t": T,
            "poly_L": float(lo[T]),
            "poly_U": float(hi[T]),
            "width": width,
            "frac_below": frac_lo,
            "frac_above": frac_hi,
            "mgf_L": float(mgf_L[T]),
            "mgf_U": float(mgf_U[T]),
            "emp_q025": float(np.quantile(C[:, T], 0.025)),
            "emp_q975": float(np.quantile(C[:, T], 0.975)),
            "exact_mean": float(mean[T]),
            "n_inconsistent": tot_incons,
            "max_eps_smooth": max_eps,
            "collapsed": False,
        }
        row["collapsed"] = _collapsed(row, emp_w)
        rows.append(row)
        ph = []
        for t in times:
            diag = diags[d].get(str(t), {})
            entry = {
                "t": int(t),
                "poly_L": float(lo[t]),
                "poly_U": float(hi[t]),
                "width": float(hi[t] - lo[t]) if np.isfinite(hi[t]) and np.isfinite(lo[t]) else float("nan"),
                "mgf_L": float(mgf_L[t]),
                "mgf_U": float(mgf_U[t]),
                "exact_mean": float(mean[t]),
                "n_inconsistent": int(diag.get("n_inconsistent", 0)),
                "max_eps_smooth": float(diag.get("max_eps_smooth", 0.0)),
                "support_L": float(Lsup[t]),
                "support_U": float(Usup[t]),
            }
            ph.append(entry)
            csv_lines.append(
                f"{d},{t},{entry['poly_L']:.12g},{entry['poly_U']:.12g},{entry['width']:.12g},"
                f"{entry['mgf_L']:.12g},{entry['mgf_U']:.12g},{entry['exact_mean']:.12g},"
                f"{entry['n_inconsistent']},{entry['max_eps_smooth']:.12g},"
                f"{entry['support_L']:.12g},{entry['support_U']:.12g}"
            )
        per_horizon[str(d)] = ph
        np.savez_compressed(
            abl / f"poly_d{d}.npz",
            lower=lo, upper=hi, support_L=Lsup, support_U=Usup,
            times=np.asarray(times, dtype=np.int64),
            degree=np.array([d]),
            mgf_lower=mgf_L, mgf_upper=mgf_U, mean=mean,
        )
        log(
            f"  d={d}: [{lo[T]:.4f},{hi[T]:.4f}] w={width:.4f} "
            f"frac={frac_lo:.4f}/{frac_hi:.4f} incons={tot_incons}"
            + (" COLLAPSED" if row["collapsed"] else ""),
            flush=True,
        )

    # Also store continuous series used for plotting (mean/MGF/C pointers)
    np.savez_compressed(
        abl / "continuous_plot_arrays.npz",
        mean=mean, mgf_lower=mgf_L, mgf_upper=mgf_U, C=C,
    )
    # Packed all-degree × all-horizon arrays for downstream use
    pack = {
        "times": np.asarray(times, dtype=np.int64),
        "degrees": np.asarray(degrees, dtype=np.int64),
        "mean": mean,
        "mgf_lower": mgf_L,
        "mgf_upper": mgf_U,
        "support_L": Lsup,
        "support_U": Usup,
    }
    for d in degrees:
        pack[f"poly_lower_d{d}"] = curves[d]["lo"]
        pack[f"poly_upper_d{d}"] = curves[d]["hi"]
    np.savez_compressed(abl / "all_horizons_all_degrees.npz", **pack)
    (abl / "per_horizon.json").write_text(json.dumps(per_horizon, indent=2))
    (abl / "horizons_table.csv").write_text("\n".join(csv_lines) + "\n")

    payload = {
        "plant": name,
        "display_name": DISPLAY[name],
        "source": str(src),
        "degrees": degrees,
        "times": times,
        "alpha": delta,
        "T": T,
        "finite_states": None,
        "rows": rows,
        "per_horizon_files": {
            "npz": "all_horizons_all_degrees.npz",
            "json": "per_horizon.json",
            "csv": "horizons_table.csv",
            "per_degree": [f"poly_d{d}.npz" for d in degrees],
        },
        "kind": "continuous_lqg",
        "nesting_repairs": int(n_nest),
        "method": "closed-form Gaussian quadratic-form MGF + continuous poly TailLP + continuous MC",
        "note": "All degrees evaluated at every horizon t=1..T; tip rows are t=T.",
    }
    (abl / "summary.json").write_text(json.dumps(payload, indent=2))
    _plot_traj(DISPLAY[name], mean, mgf_L, mgf_U, C, curves, degrees, abl)
    _plot_hist(DISPLAY[name], mean, mgf_L, mgf_U, C, curves, degrees, T, abl)
    return rows


def _plot_traj(plant, mean, mgf_L, mgf_U, C, curves, degrees, abl):
    colors = _poly_colors(degrees)
    T = C.shape[1] - 1
    t = np.arange(T + 1)
    n_draw = min(500, C.shape[0])
    idx = np.linspace(0, C.shape[0] - 1, n_draw).astype(int)
    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    ax.plot(t, (C[idx] - mean).T, color="#4c566a", alpha=0.9, lw=0.22, rasterized=True, zorder=1)
    ax.plot(t, np.zeros_like(t, dtype=float), color="#2e3440", lw=2.0, label="exact mean", zorder=3)
    q_lo = np.quantile(C, 0.025, axis=0) - mean
    q_hi = np.quantile(C, 0.975, axis=0) - mean
    ax.plot(t, q_lo, color="#2e3440", ls=":", lw=1.6, label="empirical 2.5%/97.5%", zorder=4)
    ax.plot(t, q_hi, color="#2e3440", ls=":", lw=1.6, zorder=4)
    ax.plot(t[1:], (mgf_L - mean)[1:], color=MGF_STYLE["color"], ls=MGF_STYLE["ls"],
            lw=MGF_STYLE["lw"], label="MGF", zorder=5)
    ax.plot(t[1:], (mgf_U - mean)[1:], color=MGF_STYLE["color"], ls=MGF_STYLE["ls"],
            lw=MGF_STYLE["lw"], zorder=5)
    for d in degrees:
        ts = np.asarray(curves[d]["times"], dtype=int)
        lo = curves[d]["lo"][ts] - mean[ts]
        hi = curves[d]["hi"][ts] - mean[ts]
        ax.plot(ts, lo, color=colors[d], lw=1.8, label=f"poly $d={d}$", zorder=6)
        ax.plot(ts, hi, color=colors[d], lw=1.8, zorder=6)
    span = max(
        float(np.nanmax(np.abs(mgf_L[1:] - mean[1:]))),
        float(np.nanmax(np.abs(mgf_U[1:] - mean[1:]))),
        float(np.abs(C[idx] - mean).max()),
    )
    ax.set_xlim(0, T)
    ax.set_ylim(-1.08 * span, 1.08 * span)
    ax.set_xlabel(r"time step $t$")
    ax.set_ylabel(r"$C_t-\mathbb{E}[C_t]$")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(abl / "traj_degree_ablation.png", dpi=200, bbox_inches="tight")
    fig.savefig(abl / "traj_degree_ablation.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_hist(plant, mean, mgf_L, mgf_U, C, curves, degrees, T, abl):
    colors = _poly_colors(degrees)
    x = C[:, T] - mean[T]
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    ax.hist(x, bins=60, weights=np.full(len(x), 1 / len(x)),
            color="#a3b1c6", edgecolor="#4c566a", linewidth=0.4, zorder=1)
    top = 0.74
    ax.axvline(0.0, ymax=top, color="#2e3440", lw=2.0, label="exact mean", zorder=3)
    q_lo, q_hi = np.quantile(x, [0.025, 0.975])
    ax.axvline(q_lo, ymax=top, color="#2e3440", ls=":", lw=1.6, label="empirical 2.5%/97.5%", zorder=4)
    ax.axvline(q_hi, ymax=top, color="#2e3440", ls=":", lw=1.6, zorder=4)
    ax.axvline(mgf_L[T] - mean[T], ymax=top, color=MGF_STYLE["color"], ls=MGF_STYLE["ls"],
               lw=MGF_STYLE["lw"], label="MGF", zorder=5)
    ax.axvline(mgf_U[T] - mean[T], ymax=top, color=MGF_STYLE["color"], ls=MGF_STYLE["ls"],
               lw=MGF_STYLE["lw"], zorder=5)
    for d in degrees:
        lo = curves[d]["lo"][T] - mean[T]
        hi = curves[d]["hi"][T] - mean[T]
        ax.axvline(lo, ymax=top, color=colors[d], lw=1.8, label=f"poly $d={d}$", zorder=6)
        ax.axvline(hi, ymax=top, color=colors[d], lw=1.8, zorder=6)
    half = max(abs(mgf_L[T] - mean[T]), abs(mgf_U[T] - mean[T]), abs(q_lo), abs(q_hi), 1e-9)
    lim = 1.12 * max(half, float(np.abs(x).max()))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(0, ax.get_ylim()[1] / (top - 0.03))
    ax.set_xlabel(rf"$C_{{{T}}}-\mathbb{{E}}[C_{{{T}}}]$")
    ax.set_ylabel("fraction of trajectories")
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(abl / "hist_degree_ablation.png", dpi=200, bbox_inches="tight")
    fig.savefig(abl / "hist_degree_ablation.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plants", nargs="*", default=PLANTS)
    ap.add_argument("--degrees", default=",".join(str(d) for d in DEFAULT_DEGREES))
    args = ap.parse_args()
    degrees = [int(x) for x in args.degrees.split(",") if x.strip()]
    ABL.mkdir(parents=True, exist_ok=True)
    for name in args.plants:
        run_plant(name, degrees)


if __name__ == "__main__":
    main()

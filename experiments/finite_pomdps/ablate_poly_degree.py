"""
Polynomial-degree ablation on saved finite POMDP runs
(Tiger / Network / Shopping-5 / Bridge patrol).

Reuses the fixed finite model, MGF bands, and Monte Carlo trajectories. Moments are
computed once at the largest degree; each smaller degree uses the moment prefix.

  PYTHONPATH=. python experiments/finite_pomdps/ablate_poly_degree.py
  PYTHONPATH=. python experiments/finite_pomdps/ablate_poly_degree.py tiger network
  PYTHONPATH=. python experiments/finite_pomdps/ablate_poly_degree.py bridge_patrol
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from returnbounds import io as save_artifacts
from returnbounds import moments as paper_moments
from returnbounds import polybound as pb
from returnbounds.profiling import peak_rss_kb

SRC_ROOT = _ROOT / "data" / "runs" / "finite_pomdps"
ABL = _ROOT / "data" / "runs" / "ablation_poly_degree"

RUNS = {
    "bridge_patrol": SRC_ROOT / "bridge_patrol",
    "tiger": SRC_ROOT / "tiger_tiger_hand",
    "network": SRC_ROOT / "network_network_hand",
    "shopping5": SRC_ROOT / "shopping5_shopping_hand",
}
DISPLAY = {
    "bridge_patrol": "Bridge patrol",
    "tiger": "Tiger",
    "network": "Network",
    "shopping5": "Shopping-5",
}
DEFAULT_DEGREES = [4, 8, 16, 20, 24]
TIMES_LARGE = list(range(1, 101, 5)) + [100]
TIMES_SMALL = list(range(1, 101))
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


def _poly_at_t_all_degrees(fm, t, Lsup, Usup, degrees, alpha):
    """Scaled moments once at max degree; TailLP for each degree prefix."""
    d_max = max(degrees)
    empty = {"n_inconsistent": 0, "max_eps_smooth": 0.0}
    if not np.isfinite(Lsup) or not np.isfinite(Usup) or Usup <= Lsup:
        mu = float(paper_moments.raw_moments(fm, t, 1)[t, 1])
        return {d: (mu, mu, empty) for d in degrees}

    c = 0.5 * (Lsup + Usup)
    h = 0.5 * (Usup - Lsup)
    my = paper_moments.scaled_moments_at_t(fm, t, d_max, c, h)
    out = {}
    for d in degrees:
        tau = pb.cheb_moments(my[: d + 1])
        y_lo, y_hi, diag = pb.poly_interval_t(tau, d, alpha)
        lo = float(np.clip(c + h * y_lo, Lsup, Usup))
        hi = float(np.clip(c + h * y_hi, Lsup, Usup))
        out[d] = (lo, hi, diag)
    return out


def run_one(
    key: str,
    degrees: list[int],
    alpha: float = 0.025,
    all_horizons: bool = True,
    log=print,
):
    src = RUNS[key]
    if not (src / "arrays.npz").exists() or not (src / "finite_model.npz").exists():
        raise FileNotFoundError(f"missing saved run for {key}: {src}")

    abl = ABL / key
    abl.mkdir(parents=True, exist_ok=True)
    summary = json.loads((src / "summary.json").read_text()) if (src / "summary.json").exists() else {}
    if (src / "config.json").exists():
        cfg = json.loads((src / "config.json").read_text())
    else:
        # Bridge patrol writes summary.json but not config.json.
        cfg = {
            "T": int(summary.get("T", 100)),
            "horizons": [25, 50, 100],
            "delta": float(summary.get("delta", alpha)),
            "poly_degree": int(summary.get("poly_degree", max(degrees))),
        }
    data = save_artifacts.load_run_arrays(src)
    fm = save_artifacts.load_finite_model(src / "finite_model.npz")

    T = int(cfg.get("T", summary.get("T", max(cfg.get("horizons", [100])))))
    mean = np.asarray(data["mean"], float)
    mgf_L = np.asarray(data["mgf_lower"], float)
    mgf_U = np.asarray(data["mgf_upper"], float)
    C = np.asarray(data["C"], float)
    n_states = int(fm.n_states)
    if all_horizons:
        times = [t for t in TIMES_SMALL if 1 <= t <= T]
    else:
        # Sparse horizons for traj curves; always include T for the table.
        times = sorted(set(t for t in TIMES_LARGE if 1 <= t <= T))
    if T not in times:
        times.append(T)
        times.sort()

    log("=" * 64)
    log(f"{key}: src={src}")
    log(
        f"  states={n_states}  T={T}  degrees={degrees}  n_times={len(times)}"
        + (" (all horizons)" if all_horizons else " (sparse)"),
        flush=True,
    )

    t0 = time.perf_counter()
    if "support_L" in data and "support_U" in data:
        Lsup = np.asarray(data["support_L"], float)
        Usup = np.asarray(data["support_U"], float)
        log(f"  reused stored support ({time.perf_counter() - t0:.1f}s)", flush=True)
    else:
        log("  exact support DP", flush=True)
        Lsup, Usup = paper_moments.support_interval(fm, T)
        log(f"  support in {time.perf_counter() - t0:.1f}s", flush=True)

    curves = {d: {"lo": np.full(T + 1, np.nan), "hi": np.full(T + 1, np.nan), "times": times} for d in degrees}
    diags = {d: {} for d in degrees}
    t0 = time.perf_counter()
    for i, t in enumerate(times):
        per_d = _poly_at_t_all_degrees(fm, t, float(Lsup[t]), float(Usup[t]), degrees, alpha)
        for d, (lo, hi, diag) in per_d.items():
            curves[d]["lo"][t] = lo
            curves[d]["hi"][t] = hi
            diags[d][str(t)] = diag
        if i == 0 or i == len(times) - 1 or (i + 1) % 5 == 0:
            d_show = degrees[-1]
            lo, hi = curves[d_show]["lo"][t], curves[d_show]["hi"][t]
            log(f"  t={t}: d={d_show} [{lo:.4f},{hi:.4f}]  ({i+1}/{len(times)})", flush=True)
    log(f"  all degrees/times in {time.perf_counter() - t0:.1f}s", flush=True)

    # Higher degree can only shrink the feasible set, so intervals must nest.
    # Project any TailLP numerical violations into the previous degree's interval.
    n_nest_fixes = 0
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
                n_nest_fixes += 1
                curves[d]["lo"][t] = new_lo
                curves[d]["hi"][t] = new_hi
    if n_nest_fixes:
        log(f"  nesting repairs: {n_nest_fixes}", flush=True)

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
            "seconds": None,
            "max_rss_kb": peak_rss_kb(),
            "n_times": len(times),
            "support_L": float(Lsup[T]),
            "support_U": float(Usup[T]),
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
            f"  d={d}: width={width:.4f}  exceed=({frac_lo:.4f},{frac_hi:.4f})  "
            f"[{lo[T]:.4f},{hi[T]:.4f}]  incons={tot_incons}"
            + ("  COLLAPSED" if row["collapsed"] else "")
        )

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
        "plant": key,
        "display_name": DISPLAY[key],
        "source": str(src),
        "degrees": degrees,
        "times": times,
        "alpha": alpha,
        "T": T,
        "finite_states": n_states,
        "rows": rows,
        "per_horizon_files": {
            "npz": "all_horizons_all_degrees.npz",
            "json": "per_horizon.json",
            "csv": "horizons_table.csv",
            "per_degree": [f"poly_d{d}.npz" for d in degrees],
        },
        "kind": "finite_pomdp",
        "nesting_repairs": int(n_nest_fixes),
        "note": (
            "All degrees at every horizon t=1..T"
            if all_horizons
            else "Sparse horizon grid; tip rows are t=T"
        ),
    }
    (abl / "summary.json").write_text(json.dumps(payload, indent=2))
    _plot_traj(DISPLAY[key], mean, mgf_L, mgf_U, C, curves, degrees, abl)
    _plot_hist(DISPLAY[key], mean, mgf_L, mgf_U, C, curves, degrees, T, abl)
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
    ax.plot(t[1:], (mgf_L - mean)[1:], color=MGF_STYLE["color"], ls=MGF_STYLE["ls"], lw=MGF_STYLE["lw"],
            label="MGF", zorder=5)
    ax.plot(t[1:], (mgf_U - mean)[1:], color=MGF_STYLE["color"], ls=MGF_STYLE["ls"], lw=MGF_STYLE["lw"], zorder=5)
    for d in degrees:
        ts = np.asarray(curves[d]["times"], dtype=int)
        lo = curves[d]["lo"][ts] - mean[ts]
        hi = curves[d]["hi"][ts] - mean[ts]
        ax.plot(ts, lo, color=colors[d], lw=1.8, marker="o", ms=2.5, label=f"poly $d={d}$", zorder=6)
        ax.plot(ts, hi, color=colors[d], lw=1.8, marker="o", ms=2.5, zorder=6)
    span = max(
        float(np.nanmax(np.abs(mgf_L[1:] - mean[1:]))),
        float(np.nanmax(np.abs(mgf_U[1:] - mean[1:]))),
        float(np.abs(C[idx] - mean).max()),
    )
    ax.set_xlim(0, T)
    ax.set_ylim(-1.08 * span, 1.08 * span)
    ax.set_xlabel(r"time step $t$")
    ax.set_ylabel(r"$G_t-\mathbb{E}[G_t]$")
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
        if not (np.isfinite(lo) and np.isfinite(hi)):
            continue
        ax.axvline(lo, ymax=top, color=colors[d], lw=1.8, label=f"poly $d={d}$", zorder=6)
        ax.axvline(hi, ymax=top, color=colors[d], lw=1.8, zorder=6)
    half = max(abs(mgf_L[T] - mean[T]), abs(mgf_U[T] - mean[T]), abs(q_lo), abs(q_hi), 1e-9)
    lim = 1.12 * max(half, float(np.abs(x).max()))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(0, ax.get_ylim()[1] / (top - 0.03))
    ax.set_xlabel(rf"$G_{{{T}}}-\mathbb{{E}}[G_{{{T}}}]$")
    ax.set_ylabel("fraction of trajectories")
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(abl / "hist_degree_ablation.png", dpi=200, bbox_inches="tight")
    fig.savefig(abl / "hist_degree_ablation.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", default=list(RUNS.keys()))
    ap.add_argument("--degrees", default=",".join(str(d) for d in DEFAULT_DEGREES))
    ap.add_argument(
        "--all-horizons",
        action="store_true",
        default=True,
        help="Evaluate every t=1..T (default).",
    )
    ap.add_argument(
        "--sparse-horizons",
        action="store_true",
        help="Use every-5th horizon grid instead of all t.",
    )
    args = ap.parse_args()
    degrees = [int(x) for x in args.degrees.split(",") if x.strip()]
    all_horizons = not args.sparse_horizons
    ABL.mkdir(parents=True, exist_ok=True)

    all_rows = {}
    for key in args.names:
        if key not in RUNS:
            raise SystemExit(f"unknown name {key}; choose from {list(RUNS)}")
        all_rows[key] = run_one(key, degrees, all_horizons=all_horizons)

    # append / merge into shared table.csv
    table_path = ABL / "table.csv"
    existing = {}
    if table_path.exists():
        for line in table_path.read_text().splitlines()[1:]:
            if not line.strip():
                continue
            parts = line.split(",")
            existing[(parts[0], int(parts[1]))] = line

    header = (
        "plant,degree,t,poly_L,poly_U,width,frac_below,frac_above,seconds,"
        "mgf_width,emp_width,collapsed"
    )
    for name, rows in all_rows.items():
        for r in rows:
            existing[(name, r["degree"])] = ",".join(str(x) for x in [
                name, r["degree"], r["t"], f"{r['poly_L']:.6f}", f"{r['poly_U']:.6f}",
                f"{r['width']:.6f}", f"{r['frac_below']:.6f}", f"{r['frac_above']:.6f}",
                "",
                f"{r['mgf_U']-r['mgf_L']:.6f}", f"{r['emp_q975']-r['emp_q025']:.6f}",
                int(r["collapsed"]),
            ])
    lines = [header] + [existing[k] for k in sorted(existing, key=lambda x: (x[0], x[1]))]
    table_path.write_text("\n".join(lines) + "\n")
    print("=" * 64)
    print(f"wrote {table_path}")


if __name__ == "__main__":
    main()

"""
Regenerate LQG paper traj/hist figures from the continuous closed-form MGF
plus degree-16 polynomial TailLP on closed-form moments.

  PYTHONPATH=. python experiments/lqg/plot_continuous_paper.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from returnbounds import continuous as gm
from returnbounds import plotting as figures
from returnbounds.plant import build_plant
from returnbounds.simulate_continuous import simulate as cont_simulate

HERE = Path(__file__).resolve().parent
DATA = _ROOT / "data" / "runs" / "continuous_baselines"
PLANTS = [
    "mass_spring_damper",
    "coupled_masses",
    "quarter_car",
    "two_room_thermal",
    "cartpole_linear",
]
T_MAX = 100
DELTA = 0.025
DEGREE = 16
N_TRAJ = 20000
SEED = 20261026


def _load(path):
    with open(path) as f:
        return json.load(f)


def _dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _poly_worker(payload):
    t, J, b, d0, degree, alpha = payload
    plo, phi, _, _, diag = gm.poly_bounds_from_form(J, b, d0, degree=degree, alpha=alpha)
    return int(t), float(plo), float(phi), diag


def series_for_plant(plant, T, delta, degree, log=print, n_workers=None):
    """Closed-form mean, Chernoff, and polynomial bands for t=0..T."""
    mean = np.zeros(T + 1)
    var = np.zeros(T + 1)
    lower = np.full(T + 1, np.nan)
    upper = np.full(T + 1, np.nan)
    poly_lower = np.full(T + 1, np.nan)
    poly_upper = np.full(T + 1, np.nan)
    forms = {}
    t0 = time.perf_counter()
    for t in range(1, T + 1):
        an = gm.analyze_horizon(plant, t, delta=delta)
        mean[t] = an["mean"]
        var[t] = an["var"]
        lower[t] = an["lower"]
        upper[t] = an["upper"]
        forms[t] = (an["J"], an["b"], an["d0"])
        if t in (1, 25, 50, 75, 100) or t == T:
            log(f"  t={t}: mean={mean[t]:.4f}  MGF=[{lower[t]:.4f},{upper[t]:.4f}]")
    log(f"  closed-form MGF series in {time.perf_counter() - t0:.1f}s")

    payloads = [(t, forms[t][0], forms[t][1], forms[t][2], degree, delta) for t in range(1, T + 1)]
    if n_workers is None:
        n_workers = max(1, min(len(payloads), (os.cpu_count() or 2) - 1))
    log(f"  polynomial LPs for {len(payloads)} horizons (d={degree}, workers={n_workers})")
    t0 = time.perf_counter()
    n_inconsistent = 0
    max_eps = 0.0
    max_tail_corr = 0.0
    max_tail_corr_upper = 0.0
    max_tail_corr_lower = 0.0
    max_eps_out = 0.0
    n_fallback_upper = 0
    n_guard_upper = 0
    n_fallback_lower = 0
    per_t_trunc = {}
    if n_workers <= 1:
        results = [_poly_worker(p) for p in payloads]
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futs = [pool.submit(_poly_worker, p) for p in payloads]
            results = []
            done = 0
            for fut in as_completed(futs):
                results.append(fut.result())
                done += 1
                t, plo, phi, diag = results[-1]
                if done in (1, len(payloads)) or t in (25, 50, 75, 100):
                    phi_s = f"{phi:.4f}" if np.isfinite(phi) else "inf"
                    log(f"  poly T={t}: [{plo:.4f},{phi_s}]  ({done}/{len(payloads)})")
    for t, plo, phi, diag in results:
        poly_lower[t], poly_upper[t] = plo, phi
        n_inconsistent += int(diag["n_inconsistent"])
        max_eps = max(max_eps, float(diag["max_eps_smooth"]))
        max_tail_corr = max(max_tail_corr, float(diag.get("max_tail_correction", 0.0)))
        max_tail_corr_upper = max(
            max_tail_corr_upper, float(diag.get("max_tail_correction_upper", 0.0))
        )
        max_tail_corr_lower = max(
            max_tail_corr_lower, float(diag.get("max_tail_correction_lower", 0.0))
        )
        max_eps_out = max(max_eps_out, float(diag.get("eps_out", 0.0) or 0.0))
        n_fallback_upper += int(diag.get("n_fallback_upper", 0))
        n_guard_upper += int(diag.get("n_guard_upper", 0))
        n_fallback_lower += int(diag.get("n_fallback_lower", 0))
        if t in (25, 50, 100):
            per_t_trunc[str(t)] = {
                "poly_lower": float(plo),
                "poly_upper": float(phi) if np.isfinite(phi) else float("inf"),
                "eps_plus": float(diag.get("eps_plus", 0.0)),
                "eps_minus": float(diag.get("eps_minus", 0.0)),
                "log10_eps_out": float(diag.get("log10_eps_out", float("nan"))),
                "eps_out": float(diag.get("eps_out", 0.0)),
                "max_tail_correction": float(diag.get("max_tail_correction", 0.0)),
                "max_tail_correction_upper": float(
                    diag.get("max_tail_correction_upper", 0.0)
                ),
                "max_tail_correction_lower": float(
                    diag.get("max_tail_correction_lower", 0.0)
                ),
                "n_fallback_upper": int(diag.get("n_fallback_upper", 0)),
                "n_guard_upper": int(diag.get("n_guard_upper", 0)),
                "n_fallback_lower": int(diag.get("n_fallback_lower", 0)),
                "n_inconsistent": int(diag["n_inconsistent"]),
            }
    log(f"  polynomial series in {time.perf_counter() - t0:.1f}s  "
        f"n_inconsistent={n_inconsistent} max_eps_smooth={max_eps:.2e} "
        f"max_tail_correction={max_tail_corr:.2e} "
        f"corrU={max_tail_corr_upper:.2e} corrL={max_tail_corr_lower:.2e} "
        f"fallbackU={n_fallback_upper} guardU={n_guard_upper}")
    return mean, var, lower, upper, poly_lower, poly_upper, {
        "n_inconsistent": int(n_inconsistent),
        "max_eps_smooth": float(max_eps),
        "max_eps_out": float(max_eps_out),
        "max_tail_correction": float(max_tail_corr),
        "max_tail_correction_upper": float(max_tail_corr_upper),
        "max_tail_correction_lower": float(max_tail_corr_lower),
        "n_fallback_upper": int(n_fallback_upper),
        "n_guard_upper": int(n_guard_upper),
        "n_fallback_lower": int(n_fallback_lower),
        "truncation_at": per_t_trunc,
    }


def run_plant(name, fig_dir: Path, log=print):
    cfg = _load(HERE / "configs" / f"{name}.json")
    plant = build_plant(cfg)
    delta = float(cfg.get("delta", DELTA))
    out = DATA / name
    out.mkdir(parents=True, exist_ok=True)
    log(f"=== continuous paper figures: {plant.name} ===")

    mean, var, lower, upper, poly_lower, poly_upper, poly_diag = series_for_plant(
        plant, T_MAX, delta, DEGREE, log=log
    )

    npz = out / "continuous_C.npz"
    if npz.exists():
        C = np.load(npz)["C"]
        if C.shape[1] < T_MAX + 1:
            log(f"  existing C too short ({C.shape}); resimulating")
            C = cont_simulate(plant, N_TRAJ, T_MAX, SEED)["C"]
            np.savez_compressed(npz, C=C)
        else:
            log(f"  reused {npz}")
    else:
        log(f"  simulating {N_TRAJ} continuous trajectories")
        C = cont_simulate(plant, N_TRAJ, T_MAX, SEED)["C"]
        np.savez_compressed(npz, C=C)

    trunc_report = {}
    for t_key, info in poly_diag.get("truncation_at", {}).items():
        t = int(t_key)
        trunc_report[t_key] = {
            **info,
            "frac_below_poly": float(np.mean(C[:, t] < poly_lower[t])),
            "frac_above_poly": float(np.mean(C[:, t] > poly_upper[t])),
        }
    summary = {
        "name": plant.name,
        "method": "closed-form Gaussian quadratic-form MGF + degree-16 poly TailLP + continuous MC",
        "T": T_MAX,
        "delta": delta,
        "poly_degree": DEGREE,
        "exact_mean": float(mean[T_MAX]),
        "exact_var": float(var[T_MAX]),
        "lower": float(lower[T_MAX]),
        "upper": float(upper[T_MAX]),
        "poly_lower": float(poly_lower[T_MAX]),
        "poly_upper": float(poly_upper[T_MAX]),
        "sim_mean": float(C[:, T_MAX].mean()),
        "empirical_q025": float(np.quantile(C[:, T_MAX], 0.025)),
        "empirical_q975": float(np.quantile(C[:, T_MAX], 0.975)),
        "frac_below_mgf": float(np.mean(C[:, T_MAX] < lower[T_MAX])),
        "frac_above_mgf": float(np.mean(C[:, T_MAX] > upper[T_MAX])),
        "frac_below_poly": float(np.mean(C[:, T_MAX] < poly_lower[T_MAX])),
        "frac_above_poly": float(np.mean(C[:, T_MAX] > poly_upper[T_MAX])),
        "poly_diagnostics": poly_diag,
        "truncation_report": trunc_report,
        "n_traj": int(C.shape[0]),
    }
    _dump(out / "paper_continuous_summary.json", summary)
    np.savez_compressed(
        out / "paper_continuous_arrays.npz",
        mean=mean, var=var, lower=lower, upper=upper,
        poly_lower=poly_lower, poly_upper=poly_upper, C=C,
    )

    fig_dir.mkdir(parents=True, exist_ok=True)
    traj = fig_dir / f"{name}_traj.png"
    hist = fig_dir / f"{name}_hist.png"
    figures.cost_trajectories(
        C, mean, lower, upper, traj,
        f"{plant.name} (continuous closed-form)",
        n_draw=800, poly_lower=poly_lower, poly_upper=poly_upper,
    )
    figures.histogram(
        C[:, T_MAX], mean[T_MAX], lower[T_MAX], upper[T_MAX], T_MAX,
        hist, plant.name + " (continuous)",
        poly_lower=float(poly_lower[T_MAX]), poly_upper=float(poly_upper[T_MAX]),
    )
    figures.cost_trajectories(
        C, mean, lower, upper, out / "cost_trajectories_95pct.png",
        f"{plant.name} (continuous closed-form)",
        n_draw=800, poly_lower=poly_lower, poly_upper=poly_upper,
    )
    figures.histogram(
        C[:, T_MAX], mean[T_MAX], lower[T_MAX], upper[T_MAX], T_MAX,
        out / f"histogram_T{T_MAX}.png", plant.name + " (continuous)",
        poly_lower=float(poly_lower[T_MAX]), poly_upper=float(poly_upper[T_MAX]),
    )
    pu = summary["poly_upper"]
    pu_s = f"{pu:.4f}" if np.isfinite(pu) else "inf"
    log(
        f"  wrote {traj.name}, {hist.name}  T={T_MAX} mean={summary['exact_mean']:.4f}  "
        f"MGF=[{summary['lower']:.4f},{summary['upper']:.4f}]  "
        f"poly=[{summary['poly_lower']:.4f},{pu_s}]  "
        f"exM={summary['frac_below_mgf']:.4f}/{summary['frac_above_mgf']:.4f}  "
        f"exP={summary['frac_below_poly']:.4f}/{summary['frac_above_poly']:.4f}"
    )
    return summary


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out-figures",
        type=Path,
        default=_ROOT / "figures",
    )
    p.add_argument("--plants", nargs="*", default=PLANTS)
    args = p.parse_args(argv)
    rows = []
    for name in args.plants:
        rows.append(run_plant(name, args.out_figures))
    _dump(DATA / "paper_continuous_table.json", rows)
    print("\n=== continuous closed-form @ T=100 ===")
    print(f"{'plant':20s} {'mean':>10s} {'MGF':>22s} {'poly':>22s} {'exM':>11s} {'exP':>11s}")
    for r in rows:
        pu = r["poly_upper"]
        pu_s = f"{pu:.2f}" if np.isfinite(pu) else "inf"
        print(
            f"{r['name']:20s} {r['exact_mean']:10.4f} "
            f"[{r['lower']:.2f},{r['upper']:.2f}] "
            f"[{r['poly_lower']:.2f},{pu_s}] "
            f"{r['frac_below_mgf']:.4f}/{r['frac_above_mgf']:.4f} "
            f"{r['frac_below_poly']:.4f}/{r['frac_above_poly']:.4f}"
        )


if __name__ == "__main__":
    main()

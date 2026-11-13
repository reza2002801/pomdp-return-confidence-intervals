"""
Continuous-plant MGF bounds + Monte Carlo for all LQG configs.

  PYTHONPATH=. python experiments/lqg/continuous_baselines.py

Writes data/runs/continuous_baselines/<plant>/summary.json and a suite table.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from returnbounds import continuous as gm
from returnbounds.plant import build_plant
from returnbounds.simulate_continuous import simulate as cont_simulate

HERE = Path(__file__).resolve().parent
OUT_ROOT = _ROOT / "data" / "runs" / "continuous_baselines"
HORIZONS = (25, 50, 100)
N_TRAJ = 20000
SEED = 20261026
DELTA = 0.025

PLANTS = [
    "mass_spring_damper",
    "two_room_thermal",
    "cartpole_linear",
    "coupled_masses",
    "quarter_car",
]


def _load(path):
    with open(path) as f:
        return json.load(f)


def _dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def run_plant(name, log=print):
    cfg_path = HERE / "configs" / f"{name}.json"
    cfg = _load(cfg_path)
    plant = build_plant(cfg)
    delta = float(cfg.get("delta", DELTA))
    sim_cfg = cfg.get("simulation", {})
    n_traj = int(sim_cfg.get("n_traj", N_TRAJ))
    seed = int(sim_cfg.get("seed", SEED))
    T_max = max(HORIZONS)

    log(f"=== continuous {plant.name}: analyze + MC (n={n_traj}, T≤{T_max}) ===")
    t0 = time.perf_counter()
    per_T = {}
    for T in HORIZONS:
        an = gm.analyze_horizon(plant, T, delta=delta)
        # strip large arrays before saving
        per_T[str(T)] = {
            "T": T,
            "mean": float(an["mean"]),
            "var": float(an["var"]),
            "lower": float(an["lower"]),
            "upper": float(an["upper"]),
            "lambda_upper": float(an.get("lambda_upper", float("nan"))),
            "lambda_lower": float(an.get("lambda_lower", float("nan"))),
            "lambda_max_domain": float(an.get("lambda_max_domain", float("nan"))),
        }
        log(f"  T={T}: mean={an['mean']:.4f}  MGF=[{an['lower']:.4f},{an['upper']:.4f}]")

    log(f"  simulating {n_traj} continuous trajectories to T={T_max}")
    sim = cont_simulate(plant, n_traj, T_max, seed)
    C = sim["C"]

    for T in HORIZONS:
        L, U = per_T[str(T)]["lower"], per_T[str(T)]["upper"]
        below = int(np.sum(C[:, T] < L))
        above = int(np.sum(C[:, T] > U))
        per_T[str(T)].update({
            "sim_mean": float(C[:, T].mean()),
            "empirical_q025": float(np.quantile(C[:, T], 0.025)),
            "empirical_q975": float(np.quantile(C[:, T], 0.975)),
            "frac_below_mgf": below / n_traj,
            "frac_above_mgf": above / n_traj,
            "n_below": below,
            "n_above": above,
            "n_traj": n_traj,
        })
        log(
            f"  T={T}: exceed below={below/n_traj:.4f} above={above/n_traj:.4f}  "
            f"emp q=[{per_T[str(T)]['empirical_q025']:.4f},{per_T[str(T)]['empirical_q975']:.4f}]"
        )

    out_dir = OUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "continuous_C.npz", C=C)
    summary = {
        "name": plant.name,
        "method": "continuous.analyze_horizon + simulate_continuous",
        "delta": delta,
        "horizons": list(HORIZONS),
        "n_traj": n_traj,
        "seed": seed,
        "per_horizon": per_T,
        "wall_seconds": time.perf_counter() - t0,
        "guarantee": (
            "Continuous-state linear-Gaussian reference only; "
            "not a certificate for the quantized finite model."
        ),
    }
    _dump(out_dir / "summary.json", summary)
    _dump(out_dir / "config.json", cfg)
    log(f"  wrote {out_dir} in {summary['wall_seconds']:.1f}s")
    return summary


def main(argv):
    names = argv[1:] if len(argv) > 1 else PLANTS
    rows = []
    for name in names:
        s = run_plant(name)
        for T in HORIZONS:
            h = s["per_horizon"][str(T)]
            rows.append({
                "plant": s["name"],
                "T": T,
                "mean": h["mean"],
                "mgf_L": h["lower"],
                "mgf_U": h["upper"],
                "frac_below": h["frac_below_mgf"],
                "frac_above": h["frac_above_mgf"],
            })
    _dump(OUT_ROOT / "summary_table.json", rows)
    print("\n=== continuous suite (all horizons) ===")
    print(f"{'plant':20s} {'T':>4s} {'mean':>10s} {'MGF_L':>10s} {'MGF_U':>10s} "
          f"{'P(<L)':>8s} {'P(>U)':>8s}")
    for r in rows:
        print(
            f"{r['plant']:20s} {r['T']:4d} {r['mean']:10.4f} {r['mgf_L']:10.4f} {r['mgf_U']:10.4f} "
            f"{r['frac_below']:8.4f} {r['frac_above']:8.4f}"
        )


if __name__ == "__main__":
    main(sys.argv)

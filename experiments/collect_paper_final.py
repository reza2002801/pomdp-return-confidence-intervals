"""
Aggregate t=100 paper numbers → data/runs/paper_final/FINAL_PAPER_DATA.json.

Reads stored finite / LQG run arrays only (no figure PDFs, no timing remasure).
Public `results/paper_numbers.json` is a copy of that file (see reproduce.py tables).

  PYTHONPATH=. python3 experiments/collect_paper_final.py
"""
from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from returnbounds import io as save_artifacts
from returnbounds import martingale as mg
from returnbounds import moments as paper_moments

OUT = _ROOT / "data" / "runs" / "paper_final"
DELTA = 0.025
DEGREE = 16
T_MAX = 100
N_TRAJ = 20000
LQG_SEED = 20261026
WORKERS = max(1, (os.cpu_count() or 2) - 1)

FINITE = [
    ("bridge_patrol", "Bridge patrol", "finite_pomdps/bridge_patrol"),
    ("tiger", "Tiger", "finite_pomdps/tiger_tiger_hand"),
    ("network", "Network", "finite_pomdps/network_network_hand"),
    ("shopping5", "Shopping-5", "finite_pomdps/shopping5_shopping_hand"),
]
LQG = [
    ("mass_spring_damper", "Mass spring damper"),
    ("coupled_masses", "Coupled masses"),
    ("quarter_car", "Quarter-car"),
    ("two_room_thermal", "Two-room thermal"),
    ("cartpole_linear", "Cart-pole"),
]
CONT = _ROOT / "data" / "runs" / "continuous_baselines"


def _dump(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n")


def collect_section_b():
    out = {"t": 100, "delta_per_tail": DELTA, "finite": {}, "lqg": {}, "aggregates": {}}
    fre_ratios, az_ratios, poly_reds_f = [], [], []
    all_ex = []
    for key, display, rel in FINITE:
        d = _ROOT / "data" / "runs" / rel
        arr = np.load(d / "arrays.npz")
        mean = np.asarray(arr["mean"], float)
        mgf_L = np.asarray(arr["mgf_lower"], float)
        mgf_U = np.asarray(arr["mgf_upper"], float)
        poly_L = np.asarray(arr["poly_lower"], float)
        poly_U = np.asarray(arr["poly_upper"], float)
        C = np.asarray(arr["C"], float)
        t = T_MAX
        w_mgf = float(mgf_U[t] - mgf_L[t])
        w_poly = float(poly_U[t] - poly_L[t])
        fm = save_artifacts.load_finite_model(d / "finite_model.npz")
        raw = mg.bands(fm, T_MAX, DELTA)
        aL, aU = float(raw["azuma_L"][t]), float(raw["azuma_U"][t])
        fL, fU = float(raw["freedman_L"][t]), float(raw["freedman_U"][t])
        w_fr = fU - fL
        w_az = aU - aL
        r_fr = w_fr / w_mgf
        r_az = w_az / w_mgf
        fre_ratios.append(r_fr)
        az_ratios.append(r_az)
        red = 1.0 - w_poly / w_mgf
        poly_reds_f.append(red)
        xT = C[:, t]
        ex = {
            "mgf_below": float(np.mean(xT < mgf_L[t])),
            "mgf_above": float(np.mean(xT > mgf_U[t])),
            "poly_below": float(np.mean(xT < poly_L[t])),
            "poly_above": float(np.mean(xT > poly_U[t])),
            "freedman_below": float(np.mean(xT < fL)),
            "freedman_above": float(np.mean(xT > fU)),
            "azuma_below": float(np.mean(xT < aL)),
            "azuma_above": float(np.mean(xT > aU)),
        }
        for v in ex.values():
            all_ex.append(v)
        Ls, Us = paper_moments.support_interval(fm, T_MAX)
        n_lo = n_hi = 0
        for tt in range(1, T_MAX + 1):
            if abs(poly_L[tt] - Ls[tt]) <= 1e-9 * max(1.0, abs(Ls[tt])):
                n_lo += 1
            if abs(poly_U[tt] - Us[tt]) <= 1e-9 * max(1.0, abs(Us[tt])):
                n_hi += 1
        wider = bool(w_poly > w_mgf + 1e-12)
        out["finite"][key] = {
            "display": display,
            "mgf_LU": [float(mgf_L[t]), float(mgf_U[t])],
            "poly_LU": [float(poly_L[t]), float(poly_U[t])],
            "mgf_width": w_mgf,
            "poly_width": w_poly,
            "poly_width_reduction_vs_mgf": red,
            "freedman_LU": [fL, fU],
            "azuma_LU": [aL, aU],
            "freedman_over_mgf_width_ratio": r_fr,
            "azuma_over_mgf_width_ratio": r_az,
            "exceedances": ex,
            "poly_at_support_end_counts": {
                "lower_at_support": n_lo,
                "upper_at_support": n_hi,
            },
            "poly_wider_than_mgf_at_t100": wider,
            "empirical_q025": float(np.quantile(xT, 0.025)),
            "empirical_q975": float(np.quantile(xT, 0.975)),
        }

    poly_reds_l = []
    for key, display in LQG:
        d = CONT / key
        z = np.load(d / "paper_continuous_arrays.npz")
        s = json.loads((d / "paper_continuous_summary.json").read_text())
        mgf_L = np.asarray(z["lower"], float)
        mgf_U = np.asarray(z["upper"], float)
        poly_L = np.asarray(z["poly_lower"], float)
        poly_U = np.asarray(z["poly_upper"], float)
        C = np.asarray(z["C"], float)
        t = T_MAX
        w_mgf = float(mgf_U[t] - mgf_L[t])
        w_poly = float(poly_U[t] - poly_L[t])
        red = 1.0 - w_poly / w_mgf
        poly_reds_l.append(red)
        xT = C[:, t]
        ex = {
            "mgf_below": float(np.mean(xT < mgf_L[t])),
            "mgf_above": float(np.mean(xT > mgf_U[t])),
            "poly_below": float(np.mean(xT < poly_L[t])),
            "poly_above": float(np.mean(xT > poly_U[t])),
        }
        for v in ex.values():
            all_ex.append(v)
        diag = s.get("poly_diagnostics") or {}
        trunc = diag.get("truncation_at") or {}
        n_fb_u = int(diag.get("n_fallback_upper") or 0)
        n_fb_l = int(diag.get("n_fallback_lower") or 0)
        if trunc:
            n_fb_u = sum(int(v.get("n_fallback_upper") or 0) for v in trunc.values())
            n_fb_l = sum(int(v.get("n_fallback_lower") or 0) for v in trunc.values())
            if "n_fallback_lower" in diag:
                n_fb_l = int(diag["n_fallback_lower"])
            if "n_fallback_upper" in diag:
                n_fb_u = int(diag["n_fallback_upper"])
        wider = bool(w_poly > w_mgf + 1e-12)
        out["lqg"][key] = {
            "display": display,
            "mgf_LU": [float(mgf_L[t]), float(mgf_U[t])],
            "poly_LU": [float(poly_L[t]), float(poly_U[t])],
            "mgf_width": w_mgf,
            "poly_width": w_poly,
            "poly_width_reduction_vs_mgf": red,
            "exceedances": ex,
            "poly_fallback_counts": {
                "n_fallback_upper": n_fb_u,
                "n_fallback_lower": n_fb_l,
                "n_guard_upper": int(diag.get("n_guard_upper") or 0),
            },
            "poly_wider_than_mgf_at_t100": wider,
            "empirical_q025": float(np.quantile(xT, 0.025)),
            "empirical_q975": float(np.quantile(xT, 0.975)),
        }

    out["aggregates"] = {
        "freedman_over_mgf_ratio_minmax": [min(fre_ratios), max(fre_ratios)],
        "azuma_over_mgf_ratio_minmax": [min(az_ratios), max(az_ratios)],
        "poly_width_reduction_finite_minmax": [min(poly_reds_f), max(poly_reds_f)],
        "poly_width_reduction_lqg_minmax": [min(poly_reds_l), max(poly_reds_l)],
        "poly_width_reduction_all_minmax": [
            min(poly_reds_f + poly_reds_l),
            max(poly_reds_f + poly_reds_l),
        ],
        "max_empirical_exceedance": max(all_ex),
        "any_poly_wider_than_mgf_t100": any(
            out["finite"][k]["poly_wider_than_mgf_at_t100"] for k in out["finite"]
        )
        or any(out["lqg"][k]["poly_wider_than_mgf_at_t100"] for k in out["lqg"]),
    }
    out["unrounded"] = {
        "coupled_masses_empirical_upper_quantile": out["lqg"]["coupled_masses"]["empirical_q975"],
        "tiger_poly_U": out["finite"]["tiger"]["poly_LU"][1],
        "cartpole_mgf_L": out["lqg"]["cartpole_linear"]["mgf_LU"][0],
    }
    return out


def main(argv=None):
    argv = argv or sys.argv[1:]
    if argv:
        raise SystemExit(f"unexpected arguments: {argv}")

    section_b = collect_section_b()
    env = {
        "python_version": platform.python_version(),
        "numpy": __import__("numpy").__version__,
        "scipy": __import__("scipy").__version__,
        "matplotlib": __import__("matplotlib").__version__,
        "poly_lp_workers": WORKERS,
        "cpu_count": os.cpu_count(),
    }
    payload = {
        "environment": env,
        "settings": {
            "T": T_MAX,
            "degree": DEGREE,
            "delta_per_tail": DELTA,
            "n_traj": N_TRAJ,
            "poly_workers": WORKERS,
            "lqg_seed": LQG_SEED,
        },
        "B": section_b,
    }
    out_path = OUT / "FINAL_PAPER_DATA.json"
    _dump(out_path, payload)
    print("WROTE", out_path, flush=True)


if __name__ == "__main__":
    main()

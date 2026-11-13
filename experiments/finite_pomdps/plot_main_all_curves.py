"""Shared loaders for paper curve/hist figures (4 finite + 5 LQG)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

CONT = _ROOT / "data" / "runs" / "continuous_baselines"
FS = _ROOT / "data" / "runs" / "finite_pomdps"

# Caption order: (a)-(d) finite, (e)-(i) LQG (kept for reference; plotting lives elsewhere).
PANELS = [
    ("(a) Bridge-patrol", FS / "bridge_patrol" / "arrays.npz", "fs"),
    ("(b) Tiger", FS / "tiger_tiger_hand" / "arrays.npz", "fs"),
    ("(c) Network", FS / "network_network_hand" / "arrays.npz", "fs"),
    ("(d) Shopping-5", FS / "shopping5_shopping_hand" / "arrays.npz", "fs"),
    ("(e) Mass-spring-damper", CONT / "mass_spring_damper" / "paper_continuous_arrays.npz", "cont"),
    ("(f) Coupled masses", CONT / "coupled_masses" / "paper_continuous_arrays.npz", "cont"),
    ("(g) Quarter-car", CONT / "quarter_car" / "paper_continuous_arrays.npz", "cont"),
    ("(h) Thermal", CONT / "two_room_thermal" / "paper_continuous_arrays.npz", "cont"),
    ("(i) Cart-pole", CONT / "cartpole_linear" / "paper_continuous_arrays.npz", "cont"),
]


def load(path: Path, kind: str) -> dict:
    z = np.load(path)
    if kind == "cont":
        return dict(
            mean=np.asarray(z["mean"], float),
            mgf_L=np.asarray(z["lower"], float),
            mgf_U=np.asarray(z["upper"], float),
            poly_L=np.asarray(z["poly_lower"], float),
            poly_U=np.asarray(z["poly_upper"], float),
            C=np.asarray(z["C"], float),
        )
    return dict(
        mean=np.asarray(z["mean"], float),
        mgf_L=np.asarray(z["mgf_lower"], float),
        mgf_U=np.asarray(z["mgf_upper"], float),
        poly_L=np.asarray(z["poly_lower"], float),
        poly_U=np.asarray(z["poly_upper"], float),
        C=np.asarray(z["C"], float),
    )


if __name__ == "__main__":
    raise SystemExit(
        "plot_main_all_curves.py is a shared loader only; "
        "use plot_main_curve_hist_pairs.py for paper figures."
    )

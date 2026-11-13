"""
Paper-algorithm pipeline on finite POMDP product chains (Tiger / Network / Shopping-5).

MGF Chernoff, polynomial LP on exact support,
Monte Carlo on the finite model. Outputs under data/runs/finite_pomdps/<problem>_<policy>/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from returnbounds import io as save_artifacts
from returnbounds import mgf
from returnbounds import moments as paper_moments
from returnbounds import plotting as figures
from returnbounds.build_quantized import simulate as finite_simulate
from returnbounds.profiling import StageTimer

from experiments.finite_pomdps.policies import make_policies
from experiments.finite_pomdps.pomdp_parse import parse_pomdp
from experiments.finite_pomdps.product_model import build_product
from experiments.finite_pomdps.shopping5 import build_shopping5

HERE = Path(__file__).resolve().parent
POMDP_DIR = HERE / "pomdps"
OUT_ROOT = _ROOT / "data" / "runs" / "finite_pomdps"

DEFAULT_CFG = {
    "T": 100,
    "horizons": [25, 50, 100],
    "delta": 0.025,
    "poly_degree": 16,
    "lambda_grid": {
        "min": 0.01,
        "max": 100.0,
        "points_per_decade": 8,
        "max_expansions": 2,
    },
    "simulation": {"n_traj": 20000, "seed": 0, "n_draw": 800},
}


def _dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_problem(name: str):
    name = name.lower()
    if name == "tiger":
        return parse_pomdp(POMDP_DIR / "tiger.95.POMDP", name="tiger")
    if name == "network":
        return parse_pomdp(POMDP_DIR / "network.POMDP", name="network")
    if name in ("shopping", "shopping5", "shop"):
        return build_shopping5()
    raise ValueError(f"unknown problem {name}")


def run_one(pomdp, ctrl, out_dir, cfg=None, log=print):
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    timer = StageTimer()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    T = int(cfg["T"])
    delta = float(cfg["delta"])
    degree = int(cfg["poly_degree"])
    horizons = list(cfg["horizons"])

    _dump(out_dir / "config.json", {
        **cfg,
        "pomdp": pomdp.name,
        "policy": ctrl.name,
        "reward_depends_only_on_sa": pomdp.reward_depends_only_on_sa(),
    })

    log(f"=== {pomdp.name} / {ctrl.name} ===")
    with timer.stage("build_product"):
        fm = build_product(pomdp, ctrl, log=log)
    _dump(out_dir / "finite_pilot.json", fm.pilot)
    save_artifacts.save_finite_model(fm, out_dir / "finite_model.npz")

    with timer.stage("mgf"):
        op = mgf.TiltedOperator(fm)
        reach = mgf.reachable_sets(op, fm.x0, T)
        mgf.check_stochastic(op, reach, T)
        lam_cfg = cfg["lambda_grid"]
        lams, log_m, valid, expansions, mgf_seconds = mgf.bounds_with_expansion(
            op, fm.x0, T, horizons, [delta], lam_cfg, reach, log=log
        )
        upper = np.full(T + 1, np.nan)
        for t in range(1, T + 1):
            upper[t], _ = mgf.chernoff(log_m[t], valid[t], lams, delta)
        lower, lams_neg, log_m_neg, valid_neg = mgf.lower_bounds(
            op, fm.x0, T, reach, lams, delta, horizons,
            per_decade=lam_cfg.get("points_per_decade", 8),
            max_expansions=lam_cfg.get("max_expansions", 2),
        )
        mean, var, occ = mgf.exact_moments(op, fm.x0, T)

    with timer.stage("poly"):
        poly_lower, poly_upper, Lsup, Usup, moments_0to2, poly_diagnostics = (
            paper_moments.poly_intervals(
                fm, T, degree, alpha=delta, times=list(range(1, T + 1)), log=log
            )
        )

    sim_cfg = cfg["simulation"]
    with timer.stage("simulate"):
        C = finite_simulate(fm, sim_cfg["n_traj"], T, sim_cfg["seed"])

    n = C.shape[0]
    gT = C[:, T]
    below = int(np.sum(gT < lower[T]))
    above = int(np.sum(gT > upper[T]))
    p_below = int(np.sum(gT < poly_lower[T]))
    p_above = int(np.sum(gT > poly_upper[T]))
    std = float(np.sqrt(max(var[T], 0.0)))
    h = 0.5 * (float(Usup[T]) - float(Lsup[T]))
    h_sigma = h / std if std > 0 else float("inf")
    # Distinct return values on the finite model support DP is hard; use MC unique count
    # plus exact support endpoints. Also count unique simulated values.
    n_distinct_sim = int(np.unique(np.round(gT, decimals=10)).size)

    summary = {
        "method": "paper xi-recursion MGF + paper moment recursion polynomial on finite POMDP product",
        "name": f"{pomdp.name}_{ctrl.name}",
        "pomdp": pomdp.name,
        "policy": ctrl.name,
        "T": T,
        "finite_states": fm.n_states,
        "finite_edges": fm.n_edges,
        "poly_degree": degree,
        "exact_mean": float(mean[T]),
        "exact_std": std,
        "exact_var": float(var[T]),
        "sim_mean": float(gT.mean()),
        "support_L": float(Lsup[T]),
        "support_U": float(Usup[T]),
        "h_sigma": float(h_sigma),
        "n_distinct_returns_sim": n_distinct_sim,
        "lower": float(lower[T]),
        "upper": float(upper[T]),
        "poly_lower": float(poly_lower[T]),
        "poly_upper": float(poly_upper[T]),
        "empirical_q025": float(np.quantile(gT, 0.025)),
        "empirical_q975": float(np.quantile(gT, 0.975)),
        "frac_below_mgf": below / n,
        "frac_above_mgf": above / n,
        "frac_below_poly": p_below / n,
        "frac_above_poly": p_above / n,
        "poly_diagnostics": poly_diagnostics,
        "reward_depends_only_on_sa": pomdp.reward_depends_only_on_sa(),
        "mgf_seconds": mgf_seconds,
        "build_seconds": float(fm.build_seconds),
        "lambda_expansions": expansions,
        "profiling": timer.as_dict(),
        "wall_seconds": timer.wall_seconds(),
        "guarantee": (
            "Bounds are exact certificates for this finite POMDP + controller product chain "
            "(no quantization). Continuing horizon; G_t = sum of edge rewards."
        ),
    }
    _dump(out_dir / "summary.json", summary)
    _dump(out_dir / "profiling.json", summary["profiling"])

    rows = ["t,mean,var,mgf_L,mgf_U,poly_L,poly_U,support_L,support_U"]
    for t in range(1, T + 1):
        rows.append(",".join(str(x) for x in [
            t, mean[t], var[t], lower[t], upper[t],
            poly_lower[t], poly_upper[t], Lsup[t], Usup[t],
        ]))
    (out_dir / "results.csv").write_text("\n".join(rows) + "\n")

    save_artifacts.save_run_arrays(
        out_dir,
        mean=mean, var=var, lower=lower, upper=upper,
        poly_lower=poly_lower, poly_upper=poly_upper,
        Lsup=Lsup, Usup=Usup, moments_0to2=moments_0to2, C=C,
        lams=lams, log_m=log_m, valid=valid,
        lams_neg=lams_neg, log_m_neg=log_m_neg, valid_neg=valid_neg,
    )
    figures.cost_trajectories(
        C, mean, lower, upper,
        out_dir / "cost_trajectories_95pct.png",
        f"{pomdp.name}/{ctrl.name} (finite POMDP)",
        n_draw=sim_cfg.get("n_draw", 800),
        poly_lower=poly_lower, poly_upper=poly_upper,
    )
    figures.histogram(
        gT, mean[T], lower[T], upper[T], T,
        out_dir / f"histogram_T{T}.png",
        f"{pomdp.name}/{ctrl.name}",
        poly_lower=float(poly_lower[T]), poly_upper=float(poly_upper[T]),
    )
    log(
        f"  T={T}: mean={mean[T]:.4f} std={std:.4f} h/σ={h_sigma:.3g}  "
        f"MGF=[{lower[T]:.4f},{upper[T]:.4f}] poly=[{poly_lower[T]:.4f},{poly_upper[T]:.4f}]  "
        f"n_inconsistent={poly_diagnostics['totals']['n_inconsistent']}"
    )
    log(f"  wrote {out_dir} in {summary['wall_seconds']:.1f}s")
    return summary


def policies_for(pomdp):
    return make_policies(pomdp, build_product)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--problem", choices=["tiger", "network", "shopping5", "all"], default="all")
    p.add_argument("--policy", choices=["A", "B", "A_rand", "all"], default="all")
    p.add_argument("--out-root", type=Path, default=OUT_ROOT)
    p.add_argument("--T", type=int, default=100)
    args = p.parse_args(argv)

    problems = ["tiger", "network", "shopping5"] if args.problem == "all" else [args.problem]
    policy_keys = ["A", "B", "A_rand"] if args.policy == "all" else [args.policy]

    for pname in problems:
        pomdp = load_problem(pname)
        log = print
        log(f"Building policies for {pomdp.name} (memoryless brute may take a bit)…")
        pols = policies_for(pomdp)
        for pk in policy_keys:
            ctrl = pols[pk]
            tag = f"{pomdp.name}_{ctrl.name}"
            out = args.out_root / tag
            cfg = {**DEFAULT_CFG, "T": args.T, "horizons": [h for h in DEFAULT_CFG["horizons"] if h <= args.T]}
            if args.T not in cfg["horizons"]:
                cfg["horizons"] = sorted(set(cfg["horizons"] + [args.T]))
            run_one(pomdp, ctrl, out, cfg=cfg, log=log)


if __name__ == "__main__":
    main()

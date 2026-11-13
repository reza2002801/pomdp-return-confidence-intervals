#!/usr/bin/env python3
"""
Single entry point to reproduce paper tests, runs, figures, and tables.

Usage:
  python reproduce.py tests
  python reproduce.py finite
  python reproduce.py lqg
  python reproduce.py ablation
  python reproduce.py figures
  python reproduce.py tables
  python reproduce.py all
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _run(cmd: list[str], **kwargs) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True, **kwargs)


def cmd_tests() -> None:
    _run([sys.executable, "-m", "pytest", "-q"])


def cmd_finite() -> None:
    # Paper uses only hand policies (policy A → *_hand run folders).
    _run(
        [
            sys.executable,
            "experiments/finite_pomdps/run.py",
            "--problem",
            "all",
            "--policy",
            "A",
        ]
    )
    _run([sys.executable, "experiments/finite_pomdps/run_bridge_patrol.py"])
    _run([sys.executable, "experiments/finite_pomdps/run_martingale.py"])


def cmd_lqg() -> None:
    # Closed-form MGF + Monte Carlo, then degree-16 polynomial paper arrays/figures.
    _run([sys.executable, "experiments/lqg/continuous_baselines.py"])
    _run(
        [
            sys.executable,
            "experiments/lqg/plot_continuous_paper.py",
            "--out-figures",
            str(ROOT / "figures"),
        ]
    )


def cmd_ablation() -> None:
    _run(
        [
            sys.executable,
            "experiments/lqg/ablate_poly_degree_continuous.py",
            "--degrees",
            "4,8,16,20,24",
        ]
    )
    _run([sys.executable, "experiments/finite_pomdps/ablate_poly_degree.py"])


def cmd_figures() -> None:
    fig = ROOT / "figures"
    fig.mkdir(parents=True, exist_ok=True)
    # One script per paper figure name (see README).
    _run([sys.executable, "experiments/finite_pomdps/plot_main_curve_hist_pairs.py"])
    _run([sys.executable, "experiments/finite_pomdps/plot_finite_azuma_freedman.py"])
    _run([sys.executable, "experiments/lqg/make_ablation_paper.py"])


def cmd_tables() -> None:
    results = ROOT / "results"
    results.mkdir(parents=True, exist_ok=True)
    _run([sys.executable, "experiments/collect_paper_final.py"])
    src = ROOT / "data" / "runs" / "paper_final" / "FINAL_PAPER_DATA.json"
    dst = results / "paper_numbers.json"
    if not src.exists():
        raise SystemExit(f"missing {src}")
    shutil.copy2(src, dst)
    print(f"wrote {dst}", flush=True)
    abl = ROOT / "data" / "runs" / "ablation_poly_degree" / "paper_tables.tex"
    if abl.exists():
        shutil.copy2(abl, results / "paper_tables.tex")
        print(f"wrote {results / 'paper_tables.tex'}", flush=True)


def cmd_all() -> None:
    cmd_tests()
    cmd_finite()
    cmd_lqg()
    cmd_ablation()
    cmd_figures()
    cmd_tables()


TARGETS = {
    "tests": cmd_tests,
    "finite": cmd_finite,
    "lqg": cmd_lqg,
    "ablation": cmd_ablation,
    "figures": cmd_figures,
    "tables": cmd_tables,
    "all": cmd_all,
}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("target", choices=sorted(TARGETS), help="reproduction target")
    args = p.parse_args(argv)
    TARGETS[args.target]()


if __name__ == "__main__":
    main()

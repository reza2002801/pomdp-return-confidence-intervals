# Return confidence intervals for POMDPs

Code for our paper submitted to L4DC 2027.

Given a known model and a finite-state controller, we compute finite-horizon confidence intervals for the return of agent-state policies in POMDPs:

- **MGF / Chernoff** confidence limits from the return moment-generating function;
- **moment-based polynomial LP** (TailLP) limits from return moments up to a chosen degree;
- **Azuma / Freedman** martingale baselines on finite POMDPs for comparison.

The continuous LQG examples use closed-form Gaussian quadratic-form MGFs and the same polynomial limits under a truncated support.

## Citation

```bibtex
% TODO: replace after acceptance
@inproceedings{TODO2027returnci,
  title     = {TODO},
  author    = {TODO},
  booktitle = {Learning for Dynamics and Control (L4DC)},
  year      = {2027}
}
```

## Installation

Requires **Python ≥ 3.10** (paper runs used 3.10.12).

```bash
git clone https://github.com/reza2002801/pomdp-return-confidence-intervals.git
cd pomdp-return-confidence-intervals
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# or: pip install -e .
```

Pinned paper versions: NumPy 1.26.4, SciPy 1.15.3 (HiGHS via `scipy.optimize.linprog`), matplotlib 3.10.9, pytest 9.2.0.

**Memory.** Shopping-5 builds a dense reward tensor of shape `(6, 625, 625, 650)` (~11 GiB). Plan on about **16 GB RAM** for that example. The product-build test that exercises Network + Shopping-5 is skipped unless you set `RUN_HEAVY=1`.

## Quick start

```bash
python reproduce.py tests          # pytest (expect 27 passed, 3 skipped; RUN_HEAVY=1 adds the heavy product test)
python reproduce.py all            # full paper pipeline (long)
```

Individual targets: `finite`, `lqg`, `ablation`, `figures`, `tables`.

Finite paper runs use **hand policies only**:
`python experiments/finite_pomdps/run.py --problem all --policy A`
(folders `*_hand` under `data/runs/finite_pomdps/`, plus Bridge patrol).

## Paper → command → output map

| Paper item | Script | Output |
|------------|--------|--------|
| `main_curve_hist_finite.pdf` | `experiments/finite_pomdps/plot_main_curve_hist_pairs.py` | `figures/main_curve_hist_finite.pdf` |
| `main_curve_hist_lqg.pdf` (5 panels) | `experiments/finite_pomdps/plot_main_curve_hist_pairs.py` | `figures/main_curve_hist_lqg.pdf` |
| `finite_azuma_freedman_compare.pdf` | `experiments/finite_pomdps/plot_finite_azuma_freedman.py` | `figures/finite_azuma_freedman_compare.pdf` |
| `ablation_poly_degree_lqg.pdf` | `experiments/lqg/make_ablation_paper.py` | `figures/ablation_poly_degree_lqg.pdf` |
| `ablation_poly_degree_fs.pdf` (4 finite) | `experiments/lqg/make_ablation_paper.py` | `figures/ablation_poly_degree_fs.pdf` |
| Ablation tables (LQG + finite) | `experiments/lqg/make_ablation_paper.py` (after `reproduce.py ablation`) | `results/paper_tables.tex` |
| t = 100 results table | `experiments/collect_paper_final.py` (`reproduce.py tables`) | `results/paper_numbers.json` |

Or run the same figure scripts via `python reproduce.py figures` and tables via `python reproduce.py tables`.

## Repository layout

```
returnbounds/          core algorithms
  continuous.py        closed-form LQG MGF / moments / poly path
  mgf.py               finite-model Chernoff / MGF limits
  moments.py           finite return moments + support
  polybound.py         TailLP polynomial confidence limits
  martingale.py        Azuma / Freedman bands
  build_quantized.py   finite-model container (FiniteModel) and simulator
  plant.py / systems.py LQG plants and closed loop
  simulate_continuous.py Monte Carlo for continuous plants
  io.py / plotting.py  artifacts and figures
  profiling.py         wall time + peak RSS helpers
experiments/
  finite_pomdps/       Tiger, Network, Shopping-5, Bridge patrol
  lqg/                 five LQG configs and continuous paper scripts
  collect_paper_final.py  aggregate paper numbers (no figure PDFs)
reproduce.py           single entry point
tests/                 unit / integration tests
results/               small regenerated paper numbers (after `tables`)
figures/               regenerated paper figures (after `figures`)
```

## Examples

**Finite POMDPs**

| Name | Source |
|------|--------|
| Tiger | Kaelbling, Littman & Cassandra (1998) |
| Network | [pomdp.org](https://www.pomdp.org/examples/) `network.POMDP` (due to Littman, 1996) |
| Shopping-5 | Variant of Baisero [gym-pomdps](https://github.com/abaisero/gym-pomdps) Shopping-5 (see differences below) |
| Bridge patrol | This work (custom; 7 rows × 11 columns) |

**LQG (five examples, illustrative parameters):** mass-spring-damper, coupled masses, quarter-car, two-room thermal, cart-pole (linear). Configs live in `experiments/lqg/configs/`.

### Shopping-5 vs gym-pomdps

Our in-memory model (`experiments/finite_pomdps/shopping5.py`) follows the same 5×5 “find and buy” story as gym-pomdps `shopping_5.v1.pomdp`, with these differences:

| Aspect | gym-pomdps `shopping_5.v1` | This repo |
|--------|----------------------------|-----------|
| Grid | 5×5 | 5×5 |
| Start | Agent fixed at `(0,0)`; item uniform on 25 cells | Agent and item i.i.d. uniform on 25 cells |
| Actions | query, left, right, up, down, buy | LEFT, RIGHT, UP, DOWN, QUERY, BUY |
| Move / query reward | −1 / −2 | −1 / −2 |
| Buy reward | −5 fail, +10 success | −5 fail, +10 success |
| After BUY | Identity unless success, then `reset` | Always respawn item uniformly; agent stays |
| Observations | Agent cell, or item cell on query | Agent cell, or `(agent, item)` on query |
| Discount (file) | 0.99 | 0.95 (undiscounted finite-horizon analysis in the paper) |

So the paper example is best described as **a variant of Shopping-5**.

## Expected run times and hardware

From the paper-run host (hostname omitted):

- CPU: Intel Core i7-14700K (28 logical CPUs); poly LPs use up to 27 workers
- RAM: 62.5 GB (Shopping-5 needs ~16 GB)
- Typical: finite MGF for all horizons ≪ 1 s per example; LQG form+MGF a few seconds per plant; poly d=16 per plant minutes to tens of minutes; full ablation and figure rebuild is multi-hour

See `data/runs/paper_meta.json` → `1_computing_environment` for package versions.

## License

MIT, see LICENSE.

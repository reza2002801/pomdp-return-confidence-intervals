# Azuma / Freedman vs MGF / polynomial (finite-horizon Doob)

Finite-horizon undiscounted adaptation of the discounted Azuma / Freedman CIs
(`returnbounds/martingale.py`). Center is the deterministic mean \(\mathbb{E}[G_t]\).
One-sided failure probability \(\delta=0.025\) per tail. Reuses stored MGF /
polynomial levels and 20k Monte Carlo trajectories from the existing runs
(models not rebuilt).

Tests: `pytest tests/test_martingale.py` — 4 passed. No empirical exceedance
above \(0.025\) for any method / problem at \(t\in\{25,50,100\}\).

## Table at \(T=100\)

| problem | mean | empirical [2.5, 97.5] | MGF [L,U] | poly [L,U] | Freedman [L,U] | Azuma [L,U] | Azuma per-step [L,U] | K | σ | Freedman/MGF | Azuma/MGF | poly/MGF |
|---|---:|---|---|---|---|---|---|---:|---:|---:|---:|---:|
| tiger_tiger_hand | 62.02 | [-166, 153] | [-275.2, 163.0] | [-218.0, 174.5] | [-1074, 1198] | [-2368, 2492] | [-1628, 1752] | 89.01 | 37.39 | 5.19 | 11.09 | 0.90 |
| network_network_hand | 662.4 | [-400, 1860] | [-738.6, 2371] | [-652.2, 2192] | [-2776, 4101] | [-5574, 6899] | [-5406, 6730] | 229.6 | 115.7 | 2.21 | 4.01 | 0.91 |
| shopping5_shopping_hand | 88.08 | [59, 119] | [47.68, 132.9] | [50.24, 129.4] | [-73.63, 249.8] | [-253.6, 429.7] | [-162.4, 338.6] | 12.52 | 5.331 | 3.80 | 8.02 | 0.93 |
| mass_spring_damper | 5.602 | [4.320, 7.246] | [3.965, 8.968] | [3.994, 8.074] | [-12.35, 23.55] | [-45.63, 56.83] | [-38.66, 49.87] | 1.886 | 0.569 | 7.18 | 20.48 | 0.82 |

Artifacts per run: `martingale.json`, `comparison_traj.png`, `comparison_hist.png`
(under `data/runs/finite_pomdps/<run>/` or `data/runs/paper_v2/mass_spring_damper/`).

## Per-problem notes

**tiger.** Polynomial is tightest on the lower side; MGF is slightly tighter on
the upper side (163 vs 174 vs emp 153). Freedman is ~5× wider than MGF; Azuma
~11× and dwarfs the trajectories (zoom panel without Azuma in `comparison_traj.png`).

**network.** Polynomial is tightest on both sides. Freedman (~2.2× MGF) is the
best martingale bound but still far from the empirical quantiles; Azuma and
Azuma-per-step are nearly identical here.

**shopping5 (hand A).** Polynomial is tightest on both sides; MGF close behind.
Freedman (~3.8× MGF) still covers all 20k trajectories with zero empirical
exceedance; Azuma remains very loose.

**mass_spring_damper.** Polynomial is tightest overall (width ratio 0.82 vs MGF).
Freedman (~7× MGF) and especially Azuma (~20×) are far from the empirical
spread — the per-step range \(K\) stays large relative to conditional variance
on this quantized LQG chain.

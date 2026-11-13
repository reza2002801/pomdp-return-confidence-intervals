"""
Build appendix poly-degree ablation tables + paired traj|hist composites.

Uses saved ablation_poly_degree/*/summary.json and poly_d*.npz together with the
fixed source runs (closed-form continuous LQG or finite POMDP).

  PYTHONPATH=. python3 experiments/lqg/make_ablation_paper.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from returnbounds import io as save_artifacts

ABL = _ROOT / "data" / "runs" / "ablation_poly_degree"
FIG = _ROOT / "figures"
TEX_SNIPPET = ABL / "paper_tables.tex"

# Order: 5 continuous LQG then 4 finite POMDPs (matches paper narrative).
ENVS = [
    ("mass_spring_damper", "MSD", "lqg"),
    ("coupled_masses", "Coupled", "lqg"),
    ("quarter_car", "Quarter-car", "lqg"),
    ("two_room_thermal", "Thermal", "lqg"),
    ("cartpole_linear", "Cart-pole", "lqg"),
    ("bridge_patrol", "Bridge", "pomdp"),
    ("tiger", "Tiger", "pomdp"),
    ("network", "Network", "pomdp"),
    ("shopping5", "Shopping-5", "pomdp"),
]
DEGREES = [4, 8, 16, 20, 24]
MGF_STYLE = {"color": "#2ca02c", "ls": (0, (4, 1.5)), "lw": 1.4}


def _poly_colors(degrees=DEGREES):
    cmap = plt.get_cmap("plasma")
    xs = np.linspace(0.2, 0.85, max(len(degrees), 1))
    return {d: cmap(x) for d, x in zip(degrees, xs)}


def _style(ax):
    ax.grid(True, alpha=0.25, linewidth=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def load_env(key: str):
    abl = ABL / key
    summary = json.loads((abl / "summary.json").read_text())
    src = Path(summary["source"])
    if not src.is_absolute():
        src = _ROOT / src
    elif not src.exists():
        # Fall back to repo-relative finite/LQG run dirs.
        rel = Path("data") / "runs" / (
            "finite_pomdps" if "finite" in summary.get("kind", "") else "continuous_baselines"
        ) / key
        cand = _ROOT / rel
        if cand.exists():
            src = cand
    kind = summary.get("kind", "quantized_lqg")
    if kind == "continuous_lqg":
        # Prefer local plot arrays written by continuous ablation; else paper_continuous_arrays.
        local = abl / "continuous_plot_arrays.npz"
        if local.exists():
            z = np.load(local)
            mean = np.asarray(z["mean"], float)
            mgf_L = np.asarray(z["mgf_lower"], float)
            mgf_U = np.asarray(z["mgf_upper"], float)
            C = np.asarray(z["C"], float)
        else:
            z = np.load(src / "paper_continuous_arrays.npz")
            mean = np.asarray(z["mean"], float)
            mgf_L = np.asarray(z["lower"], float)
            mgf_U = np.asarray(z["upper"], float)
            C = np.asarray(z["C"], float)
    else:
        data = save_artifacts.load_run_arrays(src)
        mean = np.asarray(data["mean"], float)
        mgf_L = np.asarray(data["mgf_lower"], float)
        mgf_U = np.asarray(data["mgf_upper"], float)
        C = np.asarray(data["C"], float)
    curves = {}
    for d in summary["degrees"]:
        z = np.load(abl / f"poly_d{d}.npz")
        curves[int(d)] = {
            "lo": np.asarray(z["lower"], float),
            "hi": np.asarray(z["upper"], float),
            "times": np.asarray(z["times"], int),
        }
    rows_by_d = {int(r["degree"]): r for r in summary["rows"]}
    return {
        "key": key,
        "summary": summary,
        "mean": mean,
        "mgf_L": mgf_L,
        "mgf_U": mgf_U,
        "C": C,
        "curves": curves,
        "rows": rows_by_d,
        "T": int(summary["T"]),
        "kind": kind,
    }


def fmt_num(x, env_key):
    ax = abs(float(x))
    if env_key in ("two_room_thermal", "quarter_car", "network"):
        if ax >= 100:
            return f"{x:.1f}"
        return f"{x:.2f}"
    if env_key in ("shopping5", "tiger", "bridge_patrol"):
        return f"{x:.1f}"
    return f"{x:.3f}"


def build_tables(envs_data):
    """Return LaTeX for two table* floats: LQG (5) and finite (4)."""
    lqg = [(k, lab, kind) for k, lab, kind in ENVS if kind == "lqg"]
    pomdp = [(k, lab, kind) for k, lab, kind in ENVS if kind == "pomdp"]

    def one_table(envs, label, caption):
        n = len(envs)
        colspec = "l" + "rr" * n
        lines = []
        lines.append(r"\begin{table*}[t]")
        lines.append(r"\centering")
        lines.append(rf"\caption{{{caption}}}")
        lines.append(rf"\label{{{label}}}")
        lines.append(r"\small")
        lines.append(r"\setlength{\tabcolsep}{3.8pt}")
        lines.append(rf"\begin{{tabular}}{{{colspec}}}")
        lines.append(r"\toprule")
        # header row 1
        h1 = [r"Degree"]
        for _, lab, _ in envs:
            h1.append(rf"\multicolumn{{2}}{{c}}{{{lab}}}")
        lines.append(" & ".join(h1) + r" \\")
        # cmidrules
        cm = []
        for i in range(n):
            start = 2 + 2 * i
            end = start + 1
            cm.append(rf"\cmidrule(lr){{{start}-{end}}}")
        lines.append(" ".join(cm))
        # header row 2
        h2 = [""]
        for _ in envs:
            h2.extend(["$L$", "$U$"])
        lines.append(" & ".join(h2) + r" \\")
        lines.append(r"\midrule")
        for d in DEGREES:
            row = [f"${d}$"]
            for key, _, _ in envs:
                r = envs_data[key]["rows"][d]
                if r.get("collapsed"):
                    row.extend([r"---", r"---"])
                else:
                    row.append(fmt_num(r["poly_L"], key))
                    row.append(fmt_num(r["poly_U"], key))
            lines.append(" & ".join(row) + r" \\")
        lines.append(r"\midrule")
        # MGF
        row = ["MGF-based CIs"]
        for key, _, _ in envs:
            r = envs_data[key]["rows"][DEGREES[0]]
            row.append(fmt_num(r["mgf_L"], key))
            row.append(fmt_num(r["mgf_U"], key))
        lines.append(" & ".join(row) + r" \\")
        row = [r"Emp.\ 2.5\% and 97.5\%"]
        for key, _, _ in envs:
            r = envs_data[key]["rows"][DEGREES[0]]
            row.append(fmt_num(r["emp_q025"], key))
            row.append(fmt_num(r["emp_q975"], key))
        lines.append(" & ".join(row) + r" \\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table*}")
        return "\n".join(lines)

    cap_lqg = (
        r"Polynomial-degree ablation at $T=100$ for the five continuous LQG plants. "
        r"Entries are interval endpoints $[L,U]$ (not widths). "
        r"Within each plant the closed-form MGF, continuous Monte Carlo sample, "
        r"policy, and failure probability $\alpha=0.025$ per tail are held fixed; "
        r"only the polynomial TailLP degree varies. "
        r"Empirical endpoints are the sample $2.5\%$ and $97.5\%$ quantiles of the "
        r"same $20{,}000$ continuous trajectories."
    )
    cap_fs = (
        r"Polynomial-degree ablation at $T=100$ for the four finite POMDPs "
        r"(Bridge patrol, Tiger, Network, Shopping-5). "
        r"Entries are interval endpoints $[L,U]$. "
        r"Model, policy, confidence level, Monte Carlo sample, and MGF comparator "
        r"are held fixed within each environment. "
        r"Empirical endpoints are the sample $2.5\%$ and $97.5\%$ quantiles."
    )
    return "\n\n".join([
        one_table(lqg, "tab:poly_degree_ablation_lqg", cap_lqg),
        one_table(pomdp, "tab:poly_degree_ablation_fs", cap_fs),
    ])


def _draw_pair(fig, outer_spec, env, colors, show_ylab=False):
    """Traj + 90deg hist sharing the trajectory y-axis (linear scale)."""
    mean, mgf_L, mgf_U, C = env["mean"], env["mgf_L"], env["mgf_U"], env["C"]
    T = env["T"]
    t = np.arange(T + 1)
    n_draw = min(400, C.shape[0])
    idx = np.linspace(0, C.shape[0] - 1, n_draw).astype(int)
    centered = C[idx] - mean
    q_lo = np.quantile(C, 0.025, axis=0) - mean
    q_hi = np.quantile(C, 0.975, axis=0) - mean
    mL, mU = mgf_L - mean, mgf_U - mean

    xT = C[:, T] - mean[T]
    qT_lo, qT_hi = np.quantile(xT, [0.025, 0.975])

    y_lo = min(
        float(np.nanpercentile(centered[:, 1:], 0.5)),
        float(np.nanmin(mL[1:])), float(np.nanmin(q_lo[1:])),
        float(np.nanpercentile(xT, 0.2)), float(qT_lo), float(mL[T]),
    )
    y_hi = max(
        float(np.nanpercentile(centered[:, 1:], 99.5)),
        float(np.nanmax(mU[1:])), float(np.nanmax(q_hi[1:])),
        float(np.nanpercentile(xT, 99.8)), float(qT_hi), float(mU[T]),
    )
    for d in DEGREES:
        if d not in env["curves"] or env["rows"].get(d, {}).get("collapsed"):
            continue
        ts = env["curves"][d]["times"]
        lo = env["curves"][d]["lo"][ts] - mean[ts]
        hi = env["curves"][d]["hi"][ts] - mean[ts]
        y_lo = min(y_lo, float(np.nanmin(lo)), float(env["curves"][d]["lo"][T] - mean[T]))
        y_hi = max(y_hi, float(np.nanmax(hi)), float(env["curves"][d]["hi"][T] - mean[T]))
    span = max(y_hi - y_lo, 1e-9)
    ylim = (y_lo - 0.08 * span, y_hi + 0.08 * span)

    inner = GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer_spec,
        wspace=0.06, width_ratios=[2.4, 1.0],
    )
    ax = fig.add_subplot(inner[0, 0])
    axh = fig.add_subplot(inner[0, 1], sharey=ax)

    ax.plot(t, centered.T, color="#4c566a", alpha=1.0, lw=0.16, rasterized=True, zorder=1)
    ax.plot(t, np.zeros_like(t), color="#2e3440", lw=1.2, zorder=3)
    ax.plot(t, q_lo, color="#2e3440", ls=":", lw=1.0, zorder=4)
    ax.plot(t, q_hi, color="#2e3440", ls=":", lw=1.0, zorder=4)
    ax.plot(t[1:], mL[1:], color=MGF_STYLE["color"], ls=MGF_STYLE["ls"],
            lw=MGF_STYLE["lw"], zorder=5)
    ax.plot(t[1:], mU[1:], color=MGF_STYLE["color"], ls=MGF_STYLE["ls"],
            lw=MGF_STYLE["lw"], zorder=5)
    for d in DEGREES:
        if d not in env["curves"] or env["rows"].get(d, {}).get("collapsed"):
            continue
        ts = env["curves"][d]["times"]
        lo = env["curves"][d]["lo"][ts] - mean[ts]
        hi = env["curves"][d]["hi"][ts] - mean[ts]
        ax.plot(ts, lo, color=colors[d], lw=1.15, zorder=6)
        ax.plot(ts, hi, color=colors[d], lw=1.15, zorder=6)
    ax.set_xlim(0, T)
    ax.set_ylim(*ylim)
    ax.set_xlabel(r"horizon $t$", fontsize=7.5)
    if show_ylab:
        ax.set_ylabel(r"$G_t^\pi-\mathbb{E}[G_t^\pi]$", fontsize=7.5)
    ax.tick_params(labelsize=6.5)
    _style(ax)

    axh.hist(
        xT, bins=45, orientation="horizontal",
        weights=np.full(len(xT), 1 / len(xT)),
        color="#a3b1c6", edgecolor="#4c566a", linewidth=0.3, zorder=1,
    )

    def _hline(v, c, ls, lw, xmax, zo):
        axh.axhline(v, xmax=xmax, color=c, ls=ls, lw=lw, zorder=zo)

    _hline(0.0, "#2e3440", "-", 1.2, 0.82, 3)
    _hline(qT_lo, "#2e3440", ":", 1.0, 0.82, 4)
    _hline(qT_hi, "#2e3440", ":", 1.0, 0.82, 4)
    _hline(mL[T], MGF_STYLE["color"], MGF_STYLE["ls"], MGF_STYLE["lw"], 0.92, 5)
    _hline(mU[T], MGF_STYLE["color"], MGF_STYLE["ls"], MGF_STYLE["lw"], 0.92, 5)
    for d in DEGREES:
        if d not in env["curves"] or env["rows"].get(d, {}).get("collapsed"):
            continue
        lo = env["curves"][d]["lo"][T] - mean[T]
        hi = env["curves"][d]["hi"][T] - mean[T]
        _hline(lo, colors[d], "-", 1.15, 0.85, 6)
        _hline(hi, colors[d], "-", 1.15, 0.85, 6)
    axh.set_ylim(*ylim)
    axh.set_xlim(0, axh.get_xlim()[1] / 0.80)
    axh.set_xlabel("relative frequency", fontsize=7)
    axh.tick_params(labelsize=6.0)
    axh.tick_params(axis="y", labelleft=False)
    _style(axh)
    # Drop traj t=T tick so it does not collide with hist fraction=0.
    ticks = [float(x) for x in ax.get_xticks() if -1e-9 <= float(x) <= T + 1e-9]
    if len(ticks) >= 2 and abs(ticks[-1] - T) <= 1e-6 * max(T, 1.0):
        ax.set_xticks(ticks[:-1])
    return ax


def build_figure(env_specs, out_stem, ncols=3, legend_fontsize: float = 11):
    """env_specs: list of (key, label); ncols settings per row."""
    colors = _poly_colors()
    n = len(env_specs)
    nrows = (n + ncols - 1) // ncols
    fig = plt.figure(figsize=(5.3 * ncols, 3.15 * nrows + 0.70))
    outer = GridSpec(
        nrows, ncols, figure=fig,
        hspace=0.28, wspace=0.22,
        left=0.045, right=0.995, top=0.80, bottom=0.06,
    )
    for idx, (key, lab) in enumerate(env_specs):
        r, c = divmod(idx, ncols)
        env = _CACHE[key]
        ax = _draw_pair(fig, outer[r, c], env, colors, show_ylab=(c == 0))
        ax.set_title(lab, fontsize=10, pad=2, loc="left")

    handles = _legend_handles(colors)
    # Two-row legend; fontsize can be bumped per figure (e.g. Fig 5 LQG ablation).
    fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995),
        ncol=4, frameon=False, fontsize=legend_fontsize,
        handlelength=2.0, columnspacing=1.2, handletextpad=0.5,
    )

    FIG.mkdir(parents=True, exist_ok=True)
    pdf = FIG / f"{out_stem}.pdf"
    png = FIG / f"{out_stem}.png"
    fig.savefig(pdf, format="pdf", bbox_inches="tight", dpi=450)
    fig.savefig(png, dpi=450, bbox_inches="tight")
    plt.close(fig)
    print("wrote", pdf, png)
    return pdf, png


def _legend_handles(colors):
    handles = [
        Line2D([0], [0], color="#2e3440", lw=1.4, label="exact mean"),
        Line2D([0], [0], color="#2e3440", ls=":", lw=1.2, label=r"Emp. 2.5% and 97.5%"),
        Line2D([0], [0], color=MGF_STYLE["color"], ls=MGF_STYLE["ls"], lw=1.5, label="MGF-based CIs"),
    ]
    for d in DEGREES:
        handles.append(
            Line2D(
                [0], [0], color=colors[d], lw=1.4,
                label=rf"Moment-based CIs ($d{{=}}{d}$)",
            )
        )
    return handles


_CACHE = {}


def main():
    missing = []
    for key, lab, kind in ENVS:
        if not (ABL / key / "summary.json").exists():
            missing.append(key)
            continue
        _CACHE[key] = load_env(key)
        # validate
        for d, r in _CACHE[key]["rows"].items():
            if r.get("collapsed"):
                print(f"WARNING collapsed: {key} d={d}")
            if r.get("n_inconsistent", 0) and r["n_inconsistent"] > 0:
                print(f"NOTE inconsistent LP flags: {key} d={d} n={r['n_inconsistent']}")

    if missing:
        print("MISSING ablation data for:", missing)
        print("Run finite ablations / LQG ablations first.")
        # still build what we can
        available = [(k, lab) for k, lab, _ in ENVS if k not in missing]
    else:
        available = [(k, lab) for k, lab, _ in ENVS]

    # Tables for whatever is available
    n_expected = sum(1 for _ in ENVS)
    if len(_CACHE) == n_expected:
        tex = build_tables(_CACHE)
        TEX_SNIPPET.write_text(tex + "\n")
        print("wrote", TEX_SNIPPET)
    else:
        print(f"skipping full tables ({len(_CACHE)}/{n_expected} envs)")

    # Continuous LQG ablation (paper Fig 5): slightly larger legend.
    lqg = [(k, lab) for k, lab, kind in ENVS if kind == "lqg" and k in _CACHE]
    if len(lqg) >= 1:
        build_figure(lqg, "ablation_poly_degree_lqg", ncols=3, legend_fontsize=14.5)
    # Finite POMDPs (2×2 when bridge is included)
    fs = [(k, lab) for k, lab, kind in ENVS if kind == "pomdp" and k in _CACHE]
    if len(fs) >= 1:
        build_figure(fs, "ablation_poly_degree_fs", ncols=2 if len(fs) == 4 else 3)


if __name__ == "__main__":
    main()

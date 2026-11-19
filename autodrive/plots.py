"""
plots.py - training charts for an experiment folder.

    training_plots(exp_dir) writes into experiments/<name>/plots/:
        training_curves.png   success rate, return, crash rates vs steps (stage bands)
        outcomes.png          how episodes ended over time (stacked shares)
        reward_terms.png      one small panel per reward term: its average
                              contribution per episode over training

Styling: one fixed categorical palette (validated for colour-blind safety),
thin lines, recessive grid, legends whenever there are >= 2 series.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

# validated categorical palette (light mode), fixed order
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
STAGE_BANDS = ["#f3f2ee", "#fcfcfb"]

OUTCOMES = [("parked", "Parked"), ("collision", "Collision"), ("pedestrian_hit", "Pedestrian hit"),
            ("off_route", "Off route"), ("stuck", "Stuck"), ("timeout", "Time out")]


def _style(plt):
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "legend.frameon": False, "lines.linewidth": 2.0,
    })


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def rolling(x, w):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return x
    w = max(1, min(w, len(x)))
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = np.empty(len(x))
    for i in range(len(x)):
        a = max(0, i - w + 1)
        out[i] = (c[i + 1] - c[a]) / (i + 1 - a)
    return out


def _stage_bands(ax, steps, stages):
    """Shade the background by curriculum stage and label each band."""
    if len(steps) == 0:
        return
    starts = [0]
    for i in range(1, len(stages)):
        if stages[i] != stages[i - 1]:
            starts.append(i)
    starts.append(len(stages))
    for k in range(len(starts) - 1):
        a, b = steps[starts[k]], steps[starts[k + 1] - 1]
        ax.axvspan(a, b, color=STAGE_BANDS[k % 2], zorder=0, lw=0)
        from .curriculum import STAGES
        label = STAGES[stages[starts[k]]]["name"].split()[0] if stages[starts[k]] < len(STAGES) else "?"
        ax.text((a + b) / 2, 1.0, label, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=8, color=INK2)


def _fmt_steps(ax):
    ax.xaxis.set_major_formatter(lambda v, p: f"{v / 1e6:.1f}M" if v >= 1e6 else f"{v / 1e3:.0f}k")


def training_plots(exp_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    exp_dir = Path(exp_dir)
    (exp_dir / "plots").mkdir(exist_ok=True)
    eps = read_csv(exp_dir / "episodes.csv")
    if len(eps) < 5:
        print("[plots] not enough episodes yet")
        return
    steps = np.array([float(e["timesteps"]) for e in eps])
    stages = [int(e["stage"]) for e in eps]
    succ = np.array([e["success"] == "True" for e in eps], float)
    ret = np.array([float(e["return"]) for e in eps])
    coll = np.array([e["status"] == "collision" for e in eps], float)
    ped = np.array([e["status"] == "pedestrian_hit" for e in eps], float)
    W = 200

    # ---------------- training curves (small multiples, one measure each) ----------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    _stage_bands(ax, steps, stages)
    ax.plot(steps, rolling(succ, W) * 100, color=PALETTE[0])
    ax.set_title("Parked successfully (% of last 200 episodes)", pad=16)
    ax.set_ylim(0, 100)
    ax = axes[1]
    _stage_bands(ax, steps, stages)
    ax.plot(steps, rolling(ret, W), color=PALETTE[0])
    ax.set_title("Episode return (rolling mean)", pad=16)
    ax = axes[2]
    _stage_bands(ax, steps, stages)
    ax.plot(steps, rolling(coll, W) * 100, color=PALETTE[1], label="Collision")
    ax.plot(steps, rolling(ped, W) * 100, color=PALETTE[6], label="Pedestrian hit")
    ax.set_title("Crash rates (% of episodes)", pad=16)
    ax.legend(loc="upper right")
    ax.set_ylim(bottom=0)
    for ax in axes:
        _fmt_steps(ax)
        ax.set_xlabel("environment steps")
    fig.suptitle(f"Training: {exp_dir.name}", x=0.01, ha="left", fontweight="bold", fontsize=13)
    fig.tight_layout()
    fig.savefig(exp_dir / "plots" / "training_curves.png", dpi=110)
    plt.close(fig)

    # ---------------- outcomes: stacked shares over time ---------------------------
    fig, ax = plt.subplots(figsize=(12, 4.2))
    shares = []
    for key, _ in OUTCOMES:
        shares.append(rolling([e["status"] == key for e in eps], W) * 100)
    cols = [PALETTE[2], PALETTE[1], PALETTE[7], PALETTE[3], PALETTE[6], PALETTE[4]]
    ax.stackplot(steps, *shares, labels=[l for _, l in OUTCOMES], colors=cols,
                 edgecolor=SURFACE, linewidth=0.6)
    _stage_bands(ax, steps, stages)
    ax.set_ylim(0, 100)
    ax.set_xlim(steps[0], steps[-1])
    ax.set_title("How episodes ended (% of last 200)", pad=16)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    _fmt_steps(ax)
    ax.set_xlabel("environment steps")
    fig.tight_layout()
    fig.savefig(exp_dir / "plots" / "outcomes.png", dpi=110)
    plt.close(fig)

    # ---------------- per-term contributions (one panel per term) -------------------
    terms = [k[2:] for k in eps[0] if k.startswith("r_")]
    used = [t for t in terms if any(abs(float(e[f"r_{t}"] or 0)) > 1e-9 for e in eps)]
    if used:
        n = len(used)
        cols_n = 4
        rows_n = int(np.ceil(n / cols_n))
        fig, axes = plt.subplots(rows_n, cols_n, figsize=(15, 2.6 * rows_n), squeeze=False)
        for i, t in enumerate(used):
            ax = axes[i // cols_n][i % cols_n]
            v = np.array([float(e[f"r_{t}"] or 0) for e in eps])
            r = rolling(v, W)
            col = PALETTE[0] if r.mean() >= 0 else PALETTE[1]
            ax.plot(steps, r, color=col, lw=1.8)
            ax.axhline(0, color=INK2, lw=0.6)
            ax.set_title(t.replace("_", " "), fontsize=10)
            _fmt_steps(ax)
            ax.tick_params(labelsize=8)
        for j in range(n, rows_n * cols_n):
            axes[j // cols_n][j % cols_n].axis("off")
        fig.suptitle("Average reward per episode from each term (blue = bonus, orange = penalty)",
                     x=0.01, ha="left", fontweight="bold", fontsize=12)
        fig.tight_layout()
        fig.savefig(exp_dir / "plots" / "reward_terms.png", dpi=100)
        plt.close(fig)
    print(f"[plots] wrote {exp_dir / 'plots'}")

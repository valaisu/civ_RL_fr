"""Plot a training run's log.jsonl into runs/<name>/plots.png.

Called automatically at the end of training (also on Ctrl-C), or manually:
    uv run civrl-plot runs/<name>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE, INK, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
# Fixed color per episode group (color follows the entity, not its rank).
GROUP_COLOR = {
    "player": "#2a78d6",
    "attacker vs scripted": "#2a78d6",
    "defender vs scripted": "#eb6834",
    "attacker vs self": "#1baf7a",
    "defender vs self": "#eda100",
    "attacker vs frozen": "#e87ba4",
    "defender vs frozen": "#008300",
    "attacker vs checkpoint": "#6250d6",
    "defender vs checkpoint": "#e34948",
}
TARGET_POINTS = 60  # training curves are binned to about this many points
# Per-update stats, plotted when the log has them (which ones depends on --algo).
STAT_PANELS = [
    ("episodes", "Episodes completed (total)"),
    ("sps", "Training speed (steps/s, running average)"),
    ("ent", "Policy entropy"),
    ("pg", "Policy loss"),
    ("vl", "Value loss"),
    ("clipfrac", "PPO/GRPO clip fraction"),
    ("group_std", "GRPO: return std within groups"),
    ("td", "DQN: TD loss (Huber)"),
    ("q", "DQN: mean Q of sampled actions"),
    ("eps", "DQN: exploration epsilon"),
    ("fit_mean", "ES: mean candidate fitness"),
    ("fit_best", "ES: best candidate fitness"),
]


def load(run_dir):
    recs = [json.loads(line) for line in open(Path(run_dir) / "log.jsonl")]
    starts = [i for i, r in enumerate(recs) if r["update"] == 1]
    recs = recs[starts[-1]:] if starts else recs  # older logs may hold several appended runs
    # older logs stored eval as eval_win_p<q> (always vs scripted): convert to eval/<group>/<metric>
    two_player = any("eval_win_p1" in r or any("attacker" in k or "defender" in k or k.startswith(("att_", "def_"))
                                                 for k in r) for r in recs)
    for r in recs:
        for q in (0, 1):
            if f"eval_win_p{q}" in r:
                g = ("attacker", "defender")[q] + " vs scripted" if two_player else "player"
                for m in ("win", "ret", "turns"):
                    if f"eval_{m}_p{q}" in r:
                        r[f"eval/{g}/{m}"] = r[f"eval_{m}_p{q}"]
    return recs


def binned_groups(recs):
    """Episode stats per group, aggregated over bins of updates (weighted by episode count)."""
    groups = sorted({k.split("/")[0] for r in recs for k in r if k.endswith("/n")})
    size = max(1, len(recs) // TARGET_POINTS)
    out = {}
    for g in groups:
        xs, cols = [], {"win": [], "ret": [], "turns": []}
        for i in range(0, len(recs), size):
            chunk = [r for r in recs[i:i + size] if f"{g}/n" in r]
            n = sum(r[f"{g}/n"] for r in chunk)
            xs.append(recs[min(i + size, len(recs)) - 1]["steps"])
            for m in cols:  # NaN where the group has no episodes (e.g. another --alternate phase): breaks the line
                cols[m].append(sum(r[f"{g}/{m}"] * r[f"{g}/n"] for r in chunk) / n if n else np.nan)
        out[g] = (np.array(xs), {m: np.array(v) for m, v in cols.items()})
    return out


def style(ax, title, steps_x=True):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)
    if steps_x:
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda v, _: f"{v / 1e6:.1f}M" if v >= 1e6 else f"{v / 1e3:.0f}k"))


def plot_run(run_dir):
    run_dir = Path(run_dir)
    recs = load(run_dir)
    if not recs:
        return None
    steps = np.array([r["steps"] for r in recs])
    groups = binned_groups(recs)
    eval_groups = sorted({k.split("/")[1] for r in recs for k in r if k.startswith("eval/")})

    single = [(k, t) for k, t in STAT_PANELS if any(k in r for r in recs)]
    rows = 1 + -(-len(single) // 3)
    fig, axes = plt.subplots(rows, 3, figsize=(16, 3.7 * rows), facecolor=SURFACE)
    axes = axes.ravel()

    # episode metrics: training curves per group + greedy eval markers
    for ax, (m, title) in zip(axes[:3], [("win", "Win rate"), ("ret", "Average episode reward"),
                                         ("turns", "Episode length (turns)")]):
        style(ax, title)
        for g, (xs, cols) in groups.items():
            ax.plot(xs, cols[m], color=GROUP_COLOR.get(g, MUTED), linewidth=1.5, label=f"{g} (training)")
        for g in eval_groups:
            key = f"eval/{g}/{m}"
            pts = [(r["steps"], r[key]) for r in recs if key in r]
            ax.plot([x for x, _ in pts], [y for _, y in pts], linestyle="none", marker="o", markersize=5,
                    markerfacecolor=SURFACE, markeredgecolor=GROUP_COLOR.get(g, MUTED), markeredgewidth=1.5,
                    label=f"{g} (eval, learner greedy)")
        if m == "win":
            ax.set_ylim(-0.03, 1.03)
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        if groups or eval_groups:
            ax.legend(fontsize=8, frameon=False, labelcolor=INK)

    # progress and optimisation metrics (one series each)
    for ax, (key, title) in zip(axes[3:], single):
        style(ax, title)
        ax.plot(steps, [r.get(key, np.nan) for r in recs], color="#2a78d6", linewidth=1.5)
    for ax in axes[3 + len(single):]:
        ax.set_visible(False)

    # --alternate: dashed line where each phase starts, labelled with the side trained
    starts = [r for i, r in enumerate(recs) if "phase" in r and (i == 0 or recs[i - 1].get("phase") != r["phase"])]
    for ax in axes[:3 + len(single)]:
        for r in starts[1:]:
            ax.axvline(r["steps"], color=MUTED, linestyle="--", linewidth=1)
    for r in starts:
        axes[0].text(r["steps"], 1.0, f" {r['side']}", color=MUTED, fontsize=8, va="top")

    for ax in axes[3 + len(single) - 3:3 + len(single)]:
        ax.set_xlabel("training steps (decisions)", color=MUTED, fontsize=9)
    name = run_dir.name
    last = recs[-1]
    algo = json.loads((run_dir / "config.json").read_text())["args"].get("algo", "ppo") \
        if (run_dir / "config.json").exists() else "ppo"
    fig.suptitle(f"{name} ({algo}): {last['steps']:,} steps, {last.get('episodes', 0):,} episodes, "
                 f"{last.get('time', 0) / 60:.1f} min", color=INK, fontsize=13, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.03 * 3 / rows))
    path = run_dir / "plots.png"
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+", help="run directories, e.g. runs/nav_v1")
    for r in ap.parse_args().runs:
        print(plot_run(r))


if __name__ == "__main__":
    main()

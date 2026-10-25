"""Compare learning algorithms: train each for the same time, then evaluate them on equal terms.

Single-player scenarios (navigate, scout; --scenario): each method trains normally, then each
final network (last.pt, greedy) plays the same --games maps. Plots: win rate, reward and
turns per method, and the eval win rate during training against time.

Skirmish (the default), each method trained with alternating sides:

1. Training: for each method, `civrl-train --alternate` for --minutes, one method after
   another (each gets the whole machine).
2. Common yardstick. A method's win rates in its own log are measured against its own frozen
   copies, so they cannot be compared between methods. Instead:
   * tournament: every method's attacker plays every method's defender, on the same maps
     for every pair, both networks greedy
   * vs scripted AI: each method's attacker and defender against the scripted opponent
   Overall score of a method = average of its attacking win rate (against all defenders)
   and its defending win rate (against all attackers), the diagonal included.
3. Plots: runs/<name>/compare.png; numbers in runs/<name>/results.json.

Which network plays a side: its newest version. For the side a run was training when time ran
out, that is last.pt; for the other side it is the latest phase<k>_<side>.pt (the version that
reached the switch threshold), or last.pt if that side never got a turn.

    uv run civrl-compare --minutes 30 --name cmp1
    uv run civrl-compare --scenario navigate --minutes 3 --name cmp_nav
    uv run civrl-compare --minutes 20 --algos ppo dqn es --name cmp2 --size 12 12
    uv run civrl-compare --name cmp1 --eval-only        # redo the tournament and plots
Options civrl-compare does not know are passed on to every civrl-train run.
"""
from __future__ import annotations

import argparse
import json
import re
import signal
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from .algos import ALGOS  # noqa: E402
from .algos.es import play  # noqa: E402
from .env import Config  # noqa: E402
from .plots import INK, MUTED, SURFACE, style  # noqa: E402
from .scenarios import make_config, make_env  # noqa: E402
from .train import make_net  # noqa: E402

# categorical slots in fixed order: a method keeps its color in every comparison
ALGO_COLOR = dict(zip(ALGOS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
SIDE_COLOR = {"attacker": "#2a78d6", "defender": "#eb6834"}
BLUES = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
SIDES = ("attacker", "defender")


def train_all(root, scenario, algos, minutes, seed, extra):
    two_player = make_env(make_config(scenario)).n_players == 2
    for algo in algos:
        print(f"\n=== {algo}: {'alternating ' if two_player else ''}training for {minutes:g} minutes "
              f"-> {root / algo}", flush=True)
        cmd = [sys.executable, "-m", "civrl.train", "--scenario", scenario, "--algo", algo,
               *(["--alternate"] if two_player else []),
               "--minutes", str(minutes), "--steps", str(10**12), "--seed", str(seed),
               "--out", str(root), "--name", algo, *extra]
        proc = subprocess.Popen(cmd)
        try:
            proc.wait()
        except KeyboardInterrupt:  # the training run saves itself; then evaluate what exists
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            proc.wait()
            print("interrupted: skipping the remaining methods")
            return
        if proc.returncode:
            print(f"!!! {algo} exited with code {proc.returncode}")


def read_log(run):
    return [json.loads(line) for line in open(run / "log.jsonl")]


def side_checkpoint(run, side):
    recs = read_log(run)
    training = recs[-1].get("side") if recs else None
    phases = sorted(run.glob(f"phase*_{side}.pt"), key=lambda p: int(re.match(r"phase(\d+)", p.name)[1]))
    return run / "last.pt" if training == side or not phases else phases[-1]


def load_net(path, cfg, cache):
    if path not in cache:
        ck = torch.load(path)
        net = make_net(make_env(cfg), ck["ch"], ck["blocks"])
        net.load_state_dict(ck["model"])
        cache[path] = net.eval()
    return cache[path]


def finished_runs(root, algos):
    runs = {a: root / a for a in algos if (root / a / "last.pt").exists()}
    if not runs:
        sys.exit(f"no finished runs in {root}")
    cfg = Config.from_dict(json.loads((runs[next(iter(runs))] / "config.json").read_text())["cfg"])
    return runs, cfg


def evaluate_single(root, algos, games, seed):
    """Single-player scenario: every final network plays the same maps."""
    runs, cfg = finished_runs(root, algos)
    algos, cache = list(runs), {}
    seeds = [int(s) for s in np.random.default_rng(seed).integers(2**31, size=games)]
    jobs = [(s, [(load_net(runs[a] / "last.pt", cfg, cache), True)]) for a in algos for s in seeds]
    print(f"\n=== evaluation: {len(jobs)} games ({games} maps per method)", flush=True)
    out = play(cfg, jobs)
    res = {"scenario": cfg.scenario, "algos": algos, "games": games, "win": {}, "ret": {}, "turns": {}}
    for k, a in enumerate(algos):
        rows = out[k * games:(k + 1) * games]
        res["win"][a] = float(np.mean([w == 0 for _, w, _, _ in rows]))
        res["ret"][a] = float(np.mean([r[0] for r, _, _, _ in rows]))
        res["turns"][a] = float(np.mean([t for _, _, t, _ in rows]))
    logs = {a: read_log(run) for a, run in runs.items()}
    res["curve"] = {a: [(r["time"] / 60, r["eval/player/win"]) for r in recs if "eval/player/win" in r]
                    for a, recs in logs.items()}
    res["steps"] = {a: recs[-1]["steps"] for a, recs in logs.items()}
    (root / "results.json").write_text(json.dumps(res, indent=1))
    return res


def plot_single(root, res):
    algos = res["algos"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), facecolor=SURFACE)
    panels = [("win", "Win rate (final network, greedy)"), ("ret", "Average episode reward"),
              ("turns", "Average episode length (turns)")]
    for ax, (key, title) in zip(axes.ravel(), panels):
        style(ax, title, steps_x=False)
        v = [res[key][a] for a in algos]
        ax.bar(algos, v, color="#2a78d6", width=0.6)
        for x, vi in enumerate(v):
            ax.text(x, vi, f"{vi:.0%}" if key == "win" else f"{vi:.2f}" if key == "ret" else f"{vi:.1f}",
                    ha="center", va="bottom", fontsize=9, color=INK)
        if key == "win":
            ax.set_ylim(0, 1.08)
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.grid(axis="x", visible=False)
    ax = axes[1, 1]
    style(ax, "Eval win rate during training", steps_x=False)
    for a in algos:
        pts = res["curve"][a]
        ax.plot([t for t, _ in pts], [w for _, w in pts], color=ALGO_COLOR.get(a, MUTED), linewidth=2, marker="o",
                markersize=4, label=f"{a} ({res['steps'][a] / 1e6:.2f}M steps)")
    ax.set_ylim(-0.03, 1.03)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("minutes of training", color=MUTED, fontsize=9)
    ax.legend(fontsize=8, frameon=False, labelcolor=INK)
    fig.suptitle(f"{root.name}: {res['scenario']}, {len(algos)} methods, {res['games']} evaluation maps",
                 color=INK, fontsize=13, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = root / "compare.png"
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)
    return path


def evaluate_all(root, algos, games, seed):
    runs, cfg = finished_runs(root, algos)
    algos = list(runs)
    cache, ckpt, nets = {}, {}, {}
    for a, run in runs.items():
        for side in SIDES:
            ckpt[a, side] = side_checkpoint(run, side)
            nets[a, side] = (load_net(ckpt[a, side], cfg, cache), True)
    seeds = [int(s) for s in np.random.default_rng(seed).integers(2**31, size=games)]
    jobs, keys = [], []
    for att in algos:
        for dfd in algos:
            for s in seeds:
                jobs.append((s, [nets[att, "attacker"], nets[dfd, "defender"]]))
                keys.append(("cross", att, dfd))
    for a in algos:
        for s in seeds:
            jobs.append((s, [nets[a, "attacker"], "scripted"]))
            keys.append(("scripted", a, "attacker"))
            jobs.append((s, ["scripted", nets[a, "defender"]]))
            keys.append(("scripted", a, "defender"))
    print(f"\n=== evaluation: {len(jobs)} games ({len(algos)}x{len(algos)} tournament + vs scripted, "
          f"{games} maps each)", flush=True)
    wins = {}
    for i in range(0, len(jobs), 300):  # in chunks, to bound memory
        for k, (_, winner, _, _) in zip(keys[i:i + 300], play(cfg, jobs[i:i + 300])):
            me = 0 if k[0] == "cross" or k[2] == "attacker" else 1
            wins[k] = wins.get(k, 0) + (winner == me) / games
        print(f"  {min(i + 300, len(jobs))}/{len(jobs)} games", flush=True)
    cross = np.array([[wins["cross", att, dfd] for dfd in algos] for att in algos])  # attacker win rate
    attack, defend = cross.mean(1), (1 - cross).mean(0)
    res = {
        "algos": algos,
        "games_per_pair": games,
        "checkpoints": {f"{a}/{s}": str(ckpt[a, s]) for a in algos for s in SIDES},
        "tournament_attacker_win": cross.tolist(),
        "attack_score": dict(zip(algos, attack.tolist())),
        "defend_score": dict(zip(algos, defend.tolist())),
        "overall_score": dict(zip(algos, ((attack + defend) / 2).tolist())),
        "vs_scripted": {a: {s: wins["scripted", a, s] for s in SIDES} for a in algos},
        "phases": {a: [(r["time"] / 60, r.get("phase", 1)) for r in read_log(run)] for a, run in runs.items()},
        "steps": {a: read_log(run)[-1]["steps"] for a, run in runs.items()},
    }
    (root / "results.json").write_text(json.dumps(res, indent=1))
    return res


def plot_compare(root, res):
    algos = res["algos"]
    cross = np.array(res["tournament_attacker_win"])
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), facecolor=SURFACE)
    (ax_t, ax_s), (ax_v, ax_p) = axes

    # tournament heatmap
    style(ax_t, "Tournament: attacker win rate (row's attacker vs column's defender)", steps_x=False)
    ax_t.grid(False)
    ax_t.imshow(cross, cmap=BLUES, vmin=0, vmax=1)
    for i in range(len(algos)):
        for j in range(len(algos)):
            ax_t.text(j, i, f"{cross[i, j]:.0%}", ha="center", va="center", fontsize=10,
                      color="white" if cross[i, j] > 0.55 else INK)
    ax_t.set_xticks(range(len(algos)), algos)
    ax_t.set_yticks(range(len(algos)), algos)
    ax_t.set_xlabel("defender", color=MUTED, fontsize=9)
    ax_t.set_ylabel("attacker", color=MUTED, fontsize=9)

    # overall score, best first
    style(ax_s, "Overall score: mean of attacking and defending win rates in the tournament", steps_x=False)
    order = sorted(algos, key=lambda a: res["overall_score"][a])
    vals = [res["overall_score"][a] for a in order]
    ax_s.barh(order, vals, color="#2a78d6", height=0.6)
    for y, (a, v) in enumerate(zip(order, vals)):
        ax_s.text(v + 0.01, y, f"{v:.0%}  (attack {res['attack_score'][a]:.0%}, defend {res['defend_score'][a]:.0%})",
                  va="center", fontsize=9, color=INK)
    ax_s.set_xlim(0, 1.25)  # room for the labels
    ax_s.set_xticks(np.linspace(0, 1, 6))
    ax_s.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax_s.grid(axis="y", visible=False)

    # vs scripted
    style(ax_v, "Win rate against the scripted AI", steps_x=False)
    x = np.arange(len(algos))
    for k, side in enumerate(SIDES):
        v = [res["vs_scripted"][a][side] for a in algos]
        ax_v.bar(x + (k - 0.5) * 0.38, v, width=0.36, color=SIDE_COLOR[side], label=f"as {side}")
        for xi, vi in zip(x, v):
            ax_v.text(xi + (k - 0.5) * 0.38, vi + 0.015, f"{vi:.0%}", ha="center", fontsize=8, color=INK)
    ax_v.set_xticks(x, algos)
    ax_v.set_ylim(0, 1.08)
    ax_v.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax_v.grid(axis="x", visible=False)
    ax_v.legend(fontsize=8, frameon=False, labelcolor=INK)

    # phases over time
    style(ax_p, "Alternating phases reached (a new phase = the side in training hit --switch-at)", steps_x=False)
    for a in algos:
        pts = res["phases"][a]
        ax_p.step([t for t, _ in pts], [p for _, p in pts], where="post", color=ALGO_COLOR.get(a, MUTED),
                  linewidth=2, label=f"{a} ({res['steps'][a] / 1e6:.2f}M steps)")
    ax_p.set_xlabel("minutes of training", color=MUTED, fontsize=9)
    ax_p.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax_p.legend(fontsize=8, frameon=False, labelcolor=INK)

    fig.suptitle(f"{root.name}: {len(algos)} methods, alternating training, {res['games_per_pair']} games per pairing",
                 color=INK, fontsize=13, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = root / "compare.png"
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(description="Train every method with alternating sides, then compare them.",
                                 epilog="Unknown options are passed on to civrl-train (e.g. --size 12 12).")
    ap.add_argument("--scenario", default="skirmish", choices=["navigate", "scout", "skirmish"])
    ap.add_argument("--algos", nargs="+", default=list(ALGOS), choices=list(ALGOS))
    ap.add_argument("--minutes", type=float, default=30, help="training time per method")
    ap.add_argument("--games", type=int, default=50, help="evaluation games per pairing")
    ap.add_argument("--name", required=True, help="output folder runs/<name>/")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-only", action="store_true", help="skip training; evaluate the existing runs")
    args, extra = ap.parse_known_args()
    root = Path(args.out) / args.name
    root.mkdir(parents=True, exist_ok=True)
    if not args.eval_only:
        print(f"comparing {', '.join(args.algos)}: {args.minutes:g} min each, about "
              f"{args.minutes * len(args.algos) / 60:.1f} h of training, then the evaluation")
        if extra:
            print(f"passed to civrl-train: {' '.join(extra)}")
        train_all(root, args.scenario, args.algos, args.minutes, args.seed, extra)
    if make_env(make_config(args.scenario)).n_players == 1:
        res = evaluate_single(root, args.algos, args.games, args.seed)
        print("\nranking (win rate, then reward):")
        for a in sorted(res["algos"], key=lambda a: (-res["win"][a], -res["ret"][a])):
            print(f"  {a:<10} win {res['win'][a]:4.0%}   reward {res['ret'][a]:+.2f}   turns {res['turns'][a]:.1f}")
        print(f"plots: {plot_single(root, res)}")
        return
    res = evaluate_all(root, args.algos, args.games, args.seed)
    print("\nranking (overall tournament score):")
    for a in sorted(res["algos"], key=lambda a: -res["overall_score"][a]):
        vs = res["vs_scripted"][a]
        print(f"  {a:<10} {res['overall_score'][a]:4.0%}   vs scripted: attacker {vs['attacker']:4.0%}, "
              f"defender {vs['defender']:4.0%}   phases {res['phases'][a][-1][1]}")
    print(f"plots: {plot_compare(root, res)}")


if __name__ == "__main__":
    main()

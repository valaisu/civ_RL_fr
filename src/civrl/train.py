"""Training entry point: picks the algorithm (--algo, see civrl/algos/), runs it, and does
the logging, evaluation, checkpoints and plots, which are the same for every algorithm.
Experience collection and the opponent handling live in rollout.py.

--alternate (two-player scenarios): train one side against a fixed opponent until its eval
win rate reaches --switch-at, then freeze a copy of the network as the opponent and train the
other side, and so on. A side that never reaches the threshold keeps training; no switch.

    uv run civrl-train --scenario navigate --steps 300000
    uv run civrl-train --scenario skirmish --algo dqn --opponent mixed --learner-side both
    uv run civrl-train --scenario skirmish --alternate --learner-side attacker
"""
from __future__ import annotations

import argparse
import copy
import json
import signal
import time
from pathlib import Path

import numpy as np
import torch

from .algos import ALGOS
from .env import N_CHANNELS
from .scenarios import make_config, make_env
from .model import Policy
from .plots import plot_run
from .rollout import Runner, batch_obs
from .scripted import scripted_action


ROLE = {1: ["player"], 2: ["attacker", "defender"]}
OPP_NAME = {"scripted": "scripted", "learner": "self", "self": "self", "frozen": "frozen", "checkpoint": "checkpoint"}


def summarize(stats, n_players):
    """Group finished episodes by (learner role, opponent): {group: {n, win, ret, turns}}."""
    groups = {}
    for s in stats:
        for q, c in enumerate(s["ctrl"]):
            if c != "learner":
                continue
            key = ROLE[n_players][q] + ("" if n_players == 1 else f" vs {OPP_NAME[s['ctrl'][1 - q]]}")
            g = groups.setdefault(key, {"n": 0, "win": 0.0, "ret": 0.0, "turns": 0.0})
            g["n"] += 1
            g["win"] += s["winner"] == q
            g["ret"] += s["returns"][q]
            g["turns"] += s["turns"]
    for g in groups.values():
        for k in ("win", "ret", "turns"):
            g[k] /= g["n"]
    return groups


def fmt_steps(n):
    return f"{n / 1e6:.2f}M" if n >= 1e6 else f"{n / 1e3:.0f}k"


def fmt_groups(groups):
    return " | ".join(f"{k}: win {g['win']:4.0%}  reward {g['ret']:+.2f}  turns {g['turns']:4.1f}  (n={g['n']})"
                      for k, g in sorted(groups.items()))


def evaluate(net, cfg, learner_side, opponents, n_eps=50, seed=12345):
    """Greedy learner vs each opponent in `opponents` ({name: None for scripted, or a policy net}).
    Opponent nets sample their actions, as they do in training. Same maps for every opponent."""
    env = make_env(cfg)
    n = env.n_players
    sides = [0] if n == 1 else ([0, 1] if learner_side == "both" else [0 if learner_side == "attacker" else 1])
    res = {}
    for opp_name, opp in opponents.items():
        for me in sides:
            rng = np.random.default_rng(seed)
            wins, turns, rets = 0, [], []
            for ep in range(n_eps):
                env.reset(int(rng.integers(2**31)))
                done, ret = False, 0.0
                while not done:
                    p = env.to_act()
                    if p == me or opp is not None:
                        o = env.observe(p)
                        with torch.no_grad():
                            logits, _ = (net if p == me else opp)(*batch_obs([o], net.device))
                        a = int(logits.argmax()) if p == me else int(torch.distributions.Categorical(logits=logits).sample())
                    else:
                        a = scripted_action(env, p)
                    r, done = env.step(a)
                    ret += float(r[me])
                wins += env.winner() == me
                turns.append(env.game.turn)
                rets.append(ret)
            g = ROLE[n][me] + ("" if n == 1 else f" vs {opp_name}")
            res[f"eval/{g}/win"] = wins / n_eps
            res[f"eval/{g}/ret"] = float(np.mean(rets))
            res[f"eval/{g}/turns"] = float(np.mean(turns))
    return res


def eval_opponents(eval_vs, opp_mode, runner, net):
    """Who to evaluate against: by default the opponent(s) the learner is training against."""
    if runner.n_players == 1 or eval_vs == "scripted" or opp_mode == "scripted":
        return {"scripted": None}
    if opp_mode in ("frozen", "checkpoint"):
        return {opp_mode: runner.opp_net}
    if opp_mode == "self":
        return {"self": net}
    return {"scripted": None, "self": net}  # mixed


COMPLETE_EPISODES = ("reinforce", "grpo")  # algorithms that collect whole episodes


def print_params(args, cfg, n_players, out, algo):
    """Print the settings that matter for a run (everything is also in config.json)."""
    rows = [("run", f"{out}" + (f"  (init from {args.init})" if args.init else "")),
            ("scenario", f"{cfg.scenario}, {fmt_steps(args.steps)} steps"
                         + (f" or {args.minutes:g} minutes, whichever comes first" if args.minutes else ""))]
    if n_players > 1:
        rows.append(("players", f"learner: {args.learner_side}, opponent: {args.opponent}"
                     + (f" (snapshot every {args.snapshot_every} updates)"
                        if args.opponent == "frozen" and not args.alternate else "")))
        if args.alternate:
            rows.append(("alternate", f"start with {args.learner_side} vs {args.opponent}; when its eval win rate "
                                      f"reaches {args.switch_at:.0%}, freeze a copy as the opponent and train the other side"))
    sk = cfg.scenario == "skirmish"
    size = (f"board {cfg.W}x{cfg.H}, island {min(cfg.island_min, cfg.W)}-{min(cfg.island_max, cfg.W)}"
            if sk else f"{cfg.W}x{cfg.H}")
    turns = f"{cfg.turns_min}-{cfg.turns_max}" if sk else f"{cfg.max_turns}"
    rows += [
        ("map", f"{size}, {'random' if cfg.random_map else f'fixed (seed {cfg.map_seed})'}, "
                f"{turns} turns, pre_scouted={cfg.pre_scouted}, full_vis={cfg.full_vis}"),
        ("reward", f"shaping x{cfg.shaping}" + (
            f", scout_w={cfg.scout_w}, intel_w={cfg.intel_w}, defense_w={cfg.defense_w}, "
            f"health_w={cfg.health_w}, fast_w={cfg.fast_w}, survive_w={cfg.survive_w}" if sk else "")),
    ]
    if sk:
        rows.append(("units", f"defender {cfg.def_min}-{cfg.def_max}, attacker defender+1 to "
                              f"{cfg.att_ratio_max}x defender, archer probability {cfg.archer_frac:.2f}"))
    if args.algo == "es":
        collect = (f"{2 * args.es_pop} candidates x {args.es_episodes} episodes per update "
                   f"(--envs/--rollout unused)")
    elif args.algo in COMPLETE_EPISODES:
        collect = f"{args.envs} envs, whole episodes, at least {args.envs * args.rollout} decisions per update"
    else:
        collect = f"{args.envs} envs x {args.rollout} steps = {args.envs * args.rollout} decisions per update"
    eval_vs = ("scripted AI" if n_players == 1 or (args.eval_vs == "scripted" and not args.alternate)
               or args.opponent == "scripted" and not args.alternate
               else "scripted AI + itself" if args.opponent == "mixed"
               else "the current training opponent" if args.alternate
               else f"training opponent ({args.opponent})")
    rows += [
        ("algo", algo.describe()),
        ("rollout", collect),
        ("network", f"{args.ch} channels, {args.blocks} residual blocks"),
        ("eval", f"every {args.eval_every} updates, 50 games, learner greedy, vs {eval_vs}"),
        ("system", f"seed={args.seed}, threads={args.threads}, device={args.device}"),
    ]
    for k, v in rows:
        print(f"  {k:<9} {v}")


def make_net(env, ch, blocks):
    return Policy(N_CHANNELS, env.geo.targets, env.n_actions, ch=ch, blocks=blocks)


OPPONENT_MODES = ["scripted", "self", "mixed", "frozen"]


def load_opponent(path, cfg, ap):
    """A fixed opponent policy from a checkpoint; must match this run's scenario and map size."""
    if not Path(path).is_file():
        ap.error(f"--opponent must be one of {OPPONENT_MODES} or an existing checkpoint file, got {path!r}")
    ck = torch.load(path)
    if ck["cfg"]["scenario"] != cfg.scenario or (ck["cfg"]["W"], ck["cfg"]["H"]) != (cfg.W, cfg.H):
        ap.error(f"opponent checkpoint is for {ck['cfg']['scenario']} {ck['cfg']['W']}x{ck['cfg']['H']}, "
                 f"this run is {cfg.scenario} {cfg.W}x{cfg.H}")
    net = make_net(make_env(cfg), ck["ch"], ck["blocks"])
    net.load_state_dict(ck["model"])
    return net.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="navigate")
    ap.add_argument("--algo", default="ppo", choices=list(ALGOS), help="learning algorithm, see civrl/algos/")
    ap.add_argument("--steps", type=int, default=300_000)
    ap.add_argument("--minutes", type=float, default=0, help="also stop after this much wall-clock time (0 = no limit)")
    ap.add_argument("--envs", type=int, default=32)
    ap.add_argument("--rollout", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=4, help="ppo, grpo: passes over each batch")
    ap.add_argument("--minibatch", type=int, default=None, help="default per algorithm: a2c, reinforce 256, others 512")
    ap.add_argument("--lr", type=float, default=None,
                    help="default per algorithm: dqn 1e-4, es 0.003, a2c and reinforce 7e-4, ppo and grpo 3e-4")
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lam", type=float, default=0.95, help="ppo, a2c: GAE lambda")
    ap.add_argument("--clip", type=float, default=0.2, help="ppo, grpo: ratio clip")
    ap.add_argument("--ent", type=float, default=0.01, help="policy-gradient algorithms: entropy bonus")
    ap.add_argument("--group-size", type=int, default=8, help="grpo: episodes per group (same start state)")
    ap.add_argument("--buffer", type=int, default=15_000, help="dqn: replay buffer size (about 22 kB each)")
    ap.add_argument("--nstep", type=int, default=3, help="dqn: n-step returns")
    ap.add_argument("--replay-ratio", type=float, default=4, help="dqn: times each new decision is sampled")
    ap.add_argument("--target-every", type=int, default=250, help="dqn: gradient steps between target updates")
    ap.add_argument("--eps-end", type=float, default=0.05, help="dqn: final exploration epsilon")
    ap.add_argument("--eps-frac", type=float, default=0.3, help="dqn: share of training to decay epsilon over")
    ap.add_argument("--per-alpha", type=float, default=0.6, help="dqn: prioritized replay exponent (0 = uniform)")
    ap.add_argument("--es-pop", type=int, default=16, help="es: antithetic pairs per generation")
    ap.add_argument("--es-sigma", type=float, default=0.02, help="es: weight noise scale")
    ap.add_argument("--es-episodes", type=int, default=2, help="es: episodes per candidate")
    ap.add_argument("--ch", type=int, default=32)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--learner-side", default="attacker", choices=["attacker", "defender", "both"])
    ap.add_argument("--opponent", default="scripted",
                    help="scripted | self | mixed | frozen | path to a checkpoint (e.g. runs/<name>/last.pt)")
    ap.add_argument("--snapshot-every", type=int, default=20, help="updates between frozen-opponent refreshes")
    ap.add_argument("--alternate", action="store_true",
                    help="train one side until it wins, then the other against a frozen copy, and so on")
    ap.add_argument("--switch-at", type=float, default=0.6, help="--alternate: eval win rate that triggers a switch")
    ap.add_argument("--fixed-map", action="store_true", help="same map every episode")
    ap.add_argument("--map-seed", type=int, default=0)
    ap.add_argument("--pre-scouted", action="store_true")
    ap.add_argument("--full-vis", action="store_true")
    ap.add_argument("--shaping", type=float, default=1.0)
    ap.add_argument("--scout-w", type=float, default=0.1, help="skirmish attacker: exploration reward")
    ap.add_argument("--intel-w", type=float, default=0.01, help="skirmish attacker: enemy-sighting reward")
    ap.add_argument("--defense-w", type=float, default=0.01, help="skirmish defender: defence-bonus reward")
    ap.add_argument("--health-w", type=float, default=0.02, help="skirmish: per-turn health-balance reward")
    ap.add_argument("--fast-w", type=float, default=0.5, help="skirmish winner: bonus for winning quickly")
    ap.add_argument("--survive-w", type=float, default=0.5, help="skirmish loser: bonus for surviving longer")
    ap.add_argument("--size", type=int, nargs=2, default=None, metavar=("W", "H"))
    ap.add_argument("--max-turns", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=10, help="updates between greedy evaluations")
    ap.add_argument("--eval-vs", default="training", choices=["training", "scripted"],
                    help="evaluate against the training opponent(s) (default) or the scripted AI")
    ap.add_argument("--log-every", type=float, default=60, help="seconds between progress lines")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--name", default=None)
    ap.add_argument("--init", default=None, help="checkpoint to start from")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", default="auto", help="auto (cuda if available, else cpu) | cpu | cuda | cuda:N")
    args = ap.parse_args()
    if args.lr is None:
        args.lr = ALGOS[args.algo].lr
    if args.minibatch is None:
        args.minibatch = ALGOS[args.algo].minibatch

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    over = dict(random_map=not args.fixed_map, map_seed=args.map_seed, pre_scouted=args.pre_scouted,
                full_vis=args.full_vis, shaping=args.shaping,
                scout_w=args.scout_w, intel_w=args.intel_w, defense_w=args.defense_w,
                health_w=args.health_w, fast_w=args.fast_w, survive_w=args.survive_w)
    if args.size:
        over.update(W=args.size[0], H=args.size[1])
    if args.max_turns:  # skirmish draws its turn limit from a range; this fixes it
        over.update(max_turns=args.max_turns, turns_min=args.max_turns, turns_max=args.max_turns)
    cfg = make_config(args.scenario, **over)

    opp_mode, opp_net = args.opponent, None
    if args.opponent not in OPPONENT_MODES:
        opp_mode = "checkpoint"
        opp_net = load_opponent(args.opponent, cfg, ap).to(args.device)
    runner = Runner(cfg, args.envs, args.learner_side, opp_mode, args.seed, opp_net,
                    group_size=args.group_size if args.algo == "grpo" else 1)
    if args.alternate and (runner.n_players == 1 or args.learner_side == "both" or opp_mode in ("self", "mixed")):
        ap.error("--alternate needs a two-player scenario, --learner-side attacker|defender and "
                 "--opponent scripted|frozen|<checkpoint>")
    net = make_net(runner.envs[0], args.ch, args.blocks)
    if args.init:
        net.load_state_dict(torch.load(args.init)["model"])
    net.to(args.device)
    if args.opponent == "frozen":
        runner.opp_net = copy.deepcopy(net).eval()
        runner.restart()  # the first episodes were seated before the snapshot existed
    algo = ALGOS[args.algo](args, net, runner)

    name = args.name or f"{args.scenario}_{time.strftime('%m%d_%H%M%S')}"
    out = Path(args.out) / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({"args": vars(args), "cfg": cfg.__dict__}, indent=2))

    def checkpoint():
        return {"model": {k: v.cpu() for k, v in net.state_dict().items()}, "cfg": cfg.__dict__, "ch": args.ch, "blocks": args.blocks,
                "algo": args.algo}

    total, update, t0 = 0, 0, time.time()
    log = open(out / "log.jsonl", "w")  # a rerun with the same name starts a fresh log
    best = -1.0
    phase, side = 1, args.learner_side
    n_episodes, window, last_print = 0, [], t0
    print_params(args, cfg, runner.n_players, out, algo)
    print(f"progress every {args.log_every:.0f}s; full per-update stats in {out / 'log.jsonl'}", flush=True)
    try:
        while total < args.steps and not (args.minutes and time.time() - t0 >= 60 * args.minutes):
            update += 1
            algo.progress = max(total / args.steps, (time.time() - t0) / (60 * args.minutes) if args.minutes else 0)
            n, s = algo.update()
            total += n
            if opp_mode == "frozen" and not args.alternate and update % args.snapshot_every == 0:
                runner.opp_net = copy.deepcopy(net).eval()
            n_episodes += len(runner.ep_stats)
            window += runner.ep_stats
            groups = summarize(runner.ep_stats, runner.n_players)
            runner.ep_stats = []
            elapsed = time.time() - t0
            rec = {"update": update, "steps": total, "episodes": n_episodes, "time": round(elapsed),
                   "sps": round(total / elapsed), **{k: round(v, 4) for k, v in s.items()},
                   **{f"{k}/{m}": round(float(v), 3) for k, g in groups.items() for m, v in g.items()}}
            if args.alternate:
                rec.update(phase=phase, side=side)
            if time.time() - last_print >= args.log_every:
                wg = summarize(window, runner.n_players)
                extra = "".join(f" | {label} {s[k]:.2f}" for k, label in
                                (("ent", "entropy"), ("eps", "epsilon"), ("fit_mean", "fitness")) if k in s)
                print(f"[{elapsed / 60:5.1f} min] steps {fmt_steps(total):>6} | episodes {n_episodes:>6} "
                      f"| {rec['sps']} steps/s{extra} || {fmt_groups(wg)}", flush=True)
                window, last_print = [], time.time()
            if update % args.eval_every == 0:
                net.eval()
                ev = evaluate(net, cfg, side, eval_opponents("training" if args.alternate else args.eval_vs,
                                                             opp_mode, runner, net))
                net.train()
                rec.update({k: round(v, 3) for k, v in ev.items()})
                score = np.mean([v for k, v in ev.items() if k.endswith("/win")])
                torch.save(checkpoint(), out / "last.pt")
                mark = ""
                if score >= best:
                    best = score
                    torch.save(checkpoint(), out / "best.pt")
                    mark = "  (new best, saved best.pt)"
                parts = [f"{k[5:-4]}: win {v:.0%}  reward {ev[k[:-4] + '/ret']:+.2f}  turns {ev[k[:-4] + '/turns']:.1f}"
                         for k, v in ev.items() if k.endswith("/win")]
                print("           eval (50 games, learner greedy): " + " | ".join(parts) + mark, flush=True)
                if args.alternate and score >= args.switch_at:
                    torch.save(checkpoint(), out / f"phase{phase}_{side}.pt")
                    old = side
                    phase, side = phase + 1, "defender" if side == "attacker" else "attacker"
                    opp_mode = runner.opponent = "frozen"
                    runner.opp_net = copy.deepcopy(net).eval()
                    runner.learner_side = side
                    runner.restart()
                    algo.on_phase_change()
                    best = -1.0
                    print(f"           >>> {old} reached {score:.0%} (saved phase{phase - 1}_{old}.pt); "
                          f"phase {phase}: training {side} vs a frozen copy of the network", flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
    except KeyboardInterrupt:
        signal.signal(signal.SIGINT, signal.SIG_IGN)  # uv forwards Ctrl-C too; don't abort the save
        print("interrupted; saving last.pt and plots")
    torch.save(checkpoint(), out / "last.pt")
    print(f"done: {fmt_steps(total)} steps, {n_episodes} episodes in {(time.time() - t0) / 60:.1f} min -> {out}")
    print(f"plots: {plot_run(out)}")


if __name__ == "__main__":
    main()

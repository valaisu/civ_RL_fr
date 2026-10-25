# Commands

Run from the project root. `uv run pytest -q` runs the rule tests.

## Setup (once per machine)

Pick the PyTorch build. Later `uv run` commands keep it; a plain `uv sync` without `--extra` would remove PyTorch.
```bash
uv sync --extra cpu      # no GPU (this laptop: saves ~3 GB of CUDA libraries)
uv sync --extra cu130    # NVIDIA RTX 20-series or newer, driver >= 580
uv sync --extra cu126    # older NVIDIA cards (GTX 10-series) or drivers >= 560
```
Training uses the GPU automatically when one is available (`--device auto`); `--device cpu` forces the CPU. The run's start-up print shows the device. Checkpoints are saved as CPU tensors, so they work on every machine.

## Train

Output goes to `runs/<name>/`: `config.json`, `log.jsonl`, `best.pt`, `last.pt` and `plots.png`. Reusing a name overwrites that run. The run prints its main settings when it starts.

```bash
uv run civrl-train --scenario navigate --steps 100000 --name nav_test
uv run civrl-train --scenario scout    --steps 150000 --name scout_test
uv run civrl-train --scenario skirmish --learner-side attacker --opponent scripted --steps 1000000 --name sk_att
uv run civrl-train --help        # all options
```

### Algorithms (`--algo`)

Default is `ppo`. All algorithms share logging, eval, checkpoints, plots and the GUI. Each file in `src/civrl/algos/` explains its method in its docstring.
```bash
uv run civrl-train --scenario navigate --algo a2c       --name nav_a2c
uv run civrl-train --scenario navigate --algo reinforce --name nav_reinforce
uv run civrl-train --scenario navigate --algo grpo      --group-size 8 --name nav_grpo
uv run civrl-train --scenario navigate --algo dqn       --name nav_dqn
uv run civrl-train --scenario navigate --algo es        --es-pop 16 --name nav_es
```
- `--lr` defaults per algorithm: dqn 1e-4, es 0.003, a2c and reinforce 7e-4, ppo and grpo 3e-4. `--minibatch`: a2c and reinforce 256, the others 512.
- Algorithm flags:
  - GRPO: `--group-size`
  - DQN: `--buffer --nstep --replay-ratio --target-every --eps-end --eps-frac --per-alpha`
  - ES: `--es-pop --es-sigma --es-episodes`
- To compare algorithms fairly, keep the scenario and `--steps` the same and run 2–3 `--seed`s each:
```bash
for algo in ppo a2c reinforce grpo dqn es; do for seed in 0 1; do
  uv run civrl-train --scenario navigate --algo $algo --steps 300000 --seed $seed --name nav_${algo}_s$seed
done; done
```

### Compare methods (skirmish, alternating training)

`civrl-compare` trains each method with `--alternate` for a fixed time, one method after another. It then plays a tournament: every method's attacker against every method's defender on the same maps, plus each method against the scripted AI. Output: `runs/<name>/compare.png`, `runs/<name>/results.json`, and one training run per method in `runs/<name>/<algo>/`.
```bash
uv run civrl-compare --minutes 30 --name cmp1                          # all 6 methods: about 3 h
uv run civrl-compare --minutes 20 --algos ppo dqn es --name cmp2 --size 12 12
uv run civrl-compare --name cmp1 --eval-only                           # redo tournament + plots only
uv run civrl-compare --scenario navigate --minutes 3 --games 100 --name cmp_nav   # single-player: no alternation; same maps for all
```
- Options it doesn't know are passed to every `civrl-train` run: algorithm flags, reward weights, `--switch-at`, ...
- `--games N` sets the games per pairing (default 50).
- Ctrl-C saves the current method's run, skips the rest, and evaluates what exists.

### Alternating sides (skirmish)

Train the attacker vs the scripted defender until its eval win rate is ≥ 60%. Then freeze a copy of the network as the attacker and train the defender until it beats that copy 60% of the time, and so on. If a side never reaches the threshold, it keeps training and the run never switches.
```bash
uv run civrl-train --scenario skirmish --alternate --learner-side attacker --switch-at 0.6 --steps 3000000 --name sk_alt
```
- The first phase plays against `--opponent`: `scripted` (default), `frozen` (the untrained network) or a checkpoint.
- Each switch saves `phase<k>_<side>.pt`. Dashed lines in `plots.png` mark the switches.
- Works with any `--algo`.

Useful flags:
- Skirmish: `--learner-side attacker|defender|both`, `--opponent scripted|self|mixed|frozen`, or a checkpoint path to train against a saved policy, e.g. `--learner-side defender --opponent runs/<name>/last.pt`. The self-play opponents haven't been tested yet.
- Skirmish reward weights: `--scout-w 0.1 --intel-w 0.01 --defense-w 0.01 --health-w 0.02 --fast-w 0.5 --survive-w 0.5` (0 turns a term off). They're described in `scenarios/skirmish.py`. Island size, turn limit and army sizes vary per episode; the ranges are Config fields in `env.py`, and `--max-turns N` fixes the turn limit.
- Environment: `--fixed-map --map-seed N`, `--pre-scouted`, `--full-vis`, `--shaping X` (0 = sparse rewards only), `--size W H`, `--max-turns N`.
- PPO and shared: `--lr --gamma --lam --clip --ent --epochs --minibatch --envs --rollout --ch --blocks`.
- `--init runs/<name>/last.pt` continues from a checkpoint.
- `--minutes M` also stops training after M minutes of wall-clock time.

Run in the background:
```bash
nohup uv run civrl-train --scenario scout --name scout_bg > scout_bg.log 2>&1 &
tail -f scout_bg.log
pkill -f "name scout_bg"
```

## Watch agents

```bash
uv run civrl-gui                 # pick scenario + model(s), replay with arrow keys
```

## Plots

Every run writes `runs/<name>/plots.png` at the end, including when stopped with Ctrl-C. To redraw it for any run:
```bash
uv run civrl-plot runs/nav_test
```

## Reading training output

The terminal shows two kinds of lines:
- A **progress line** every `--log-every` seconds (default 60). It gives steps, total episodes, speed and policy entropy, then **win rate, average episode reward and length** for the episodes finished since the last line. Episodes are grouped by role and opponent, e.g. `attacker vs scripted`.
- An **eval line** every `--eval-every` updates (default 10). It shows 50 games with the learner playing greedily against the opponent(s) it is training against; `--eval-vs scripted` evaluates against the scripted AI instead. `best.pt` is saved when the eval win rate matches or beats the best so far.

`runs/<name>/log.jsonl` has one line per update with everything:
- `pg`, `vl`: policy and value loss
- `ent`: entropy
- `clipfrac`: how often the PPO/GRPO clip was active
- Algorithm-specific stats:
  - GRPO: `group_std`
  - DQN: `td`, `q`, `eps`, `buffer`
  - ES: `fit_mean`, `fit_best`
- `phase`, `side`: the current phase with `--alternate`
- `<group>/win|ret|turns|n`: per-group episode stats
- `eval/<group>/win|ret|turns`: evaluation results

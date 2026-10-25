# civrl: RL on a simplified Civilization 6 combat game

Small hex-map scenarios with Civ6 combat rules. Six interchangeable learning algorithms (`--algo`): PPO, A2C, REINFORCE, GRPO, DQN and evolution strategies.

```bash
uv sync --extra cpu                                # or cu130 / cu126 for an NVIDIA GPU, see COMMANDS.md
uv run pytest -q                                   # rule tests
uv run civrl-train --scenario navigate --steps 300000
uv run civrl-gui                                   # pick scenario + models, replay with arrow keys
uv run civrl-plot  runs/<run>                       # redraw runs/<run>/plots.png
uv run civrl-compare --minutes 30 --name cmp1       # train all methods, then compare them
```

## Scenarios

| # | `--scenario` | Goal | Status |
|---|---|---|---|
| 1 | `navigate` | Warrior goes from A to the gold tile | done |
| 2 | `scout` | Scout reveals every land tile | done |
| 3 | `skirmish` | Player 0 (attacker, outnumbering by 1 unit up to 75%) kills all of player 1 before the turn limit; player 1 survives. Island size, turn limit and army sizes vary per episode | done |
| 4–6 | | Cities, production, walls, land control | next |
| 7–8 | | Science/tech tree, strategic resources (owning the tile suffices) | later |

## Flags (train)

- `--fixed-map [--map-seed N]`: same map (and spawns) every episode; default is a random island per episode.
- `--pre-scouted`: whole map revealed at the start; enemy units still need line of sight.
- `--full-vis`: everything, including enemy units, always visible.
- `--shaping X`: multiplier for the dense shaping rewards (0 = only win/loss).
- `--size W H`, `--max-turns N`: map size and turn limit.
- Skirmish: `--learner-side attacker|defender|both`, `--opponent scripted|self|mixed|frozen`.
- `--algo ppo|a2c|reinforce|grpo|dqn|es`: learning algorithm; each one's docstring in `algos/` explains how it works.
- `--alternate [--switch-at 0.6]`: train one side until its eval win rate reaches the threshold, then the other side against a frozen copy, and so on.

## Rules implemented

**Map.** Pointy-top hexes, island surrounded by water. Terrain: grassland, plains, desert, mountains (impassable), water (impassable for land units). Features: hills, woods, marsh, rivers on hex edges.

**Visibility.** Each player sees tiles as unexplored, revealed (terrain remembered) or visible. Sight is 2 tiles (+1 on hills). Mountains block line of sight, and woods and hills block it unless the viewer is on hills.

**Memory.** Every enemy unit ever seen is remembered at its last seen tile with the turn it was seen. The entry is updated when the unit is seen again and dropped when it is seen dying. This memory is shown to the agent as observation channels, so the policy can stay feed-forward.

**Military scores.** Both sides' military scores are common knowledge, like Civ6's military strength ranking. A side's score is the sum over its living units of max(melee CS, ranged RS) × HP/100, so a full-health warrior counts 20 and an archer 25. Each player's observation has two constant planes: its own score and the enemy's, each divided by 400.

**Turns.** Players alternate. Within a turn the player's units act in a fixed order, one decision at a time:

- An action targets one of the 18 tiles within distance 2 (move to an adjacent tile, melee an adjacent enemy, or make a ranged attack), or fortifies.
- Fortify also ends the unit's turn.
- Illegal actions are masked out.
- A decision with only one legal option is applied automatically.

**Movement.**

- Cost is 1, +1 each for hills, woods and marsh.
- Crossing a river costs +2.
- A unit with full MP can always move one tile.
- Entering a tile adjacent to an enemy unit (zone of control) ends movement.
- One unit per tile.

**Combat (Civ6).**

- Damage to the defender is `30 · e^(ΔCS/25) · U(0.8, 1.2)`, where ΔCS is attacker strength minus defender strength. In melee the attacker also takes damage with Δ reversed. Ranged attacks take no return damage.
- Every unit has 100 HP. A unit loses 1 CS for every 10 HP missing.
- Modifiers:
  - Defender on hills or in woods: +3.
  - Defender in marsh: −2.
  - Melee attack across a river: −5.
  - Flanking: +2 per other friendly unit adjacent to the target.
  - Support: +2 per friendly unit adjacent to the defender.
  - Anti-cavalry vs cavalry: +10.
  - Fortified: +6 if the unit fortified with full MP, +3 if it had already moved. The bonus lasts until it moves or attacks.
- Ranged attacks need line of sight.
- Attacking ends the unit's turn. A melee attacker that kills the defender moves into its tile.
- Healing: +10 HP at the end of a turn in which the unit didn't move or attack.
- Ranged vs cities: −12 (planned house rule; siege units are left out for now).

**Units.** Scout, warrior, slinger, archer, spearman, heavy chariot, horseman, swordsman with Civ6 stats (`src/civrl/units.py`).

## Code map

- `hexmap.py`: geometry and island generator
- `game.py`: rules engine
- `env.py`: base environment, config and observation encoding
- `scenarios/`: one file per scenario (`navigate.py`, `scout.py`, `skirmish.py`), each with its reward function and a short description of it
- `scripted.py`: baseline and opponent AIs
- `model.py`: CNN policy with a per-target-tile action head
- `rollout.py`: experience collection, opponents and multi-agent trajectory handling
- `algos/`: learning algorithms: `pg.py` (PPO, A2C, REINFORCE, GRPO), `dqn.py`, `es.py`
- `train.py`: entry point: arguments, `--alternate`, evaluation, logging, checkpoints
- `compare.py`: trains every method with alternating sides, then a cross-method tournament and comparison plots
- `render.py`: pygame renderer
- `plots.py`: training plots from `log.jsonl`
- `gui.py`: replay viewer GUI

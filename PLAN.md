# Plan

## Rules and agreed design (see README.md for details)
- [x] Agree on rule set and adjustments: river crossing +2 MP, fortify +6 with full MP / +3 after moving, ranged vs city −12, land control = tiles occupied or in your zone of control, units move one tile per action, alternating turns, enemy memory by unit identity, own-the-tile strategic resources
- [x] Set up the project with uv; add the uv preference to the global CLAUDE.md

## Phase 1: engine and scenarios 1–3
- [x] Hex map and random island generator; flags for fixed map, pre-scouted and full vis
- [x] Rules engine: movement, ZoC, visibility/LoS, Civ6 combat, fortify, healing, enemy memory
- [x] Rule unit tests
- [x] Scenario envs: 1 navigate, 2 scout, 3 skirmish; scripted baselines and opponents
- [x] Masked PPO trainer with multi-agent trajectory handling
- [x] Renderer (GIF demos later removed in favour of the GUI)
- [x] Training plots at end of run
- [x] Replay GUI (choose scenario and models, step with arrow keys)
- [ ] Self-play modes (`--opponent self|mixed|frozen`): implemented, never run

## Algorithm comparison
- [x] `--algo`: PPO, A2C, REINFORCE, GRPO, DQN (Rainbow-lite), ES; shared rollout, eval and plots
- [x] `--alternate`: train sides in turn, switching when the eval win rate reaches `--switch-at`
- [x] `civrl-compare`: time-budgeted alternating training per method, cross-method tournament, comparison plots
- [ ] Compare the algorithms on navigate / scout / skirmish (user's experiments)

## Phase 2: cities (scenarios 4–6)
- [ ] Cities: HP, walls, city strength, ranged strike with walls, capture by melee
- [ ] Automatic growth, tile working and borders; the agent chooses only production
- [ ] Scenario 4: conquer or defend a city
- [ ] Scenario 5: two sides with cities; kill the opponent, otherwise control the middle land
- [ ] Scenario 6: scenario 5 plus walls and unit production

## Phase 3: economy (scenarios 7–8)
- [ ] Science, Library, small military tech tree
- [ ] Strategic resources (Horses, Iron); owning the tile is enough

## Other
- [x] GPU support: per-machine PyTorch extras (cpu / cu126 / cu130), `--device auto`

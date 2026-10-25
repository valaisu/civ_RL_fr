"""Base environment: config, map/unit setup helpers and observation encoding.
The scenarios (with their reward functions) live in civrl/scenarios/, one file each.

API (multi-agent, turn based):
    env.reset(seed)
    p = env.to_act()                     # whose decision it is
    obs = env.observe(p)                 # dict: grid [C,H,W], focus (tile or N), mask [A]
    rewards, done = env.step(action)     # rewards: np.array per player
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, fields

import numpy as np

from .game import N_ACTIONS, Game
from .hexmap import DESERT, GRASS, MOUNTAIN, PLAINS, WATER, HexGeometry, generate_island
from .units import N_UNIT_TYPES, UNIT_INDEX

T = N_UNIT_TYPES
N_CHANNELS = 35 + 3 * T
MIL_SCALE = 400  # military score -> observation value (kept <= 1: DQN stores observations as 8-bit [0, 1])


@dataclass
class Config:
    scenario: str = "navigate"  # navigate | scout | skirmish
    W: int = 10
    H: int = 10
    random_map: bool = True
    map_seed: int = 0  # used when random_map is False
    pre_scouted: bool = False
    full_vis: bool = False
    max_turns: int = 20
    shaping: float = 1.0  # multiplier for dense shaping rewards (0 = sparse only)
    # skirmish: per-episode variation (min == max makes a value fixed)
    island_min: int = 9  # island region side length, clamped to the board (W x H)
    island_max: int = 14
    turns_min: int = 15
    turns_max: int = 30
    def_min: int = 2  # defender unit count
    def_max: int = 5
    att_ratio_max: float = 1.75  # attacker count in [defenders + 1, floor(att_ratio_max * defenders)]
    archer_frac: float = 1 / 3  # each unit is an archer with this probability, else a warrior
    # skirmish reward weights (see scenarios/skirmish.py)
    scout_w: float = 0.1  # attacker: total reward for revealing every land tile
    intel_w: float = 0.01  # attacker: per-turn reward when all enemies are in sight
    defense_w: float = 0.01  # defender: per-turn reward per 10 points of average defence bonus
    health_w: float = 0.02  # both: per-turn reward per unit of strength-weighted health balance
    fast_w: float = 0.5  # winner: bonus * (turns left / max turns)
    survive_w: float = 0.5  # loser: bonus * (turns survived / max turns)

    @classmethod
    def from_dict(cls, d):
        """Build from a saved config, ignoring fields that no longer exist."""
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


def military_score(game, p):
    """Public military strength of player p: sum over living units of
    max(melee CS, ranged RS) * HP / 100 (a full-health warrior counts 20, an archer 25)."""
    return sum(max(u.ut.cs, u.ut.rs) * u.hp / 100 for u in game.alive_units(p))


def bfs_dist(gmap, src):
    d = np.full(gmap.geo.N, 10**6, dtype=np.int64)
    d[src] = 0
    q = deque([src])
    while q:
        t = q.popleft()
        for n in gmap.geo.nbr[t]:
            if n >= 0 and gmap.passable[n] and d[n] > d[t] + 1:
                d[n] = d[t] + 1
                q.append(n)
    return d


_GEO_CACHE = {}


def _geometry(W, H):
    if (W, H) not in _GEO_CACHE:
        _GEO_CACHE[(W, H)] = HexGeometry(W, H)
    return _GEO_CACHE[(W, H)]


class CivEnv:
    n_players = 1

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.geo = _geometry(cfg.W, cfg.H)
        self.N = self.geo.N
        self.n_actions = N_ACTIONS
        self.game: Game | None = None

    # ---------------------------------------------------------------- lifecycle
    def reset(self, seed=None):
        cfg = self.cfg
        rng = np.random.default_rng(seed)
        map_rng = rng if cfg.random_map else np.random.default_rng(cfg.map_seed)
        gmap = self._make_map(map_rng)
        self.game = Game(gmap, self.n_players, rng, max_turns=self._max_turns(map_rng),
                         pre_scouted=cfg.pre_scouted, full_vis=cfg.full_vis)
        # With a fixed map, keep spawns fixed too (map_rng continues deterministically).
        self._setup(map_rng)
        self.game.start()
        self._after_setup()
        self.done = False
        return self

    def to_act(self):
        return self.game.player

    def step(self, action):
        g = self.game
        p = g.player
        turn_before = g.turn
        g.dmg_dealt[:] = 0
        g.kills[:] = 0
        g.step(int(action))
        r = self._reward(p, g.turn != turn_before)
        if g.over:
            self.done = True
        return r, self.done

    # ---------------------------------------------------------------- scenario hooks
    def _make_map(self, rng):
        return generate_island(self.geo, rng)

    def _max_turns(self, rng):
        return self.cfg.max_turns

    def _setup(self, rng):
        raise NotImplementedError

    def _after_setup(self):
        pass

    def _reward(self, p, new_turn):
        raise NotImplementedError

    def role(self, p):
        return 0.0

    def goal_tiles(self):
        return []

    def winner(self):
        return self.game.winner

    # ---------------------------------------------------------------- spawning
    def _spawn_group(self, rng, owner, names, anchor):
        g = self.game
        d = bfs_dist(g.map, anchor)
        cand = [t for t in np.argsort(d + rng.random(self.N) * 0.5) if d[t] < 10**6]
        i = 0
        for name in names:
            while g.tile_unit[cand[i]] >= 0:
                i += 1
            g.add_unit(owner, UNIT_INDEX[name], int(cand[i]))
            i += 1

    # ---------------------------------------------------------------- observation
    def observe(self, p):
        g, m, geo = self.game, self.game.map, self.geo
        N = self.N
        X = np.zeros((N_CHANNELS, N), dtype=np.float32)
        ex = g.explored[p]
        vis = g.visible[p]
        X[0] = (m.terrain == WATER) & ex
        X[1] = (m.terrain == MOUNTAIN) & ex
        X[2] = m.hills & ex
        X[3] = m.woods & ex
        X[4] = m.marsh & ex
        X[5] = (m.terrain == GRASS) & ex
        X[6] = (m.terrain == PLAINS) & ex
        X[7] = (m.terrain == DESERT) & ex
        X[8] = ex
        X[9] = vis
        X[10:16] = (m.river & ex[:, None]).T
        own, enemy, memc = 16, 21 + T, 24 + 2 * T
        for u in g.units:
            if not u.alive:
                continue
            t = u.tile
            if u.owner == p:
                X[own, t] = 1
                X[own + 1, t] = u.hp / 100
                X[own + 2, t] = u.mp / u.ut.mp
                X[own + 3, t] = u.fort / 6
                X[own + 4, t] = u.done
                X[own + 5 + u.type, t] = 1
            elif vis[t]:
                X[enemy, t] = 1
                X[enemy + 1, t] = u.hp / 100
                X[enemy + 2, t] = u.fort / 6
                X[enemy + 3 + u.type, t] = 1
        for uid, e in g.memory[p].items():
            if not vis[g.units[uid].tile] or not g.units[uid].alive:
                X[memc, e.tile] = 1
                X[memc + 1, e.tile] = min(1.0, (g.turn - e.turn + 1) / 10)
                X[memc + 2, e.tile] = e.hp / 100
                X[memc + 3 + e.type, e.tile] = 1
        base = 27 + 3 * T
        focus = N
        mask = np.zeros(N_ACTIONS, dtype=bool)
        if p == g.player and not g.over:
            u = g.focus_unit()
            if u is not None:
                focus = u.tile
                X[base, focus] = 1
                mask = g.legal_mask(u)
        # enemy zone of control from visible enemies
        for u in g.units:
            if u.alive and u.owner != p and vis[u.tile]:
                n = geo.nbr[u.tile]
                X[base + 1, n[n >= 0]] = 1
        for t in self.goal_tiles():
            X[base + 2, t] = 1
        X[base + 3] = geo.parity
        X[base + 4] = g.turn / g.max_turns
        X[base + 5] = self.role(p)
        # common knowledge: both sides' military scores (like Civ6's military strength ranking)
        X[base + 6] = min(1.0, military_score(g, p) / MIL_SCALE)
        X[base + 7] = min(1.0, sum(military_score(g, q) for q in range(g.n_players) if q != p) / MIL_SCALE)
        return {"grid": X.reshape(N_CHANNELS, geo.H, geo.W), "focus": focus, "mask": mask}

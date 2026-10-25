"""Core game rules: units, movement, visibility, combat, memory, turns.

Turns alternate: player 0 moves all its units (one decision at a time, in unit-id
order), then player 1, and so on. A decision is one action for the "focus" unit:
  0..17  target tile offset (hexmap.ACTION_OFFSETS): move / melee / ranged attack
  18     fortify (also ends the unit's turn)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .hexmap import ACTION_OFFSETS, MOUNTAIN, GameMap
from .units import CAVALRY, UNIT_TYPES

N_OFFSETS = len(ACTION_OFFSETS)
A_FORTIFY = N_OFFSETS
N_ACTIONS = N_OFFSETS + 1

SIGHT = 2
FORT_FULL, FORT_PARTIAL = 6, 3
HEAL_NEUTRAL = 10


@dataclass
class Unit:
    id: int
    owner: int
    type: int
    tile: int
    hp: int = 100
    mp: int = 0
    fort: int = 0  # fortify bonus currently held
    acted: bool = False  # moved or attacked this turn (blocks healing)
    done: bool = False  # finished for this turn
    alive: bool = True

    @property
    def ut(self):
        return UNIT_TYPES[self.type]


@dataclass
class Memory:
    tile: int
    type: int
    hp: int
    turn: int


class Game:
    def __init__(self, gmap: GameMap, n_players: int, rng: np.random.Generator,
                 max_turns: int = 50, pre_scouted=False, full_vis=False):
        gmap.freeze()
        self.map = gmap
        self.geo = gmap.geo
        self.n_players = n_players
        self.rng = rng
        self.max_turns = max_turns
        self.full_vis = full_vis
        N = self.geo.N
        self.units: list[Unit] = []
        self.tile_unit = np.full(N, -1, dtype=np.int64)
        self.explored = np.zeros((n_players, N), dtype=bool)
        self.visible = np.zeros((n_players, N), dtype=bool)
        if pre_scouted or full_vis:
            self.explored[:] = True
        self.memory: list[dict[int, Memory]] = [dict() for _ in range(n_players)]
        self.turn = 0
        self.player = 0
        self.over = False
        self.winner = -1
        self.end_check = None  # optional callable -> bool, checked between decisions
        # Event counters since last read (for rewards): dmg[p] dealt by p, kills[p] by p
        self.dmg_dealt = np.zeros(n_players)
        self.kills = np.zeros(n_players)
        self._vis_from = self._precompute_vis()
        self._vis_mat = np.zeros((N, N), dtype=bool)
        for v, ts in enumerate(self._vis_from):
            self._vis_mat[v, ts] = True

    # ------------------------------------------------------------ setup
    def add_unit(self, owner, type_idx, tile):
        assert self.tile_unit[tile] < 0 and self.map.passable[tile]
        u = Unit(len(self.units), owner, type_idx, tile)
        u.mp = u.ut.mp
        self.units.append(u)
        self.tile_unit[tile] = u.id
        return u

    def start(self):
        for p in range(self.n_players):
            self.update_vis(p)
        self._begin_turn(0)

    # ------------------------------------------------------------ visibility
    def _precompute_vis(self):
        """For each tile, the tiles visible from it with base sight (+1 on hills)."""
        geo, m = self.geo, self.map
        lines = geo.sight_lines(SIGHT + 1)
        # blocked[x] for x in [0, N) plus padding slot N (clear) and -1 (off-map: blocked)
        blk_flat = np.concatenate([(m.terrain == MOUNTAIN) | m.hills | m.woods, [False, True]])
        blk_high = np.concatenate([m.terrain == MOUNTAIN, [False, True]])
        out = []
        for v in range(geo.N):
            on_hills = bool(m.hills[v])
            s = SIGHT + (1 if on_hills else 0)
            blk = blk_high if on_hills else blk_flat
            ts, la, lb = lines[v]
            ok = (geo.dist[v, ts] <= s) & (~blk[la].any(1) | ~blk[lb].any(1))
            near = np.nonzero(geo.dist[v] <= 1)[0]
            out.append(np.concatenate([near, ts[ok]]))
        return out

    def vis_from(self, tile):
        return self._vis_from[tile]

    def update_vis(self, p):
        vis = np.zeros(self.geo.N, dtype=bool)
        if self.full_vis:
            vis[:] = True
        else:
            for u in self.units:
                if u.alive and u.owner == p:
                    vis[self.vis_from(u.tile)] = True
        self.visible[p] = vis
        self.explored[p] |= vis
        # Memory: last place each enemy unit was seen. Kept until seen again or known dead.
        mem = self.memory[p]
        for u in self.units:
            if u.alive and u.owner != p and vis[u.tile]:
                mem[u.id] = Memory(u.tile, u.type, u.hp, self.turn)

    def unit_visible_to(self, u: Unit, p: int):
        return u.owner == p or self.visible[p, u.tile]

    # ------------------------------------------------------------ helpers
    def unit_at(self, tile):
        i = self.tile_unit[tile]
        return self.units[i] if i >= 0 else None

    def enemy_zoc(self, p):
        """Tiles adjacent to (or occupied by) enemy military units of player p."""
        z = np.zeros(self.geo.N, dtype=bool)
        for u in self.units:
            if u.alive and u.owner != p:
                z[u.tile] = True
                n = self.geo.nbr[u.tile]
                z[n[n >= 0]] = True
        return z

    def in_enemy_zoc(self, p, tile):
        for n in self.geo.nbr[tile]:
            if n >= 0:
                o = self.unit_at(n)
                if o is not None and o.owner != p:
                    return True
        return False

    def los(self, a, b):
        return bool(self._vis_mat[a, b])

    # ------------------------------------------------------------ decisions
    def focus_unit(self) -> Unit | None:
        for u in self.units:
            if u.alive and u.owner == self.player and not u.done:
                return u
        return None

    def legal_mask(self, u: Unit):
        mask = np.zeros(N_ACTIONS, dtype=bool)
        if u.mp <= 0:
            return mask
        mask[A_FORTIFY] = True
        geo, m = self.geo, self.map
        ut = u.ut
        for k, t in enumerate(geo.targets[u.tile]):
            if t < 0:
                continue
            o = self.unit_at(t)
            if k < 6:
                if o is None:
                    if m.passable[t] and (u.mp >= ut.mp or m.move_cost(u.tile, t) <= u.mp):
                        mask[k] = True
                elif o.owner != u.owner and ut.rs == 0:
                    mask[k] = True  # melee attack
            if ut.rs > 0 and o is not None and o.owner != u.owner:
                if geo.dist[u.tile, t] <= ut.range and self.visible[u.owner, t] and self.los(u.tile, t):
                    mask[k] = True
        return mask

    def step(self, action: int):
        """Apply one action for the focus unit, then auto-advance turns."""
        u = self.focus_unit()
        assert u is not None and not self.over
        mask = self.legal_mask(u)
        assert mask[action], f"illegal action {action}"
        if action == A_FORTIFY:
            self._fortify(u)
        else:
            t = int(self.geo.targets[u.tile, action])
            o = self.unit_at(t)
            if o is None:
                self._move(u, t)
            elif u.ut.rs > 0:
                self._ranged(u, o)
            else:
                self._melee(u, o)
        self._advance()

    def _advance(self):
        """Skip units without choices; end turns when no unit is left to move."""
        while not self.over:
            if self.end_check is not None and self.end_check():
                self.over = True
                return
            u = self.focus_unit()
            if u is None:
                self._end_turn()
                continue
            mask = self.legal_mask(u)
            n = mask.sum()
            if n == 0:
                u.done = True
            elif n == 1:  # only option is fortify: no real decision, apply it
                self._fortify(u)
            else:
                return

    def _fortify(self, u: Unit):
        u.fort = max(u.fort, FORT_FULL if u.mp >= u.ut.mp else FORT_PARTIAL)
        u.done = True

    # ------------------------------------------------------------ movement
    def _move(self, u: Unit, t: int):
        cost = self.map.move_cost(u.tile, t)
        self.tile_unit[u.tile] = -1
        u.tile = t
        self.tile_unit[t] = u.id
        u.mp = max(0, u.mp - cost)
        u.fort = 0
        u.acted = True
        if self.in_enemy_zoc(u.owner, t):
            u.mp = 0
        if u.mp == 0:
            u.done = True
        self.update_vis(u.owner)
        # Our new position might now be seen by the enemy (only matters for memory).
        for p in range(self.n_players):
            if p != u.owner:
                self.update_vis(p)

    # ------------------------------------------------------------ combat
    def _wound(self, u: Unit):
        return (100 - u.hp) // 10

    def _adjacent_count(self, owner, tile, exclude=-1):
        c = 0
        for n in self.geo.nbr[tile]:
            if n >= 0:
                o = self.unit_at(n)
                if o is not None and o.owner == owner and o.id != exclude:
                    c += 1
        return c

    def defense_bonus(self, d: Unit):
        """Situational defence modifiers: fortify, terrain and support (not vs a specific attacker)."""
        m = self.map
        s = d.fort
        if m.hills[d.tile]:
            s += 3
        if m.woods[d.tile]:
            s += 3
        if m.marsh[d.tile]:
            s -= 2
        s += 2 * self._adjacent_count(d.owner, d.tile)  # support
        return s

    def defense_strength(self, d: Unit, a: Unit):
        s = d.ut.cs - self._wound(d) + self.defense_bonus(d)
        if d.ut.cls == "anti_cav" and a.ut.cls in CAVALRY:
            s += 10
        return s

    def attack_strength(self, a: Unit, d: Unit, ranged: bool):
        s = (a.ut.rs if ranged else a.ut.cs) - self._wound(a)
        if not ranged:
            s += 2 * self._adjacent_count(a.owner, d.tile, exclude=a.id)  # flanking
            dirn = self.geo.dir_between(a.tile, d.tile)
            if self.map.river[a.tile, dirn]:
                s -= 5
        if a.ut.cls == "anti_cav" and d.ut.cls in CAVALRY:
            s += 10
        return s

    @staticmethod
    def damage(diff, rng):
        return int(round(30 * math.exp(diff / 25) * rng.uniform(0.8, 1.2)))

    def _ranged(self, a: Unit, d: Unit):
        diff = self.attack_strength(a, d, True) - self.defense_strength(d, a)
        dmg = min(d.hp, self.damage(diff, self.rng))
        d.hp -= dmg
        self.dmg_dealt[a.owner] += dmg
        self._end_attack(a)
        if d.hp <= 0:
            self._kill(d, a.owner)

    def _melee(self, a: Unit, d: Unit):
        diff = self.attack_strength(a, d, False) - self.defense_strength(d, a)
        dmg_d = min(d.hp, self.damage(diff, self.rng))
        dmg_a = min(a.hp, self.damage(-diff, self.rng))
        d.hp -= dmg_d
        a.hp -= dmg_a
        self.dmg_dealt[a.owner] += dmg_d
        self.dmg_dealt[d.owner] += dmg_a
        self._end_attack(a)
        if d.hp <= 0:
            a.hp = max(a.hp, 1)
            t = d.tile
            self._kill(d, a.owner)
            self.tile_unit[a.tile] = -1
            a.tile = t
            self.tile_unit[t] = a.id
            self.update_vis(a.owner)
        elif a.hp <= 0:
            self._kill(a, d.owner)

    def _end_attack(self, a: Unit):
        a.mp = 0
        a.fort = 0
        a.acted = True
        a.done = True

    def _kill(self, u: Unit, killer: int):
        u.alive = False
        self.tile_unit[u.tile] = -1
        self.kills[killer] += 1
        for p in range(self.n_players):
            if p != u.owner and (self.visible[p, u.tile] or p == killer):
                self.memory[p].pop(u.id, None)

    # ------------------------------------------------------------ turns
    def _end_turn(self):
        p = self.player
        for u in self.units:
            if u.alive and u.owner == p and not u.acted:
                u.hp = min(100, u.hp + HEAL_NEUTRAL)
        nxt = (p + 1) % self.n_players
        if nxt == 0:
            self.turn += 1
            if self.turn >= self.max_turns:
                self.over = True
                return
        self._begin_turn(nxt)

    def _begin_turn(self, p):
        self.player = p
        for u in self.units:
            if u.alive and u.owner == p:
                u.mp = u.ut.mp
                u.acted = False
                u.done = False
        self.update_vis(p)

    def alive_units(self, p):
        return [u for u in self.units if u.alive and u.owner == p]

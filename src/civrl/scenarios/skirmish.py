"""Scenario 3 - skirmish: player 0 (attacker) must kill every unit of player 1 (defender)
before the turn limit; the defender wins by surviving.

Setup (varies every episode; ranges are Config fields, min == max makes a value fixed):
  * board 14x14; the island is generated in a random region of 9..14 x 9..14 tiles
  * turn limit 15..30
  * defender 2..5 units; attacker between defenders+1 and floor(1.75 * defenders) units
  * each unit is an archer with probability 1/3, otherwise a warrior
The two armies spawn as far apart as possible. The attacker moves first. The observation's
role channel is 1 for the attacker, 0 for the defender; the turn channel is turn / turn limit.

Episode ends: all defenders dead (attacker wins), all attackers dead or turn limit
reached (defender wins).

Reward. Both players get a reward after every decision, whoever made it. All terms except
the base +-1 are multiplied by `shaping`; weights are Config fields (CLI: --scout-w etc.).
  Both sides (zero-sum):
  * damage:   +0.5 * (HP damage I dealt / enemy's total starting HP)
              -0.5 * (HP damage I took  / my total starting HP)          (healing not counted)
  * health:   once per turn: +health_w * (my health - enemy health), where a side's health is
              sum(strength * HP) over its living units / sum(strength * 100) over its starting units,
              strength = max(melee CS, ranged RS) (warrior 20, archer 25). Healing raises it.
  * outcome:  winner +1 + fast_w * (turns left / turn limit)
              loser  -1 + survive_w * (turns survived / turn limit)
  Attacker only:
  * scouting: +scout_w * (newly revealed land tiles / all land tiles), whenever tiles get revealed
  * intel:    once per turn: +intel_w * (sum of "freshness" over living enemies / enemy starting count)
              freshness = 1 if the enemy is in sight now, falling linearly to 0 over 5 turns
              since it was last seen (1, 0.8, 0.6, 0.4, 0.2, 0)
  Defender only:
  * defence:  once per turn: +defense_w * (average defence bonus of living defenders / 10)
              defence bonus = fortify + terrain (hills/woods +3, marsh -2) + support (+2 per adjacent friend)
  "Once per turn" means when a new turn starts, i.e. right after the defender's moves.

Note: rewards earned during the opponent's moves (e.g. damage taken) are credited by the
trainer to the player's most recent decision (see Runner in train.py).
"""
import numpy as np

from ..env import CivEnv, bfs_dist
from ..hexmap import generate_island

PRESET = dict(scenario="skirmish", W=14, H=14)
INTEL_FADE_TURNS = 5


def strength(u):
    return max(u.ut.cs, u.ut.rs)


class SkirmishEnv(CivEnv):
    n_players = 2

    def _make_map(self, rng):
        cfg = self.cfg
        rw = int(rng.integers(min(cfg.island_min, cfg.W), min(cfg.island_max, cfg.W) + 1))
        rh = int(rng.integers(min(cfg.island_min, cfg.H), min(cfg.island_max, cfg.H) + 1))
        c0 = int(rng.integers(0, cfg.W - rw + 1))
        r0 = int(rng.integers(0, cfg.H - rh + 1))
        return generate_island(self.geo, rng, region=(c0, r0, rw, rh))

    def _max_turns(self, rng):
        return int(rng.integers(self.cfg.turns_min, self.cfg.turns_max + 1))

    def _setup(self, rng):
        cfg = self.cfg
        n_def = int(rng.integers(cfg.def_min, cfg.def_max + 1))
        n_att = int(rng.integers(n_def + 1, max(n_def + 1, int(cfg.att_ratio_max * n_def)) + 1))
        armies = [["archer" if rng.random() < cfg.archer_frac else "warrior" for _ in range(n)]
                  for n in (n_att, n_def)]
        m = self.game.map
        land = np.nonzero(m.passable)[0]
        a = int(rng.choice(land))
        da = bfs_dist(m, a)
        b = int(land[np.argmax(np.where(da[land] < 10**6, da[land], -1))])
        db = bfs_dist(m, b)
        a = int(land[np.argmax(np.where(db[land] < 10**6, db[land], -1))])  # farthest from b
        self._spawn_group(rng, 0, armies[0], a)
        self._spawn_group(rng, 1, armies[1], b)
        g = self.game
        g.end_check = lambda: not g.alive_units(0) or not g.alive_units(1)
        self.n_start = [n_att, n_def]
        self.start_hp = np.array([100.0 * n_att, 100.0 * n_def])
        self.start_strength = [100.0 * sum(strength(u) for u in g.units if u.owner == q) for q in range(2)]

    def _after_setup(self):
        self.prev_explored = int((self.game.explored[0] & self.game.map.land).sum())

    def role(self, p):
        return 1.0 if p == 0 else 0.0

    def _intel(self):
        """Attacker's knowledge of enemy positions, 0..1 (see module docstring)."""
        g = self.game
        total = 0.0
        for u in g.alive_units(1):
            if g.visible[0, u.tile]:
                total += 1.0
            elif u.id in g.memory[0]:
                age = g.turn - g.memory[0][u.id].turn
                total += max(0.0, 1.0 - age / INTEL_FADE_TURNS)
        return total / self.n_start[1]

    def _defence(self):
        g = self.game
        units = g.alive_units(1)
        return np.mean([g.defense_bonus(u) for u in units]) / 10 if units else 0.0

    def _health(self, q):
        """Strength-weighted remaining health of player q's starting army, 0..1."""
        return sum(strength(u) * u.hp for u in self.game.alive_units(q)) / self.start_strength[q]

    def _reward(self, p, new_turn):
        g, cfg = self.game, self.cfg
        r = np.zeros(2, dtype=np.float32)
        w = 0.5 * cfg.shaping
        r[0] += w * (g.dmg_dealt[0] / self.start_hp[1] - g.dmg_dealt[1] / self.start_hp[0])
        r[1] += w * (g.dmg_dealt[1] / self.start_hp[0] - g.dmg_dealt[0] / self.start_hp[1])

        explored = int((g.explored[0] & g.map.land).sum())
        r[0] += cfg.shaping * cfg.scout_w * (explored - self.prev_explored) / g.map.land.sum()
        self.prev_explored = explored
        if new_turn:
            r[0] += cfg.shaping * cfg.intel_w * self._intel()
            r[1] += cfg.shaping * cfg.defense_w * self._defence()
            balance = self._health(0) - self._health(1)
            r[0] += cfg.shaping * cfg.health_w * balance
            r[1] -= cfg.shaping * cfg.health_w * balance

        alive = [len(g.alive_units(q)) for q in range(2)]
        if alive[1] == 0:
            g.winner = 0
        elif alive[0] == 0 or g.over:
            g.winner = 1
        if g.winner >= 0:
            g.over = True
            frac = min(g.turn, g.max_turns) / g.max_turns  # share of the turn limit used
            r[g.winner] += 1 + cfg.shaping * cfg.fast_w * (1 - frac)
            r[1 - g.winner] += -1 + cfg.shaping * cfg.survive_w * frac
        return r

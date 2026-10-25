"""Scenario 1 - navigate: move a single warrior from its start tile to the goal tile.

Setup: one warrior and a goal tile at least 6 steps away (shortest walkable path).
The goal is always marked in the observation, even under fog.

Episode ends: the warrior reaches the goal (win), or the turn limit runs out (no win).

Reward (for player 0, after every decision):
  * progress:  +0.05 * shaping per tile the warrior got closer to the goal
               (shortest walkable path on the true map; moving away gives the same amount negative)
  * time:      -0.01 every time a new turn starts
  * goal:      +1 when the warrior enters the goal tile
"""
import numpy as np

from ..env import CivEnv, bfs_dist
from ..units import UNIT_INDEX

PRESET = dict(scenario="navigate", W=10, H=10, max_turns=20)


class NavigateEnv(CivEnv):
    def _setup(self, rng):
        m = self.game.map
        land = np.nonzero(m.passable)[0]
        for _ in range(100):
            a = int(rng.choice(land))
            d = bfs_dist(m, a)
            far = land[(d[land] >= 6) & (d[land] < 10**6)]
            if len(far):
                break
        self.goal = int(rng.choice(far))
        self.game.add_unit(0, UNIT_INDEX["warrior"], a)
        self.dist_to_goal = bfs_dist(m, self.goal)
        self.game.end_check = lambda: self.game.units[0].tile == self.goal

    def _after_setup(self):
        self.prev_d = self.dist_to_goal[self.game.units[0].tile]

    def goal_tiles(self):
        return [self.goal]

    def _reward(self, p, new_turn):
        g = self.game
        u = g.units[0]
        d = self.dist_to_goal[u.tile]
        r = self.cfg.shaping * 0.05 * (self.prev_d - d)
        self.prev_d = d
        if new_turn:
            r -= 0.01
        if u.tile == self.goal:
            r += 1.0
            g.over = True
            g.winner = 0
        return np.array([r], dtype=np.float32)

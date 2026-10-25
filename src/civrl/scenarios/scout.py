"""Scenario 2 - scout: reveal every land tile of the island with a single scout.

Setup: one scout (3 MP) on a random land tile. The "target" tiles are all land tiles
that can be seen from at least one walkable tile, so revealing all of them is always possible.

Episode ends: every target tile has been revealed (win), or the turn limit runs out (no win).

Reward (for player 0, after every decision):
  * exploration: +shaping * (newly revealed target tiles / number of target tiles)
                 (adds up to at most +1 * shaping over an episode)
  * time:        -0.01 every time a new turn starts
  * completion:  +1 when the last target tile is revealed
"""
import numpy as np

from ..env import CivEnv
from ..units import UNIT_INDEX

PRESET = dict(scenario="scout", W=10, H=10, max_turns=40)


class ScoutEnv(CivEnv):
    def _setup(self, rng):
        g = self.game
        m = g.map
        land = np.nonzero(m.passable)[0]
        g.add_unit(0, UNIT_INDEX["scout"], int(rng.choice(land)))
        seeable = np.zeros(self.N, dtype=bool)
        for t in land:
            seeable[g.vis_from(t)] = True
        self.targets = seeable & m.land
        g.end_check = lambda: bool((g.explored[0] & self.targets).sum() == self.targets.sum())

    def _after_setup(self):
        self.prev_seen = int((self.game.explored[0] & self.targets).sum())

    def _reward(self, p, new_turn):
        g = self.game
        seen = int((g.explored[0] & self.targets).sum())
        r = self.cfg.shaping * (seen - self.prev_seen) / self.targets.sum()
        self.prev_seen = seen
        if new_turn:
            r -= 0.01
        if seen == self.targets.sum():
            r += 1.0
            g.over = True
            g.winner = 0
        return np.array([r], dtype=np.float32)

    def coverage(self):
        return (self.game.explored[0] & self.targets).sum() / self.targets.sum()

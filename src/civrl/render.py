"""Pygame hex renderer (headless capable), used by the GUI."""
from __future__ import annotations

import math
import os

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame  # noqa: E402

from .hexmap import DESERT, GRASS, MOUNTAIN, PLAINS, WATER  # noqa: E402

TERRAIN_COL = {
    WATER: (52, 102, 168),
    GRASS: (98, 152, 62),
    PLAINS: (165, 168, 82),
    DESERT: (222, 200, 140),
    MOUNTAIN: (128, 120, 112),
}
PLAYER_COL = [(70, 130, 230), (215, 60, 50), (240, 200, 40), (160, 80, 200)]
RIVER_COL = (70, 150, 230)
SQ3 = math.sqrt(3)


class Renderer:
    def __init__(self, env, hex_size=26, views=(None,)):
        """views: list of player ids (perspective with fog) or None (omniscient)."""
        if not pygame.get_init():
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            pygame.init()
        self.env = env
        self.s = hex_size
        self.views = list(views)
        geo = env.geo
        self.margin = hex_size
        self.top = 34
        self.pw = int(SQ3 * hex_size * (geo.W + 0.5) + 2 * self.margin)
        self.ph = int(1.5 * hex_size * (geo.H - 1) + 2 * hex_size + 2 * self.margin) + self.top
        self.surf = pygame.Surface((self.pw * len(self.views), self.ph))
        self.font = pygame.font.Font(None, int(hex_size * 0.8))
        self.small = pygame.font.Font(None, int(hex_size * 0.6))
        self.title = pygame.font.Font(None, 24)

    def center(self, t):
        geo = self.env.geo
        c, r = geo.col[t], geo.row[t]
        x = self.margin + SQ3 * self.s * (c + 0.5 * (r & 1)) + SQ3 * self.s / 2
        y = self.top + self.margin + 1.5 * self.s * r + self.s
        return x, y

    def corners(self, t, scale=1.0):
        x, y = self.center(t)
        return [(x + scale * self.s * math.cos(math.radians(60 * i - 30)),
                 y + scale * self.s * math.sin(math.radians(60 * i - 30))) for i in range(6)]

    def edge(self, t, d):
        """Screen endpoints of the edge between tile t and its neighbour in direction d."""
        x, y = self.center(t)
        out = []
        for ang in (-60 * d - 30, -60 * d + 30):
            out.append((x + self.s * math.cos(math.radians(ang)), y + self.s * math.sin(math.radians(ang))))
        return out

    # ------------------------------------------------------------------ drawing
    def draw(self, caption=""):
        self.surf.fill((18, 18, 24))
        for i, v in enumerate(self.views):
            panel = pygame.Surface((self.pw, self.ph))
            self._draw_panel(panel, v, caption)
            self.surf.blit(panel, (i * self.pw, 0))
        return self.surf

    def _draw_panel(self, S, viewer, caption):
        env, g = self.env, self.env.game
        m, geo = g.map, g.geo
        S.fill((18, 18, 24))
        label = "omniscient" if viewer is None else f"player {viewer} view"
        txt = f"{env.cfg.scenario}  turn {g.turn + 1}/{g.max_turns}  [{label}]  {caption}"
        S.blit(self.title.render(txt, True, (230, 230, 230)), (8, 8))
        explored = np.ones(geo.N, bool) if viewer is None else g.explored[viewer]
        visible = np.ones(geo.N, bool) if viewer is None else g.visible[viewer]
        for t in range(geo.N):
            pts = self.corners(t)
            if not explored[t]:
                pygame.draw.polygon(S, (8, 8, 10), pts)
                pygame.draw.polygon(S, (30, 30, 34), pts, 1)
                continue
            pygame.draw.polygon(S, TERRAIN_COL[int(m.terrain[t])], pts)
            x, y = self.center(t)
            s = self.s
            if m.terrain[t] == MOUNTAIN:
                pygame.draw.polygon(S, (90, 84, 80), [(x - 0.6 * s, y + 0.4 * s), (x, y - 0.6 * s), (x + 0.6 * s, y + 0.4 * s)])
                pygame.draw.polygon(S, (240, 240, 245), [(x - 0.2 * s, y - 0.3 * s), (x, y - 0.6 * s), (x + 0.2 * s, y - 0.3 * s)])
            if m.hills[t]:
                for dx in (-0.35, 0.3):
                    pygame.draw.arc(S, (110, 80, 45), (x + dx * s - 0.3 * s, y + 0.05 * s, 0.6 * s, 0.5 * s), 0, math.pi, 3)
            if m.woods[t]:
                for dx, dy in ((-0.35, -0.25), (0.3, -0.3), (0.0, 0.05)):
                    cx, cy = x + dx * s, y + dy * s
                    pygame.draw.polygon(S, (34, 85, 34), [(cx - 0.17 * s, cy + 0.22 * s), (cx, cy - 0.22 * s), (cx + 0.17 * s, cy + 0.22 * s)])
            if m.marsh[t]:
                for dy in (-0.15, 0.15):
                    pygame.draw.line(S, (60, 120, 150), (x - 0.4 * s, y + dy * s), (x + 0.4 * s, y + dy * s), 2)
            pygame.draw.polygon(S, (40, 40, 40), pts, 1)
        # rivers
        for t in range(geo.N):
            if not explored[t]:
                continue
            for d in range(3):  # each edge once (E, NE, NW from this side)
                if m.river[t, d]:
                    pygame.draw.line(S, RIVER_COL, *self.edge(t, d), 4)
            for d in range(3, 6):
                n = geo.nbr[t, d]
                if m.river[t, d] and (n < 0 or not explored[n]):
                    pygame.draw.line(S, RIVER_COL, *self.edge(t, d), 4)
        # fog
        fog = pygame.Surface((self.pw, self.ph), pygame.SRCALPHA)
        for t in range(geo.N):
            if explored[t] and not visible[t]:
                pygame.draw.polygon(fog, (0, 0, 0, 120), self.corners(t))
        S.blit(fog, (0, 0))
        # goals
        for t in env.goal_tiles():
            pygame.draw.polygon(S, (255, 215, 0), self.corners(t, 0.85), 3)
        # remembered enemies (ghosts)
        if viewer is not None:
            for uid, e in g.memory[viewer].items():
                u = g.units[uid]
                if u.alive and visible[u.tile]:
                    continue
                self._draw_unit(S, e.tile, u.owner, e.type, e.hp, 0, ghost=g.turn - e.turn)
        for u in g.units:
            if u.alive and (viewer is None or u.owner == viewer or visible[u.tile]):
                self._draw_unit(S, u.tile, u.owner, u.type, u.hp, u.fort)
        if g.over:
            msg = "GAME OVER" + (f" - player {g.winner} wins" if g.winner >= 0 else "")
            img = self.title.render(msg, True, (255, 255, 255))
            S.blit(img, (self.pw // 2 - img.get_width() // 2, self.ph - 26))

    def _draw_unit(self, S, t, owner, utype, hp, fort, ghost=None):
        from .units import UNIT_TYPES

        x, y = self.center(t)
        s = self.s
        col = PLAYER_COL[owner]
        r = int(0.48 * s)
        if ghost is not None:
            pygame.draw.circle(S, col, (x, y), r, 2)
            img = self.small.render(f"{UNIT_TYPES[utype].code}?{ghost}", True, col)
            S.blit(img, (x - img.get_width() / 2, y - img.get_height() / 2))
            return
        pygame.draw.circle(S, col, (x, y), r)
        pygame.draw.circle(S, (15, 15, 15), (x, y), r, 2)
        img = self.font.render(UNIT_TYPES[utype].code, True, (255, 255, 255))
        S.blit(img, (x - img.get_width() / 2, y - img.get_height() / 2))
        if fort:  # shield badge with the fortify bonus
            cx, cy, w, h = x + 0.45 * s, y - 0.4 * s, 0.24 * s, 0.3 * s
            shield = [(cx - w, cy - h), (cx + w, cy - h), (cx + w, cy + 0.2 * h), (cx, cy + h), (cx - w, cy + 0.2 * h)]
            pygame.draw.polygon(S, (215, 215, 225), shield)
            pygame.draw.polygon(S, (20, 20, 20), shield, 2)
            img = self.small.render(str(fort), True, (20, 20, 20))
            S.blit(img, (cx - img.get_width() / 2, cy - 0.15 * h - img.get_height() / 2))
        w = 0.9 * s
        bh = max(5, int(0.12 * s))
        pygame.draw.rect(S, (30, 30, 30), (x - w / 2, y + 0.55 * s, w, bh))
        hc = (60, 200, 60) if hp > 60 else (230, 200, 40) if hp > 30 else (220, 50, 40)
        pygame.draw.rect(S, hc, (x - w / 2, y + 0.55 * s, w * hp / 100, bh))

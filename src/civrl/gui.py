"""Interactive GUI: pick a scenario and policies, then replay the game step by step.

    uv run civrl-gui

Viewer keys:
    Left / Right        previous / next decision
    Up / Down           previous / next player-turn
    Home / End          first / last state
    Space               autoplay on/off
    V                   cycle views: omniscient, each player's fog-of-war view, all side by side
    F11                 toggle fullscreen
    N                   new episode (next seed), same settings
    Esc                 back to menu
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame  # noqa: E402
import torch  # noqa: E402

from .env import Config
from .scenarios import PRESETS, make_config, make_env
from .game import A_FORTIFY
from .render import PLAYER_COL, Renderer
from .scripted import random_action, scripted_action
from .model import Policy

DIR_NAMES = ["E", "NE", "NW", "W", "SW", "SE"]
BG = (24, 24, 30)
FG = (225, 225, 225)
DIM = (140, 140, 150)
HL = (255, 215, 0)
ATTACK = (255, 70, 50)
SIDEBAR = 360
LIST_TOP = 110  # menu: top of the per-player option lists


def find_checkpoints(runs_dir="runs"):
    """{scenario: [(label, path), ...]} for every runs/<name>/{best,last}.pt
    (and runs/<comparison>/<algo>/{best,last}.pt from civrl-compare)."""
    out = {}
    root = Path(runs_dir)
    for cfg_file in sorted([*root.glob("*/config.json"), *root.glob("*/*/config.json")]):
        scen = json.loads(cfg_file.read_text())["cfg"]["scenario"]
        name = cfg_file.parent.relative_to(root).as_posix()
        for which in ("best", "last"):
            pt = cfg_file.parent / f"{which}.pt"
            if pt.exists():
                out.setdefault(scen, []).append((f"{name}/{which}", str(pt)))
    return out


def load_policy(spec, env, rng, greedy=True):
    """spec: 'scripted', 'random', or a checkpoint path. Returns fn(env, p) -> action.
    For checkpoints, greedy=True takes the most likely action, otherwise samples."""
    if spec == "scripted":
        return lambda env, p: scripted_action(env, p)
    if spec == "random":
        return lambda env, p: random_action(env, rng)
    ck = torch.load(spec)
    # older checkpoints predate observation channels that were appended later; they see only the first in_ch
    in_ch = ck["model"]["stem.weight"].shape[1]
    net = Policy(in_ch, env.geo.targets, env.n_actions, ch=ck["ch"], blocks=ck["blocks"])
    net.load_state_dict(ck["model"])
    net.eval()

    def act(env, p):
        o = env.observe(p)
        with torch.no_grad():
            logits, _ = net(torch.as_tensor(o["grid"][None, :in_ch]), torch.as_tensor([o["focus"]]),
                            torch.as_tensor(o["mask"][None]))
        if greedy:
            return int(logits.argmax())
        return int(torch.distributions.Categorical(logits=logits).sample())

    return act


# ============================================================== episode recording
def snapshot(game):
    """Deep copy of the mutable game state; static map data is shared."""
    memo = {id(game.map): game.map, id(game.geo): game.geo, id(game.rng): game.rng,
            id(game._vis_mat): game._vis_mat, id(game._vis_from): game._vis_from}
    return copy.deepcopy(game, memo)


def describe(game, action):
    """Text for the action about to be applied, (from, to) tiles for highlighting,
    and the ids of units in combat."""
    u = game.focus_unit()
    who = f"P{u.owner} {u.ut.name}#{u.id}"
    if action == A_FORTIFY:
        bonus = 6 if u.mp >= u.ut.mp else 3
        return f"{who} fortifies (+{bonus})", (u.tile, u.tile), []
    t = int(game.geo.targets[u.tile, action])
    o = game.unit_at(t)
    if o is None:
        return f"{who} moves {DIR_NAMES[action]}", (u.tile, t), []
    kind = "shoots" if u.ut.rs > 0 else "attacks"
    return f"{who} {kind} P{o.owner} {o.ut.name}#{o.id}", (u.tile, t), [u.id, o.id]


def record_episode(env, policies, seed):
    """Play one episode. Returns list of states: dict(game, text, move, player)."""
    env.reset(seed)
    states = [dict(game=snapshot(env.game), text="start", move=None, player=env.to_act(), attack=False)]
    done = False
    while not done:
        g = env.game
        p = env.to_act()
        a = policies[p](env, p)
        text, move, fighters = describe(g, a)
        hp_before = {i: g.units[i].hp for i in fighters}
        _, done = env.step(a)
        if fighters:  # combat outcome
            us = [env.game.units[i] for i in fighters if env.game.units[i].hp != hp_before[i]]
            text += ": " + ", ".join(f"{u.ut.name}#{u.id} {hp_before[u.id]}->{max(u.hp, 0)}"
                                     + (" (dead)" if not u.alive else "") for u in us)
        states.append(dict(game=snapshot(env.game), text=text, move=move, player=p, attack=bool(fighters)))
    return states


# ============================================================== widgets
class Button:
    def __init__(self, rect, label, on_click, selected=False):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.on_click = on_click
        self.selected = selected

    def draw(self, S, font):
        col = (70, 90, 140) if self.selected else (50, 50, 60)
        pygame.draw.rect(S, col, self.rect, border_radius=4)
        pygame.draw.rect(S, (90, 90, 100), self.rect, 1, border_radius=4)
        img = font.render(self.label, True, FG)
        S.blit(img, (self.rect.x + 8, self.rect.centery - img.get_height() // 2))


class App:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("civrl")
        # (0, 0) = desktop size; resizable so the window manager can maximize it
        self.screen = pygame.display.set_mode((0, 0), pygame.RESIZABLE)
        self.font = pygame.font.Font(None, 24)
        self.small = pygame.font.Font(None, 20)
        self.big = pygame.font.Font(None, 34)
        self.clock = pygame.time.Clock()
        self.ckpts = find_checkpoints()
        self.scenario = "navigate"
        self.choice = ["scripted", "scripted"]  # per player: 'scripted' | 'random' | path
        self.scroll = [0, 0]  # per player: first visible checkpoint row
        self.seed = 0
        self.fixed_map = False
        self.pre_scouted = False
        self.full_vis = False
        self.greedy = True
        self.mode = "menu"
        self.message = ""

    # ---------------------------------------------------------------- menu
    def options(self):
        return [("scripted", "scripted"), ("random", "random")] + self.ckpts.get(self.scenario, [])

    def n_players(self):
        return 2 if self.scenario == "skirmish" else 1

    def menu_layout(self):
        """y of the settings bar, and how many option rows fit above it (lists scroll beyond that)."""
        bar_y = max(300, self.screen.get_height() - 130)
        return bar_y, max(1, (bar_y - 30 - LIST_TOP) // 32)

    def menu_buttons(self):
        bs = []
        x, y = 30, 80
        for name in PRESETS:
            def pick(n=name):
                self.scenario = n
                self.choice = ["scripted", "scripted"]
                self.scroll = [0, 0]
            bs.append(Button((x, y, 150, 32), name, pick, self.scenario == name))
            y += 40
        bar_y, rows = self.menu_layout()
        opts = self.options()
        self.col_rects = []
        for p in range(self.n_players()):
            x = 220 + p * 400
            self.scroll[p] = max(0, min(self.scroll[p], len(opts) - rows))
            for j, (label, val) in enumerate(opts[self.scroll[p]: self.scroll[p] + rows]):
                def pick(p=p, v=val):
                    self.choice[p] = v
                bs.append(Button((x, LIST_TOP + j * 32, 380, 28), label, pick, self.choice[p] == val))
            self.col_rects.append(pygame.Rect(x, LIST_TOP, 380, rows * 32))
        # settings row
        y = bar_y
        def toggle(attr):
            def f():
                setattr(self, attr, not getattr(self, attr))
            return f
        for i, (attr, label) in enumerate([("fixed_map", "fixed map"), ("pre_scouted", "pre-scouted"),
                                           ("full_vis", "full vis"), ("greedy", "greedy (vs sample)")]):
            bs.append(Button((30 + i * 190, y, 180, 30), f"[{'x' if getattr(self, attr) else ' '}] {label}",
                             toggle(attr), getattr(self, attr)))
        def seed_delta(d):
            def f():
                self.seed = max(0, self.seed + d)
            return f
        bs.append(Button((30, y + 45, 40, 30), "-", seed_delta(-1)))
        bs.append(Button((180, y + 45, 40, 30), "+", seed_delta(1)))
        bs.append(Button((230, y + 45, 110, 30), "random", lambda: setattr(self, "seed", int(np.random.randint(10**6)))))
        bs.append(Button((360, y + 45, 170, 30), "PLAY  (Enter)", self.start, True))
        return bs

    def draw_menu(self, buttons):
        S = self.screen
        S.fill(BG)
        S.blit(self.big.render("civrl replay viewer", True, FG), (30, 20))
        S.blit(self.font.render("scenario", True, DIM), (30, 58))
        for p in range(self.n_players()):
            lab = "player 0 (attacker)" if self.n_players() == 2 and p == 0 else (
                "player 1 (defender)" if p == 1 else "player 0")
            S.blit(self.font.render(lab, True, PLAYER_COL[p]), (220 + p * 400, 88))
        bar_y, rows = self.menu_layout()
        S.blit(self.font.render(f"seed {self.seed}", True, FG), (80, bar_y + 52))
        if not self.ckpts.get(self.scenario):
            S.blit(self.small.render(f"(no checkpoints for '{self.scenario}' in runs/)", True, DIM), (220, 60))
        n_opts = len(self.options())
        for p in range(self.n_players()):
            if n_opts > rows:  # scroll hint under the list
                s = self.scroll[p]
                hint = f"rows {s + 1}-{s + rows} of {n_opts}  (mouse wheel to scroll)"
                S.blit(self.small.render(hint, True, DIM), (220 + p * 400, LIST_TOP + rows * 32 + 2))
        for b in buttons:
            b.draw(S, self.small)
        if self.message:
            S.blit(self.font.render(self.message, True, (240, 120, 100)), (30, bar_y + 95))

    # ---------------------------------------------------------------- episode
    def start(self):
        paths = [c for c in self.choice[: self.n_players()] if c not in ("scripted", "random")]
        if paths:
            cfg = Config.from_dict(torch.load(paths[0])["cfg"])
            for other in paths[1:]:
                oc = Config.from_dict(torch.load(other)["cfg"])
                if (oc.W, oc.H) != (cfg.W, cfg.H):
                    self.message = "selected models were trained on different map sizes"
                    return
        else:
            cfg = make_config(self.scenario)
        if self.fixed_map:
            cfg.random_map = False
        cfg.pre_scouted |= self.pre_scouted
        cfg.full_vis |= self.full_vis
        self.env = make_env(cfg)
        rng = np.random.default_rng(self.seed)
        self.policies = [load_policy(self.choice[p], self.env, rng, greedy=self.greedy)
                         for p in range(self.n_players())]
        self.new_episode()

    def new_episode(self):
        self.message = ""
        self.states = record_episode(self.env, self.policies, self.seed)
        self.idx = 0
        self.autoplay = False
        n = self.env.n_players
        self.view_sets = [[None]] + [[p] for p in range(n)] + [[None] + list(range(n))]
        self.view_i = 0
        self.make_renderer()
        self.mode = "view"

    def make_renderer(self):
        views = self.view_sets[self.view_i]
        # largest hexes such that all panels fill the window next to the sidebar
        geo = self.env.geo
        win_w, win_h = self.screen.get_size()
        avail_w = (win_w - SIDEBAR) / len(views)
        size = min(avail_w / (1.732 * (geo.W + 0.5) + 2), (win_h - 40) / (1.5 * (geo.H - 1) + 4))
        self.R = Renderer(self.env, hex_size=max(8, int(size)), views=views)

    def jump_turn(self, d):
        """Move to the first state of the next/previous player-turn."""
        st = self.states
        key = lambda i: (st[i]["game"].turn, st[i]["game"].player)
        i = self.idx
        if d > 0:
            k = key(i)
            while i < len(st) - 1 and key(i) == k:
                i += 1
        else:
            if i > 0:
                i -= 1
            k = key(i)
            while i > 0 and key(i - 1) == k:
                i -= 1
        self.idx = i

    def draw_view(self):
        S = self.screen
        S.fill(BG)
        st = self.states[self.idx]
        g = st["game"]
        self.env.game = g
        surf = self.R.draw()
        # center the map area left of the sidebar
        mx = (S.get_width() - SIDEBAR - surf.get_width()) // 2
        my = (S.get_height() - surf.get_height()) // 2
        S.blit(surf, (mx, my))
        # line for the move/attack that led to this state (fortify shows as a shield badge)
        if st["move"] is not None and st["move"][0] != st["move"][1]:
            a, b = st["move"]
            col = ATTACK if st["attack"] else HL
            for i, v in enumerate(self.R.views):
                if not (v is None or v == st["player"] or g.visible[v, a] or g.visible[v, b]):
                    continue  # this player could not see the move
                ox = mx + i * self.R.pw
                pa, pb = self.R.center(a), self.R.center(b)
                pa, pb = (pa[0] + ox, pa[1] + my), (pb[0] + ox, pb[1] + my)
                # stop the line at the unit circles so the labels stay readable
                dx, dy = pb[0] - pa[0], pb[1] - pa[1]
                k = 0.5 * self.R.s / max(1e-6, (dx * dx + dy * dy) ** 0.5)
                pa, pb = (pa[0] + k * dx, pa[1] + k * dy), (pb[0] - k * dx, pb[1] - k * dy)
                pygame.draw.line(S, col, pa, pb, max(3, int(self.R.s * 0.1)))
        # sidebar
        x = S.get_width() - SIDEBAR + 12
        y = 12
        def line(txt, col=FG, f=None):
            nonlocal y
            S.blit((f or self.font).render(txt, True, col), (x, y))
            y += (f or self.font).get_height() + 4
        line(f"{self.env.cfg.scenario}  seed {self.seed}", f=self.big)
        for p in range(self.env.n_players):
            name = Path(self.choice[p]).parent.name + "/" + Path(self.choice[p]).stem \
                if self.choice[p] not in ("scripted", "random") else self.choice[p]
            line(f"P{p}: {name}", PLAYER_COL[p], self.small)
        y += 6
        line(f"state {self.idx}/{len(self.states) - 1}   turn {g.turn + 1}/{g.max_turns}")
        line(f"to move: P{g.player}" if not g.over else
             ("GAME OVER - " + (f"P{g.winner} wins" if g.winner >= 0 else "no winner")), HL)
        y += 6
        line("last actions:", DIM, self.small)
        lh = self.small.get_height() + 4
        help_y = S.get_height() - 8 * lh - 10
        log = []  # newest entries last; keep only as many lines as fit above the help
        for i in range(max(1, self.idx - 13), self.idx + 1):
            s = self.states[i]
            col = PLAYER_COL[s["player"]] if i == self.idx else DIM
            log += [(chunk, col) for chunk in _wrap(s["text"], 44)]
        n_fit = max(0, (help_y - 8 - y) // lh)
        for chunk, col in log[len(log) - n_fit:]:
            line(chunk, col, self.small)
        y = help_y
        for h in ["Left/Right: decision", "Up/Down: player-turn", "Home/End: start/end",
                  "Space: autoplay", "V: change view", "N: next seed", "F11: fullscreen", "Esc: menu"]:
            line(h, DIM, self.small)

    # ---------------------------------------------------------------- loop
    def run(self):
        while True:
            buttons = self.menu_buttons() if self.mode == "menu" else []
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    return
                if ev.type == pygame.VIDEORESIZE or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_F11):
                    if ev.type == pygame.KEYDOWN:
                        pygame.display.toggle_fullscreen()
                    self.screen = pygame.display.get_surface()
                    if self.mode == "view":
                        self.make_renderer()
                    continue
                if self.mode == "menu":
                    if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                        for b in buttons:
                            if b.rect.collidepoint(ev.pos):
                                b.on_click()
                    elif ev.type == pygame.MOUSEWHEEL:
                        for p, r in enumerate(self.col_rects):
                            if r.collidepoint(pygame.mouse.get_pos()):
                                self.scroll[p] -= ev.y
                    elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_RETURN:
                        self.start()
                    elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                        return
                elif ev.type == pygame.KEYDOWN:
                    last = len(self.states) - 1
                    if ev.key == pygame.K_RIGHT:
                        self.idx = min(last, self.idx + 1)
                    elif ev.key == pygame.K_LEFT:
                        self.idx = max(0, self.idx - 1)
                    elif ev.key == pygame.K_DOWN:
                        self.jump_turn(+1)
                    elif ev.key == pygame.K_UP:
                        self.jump_turn(-1)
                    elif ev.key == pygame.K_HOME:
                        self.idx = 0
                    elif ev.key == pygame.K_END:
                        self.idx = last
                    elif ev.key == pygame.K_SPACE:
                        self.autoplay = not self.autoplay
                    elif ev.key == pygame.K_v:
                        self.view_i = (self.view_i + 1) % len(self.view_sets)
                        self.make_renderer()
                    elif ev.key == pygame.K_n:
                        self.seed += 1
                        self.new_episode()
                    elif ev.key == pygame.K_ESCAPE:
                        self.mode = "menu"
            if self.mode == "menu":
                self.draw_menu(buttons)
            else:
                if self.autoplay:
                    self.idx = min(len(self.states) - 1, self.idx + 1)
                    self.autoplay = self.idx < len(self.states) - 1
                self.draw_view()
            pygame.display.flip()
            self.clock.tick(4 if self.mode == "view" and self.autoplay else 30)


def _wrap(text, n):
    words, out, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > n and cur:
            out.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    return out + [cur] if cur else out


def main():
    App().run()


if __name__ == "__main__":
    main()

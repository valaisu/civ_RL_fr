"""Hand-written policies: baselines and opponents. They only use information the
player could know (own units, visible enemies, remembered enemies, explored map)."""
from __future__ import annotations

from collections import deque

import numpy as np

from .game import A_FORTIFY, Game

INF = 10**6


def multi_bfs(game: Game, sources, passable=None):
    m = game.map
    if passable is None:
        passable = m.passable
    d = np.full(game.geo.N, INF, dtype=np.int64)
    q = deque()
    for s in sources:
        d[s] = 0
        q.append(s)
    while q:
        t = q.popleft()
        for n in game.geo.nbr[t]:
            if n >= 0 and passable[n] and d[n] > d[t] + 1:
                d[n] = d[t] + 1
                q.append(n)
    return d


def step_towards(game: Game, u, mask, dist):
    """Best legal move (k<6, empty tile) that lowers `dist`; None if no progress."""
    best, best_d = None, dist[u.tile]
    for k in range(6):
        t = game.geo.targets[u.tile, k]
        if mask[k] and game.unit_at(t) is None and dist[t] < best_d:
            best, best_d = k, dist[t]
    return best


def known_enemies(game: Game, p):
    tiles = [u.tile for u in game.units if u.alive and u.owner != p and game.visible[p, u.tile]]
    tiles += [e.tile for e in game.memory[p].values() if not game.visible[p, e.tile]]
    return tiles


def explore_target_dist(game: Game, p):
    m = game.map
    # passable tiles we have not explored are treated as passable (optimistic)
    passable = m.passable | ~game.explored[p]
    # tiles from which something unexplored would come into view
    reveals = (game._vis_mat & ~game.explored[p][None, :]).any(axis=1) & passable
    return multi_bfs(game, np.nonzero(reveals)[0], passable)


def pick_attack(game: Game, u, mask, min_diff):
    best, best_score = None, -1e9
    for k in range(18):
        if not mask[k]:
            continue
        t = game.geo.targets[u.tile, k]
        o = game.unit_at(t)
        if o is None:
            continue
        ranged = u.ut.rs > 0
        diff = game.attack_strength(u, o, ranged) - game.defense_strength(o, u)
        exp_dmg = 30 * np.exp(diff / 25)
        kill = exp_dmg >= o.hp
        if not ranged and diff < min_diff and not kill:
            continue
        score = exp_dmg + (100 if kill else 0) - (0 if ranged else 30 * np.exp(-diff / 25))
        if score > best_score:
            best, best_score = k, score
    return best


def attacker_action(game: Game, p):
    u = game.focus_unit()
    mask = game.legal_mask(u)
    k = pick_attack(game, u, mask, min_diff=-4)
    if k is not None:
        return k
    enemies = known_enemies(game, p)
    if enemies:
        if u.ut.rs > 0 and min(game.geo.dist[u.tile, e] for e in enemies) <= u.ut.range:
            return A_FORTIFY  # in range; wait for a shot
        dist = multi_bfs(game, enemies)
    else:
        dist = explore_target_dist(game, p)
    k = step_towards(game, u, mask, dist)
    return A_FORTIFY if k is None else k


def defender_action(game: Game, p):
    u = game.focus_unit()
    mask = game.legal_mask(u)
    k = pick_attack(game, u, mask, min_diff=0)
    if k is not None:
        return k
    m = game.map
    if m.hills[u.tile] or m.woods[u.tile]:
        return A_FORTIFY
    good = [t for t in np.nonzero((m.hills | m.woods) & m.passable)[0]
            if game.unit_at(t) is None and game.geo.dist[u.tile, t] <= 3]
    if good:
        k = step_towards(game, u, mask, multi_bfs(game, good))
        if k is not None:
            return k
    return A_FORTIFY


def navigate_action(env):
    g = env.game
    u = g.focus_unit()
    mask = g.legal_mask(u)
    # plan on the explored map only (unexplored counted as passable)
    passable = g.map.passable | ~g.explored[0]
    k = step_towards(g, u, mask, multi_bfs(g, [env.goal], passable))
    return A_FORTIFY if k is None else k


def scout_action(env):
    g = env.game
    u = g.focus_unit()
    mask = g.legal_mask(u)
    k = step_towards(g, u, mask, explore_target_dist(g, 0))
    return A_FORTIFY if k is None else k


def random_action(env, rng):
    mask = env.game.legal_mask(env.game.focus_unit())
    return int(rng.choice(np.nonzero(mask)[0]))


def scripted_action(env, p):
    s = env.cfg.scenario
    if s == "navigate":
        return navigate_action(env)
    if s == "scout":
        return scout_action(env)
    if s == "skirmish":
        return attacker_action(env.game, p) if p == 0 else defender_action(env.game, p)
    raise ValueError(s)

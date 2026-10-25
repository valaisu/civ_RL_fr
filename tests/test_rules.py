import numpy as np
import pytest

from civrl.game import A_FORTIFY, Game, Memory
from civrl.hexmap import GRASS, HexGeometry, GameMap
from civrl.units import UNIT_INDEX


def flat_game(W=8, H=8, n_players=2):
    geo = HexGeometry(W, H)
    m = GameMap(geo)
    m.terrain[:] = GRASS
    return Game(m, n_players, np.random.default_rng(0), max_turns=99)


def k_to(g, a, b):
    return list(g.geo.targets[a]).index(b)


def test_river_crossing_costs_two_extra():
    g = flat_game()
    a = 3 * 8 + 3
    b = g.geo.nbr[a, 0]
    g.map.river[a, 0] = g.map.river[b, 3] = True
    assert g.map.move_cost(a, b) == 3
    g.map.hills[b] = True
    assert g.map.move_cost(a, b) == 4


def test_full_mp_can_always_move_one_tile():
    g = flat_game(n_players=1)
    a = 3 * 8 + 3
    b = g.geo.nbr[a, 0]
    g.map.hills[b] = g.map.woods[b] = True  # cost 3 > warrior's 2 MP
    u = g.add_unit(0, UNIT_INDEX["warrior"], a)
    g.start()
    assert g.legal_mask(u)[0]
    g.step(0)
    assert u.tile == b and g.turn == 1  # move used all MP, so the turn ended


def test_fortify_bonus_full_vs_partial():
    g = flat_game(n_players=1)
    u1 = g.add_unit(0, UNIT_INDEX["warrior"], 10)
    u2 = g.add_unit(0, UNIT_INDEX["warrior"], 30)
    g.start()
    g.step(A_FORTIFY)  # u1 with full MP
    assert u1.fort == 6
    g.step(0)  # u2 moves one tile (1 of 2 MP)
    g.step(A_FORTIFY)
    assert u2.fort == 3
    # moving resets fortification
    g.step(0)
    assert u1.fort == 0


def test_zoc_stops_movement():
    g = flat_game()
    a = 3 * 8 + 2
    u = g.add_unit(0, UNIT_INDEX["horseman"], a)
    e_tile = g.geo.nbr[g.geo.nbr[a, 0], 0]
    g.add_unit(1, UNIT_INDEX["warrior"], e_tile)
    g.start()
    g.step(0)  # step next to the enemy
    assert u.mp == 0 and u.done


def test_damage_formula_equal_strength():
    rng = np.random.default_rng(0)
    d = [Game.damage(0, rng) for _ in range(2000)]
    assert 24 <= min(d) and max(d) <= 36 and abs(np.mean(d) - 30) < 0.5


def test_flanking_support_and_terrain():
    g = flat_game()
    c = 4 * 8 + 4
    d = g.add_unit(1, UNIT_INDEX["warrior"], c)
    a = g.add_unit(0, UNIT_INDEX["warrior"], g.geo.nbr[c, 3])
    assert g.attack_strength(a, d, False) == 20
    g.add_unit(0, UNIT_INDEX["warrior"], g.geo.nbr[c, 0])  # flanker
    assert g.attack_strength(a, d, False) == 22
    g.add_unit(1, UNIT_INDEX["warrior"], g.geo.nbr[c, 1])  # supporter
    g.map.hills[c] = True
    assert g.defense_strength(d, a) == 20 + 2 + 3


def test_wounded_penalty_and_anticav():
    g = flat_game()
    sp = g.add_unit(1, UNIT_INDEX["spearman"], 20)
    h = g.add_unit(0, UNIT_INDEX["horseman"], 21)
    sp.hp = 55
    assert g.defense_strength(sp, h) == 25 - 4 + 10


def test_ranged_attack_no_counter_and_melee_advances():
    g = flat_game()
    c = 4 * 8 + 4
    ar = g.add_unit(0, UNIT_INDEX["archer"], g.geo.targets[c, 6 + 0])  # 2 tiles away
    wa = g.add_unit(0, UNIT_INDEX["warrior"], g.geo.nbr[c, 3])
    e = g.add_unit(1, UNIT_INDEX["warrior"], c)
    g.start()
    g.step(k_to(g, ar.tile, c))
    assert ar.hp == 100 and e.hp < 100
    e.hp = 1
    g.step(k_to(g, wa.tile, c))
    assert not e.alive and wa.tile == c


def test_memory_tracks_last_seen_enemy():
    g = flat_game(W=12, H=4)
    me = g.add_unit(0, UNIT_INDEX["scout"], 1 * 12 + 1)
    en = g.add_unit(1, UNIT_INDEX["warrior"], 1 * 12 + 3)
    g.start()
    assert en.id in g.memory[0]
    # enemy walks out of sight: memory stays at the old tile
    g.tile_unit[en.tile] = -1
    en.tile = 1 * 12 + 10
    g.tile_unit[en.tile] = en.id
    g.update_vis(0)
    assert g.memory[0][en.id].tile == 1 * 12 + 3
    # seen again -> memory updates
    me.tile, g.tile_unit[1 * 12 + 1], g.tile_unit[1 * 12 + 9] = 1 * 12 + 9, -1, me.id
    g.update_vis(0)
    assert g.memory[0][en.id].tile == 1 * 12 + 10


def test_defense_bonus_terrain_fortify_support():
    g = flat_game()
    c = 4 * 8 + 4
    d = g.add_unit(1, UNIT_INDEX["warrior"], c)
    g.map.hills[c] = True
    d.fort = 6
    g.add_unit(1, UNIT_INDEX["warrior"], g.geo.nbr[c, 0])
    assert g.defense_bonus(d) == 3 + 6 + 2


def test_skirmish_intel_fades_over_five_turns():
    from civrl.scenarios import make_config, make_env

    env = make_env(make_config("skirmish", def_min=2, def_max=2))
    env.reset(0)
    g = env.game
    enemy = g.alive_units(1)[0]
    g.visible[0, :] = False
    g.memory[0].clear()
    assert env._intel() == 0
    g.visible[0, enemy.tile] = True
    assert env._intel() == 0.5  # 1 of 2 enemies in sight
    g.visible[0, :] = False
    g.memory[0][enemy.id] = Memory(enemy.tile, enemy.type, enemy.hp, g.turn - 2)
    assert abs(env._intel() - 0.5 * 0.6) < 1e-9
    g.memory[0][enemy.id].turn = g.turn - 5
    assert env._intel() == 0


def test_skirmish_variable_setup_in_ranges():
    from civrl.scenarios import make_config, make_env

    cfg = make_config("skirmish")
    env = make_env(cfg)
    seen_turns, seen_def = set(), set()
    for seed in range(60):
        env.reset(seed)
        g = env.game
        n_att, n_def = len(g.alive_units(0)), len(g.alive_units(1))
        assert cfg.def_min <= n_def <= cfg.def_max
        assert n_def + 1 <= n_att <= max(n_def + 1, int(cfg.att_ratio_max * n_def))
        assert cfg.turns_min <= g.max_turns <= cfg.turns_max
        cols, rows = g.geo.col[g.map.land], g.geo.row[g.map.land]
        assert cols.max() - cols.min() + 1 <= cfg.island_max and rows.max() - rows.min() + 1 <= cfg.island_max
        seen_turns.add(g.max_turns)
        seen_def.add(n_def)
    assert len(seen_turns) > 5 and seen_def == set(range(cfg.def_min, cfg.def_max + 1))


def test_skirmish_health_is_strength_weighted():
    from civrl.scenarios import make_config, make_env

    env = make_env(make_config("skirmish"))
    env.reset(3)
    g = env.game
    assert env._health(0) == 1 and env._health(1) == 1
    u = g.alive_units(1)[0]
    u.hp = 50
    s = 25 if u.ut.name == "archer" else 20
    assert abs(env._health(1) - (1 - s * 50 / env.start_strength[1])) < 1e-9


def test_skirmish_outcome_rewards_time():
    from civrl.scenarios import make_config, make_env

    env = make_env(make_config("skirmish", def_min=2, def_max=2))
    env.reset(0)
    g = env.game
    g.turn = g.max_turns // 2
    for u in g.alive_units(1):
        g._kill(u, 0)
    r = env._reward(0, False)
    frac = g.turn / g.max_turns
    assert abs(r[0] - (1 + 0.5 * (1 - frac))) < 1e-6
    assert abs(r[1] - (-1 + 0.5 * frac)) < 1e-6


def test_military_scores_are_public():
    from civrl.env import MIL_SCALE, military_score
    from civrl.scenarios import make_config, make_env
    env = make_env(make_config("skirmish", full_vis=False)).reset(5)
    g = env.game
    g.units[0].hp = 50
    for p in (0, 1):
        x = env.observe(p)["grid"]
        own, enemy = x[-2].ravel(), x[-1].ravel()
        assert np.allclose(own, military_score(g, p) / MIL_SCALE)  # same value on every tile
        assert np.allclose(enemy, military_score(g, 1 - p) / MIL_SCALE)  # known even under fog
    assert military_score(g, g.units[0].owner) == sum(
        max(u.ut.cs, u.ut.rs) * u.hp / 100 for u in g.alive_units(g.units[0].owner))

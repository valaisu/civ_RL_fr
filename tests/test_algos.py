"""Plumbing checks for the learning algorithms (that they run, not that they learn)."""
import argparse

import numpy as np
import pytest

from civrl.algos import ALGOS
from civrl.algos.dqn import nstep_transitions
from civrl.rollout import Runner
from civrl.scenarios import make_config
from civrl.train import make_net


def test_grpo_groups_share_start_and_complete():
    cfg = make_config("navigate")
    r = Runner(cfg, 3, "attacker", "scripted", 0, group_size=4)
    net = make_net(r.envs[0], 8, 1)
    from civrl.algos.pg import sample_policy
    b, boot = r.rollout(sample_policy(net), 2, complete=True)
    assert b["done"].sum() == len(set(b["tid"].tolist())) and not boot  # every episode finished
    groups = {}
    for tid, g in zip(b["tid"], b["gkey"]):
        groups.setdefault(g, set()).add(tid)
    assert all(len(t) == 4 for t in groups.values())  # groups are always completed
    starts = {}
    for i in np.unique(b["tid"], return_index=True)[1]:
        starts.setdefault(b["gkey"][i], set()).add(b["grid"][i].tobytes())
    assert all(len(s) == 1 for s in starts.values())  # same first observation within a group


def test_nstep_targets():
    g = np.zeros((4, 1, 1, 1), np.float16)
    b = {"tid": np.array([0, 0, 0, 0]), "rew": np.array([1, 1, 1, 1], np.float32),
         "done": np.array([0, 0, 0, 1], bool), "grid": g, "focus": np.zeros(4, int),
         "mask": np.ones((4, 2), bool), "act": np.zeros(4, int)}
    rows = nstep_transitions(b, {}, 2, 0.5)
    assert rows["ret"] == [1.5, 1.5, 1.5, 1.0]
    assert rows["disc"] == [0.25, 0.25, 0.0, 0.0]


def args_for(algo, **kw):
    a = dict(algo=algo, steps=1000, envs=2, rollout=4, epochs=1, minibatch=16, lr=None, gamma=0.99, lam=0.95,
             clip=0.2, ent=0.01, group_size=2, buffer=200, nstep=2, replay_ratio=1, target_every=5,
             eps_end=0.05, eps_frac=0.3, per_alpha=0.6, es_pop=2, es_sigma=0.02, es_episodes=1, seed=0)
    a.update(kw)
    a["lr"] = a["lr"] or ALGOS[algo].lr
    return argparse.Namespace(**a)


@pytest.mark.parametrize("algo", list(ALGOS))
@pytest.mark.parametrize("opponent", ["scripted", "self"])
def test_one_update(algo, opponent):
    cfg = make_config("skirmish", W=10, H=10, island_max=10, turns_min=3, turns_max=3, def_min=1, def_max=1)
    a = args_for(algo, minibatch=4 if algo == "dqn" else 16)
    r = Runner(cfg, a.envs, "both", opponent, 0, group_size=a.group_size if algo == "grpo" else 1)
    net = make_net(r.envs[0], 8, 1)
    alg = ALGOS[algo](a, net, r)
    for _ in range(3):
        n, stats = alg.update()
        assert n > 0 and all(np.isfinite(v) for v in stats.values())
    alg.on_phase_change()

"""Evolution strategies (OpenAI-ES, Salimans et al. 2017): no backpropagation through the
policy at all.

Each update ("generation"):
  * draw --es-pop noise vectors eps_i and form antithetic candidates theta +- sigma * eps_i
    (sigma = --es-sigma), so 2 * pop networks
  * every candidate plays the same --es-episodes episodes (same seeds and seating: common
    random numbers, so differences come from the weights, not from easier maps). The
    candidate acts greedily (argmax), opponents act as in training.
  * fitness = the candidate's mean episode return; fitnesses are replaced by their centered
    ranks (-0.5 .. 0.5), which ignores the reward scale and outliers
  * gradient estimate: sum_i (rank(+i) - rank(-i)) * eps_i / (2 * pop * sigma), applied
    with Adam (default lr 0.003: with few candidates the estimate is noisy, and Adam would
    turn each noisy estimate into a full-size step for every weight)
Credit assignment inside an episode does not exist here: only the total return counts.
With --opponent self the opponent is the unperturbed network theta.
"""
import copy

import numpy as np
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from ..rollout import batch_obs
from ..scenarios import make_env
from ..scripted import scripted_action
from .base import Algo


def play(cfg, jobs):
    """Play episodes in parallel. jobs: list of (seed, controllers), one controller per player:
    "scripted" or (net, greedy). Returns per job (returns per player, winner, turns, decisions)."""
    envs = [make_env(cfg).reset(seed) for seed, _ in jobs]
    ret = np.zeros((len(jobs), envs[0].n_players))
    steps = np.zeros(len(jobs), dtype=np.int64)
    live = list(range(len(jobs)))
    while live:
        by_net = {}
        for j in live:
            env = envs[j]
            while not env.done and jobs[j][1][env.to_act()] == "scripted":
                ret[j] += env.step(scripted_action(env, env.to_act()))[0]
            if not env.done:
                by_net.setdefault(id(jobs[j][1][env.to_act()]), []).append(j)
        for js in by_net.values():
            net, greedy = jobs[js[0]][1][envs[js[0]].to_act()]
            with torch.no_grad():
                logits, _ = net(*batch_obs([envs[j].observe(envs[j].to_act()) for j in js], net.device))
            acts = logits.argmax(1) if greedy else torch.distributions.Categorical(logits=logits).sample()
            for j, a in zip(js, acts.tolist()):
                steps[j] += 1
                ret[j] += envs[j].step(a)[0]
        live = [j for j in live if not envs[j].done]
    return [(ret[j], envs[j].winner(), envs[j].game.turn, steps[j]) for j in range(len(jobs))]


class ES(Algo):
    lr = 0.003

    def __init__(self, args, net, runner):
        super().__init__(args, net, runner)
        self.cands = [copy.deepcopy(net) for _ in range(2 * args.es_pop)]
        self.rng = np.random.default_rng(args.seed)

    def update(self):
        a, r = self.args, self.runner
        theta = parameters_to_vector(self.net.parameters()).detach()
        noise = torch.as_tensor(self.rng.standard_normal((a.es_pop, len(theta))), dtype=torch.float32,
                                device=theta.device)
        for i, c in enumerate(self.cands):
            sign = 1 if i % 2 == 0 else -1
            vector_to_parameters(theta + sign * a.es_sigma * noise[i // 2], c.parameters())
        episodes = [(int(r.rng.integers(2**31)), r.assign()) for _ in range(a.es_episodes)]
        opp = {"learner": (self.net, False), "frozen": (r.opp_net, False),
               "checkpoint": (r.opp_net, False), "scripted": "scripted"}
        jobs, owner = [], []
        for ci, c in enumerate(self.cands):
            for seed, ctrl in episodes:
                me = ctrl.index("learner")  # the seat the candidate plays
                jobs.append((seed, [(c, True) if q == me else opp[x] for q, x in enumerate(ctrl)]))
                owner.append((ci, me, ctrl))
        results = play(r.cfg, jobs)
        fit = np.zeros(len(self.cands))
        n = 0
        for (ci, me, ctrl), (ret, winner, turns, steps) in zip(owner, results):
            fit[ci] += ret[me] / a.es_episodes
            n += int(steps)
            # log as an episode of the learner in its seat (self-play opponent = unperturbed net)
            seats = tuple("learner" if q == me else ("self" if x == "learner" else x) for q, x in enumerate(ctrl))
            r.ep_stats.append({"ctrl": seats, "winner": winner, "turns": turns, "returns": ret.tolist()})
        ranks = np.empty(len(fit))
        ranks[np.argsort(fit)] = np.arange(len(fit))
        ranks = ranks / (len(fit) - 1) - 0.5
        w = torch.as_tensor(ranks[0::2] - ranks[1::2], dtype=torch.float32, device=theta.device)
        grad = -(w @ noise) / (2 * a.es_pop * a.es_sigma)  # minus: Adam minimizes
        self.opt.zero_grad()
        for p, g in zip(self.net.parameters(), _split(grad, self.net)):
            p.grad = g
        self.opt.step()
        return n, {"fit_mean": float(fit.mean()), "fit_best": float(fit.max())}

    def describe(self):
        a = self.args
        return (f"ES: lr={a.lr}, {a.es_pop} antithetic pairs ({2 * a.es_pop} candidates), sigma={a.es_sigma}, "
                f"{a.es_episodes} episodes per candidate, candidates act greedily")


def _split(vec, net):
    out, i = [], 0
    for p in net.parameters():
        out.append(vec[i:i + p.numel()].view_as(p).clone())
        i += p.numel()
    return out

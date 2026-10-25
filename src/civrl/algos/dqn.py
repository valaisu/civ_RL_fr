"""DQN with most Rainbow ingredients: Double DQN, dueling head, n-step returns and
prioritized replay (no distributional values, no noisy nets).

The network learns Q(s, a): the expected discounted return after taking action a in s.
  * acting: epsilon-greedy over the legal actions; epsilon falls linearly from 1 to
    --eps-end during the first --eps-frac of training
  * every decision goes into a replay buffer (--buffer transitions, oldest dropped)
    as (s, a, n-step return R, s'), where s' is the same player's decision --nstep decisions
    later (rewards earned during the opponent's moves are included in R)
  * target: R + gamma^n * Q_target(s', argmax_a' Q(s', a'))  (Double DQN: the online net
    picks a', a slowly updated copy evaluates it; the copy is refreshed every
    --target-every gradient steps). Terminal s': 0.
  * dueling: Q(s, a) = V(s) + A(s, a) - mean of A over legal actions. The policy network's
    action logits serve as A and its value head as V, so a DQN checkpoint plays greedily
    (argmax of the logits) in the GUI and in evaluation like any other.
  * prioritized replay: transitions are sampled with probability ~ |TD error|^alpha
    (--per-alpha, 0 = uniform), corrected with importance weights (beta 0.4 -> 1)
  * each update collects one rollout and then takes about --replay-ratio * new decisions
    / --minibatch gradient steps (Huber loss)
Fixed opponents and self-play work as for the other algorithms; with self-play both seats'
decisions go into the same buffer.
"""
import copy
import math

import numpy as np
import torch
import torch.nn.functional as F

from ..rollout import batch_obs
from .base import Algo


def q_values(logits, val, mask):
    a = logits.masked_fill(~mask, 0)
    m = mask.float()
    mean = (a * m).sum(1) / m.sum(1).clamp_min(1)
    return (val[:, None] + a - mean[:, None]).masked_fill(~mask, -1e9)


class Replay:
    FIELDS = ("grid", "focus", "mask", "act", "ret", "disc", "next_grid", "next_focus", "next_mask")

    def __init__(self, cap, grid_shape, n_actions):
        self.cap, self.n, self.i = cap, 0, 0
        self.d = {"grid": np.zeros((cap, *grid_shape), np.uint8),  # observations are in [0, 1]
                  "next_grid": np.zeros((cap, *grid_shape), np.uint8),
                  "focus": np.zeros(cap, np.int64), "next_focus": np.zeros(cap, np.int64),
                  "mask": np.zeros((cap, n_actions), bool), "next_mask": np.zeros((cap, n_actions), bool),
                  "act": np.zeros(cap, np.int64), "ret": np.zeros(cap, np.float32),
                  "disc": np.zeros(cap, np.float32)}
        self.prio = np.zeros(cap)

    def add(self, items):
        k = len(items["act"])
        idx = (self.i + np.arange(k)) % self.cap
        for f in self.FIELDS:
            v = items[f]
            self.d[f][idx] = np.round(np.asarray(v, np.float32) * 255) if f.endswith("grid") else v
        self.prio[idx] = self.prio[:self.n].max() if self.n else 1.0
        self.i = (self.i + k) % self.cap
        self.n = min(self.cap, self.n + k)

    def sample(self, batch, alpha, beta, rng):
        p = self.prio[:self.n] ** alpha
        p /= p.sum()
        idx = rng.choice(self.n, batch, p=p)
        w = (self.n * p[idx]) ** -beta
        out = {f: torch.as_tensor(self.d[f][idx]) for f in self.FIELDS}
        for f in ("grid", "next_grid"):
            out[f] = out[f].float() / 255
        return idx, torch.as_tensor(w / w.max(), dtype=torch.float32), out


def nstep_transitions(b, boot_obs, n, gamma):
    """Turn a rollout into (s, a, R, discount, s') with s' the same player's decision n later."""
    rows = {f: [] for f in Replay.FIELDS}
    order = np.lexsort((np.arange(len(b["tid"])), b["tid"]))
    runs = np.split(order, np.nonzero(np.diff(b["tid"][order]))[0] + 1)
    for seq in runs:
        if not len(seq):
            continue
        L = len(seq)
        for j in range(L):
            R, d, m, term = 0.0, 1.0, 0, False
            while m < n and j + m < L:
                i = seq[j + m]
                R += d * b["rew"][i]
                d *= gamma
                m += 1
                if b["done"][i]:
                    term = True
                    break
            s = seq[j]
            if term:
                nxt = (b["grid"][s], b["focus"][s], b["mask"][s])  # unused: discount is 0
                d = 0.0
            elif j + m < L:
                k = seq[j + m]
                nxt = (b["grid"][k], b["focus"][k], b["mask"][k])
            else:
                o = boot_obs[b["tid"][s]]
                nxt = (o["grid"], o["focus"], o["mask"])
            for f, v in zip(Replay.FIELDS, (b["grid"][s], b["focus"][s], b["mask"][s], b["act"][s], R, d, *nxt)):
                rows[f].append(v)
    return rows


class DQN(Algo):
    lr = 1e-4

    def __init__(self, args, net, runner):
        super().__init__(args, net, runner)
        env = runner.envs[0]
        self.target = copy.deepcopy(net)
        self.replay = Replay(args.buffer, (env.observe(0)["grid"].shape), env.n_actions)
        self.rng = np.random.default_rng(args.seed)
        self.grad_steps = 0

    def eps(self):
        a = self.args
        return max(a.eps_end, 1 - (1 - a.eps_end) * self.progress / a.eps_frac)

    def policy(self, runner, env_ids, obs):
        logits, _ = self.net(*batch_obs(obs))
        act = logits.argmax(1).numpy()
        eps = self.eps()
        for i, o in enumerate(obs):
            if self.rng.random() < eps:
                act[i] = self.rng.choice(np.nonzero(o["mask"])[0])
        return {"act": act}

    def update(self):
        a = self.args
        b, boot_obs = self.runner.rollout(self.policy, a.rollout)
        n = len(b["tid"])
        self.replay.add(nstep_transitions(b, boot_obs, a.nstep, a.gamma))
        stats = {"eps": self.eps(), "buffer": self.replay.n}
        if self.replay.n < 4 * a.minibatch:
            return n, stats  # let the buffer fill a little first
        beta = 0.4 + 0.6 * min(1.0, self.progress)
        rows = []
        for _ in range(math.ceil(a.replay_ratio * n / a.minibatch)):
            idx, w, t = self.replay.sample(a.minibatch, a.per_alpha, beta, self.rng)
            logits, val = self.net(t["grid"], t["focus"], t["mask"])
            q = q_values(logits, val, t["mask"]).gather(1, t["act"][:, None]).squeeze(1)
            with torch.no_grad():
                nl, nv = self.net(t["next_grid"], t["next_focus"], t["next_mask"])
                best = q_values(nl, nv, t["next_mask"]).argmax(1, keepdim=True)
                tl, tv = self.target(t["next_grid"], t["next_focus"], t["next_mask"])
                tq = q_values(tl, tv, t["next_mask"]).gather(1, best).squeeze(1)
                # s' observed outside that player's turn has no legal actions: use V(s')
                nxt = torch.where(t["next_mask"].any(1), tq, tv)
                y = t["ret"] + t["disc"] * nxt
            loss = (w * F.smooth_l1_loss(q, y, reduction="none")).mean()
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10)
            self.opt.step()
            self.replay.prio[idx] = (q - y).abs().detach().numpy() + 1e-3
            self.grad_steps += 1
            if self.grad_steps % a.target_every == 0:
                self.target.load_state_dict(self.net.state_dict())
            rows.append((loss.item(), q.mean().item()))
        stats.update(td=float(np.mean([r[0] for r in rows])), q=float(np.mean([r[1] for r in rows])))
        return n, stats

    def describe(self):
        a = self.args
        return (f"DQN (double, dueling, {a.nstep}-step, prioritized alpha={a.per_alpha}): lr={a.lr}, "
                f"gamma={a.gamma}, buffer {a.buffer}, replay ratio {a.replay_ratio}, minibatch {a.minibatch}, "
                f"target every {a.target_every} steps, epsilon 1 -> {a.eps_end} over {a.eps_frac:.0%} of training")

    def on_phase_change(self):
        self.replay.n = self.replay.i = 0  # old transitions were against another opponent

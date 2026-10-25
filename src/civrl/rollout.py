"""Experience collection shared by the learning algorithms (see civrl/algos/).

Each env decision belongs to one player. Players are controlled by the learner, the
scripted AI, or a fixed opponent network (a periodic snapshot of the learner, or a
checkpoint given as --opponent runs/<name>/last.pt). Learner decisions are stored per
(env, player) trajectory; rewards a player receives while others act are credited to
that player's most recent decision.

How the learner picks its actions is a `policy(runner, env_ids, obs) -> dict` hook: the
dict holds "act" (one action per env) plus any per-decision values the algorithm wants
stored with the transition (e.g. "logp" and "val" for PPO); all of them end up in the batch.
"""
from __future__ import annotations

import numpy as np
import torch

from .scenarios import make_env
from .scripted import scripted_action


def batch_obs(obs, device="cpu"):
    return (torch.as_tensor(np.stack([o["grid"] for o in obs]), device=device),
            torch.as_tensor(np.array([o["focus"] for o in obs]), device=device),
            torch.as_tensor(np.stack([o["mask"] for o in obs]), device=device))


@torch.no_grad()
def sample_actions(net, obs):
    logits, _ = net(*batch_obs(obs, net.device))
    return torch.distributions.Categorical(logits=logits).sample().cpu().numpy()


class Runner:
    def __init__(self, cfg, n_envs, learner_side, opponent, seed, opp_net=None, group_size=1):
        self.cfg = cfg
        self.envs = [make_env(cfg) for _ in range(n_envs)]
        self.n_players = self.envs[0].n_players
        self.learner_side = learner_side
        self.opponent = opponent
        self.opp_net = opp_net
        self.group_size = group_size  # consecutive episodes that share seed and seating (GRPO)
        self.rng = np.random.default_rng(seed)
        self.ctrl = [None] * n_envs
        self.pending = np.zeros((n_envs, self.n_players), dtype=np.float32)
        self.ep_ret = np.zeros((n_envs, self.n_players))  # undiscounted episode return per player
        self.open = [dict() for _ in range(n_envs)]  # player -> index of open transition
        self.traj = [dict() for _ in range(n_envs)]  # player -> trajectory id
        self.tid_group = {}  # trajectory id -> group key (episode group, player)
        self.next_tid = 0
        self.next_gid = 0
        self.queue = []  # episodes still to start: (seed, ctrl, group id)
        self.idle = [False] * n_envs
        self.stopping = False
        self.ep_stats = []
        self.restart()

    # ---------------------------------------------------------------- episodes
    def assign(self):
        """Who controls each player in a new episode."""
        if self.n_players == 1:
            return ("learner",)
        side = self.learner_side
        if side == "both":
            side = ["attacker", "defender"][self.rng.integers(2)]
        me = 0 if side == "attacker" else 1
        opp = self.opponent
        if opp == "mixed":
            opp = "scripted" if self.rng.random() < 0.5 else "self"
        ctrl = ["scripted"] * self.n_players
        ctrl[me] = "learner"
        for q in range(self.n_players):
            if q != me:
                ctrl[q] = {"scripted": "scripted", "self": "learner", "frozen": "frozen",
                           "checkpoint": "checkpoint"}[opp]
        if opp == "frozen" and self.opp_net is None:
            ctrl[1 - me] = "scripted"
        return tuple(ctrl)

    def restart(self):
        """Abandon the episodes in progress and start new ones (e.g. after the sides change)."""
        self.queue = []
        for e in range(len(self.envs)):
            self._reset(e)

    def _reset(self, e):
        if not self.queue:
            self.queue = [(int(self.rng.integers(2**31)), self.assign(), self.next_gid)] * self.group_size
            self.next_gid += 1
        seed, ctrl, gid = self.queue.pop()
        self.envs[e].reset(seed)
        self.ctrl[e] = ctrl
        self.idle[e] = False
        self.pending[e] = 0
        self.ep_ret[e] = 0
        self.open[e] = {}
        self.traj[e] = {}
        for q, c in enumerate(ctrl):
            if c == "learner":
                self.traj[e][q] = self.next_tid
                self.tid_group[self.next_tid] = gid * self.n_players + q
                self.next_tid += 1

    def _finish(self, e, buf):
        env = self.envs[e]
        for q, idx in self.open[e].items():
            buf["rew"][idx] += self.pending[e, q]
            buf["done"][idx] = True
        self.open[e] = {}
        self.ep_stats.append({"ctrl": self.ctrl[e], "winner": env.winner(), "turns": env.game.turn,
                              "returns": self.ep_ret[e].tolist()})
        if self.stopping and not self.queue:
            self.idle[e] = True
        else:
            self._reset(e)

    def _apply(self, e, action, buf):
        r, done = self.envs[e].step(action)
        self.pending[e] += r
        self.ep_ret[e] += r
        if done:
            self._finish(e, buf)

    def _advance_scripted(self, e, buf):
        env = self.envs[e]
        while not self.idle[e] and self.ctrl[e][env.to_act()] == "scripted":
            self._apply(e, scripted_action(env, env.to_act()), buf)

    # ---------------------------------------------------------------- rollout
    @torch.no_grad()
    def rollout(self, policy, n_steps, complete=False):
        """Collect learner decisions.

        complete=False: n_steps rounds of one decision per env. Episodes continue into the next
            rollout; the last open decision of each trajectory is returned in `boot_obs`
            (trajectory id -> current observation of that player) for bootstrapping.
        complete=True: whole episodes only. New episodes start until about n_steps * n_envs
            learner decisions are collected (and until the current episode group is complete),
            then the running episodes are played to the end.
        Returns (batch dict of arrays, boot_obs).
        """
        E = len(self.envs)
        buf = {k: [] for k in ("grid", "focus", "mask", "rew", "done", "tid", "gkey")}
        quota = n_steps * E
        if complete:
            self.stopping = False
            for e in range(E):
                if self.idle[e]:
                    self._reset(e)
        it = 0
        while complete or it < n_steps:
            it += 1
            if complete and len(buf["tid"]) >= quota:
                self.stopping = True
            for e in range(E):
                self._advance_scripted(e, buf)
            live = [e for e in range(E) if not self.idle[e]]
            if not live:
                break
            frozen = [e for e in live if self.ctrl[e][self.envs[e].to_act()] in ("frozen", "checkpoint")]
            learn = [e for e in live if e not in frozen]
            if frozen:
                acts = sample_actions(self.opp_net, [self.envs[e].observe(self.envs[e].to_act()) for e in frozen])
                for e, a in zip(frozen, acts):
                    self._apply(e, int(a), buf)
            if not learn:
                continue
            obs = [self.envs[e].observe(self.envs[e].to_act()) for e in learn]
            out = policy(self, learn, obs)
            for i, e in enumerate(learn):
                p = self.envs[e].to_act()
                # close previous transition of this player, open a new one
                if p in self.open[e]:
                    buf["rew"][self.open[e][p]] += self.pending[e, p]
                self.pending[e, p] = 0
                self.open[e][p] = len(buf["tid"])
                for k in ("grid", "focus", "mask"):
                    buf[k].append(obs[i][k])
                for k, v in out.items():
                    buf.setdefault(k, []).append(v[i])
                buf["rew"].append(0.0)
                buf["done"].append(False)
                buf["tid"].append(self.traj[e][p])
                buf["gkey"].append(self.tid_group[self.traj[e][p]])
                self._apply(e, int(out["act"][i]), buf)
        boot_obs = {}
        for e in range(E):
            for q, idx in self.open[e].items():
                buf["rew"][idx] += self.pending[e, q]
                self.pending[e, q] = 0
                boot_obs[self.traj[e][q]] = self.envs[e].observe(q)
            self.open[e] = {}  # transitions are now owned by this batch
        b = {k: np.array(v) for k, v in buf.items()}
        b["grid"] = b["grid"].astype(np.float16) if len(b["grid"]) else b["grid"]
        b["rew"] = b["rew"].astype(np.float32)
        return b, boot_obs


def compute_gae(b, boot, gamma, lam):
    """GAE(lambda) per trajectory; `boot` maps trajectory id -> value of its unfinished end.
    With lam=1 and complete episodes the advantage is the Monte Carlo return minus V(s)."""
    n = len(b["rew"])
    adv = np.zeros(n, dtype=np.float32)
    order = np.lexsort((np.arange(n), b["tid"]))  # group by trajectory, keep time order
    last_adv, next_val, cur_tid = 0.0, 0.0, None
    for i in order[::-1]:
        tid = b["tid"][i]
        if tid != cur_tid:
            cur_tid = tid
            next_val = boot.get(tid, 0.0)
            last_adv = 0.0
        if b["done"][i]:
            next_val, last_adv = 0.0, 0.0
        delta = b["rew"][i] + gamma * next_val - b["val"][i]
        last_adv = delta + gamma * lam * last_adv
        adv[i] = last_adv
        next_val = b["val"][i]
    return adv, adv + b["val"]

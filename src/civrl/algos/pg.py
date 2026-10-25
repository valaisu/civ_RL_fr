"""Policy-gradient family: REINFORCE, A2C, PPO and GRPO.

All four push up the log-probability of each action in proportion to an advantage A
(how much better than expected the action turned out):  loss = -mean(A * log pi(a|s)).
They differ in how A is estimated and in how hard each batch is used.

  REINFORCE  Whole episodes. A = discounted return-to-go G_t - V(s_t); the value network is
             only a baseline (it lowers the variance, it does not change the expected
             gradient).
  A2C        Fixed-length rollouts, episodes continue across batches. A from GAE(lambda):
             a mix of one-step TD errors and longer returns, bootstrapped with V at the cut.
  REINFORCE and A2C use every decision once: one pass over the batch, one gradient step per
             minibatch (default 256, lr 7e-4), no clipping.
  PPO        A2C's advantages, but each batch is reused for --epochs passes in minibatches.
             The probability ratio new/old policy is clipped to [1-clip, 1+clip] so the
             policy cannot move far from the one that collected the data.
  GRPO       No value network. Episodes are played in groups of --group-size that start from
             the same state (same map, armies, turn limit and seating). Every decision of an
             episode gets A = (episode return - group mean) / group std. PPO's clipped update.
             The KL penalty to a reference model used for LLMs is left out (we train from scratch).

Advantages are normalized to mean 0, std 1 over the batch for REINFORCE, A2C and PPO.
The entropy bonus (--ent) applies to all four.
"""
import numpy as np
import torch

from ..rollout import batch_obs, compute_gae
from .base import Algo


def sample_policy(net):
    def policy(runner, env_ids, obs):
        logits, val = net(*batch_obs(obs))
        dist = torch.distributions.Categorical(logits=logits)
        act = dist.sample()
        return {"act": act.numpy(), "logp": dist.log_prob(act).numpy(), "val": val.numpy()}
    return policy


@torch.no_grad()
def boot_values(net, boot_obs):
    if not boot_obs:
        return {}
    tids = list(boot_obs)
    _, v = net(*batch_obs([boot_obs[t] for t in tids]))
    return dict(zip(tids, v.tolist()))


def normalize(adv):
    return (adv - adv.mean()) / (adv.std() + 1e-8)


def policy_update(algo, b, adv, ret, epochs, minibatch, clip):
    """Policy-gradient update on batch b. clip=None gives the plain policy gradient.
    ret=None: no value loss."""
    net, opt, args = algo.net, algo.opt, algo.args
    T = {k: torch.as_tensor(b[k]) for k in ("grid", "focus", "mask", "act", "logp")}
    T["grid"] = T["grid"].float()
    adv_t = torch.as_tensor(adv, dtype=torch.float32)
    ret_t = None if ret is None else torch.as_tensor(ret, dtype=torch.float32)
    n = len(adv_t)
    stats = []
    for _ in range(epochs):
        perm = torch.randperm(n)
        for s in range(0, n, minibatch):
            i = perm[s:s + minibatch]
            logits, v = net(T["grid"][i], T["focus"][i], T["mask"][i])
            dist = torch.distributions.Categorical(logits=logits)
            ratio = (dist.log_prob(T["act"][i]) - T["logp"][i]).exp()
            a = adv_t[i]
            surr = ratio * a
            if clip is not None:
                surr = torch.min(surr, ratio.clamp(1 - clip, 1 + clip) * a)
            pg = -surr.mean()
            # entropy over legal actions only (illegal ones have probability 0)
            p = dist.probs
            ent = -(p * torch.log(p.clamp_min(1e-12))).sum(1).mean()
            loss = pg - args.ent * ent
            row = {"pg": pg.item(), "ent": ent.item()}
            if ret_t is not None:
                vl = 0.5 * ((v - ret_t[i]) ** 2).mean()
                loss = loss + vl
                row["vl"] = vl.item()
            if clip is not None:
                row["clipfrac"] = ((ratio - 1).abs() > clip).float().mean().item()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()
            stats.append(row)
    return {k: float(np.mean([r[k] for r in stats])) for k in stats[0]}


class PPO(Algo):
    def update(self):
        a = self.args
        b, boot_obs = self.runner.rollout(sample_policy(self.net), a.rollout)
        adv, ret = compute_gae(b, boot_values(self.net, boot_obs), a.gamma, a.lam)
        return len(adv), policy_update(self, b, normalize(adv), ret, a.epochs, a.minibatch, a.clip)

    def describe(self):
        a = self.args
        return (f"PPO: lr={a.lr}, gamma={a.gamma}, lambda={a.lam}, clip={a.clip}, entropy={a.ent}, "
                f"epochs={a.epochs}, minibatch={a.minibatch}")


class A2C(Algo):
    lr = 7e-4
    minibatch = 256

    def update(self):
        a = self.args
        b, boot_obs = self.runner.rollout(sample_policy(self.net), a.rollout)
        adv, ret = compute_gae(b, boot_values(self.net, boot_obs), a.gamma, a.lam)
        return len(adv), policy_update(self, b, normalize(adv), ret, 1, a.minibatch, None)

    def describe(self):
        a = self.args
        return (f"A2C: lr={a.lr}, gamma={a.gamma}, lambda={a.lam}, entropy={a.ent}, "
                f"one pass, one gradient step per minibatch of {a.minibatch}")


class Reinforce(Algo):
    lr = 7e-4
    minibatch = 256

    def update(self):
        a = self.args
        b, _ = self.runner.rollout(sample_policy(self.net), a.rollout, complete=True)
        adv, ret = compute_gae(b, {}, a.gamma, 1.0)  # = return-to-go minus V(s): whole episodes, lambda 1
        return len(adv), policy_update(self, b, normalize(adv), ret, 1, a.minibatch, None)

    def describe(self):
        a = self.args
        return (f"REINFORCE with value baseline: lr={a.lr}, gamma={a.gamma}, entropy={a.ent}, "
                f"whole episodes, one pass, one gradient step per minibatch of {a.minibatch}")


class GRPO(Algo):
    def update(self):
        a = self.args
        b, _ = self.runner.rollout(sample_policy(self.net), a.rollout, complete=True)
        ret = {}  # trajectory -> (group key, undiscounted episode return)
        for tid, g, r in zip(b["tid"], b["gkey"], b["rew"]):
            ret[tid] = (g, ret.get(tid, (g, 0.0))[1] + r)
        groups = {}
        for g, r in ret.values():
            groups.setdefault(g, []).append(r)
        stat = {g: (np.mean(rs), np.std(rs)) for g, rs in groups.items()}
        ep_adv = {tid: (r - stat[g][0]) / (stat[g][1] + 1e-8) for tid, (g, r) in ret.items()}
        adv = np.array([ep_adv[t] for t in b["tid"]], dtype=np.float32)
        stats = policy_update(self, b, adv, None, a.epochs, a.minibatch, a.clip)
        stats["group_std"] = float(np.mean([s for _, s in stat.values()]))
        return len(adv), stats

    def describe(self):
        a = self.args
        return (f"GRPO: lr={a.lr}, group size {a.group_size}, clip={a.clip}, entropy={a.ent}, "
                f"epochs={a.epochs}, minibatch={a.minibatch}, no value network")

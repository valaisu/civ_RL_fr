"""Policy/value network.

A conv trunk turns the map into per-tile features. The 18 tile-target logits are
scored from (target tile feature, focus unit feature, global feature, offset
embedding), so the head is independent of where on the map the unit stands.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .game import N_OFFSETS


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return F.relu(x + self.c2(F.relu(self.c1(x))))


class Policy(nn.Module):
    def __init__(self, in_ch, targets, n_actions, ch=48, blocks=2):
        super().__init__()
        N = targets.shape[0]
        t = torch.as_tensor(targets, dtype=torch.long).clone()
        t[t < 0] = N  # off-map -> zero padding slot
        t = torch.cat([t, torch.full((1, t.shape[1]), N, dtype=torch.long)])  # focus == N (none)
        self.register_buffer("targets", t)
        self.N = N
        self.n_actions = n_actions
        self.stem = nn.Conv2d(in_ch, ch, 3, padding=1)
        self.blocks = nn.Sequential(*[ResBlock(ch) for _ in range(blocks)])
        self.off_emb = nn.Embedding(N_OFFSETS, 16)
        hid = 128
        self.act_head = nn.Sequential(nn.Linear(ch * 4 + 16, hid), nn.ReLU(), nn.Linear(hid, 1))
        self.fort_head = nn.Sequential(nn.Linear(ch * 3, hid), nn.ReLU(), nn.Linear(hid, n_actions - N_OFFSETS))
        self.value_head = nn.Sequential(nn.Linear(ch * 3, hid), nn.ReLU(), nn.Linear(hid, 1))

    @property
    def device(self):
        return self.targets.device

    def forward(self, grid, focus, mask=None):
        B = grid.shape[0]
        x = self.blocks(F.relu(self.stem(grid)))  # [B,C,H,W]
        C = x.shape[1]
        feat = x.flatten(2).transpose(1, 2)  # [B,N,C]
        feat = torch.cat([feat, feat.new_zeros(B, 1, C)], 1)  # [B,N+1,C]
        g = torch.cat([feat[:, :-1].mean(1), feat[:, :-1].amax(1)], 1)  # [B,2C]
        bi = torch.arange(B, device=grid.device)
        u = feat[bi, focus]  # [B,C]
        tidx = self.targets[focus]  # [B,18]
        tf = feat[bi[:, None], tidx]  # [B,18,C]
        k = tf.shape[1]
        h = torch.cat([tf, u[:, None].expand(-1, k, -1), g[:, None].expand(-1, k, -1),
                       self.off_emb.weight[None].expand(B, -1, -1)], 2)
        logits = torch.cat([self.act_head(h).squeeze(-1), self.fort_head(torch.cat([u, g], 1))], 1)
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)
        value = self.value_head(torch.cat([g, u], 1)).squeeze(-1)
        return logits, value

"""Hex geometry (pointy-top, odd-r offset storage) and random island generation.

Tiles are indexed i = row * W + col. Directions 0..5 = E, NE, NW, W, SW, SE.
"""
from __future__ import annotations

import numpy as np

AXIAL_DIRS = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]

# Base terrain
WATER, GRASS, PLAINS, DESERT, MOUNTAIN = range(5)
TERRAIN_NAMES = ["water", "grass", "plains", "desert", "mountain"]


def offset_to_axial(col, row):
    return col - (row - (row & 1)) // 2, row


def axial_to_offset(q, r):
    return q + (r - (r & 1)) // 2, r


def _ring_offsets(radius):
    """Axial offsets on the ring at distance `radius`, in a stable order."""
    if radius == 1:
        return list(AXIAL_DIRS)
    out = []
    q, r = AXIAL_DIRS[4][0] * radius, AXIAL_DIRS[4][1] * radius
    for d in range(6):
        for _ in range(radius):
            out.append((q, r))
            q, r = q + AXIAL_DIRS[d][0], r + AXIAL_DIRS[d][1]
    return out


# 18 action offsets: ring 1 (indices 0..5 == directions) then ring 2.
ACTION_OFFSETS = _ring_offsets(1) + _ring_offsets(2)


class HexGeometry:
    """Terrain-independent lookup tables for a W x H map."""

    def __init__(self, W: int, H: int):
        self.W, self.H, self.N = W, H, W * H
        cols, rows = np.meshgrid(np.arange(W), np.arange(H))
        self.col = cols.ravel()
        self.row = rows.ravel()
        self.q = self.col - (self.row - (self.row & 1)) // 2
        self.r = self.row.copy()

        def tile_at(q, r):
            c, rr = axial_to_offset(q, r)
            if 0 <= c < W and 0 <= rr < H:
                return rr * W + c
            return -1

        self.nbr = np.full((self.N, 6), -1, dtype=np.int64)
        self.targets = np.full((self.N, len(ACTION_OFFSETS)), -1, dtype=np.int64)
        for i in range(self.N):
            q, r = self.q[i], self.r[i]
            for d, (dq, dr) in enumerate(AXIAL_DIRS):
                self.nbr[i, d] = tile_at(q + dq, r + dr)
            for k, (dq, dr) in enumerate(ACTION_OFFSETS):
                self.targets[i, k] = tile_at(q + dq, r + dr)
        dq = self.q[:, None] - self.q[None, :]
        dr = self.r[:, None] - self.r[None, :]
        self.dist = ((np.abs(dq) + np.abs(dr) + np.abs(dq + dr)) // 2).astype(np.int64)
        self.parity = (self.row & 1).astype(np.float32)

    def dir_between(self, a, b):
        """Direction index from tile a to adjacent tile b, or -1."""
        hits = np.nonzero(self.nbr[a] == b)[0]
        return int(hits[0]) if len(hits) else -1

    def line(self, a, b):
        """Two hex lines (nudged both ways) from a to b, excluding endpoints."""
        n = int(self.dist[a, b])
        aq, ar = float(self.q[a]), float(self.r[a])
        bq, br = float(self.q[b]), float(self.r[b])
        lines = []
        for eps in (1e-6, -1e-6):
            tiles = []
            for s in range(1, n):
                t = s / n
                fq = aq + (bq - aq) * t + eps
                fr = ar + (br - ar) * t + eps
                fs = -fq - fr
                rq, rr, rs = round(fq), round(fr), round(fs)
                dq_, dr_, ds_ = abs(rq - fq), abs(rr - fr), abs(rs - fs)
                if dq_ > dr_ and dq_ > ds_:
                    rq = -rr - rs
                elif dr_ > ds_:
                    rr = -rq - rs
                c, row = axial_to_offset(int(rq), int(rr))
                tiles.append(row * self.W + c if 0 <= c < self.W and 0 <= row < self.H else -1)
            lines.append(tiles)
        return lines

    def sight_lines(self, max_sight=3):
        """Per viewer tile: (targets, lineA, lineB) for targets at distance 2..max_sight.
        Line entries index an extended array: N = always clear (padding), -1 = off-map."""
        if getattr(self, "_sight", None) is None:
            out = []
            for v in range(self.N):
                ts = np.nonzero((self.dist[v] >= 2) & (self.dist[v] <= max_sight))[0]
                la = np.full((len(ts), max_sight - 1), self.N, dtype=np.int64)
                lb = la.copy()
                for j, t in enumerate(ts):
                    l1, l2 = self.line(v, t)
                    la[j, :len(l1)] = l1
                    lb[j, :len(l2)] = l2
                out.append((ts, la, lb))
            self._sight = out
        return self._sight


class GameMap:
    def __init__(self, geo: HexGeometry):
        self.geo = geo
        N = geo.N
        self.terrain = np.zeros(N, dtype=np.int64)  # WATER by default
        self.hills = np.zeros(N, dtype=bool)
        self.woods = np.zeros(N, dtype=bool)
        self.marsh = np.zeros(N, dtype=bool)
        self.river = np.zeros((N, 6), dtype=bool)  # river on edge (tile, dir)
        self.resource = np.zeros(N, dtype=np.int64)  # 0 none (used later)
        self._frozen = None

    def freeze(self):
        """Cache derived arrays once terrain is final (called by Game)."""
        self._frozen = (self.terrain != WATER, (self.terrain != WATER) & (self.terrain != MOUNTAIN))

    @property
    def land(self):
        return self._frozen[0] if self._frozen else self.terrain != WATER

    @property
    def passable(self):
        return self._frozen[1] if self._frozen else (self.terrain != WATER) & (self.terrain != MOUNTAIN)

    def move_cost(self, a, b):
        """Movement cost of stepping from a into adjacent b (Civ6 + house river rule)."""
        cost = 1 + int(self.hills[b]) + int(self.woods[b]) + int(self.marsh[b])
        d = self.geo.dir_between(a, b)
        if self.river[a, d]:
            cost += 2
        return cost

    def components(self, mask):
        """Connected components (label array, -1 outside mask)."""
        geo = self.geo
        lab = np.full(geo.N, -1, dtype=np.int64)
        cur = 0
        for s in range(geo.N):
            if not mask[s] or lab[s] >= 0:
                continue
            stack = [s]
            lab[s] = cur
            while stack:
                t = stack.pop()
                for n in geo.nbr[t]:
                    if n >= 0 and mask[n] and lab[n] < 0:
                        lab[n] = cur
                        stack.append(n)
            cur += 1
        return lab


def _smooth_noise(geo, rng, iters=3):
    v = rng.random(geo.N)
    nbr = geo.nbr
    for _ in range(iters):
        acc = v.copy()
        cnt = np.ones(geo.N)
        for d in range(6):
            n = nbr[:, d]
            ok = n >= 0
            acc[ok] += v[n[ok]]
            cnt[ok] += 1
        v = acc / cnt
    v = (v - v.min()) / (v.max() - v.min() + 1e-9)
    return v


def generate_island(geo: HexGeometry, rng: np.random.Generator, land_frac=0.55, region=None) -> GameMap:
    """Random island. `region` = (col0, row0, w, h) limits it to a sub-rectangle (rest is water)."""
    m = GameMap(geo)
    c0, r0, W, H = region if region is not None else (0, 0, geo.W, geo.H)
    # Island shape: radial falloff + noise, region border always water.
    cx, cy = c0 + (W - 1) / 2, r0 + (H - 1) / 2
    x = geo.col + 0.5 * geo.parity
    y = geo.row * 0.866
    rad = np.sqrt(((x - cx) / (W / 2)) ** 2 + ((y - cy * 0.866) / (H * 0.866 / 2)) ** 2)
    score = (1 - rad) + 0.6 * (_smooth_noise(geo, rng) - 0.5)
    inside = (geo.col >= c0) & (geo.col < c0 + W) & (geo.row >= r0) & (geo.row < r0 + H)
    border = ~inside | (geo.col == c0) | (geo.row == r0) | (geo.col == c0 + W - 1) | (geo.row == r0 + H - 1)
    score[border] = -np.inf
    thresh = np.quantile(score[~border], 1 - land_frac)
    land = score >= thresh
    lab = m.components(land)
    if lab.max() >= 0:
        biggest = np.bincount(lab[lab >= 0]).argmax()
        land = lab == biggest
    land_idx = np.nonzero(land)[0]

    # Base terrain
    temp = _smooth_noise(geo, rng)
    m.terrain[land_idx] = GRASS
    m.terrain[land_idx[temp[land_idx] > 0.55]] = PLAINS
    m.terrain[land_idx[temp[land_idx] > 0.85]] = DESERT

    # Mountains: few, never disconnecting passable land.
    n_mtn = max(1, int(0.06 * len(land_idx)))
    for t in rng.permutation(land_idx)[: n_mtn * 2]:
        if n_mtn == 0:
            break
        m.terrain[t] = MOUNTAIN
        plab = m.components(m.passable)
        if len(np.unique(plab[plab >= 0])) > 1:
            m.terrain[t] = GRASS
        else:
            n_mtn -= 1

    passable = np.nonzero(m.passable)[0]
    elev = _smooth_noise(geo, rng)
    m.hills[passable[elev[passable] > 0.68]] = True
    wood_noise = _smooth_noise(geo, rng)
    m.woods[passable[(wood_noise[passable] > 0.62) & (m.terrain[passable] != DESERT)]] = True
    flat = passable[~m.hills[passable] & ~m.woods[passable] & (m.terrain[passable] == GRASS)]
    if len(flat):
        m.marsh[rng.choice(flat, size=max(0, len(flat) // 20), replace=False)] = True

    for _ in range(rng.integers(1, 3)):
        _add_river(m, rng)
    return m


def _common(geo, a, b):
    """The two tiles adjacent to both a and b (may be -1)."""
    d = geo.dir_between(a, b)
    return geo.nbr[a, (d + 1) % 6], geo.nbr[a, (d - 1) % 6]


def _add_river(m: GameMap, rng, max_len=10):
    """Walk along hex edges from a high tile towards the sea."""
    geo = m.geo
    land = np.nonzero(m.passable)[0]
    if len(land) == 0:
        return
    water = np.nonzero(~m.land)[0]
    coast_dist = geo.dist[:, water].min(axis=1)
    start_pool = land[coast_dist[land] >= 2]
    if len(start_pool) == 0:
        return
    a = int(rng.choice(start_pool))
    nb = [n for n in geo.nbr[a] if n >= 0 and m.land[n]]
    if not nb:
        return
    b = int(rng.choice(nb))
    c = _common(geo, a, b)[rng.integers(2)]
    edges = [(a, b)]
    for _ in range(max_len):
        if c < 0 or not m.land[c]:
            break
        options = [(a, c, b), (b, c, a)]  # (x, c, other): new edge (x,c)
        options.sort(key=lambda o: coast_dist[o[0]] + coast_dist[o[1]])
        x, cc, other = options[0] if rng.random() < 0.75 else options[1]
        edges.append((x, cc))
        nxt = [t for t in _common(geo, x, cc) if t != other]
        a, b, c = x, cc, (nxt[0] if nxt else -1)
    for a, b in edges:
        if m.land[a] and m.land[b]:
            d = geo.dir_between(a, b)
            m.river[a, d] = True
            m.river[b, (d + 3) % 6] = True

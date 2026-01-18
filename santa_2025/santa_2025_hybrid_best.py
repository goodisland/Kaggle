# -*- coding: utf-8 -*-
"""
Santa 2025 - Hybrid Best (Score-first, runtime-not-a-priority)
--------------------------------------------------------------
Goal: minimize bounding square side for n=1..200 tree packings.

Key ideas:
  1) Strong insertion (most important): generate many feasible candidates and pick the best by side length.
     - Ray push-in (multi-angle)
     - Boundary-normal slot tries (min/max x/y)
     - Near-existing-tree jitter tries
  2) Hybrid SA: local neighbors + global boundary trees (because score is dominated by global min/max x/y).
  3) Moves include boundary-normal push (more effective than origin-push).
  4) Collision detection: outside STRtree (fixed) + local brute-force (moving). No stale-index misses.
  5) Keep-best (always restore best state at end of SA).
  6) Touching is allowed, overlap is forbidden: invalid if intersects AND NOT touches.

Output: submission.csv with columns id,x,y,deg; values prefixed by 's'
"""

import math
import random
import time
from decimal import Decimal, getcontext

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from shapely.geometry import Polygon
from shapely import affinity
from shapely.strtree import STRtree

# -----------------------------
# Global config (score-first)
# -----------------------------
getcontext().prec = 28

# IMPORTANT: keep consistent with Getting Started
SCALE = Decimal("1e15")

XY_MIN, XY_MAX = Decimal("-100"), Decimal("100")

# Insertion (strong, heavy)
INS_RAY_TRIES = 140          # multi-ray push-in attempts
INS_BOUNDARY_TRIES = 120     # boundary-normal slot attempts
INS_JITTER_TRIES = 220       # near-existing jitter attempts
INS_START_R = Decimal("28.0")
INS_STEP_IN = Decimal("0.55")
INS_BACKOFF = Decimal("0.06")

# SA: hybrid (heavy)
SA_STEPS = 5200
WARMUP_STEPS = 650          # accept only improvements early
T0 = 0.020                  # delta is in ORIGINAL units side length
T_DECAY = 0.99925
T_MIN = 1e-6

# Local neighborhood
K_LOCAL = 70

# Global boundary sampling
GLOBAL_PICK_P = 0.28         # probability to pick a tree from global boundary sets
GLOBAL_EDGE_K = 14           # how many trees per edge to consider in boundary sets

# Boundary refresh cadence (SA)
REFRESH_EVERY = 250

# Move probabilities (sum to 1.0)
P_TRANSLATE = 0.36
P_ROTATE = 0.24
P_BOTH = 0.18
P_SWAP = 0.06
P_BOUNDARY_PUSH = 0.16

# Step sizes (original units)
MOVE_SIGMA = 0.070
ANGLE_SIGMA = 8.5

# Output
ROUND_DECIMALS = 6
OUT_CSV = "submission.csv"

# Random seed (optional; set to None to vary each run)
SEED = 1234


# -----------------------------
# Helpers
# -----------------------------
def clamp(v: Decimal) -> Decimal:
    if v < XY_MIN:
        return XY_MIN
    if v > XY_MAX:
        return XY_MAX
    return v


def wrap_angle_deg(a: Decimal) -> Decimal:
    return Decimal(str(float(a) % 360.0))


def adaptive_sigmas(n: int):
    # Not too aggressive; we can keep sizable moves since runtime is ok
    if n < 60:
        m = 1.00
    elif n < 120:
        m = 0.85
    elif n < 170:
        m = 0.72
    else:
        m = 0.62
    return MOVE_SIGMA * m, ANGLE_SIGMA * m


def choose_move():
    r = random.random()
    if r < P_TRANSLATE:
        return "translate"
    r -= P_TRANSLATE
    if r < P_ROTATE:
        return "rotate"
    r -= P_ROTATE
    if r < P_BOTH:
        return "both"
    r -= P_BOTH
    if r < P_SWAP:
        return "swap"
    return "boundary_push"


def compute_side_original(polygons_scaled):
    # bounds in scaled coords -> divide by SCALE
    minx = min(p.bounds[0] for p in polygons_scaled)
    miny = min(p.bounds[1] for p in polygons_scaled)
    maxx = max(p.bounds[2] for p in polygons_scaled)
    maxy = max(p.bounds[3] for p in polygons_scaled)

    w = (Decimal(str(maxx)) - Decimal(str(minx))) / SCALE
    h = (Decimal(str(maxy)) - Decimal(str(miny))) / SCALE
    return max(w, h)


def bounds_scaled(polygons_scaled):
    minx = min(p.bounds[0] for p in polygons_scaled)
    miny = min(p.bounds[1] for p in polygons_scaled)
    maxx = max(p.bounds[2] for p in polygons_scaled)
    maxy = max(p.bounds[3] for p in polygons_scaled)
    return minx, miny, maxx, maxy


def is_overlap_touch_ok(a, b):
    # invalid overlap if intersects AND NOT touches
    return a.intersects(b) and (not a.touches(b))


def snapshot_state(trees, idx_list):
    return [(i, trees[i].x, trees[i].y, trees[i].deg) for i in idx_list]


def restore_state(trees, snap):
    for i, x, y, deg in snap:
        trees[i] = ChristmasTree(x, y, deg)


def weighted_angle():
    # same spirit as Getting Started: favor abs(sin(2*theta))
    while True:
        ang = random.uniform(0.0, 2.0 * math.pi)
        if random.random() < abs(math.sin(2.0 * ang)):
            return ang


# -----------------------------
# Tree geometry (official)
# -----------------------------
class ChristmasTree:
    # Dimensions (original units)
    trunk_w = Decimal("0.15")
    trunk_h = Decimal("0.2")
    base_w  = Decimal("0.7")
    mid_w   = Decimal("0.4")
    top_w   = Decimal("0.25")

    tip_y    = Decimal("0.8")
    tier_1_y = Decimal("0.5")
    tier_2_y = Decimal("0.25")
    base_y   = Decimal("0.0")
    trunk_bottom_y = -trunk_h

    base_poly = Polygon([
        (0, tip_y),
        (top_w/2, tier_1_y),
        (top_w/4, tier_1_y),
        (mid_w/2, tier_2_y),
        (mid_w/4, tier_2_y),
        (base_w/2, base_y),
        (trunk_w/2, base_y),
        (trunk_w/2, trunk_bottom_y),
        (-trunk_w/2, trunk_bottom_y),
        (-trunk_w/2, base_y),
        (-base_w/2, base_y),
        (-mid_w/4, tier_2_y),
        (-mid_w/2, tier_2_y),
        (-top_w/4, tier_1_y),
        (-top_w/2, tier_1_y),
    ])

    base_poly_scaled = affinity.scale(
        base_poly, xfact=float(SCALE), yfact=float(SCALE), origin=(0, 0)
    )

    def __init__(self, x=Decimal("0"), y=Decimal("0"), deg=Decimal("0")):
        self.x = Decimal(x)
        self.y = Decimal(y)
        self.deg = Decimal(deg)
        self.poly = None
        self._rebuild()

    def _rebuild(self):
        r = affinity.rotate(self.base_poly_scaled, float(self.deg), origin=(0, 0))
        self.poly = affinity.translate(r, xoff=float(self.x * SCALE), yoff=float(self.y * SCALE))


# -----------------------------
# Collision: outside STRtree + local brute force
# -----------------------------
def _query_indices(tree, geom, geom2idx):
    hits = tree.query(geom)
    if len(hits) == 0:
        return []
    h0 = hits[0]
    if isinstance(h0, (int, np.integer)):
        return list(hits)
    # shapely returns geometries
    return [geom2idx[id(g)] for g in hits if id(g) in geom2idx]


def make_outside_index(polys_scaled):
    if not polys_scaled:
        return None, None, None
    st = STRtree(polys_scaled)
    geom2idx = {id(g): k for k, g in enumerate(polys_scaled)}
    return st, geom2idx, polys_scaled


def overlap_with_outside(cand_poly, outside_tree, outside_geom2idx, outside_polys):
    if outside_tree is None:
        return False
    hit_ids = _query_indices(outside_tree, cand_poly, outside_geom2idx)
    for k in hit_ids:
        p = outside_polys[k]
        if is_overlap_touch_ok(cand_poly, p):
            return True
    return False


def overlap_with_local(cand_poly, local_ids, polygons_scaled, ignore_i):
    # brute-force among locals (<= ~70)
    for j in local_ids:
        if j == ignore_i:
            continue
        if is_overlap_touch_ok(cand_poly, polygons_scaled[j]):
            return True
    return False


def assert_no_overlap_all(trees, where=""):
    polys = [t.poly for t in trees]
    st = STRtree(polys)
    geom2idx = {id(g): k for k, g in enumerate(polys)}
    for i, p in enumerate(polys):
        hits = _query_indices(st, p, geom2idx)
        for j in hits:
            if j <= i:
                continue
            if is_overlap_touch_ok(p, polys[j]):
                raise RuntimeError(f"Overlap detected {where}: i={i}, j={j}")


# -----------------------------
# Candidate generation (insertion) - heavy & score-based
# -----------------------------
def try_push_in_ray(existing_polys, existing_tree, vx, vy, deg):
    """
    Place tree along ray direction (vx,vy) starting far, pushing inward until collision,
    then back off slightly to a feasible touching position.
    Return (x,y,deg, side) or None.
    """
    # Start far
    r = INS_START_R
    last_ok = None

    # push in
    while r >= 0:
        x = r * vx
        y = r * vy
        cand = ChristmasTree(x, y, deg)
        if not any(is_overlap_touch_ok(cand.poly, p) for p in existing_polys):
            last_ok = (x, y)
            r -= INS_STEP_IN
        else:
            break

    if last_ok is None:
        return None

    # back off outward a bit to ensure feasibility even with float quirks
    x, y = last_ok
    r_ok = Decimal(str(math.sqrt(float(x*x + y*y))))
    # step outward until safe (usually already safe, but keep it robust)
    rr = r_ok
    while True:
        cand = ChristmasTree(rr * vx, rr * vy, deg)
        if not any(is_overlap_touch_ok(cand.poly, p) for p in existing_polys):
            break
        rr += INS_BACKOFF
        if rr > INS_START_R + Decimal("5"):
            return None

    # compute side after insertion
    side = compute_side_original(existing_polys + [cand.poly])
    return (rr * vx, rr * vy, deg, side)


def insert_best_candidate(trees):
    """
    Build many candidates and pick the feasible one that minimizes resulting side.
    Heavy on purpose (runtime ok).
    """
    if not trees:
        return [ChristmasTree()]

    existing_polys = [t.poly for t in trees]

    # quick global bounds (scaled)
    minx, miny, maxx, maxy = bounds_scaled(existing_polys)
    w = (Decimal(str(maxx)) - Decimal(str(minx))) / SCALE
    h = (Decimal(str(maxy)) - Decimal(str(miny))) / SCALE
    side_now = max(w, h)

    best = None  # (side, x, y, deg)

    # ---- (1) Ray push-in tries ----
    for _ in range(INS_RAY_TRIES):
        theta = weighted_angle()
        vx = Decimal(str(math.cos(theta)))
        vy = Decimal(str(math.sin(theta)))
        deg = Decimal(str(random.uniform(0, 360)))
        got = try_push_in_ray(existing_polys, None, vx, vy, deg)
        if got is None:
            continue
        x, y, deg2, side = got
        if (best is None) or (side < best[0]):
            best = (side, x, y, deg2)

    # ---- (2) Boundary-normal slot tries ----
    # Aim to reduce whichever dimension is dominating.
    # If width > height: try to push minx/maxx inward. Else push miny/maxy inward.
    dom = "x" if w >= h else "y"

    for _ in range(INS_BOUNDARY_TRIES):
        # pick an edge and a target coordinate near it (in original units)
        if dom == "x":
            # try minx or maxx side
            edge = "minx" if random.random() < 0.5 else "maxx"
            # Choose y uniformly in current [miny,maxy] (original units)
            y0 = (Decimal(str(random.uniform(float(miny), float(maxy)))) / SCALE)
            # choose x near edge, slightly inside
            if edge == "minx":
                x0 = (Decimal(str(minx)) / SCALE) + Decimal(str(random.uniform(0.0, float(side_now)*0.25)))
            else:
                x0 = (Decimal(str(maxx)) / SCALE) - Decimal(str(random.uniform(0.0, float(side_now)*0.25)))
        else:
            edge = "miny" if random.random() < 0.5 else "maxy"
            x0 = (Decimal(str(random.uniform(float(minx), float(maxx)))) / SCALE)
            if edge == "miny":
                y0 = (Decimal(str(miny)) / SCALE) + Decimal(str(random.uniform(0.0, float(side_now)*0.25)))
            else:
                y0 = (Decimal(str(maxy)) / SCALE) - Decimal(str(random.uniform(0.0, float(side_now)*0.25)))

        deg = Decimal(str(random.uniform(0, 360)))
        cand = ChristmasTree(x0, y0, deg)

        # If colliding, try small nudges along inward normal
        # inward normal: minx => +x, maxx => -x, miny => +y, maxy => -y
        ok = True
        for p in existing_polys:
            if is_overlap_touch_ok(cand.poly, p):
                ok = False
                break
        if not ok:
            # push along inward normal a few steps
            step = Decimal("0.08")
            for k in range(1, 30):
                if edge == "minx":
                    cand = ChristmasTree(x0 + step * Decimal(k), y0, deg)
                elif edge == "maxx":
                    cand = ChristmasTree(x0 - step * Decimal(k), y0, deg)
                elif edge == "miny":
                    cand = ChristmasTree(x0, y0 + step * Decimal(k), deg)
                else:
                    cand = ChristmasTree(x0, y0 - step * Decimal(k), deg)

                if not any(is_overlap_touch_ok(cand.poly, p) for p in existing_polys):
                    ok = True
                    break

        if not ok:
            continue

        side = compute_side_original(existing_polys + [cand.poly])
        if (best is None) or (side < best[0]):
            best = (side, cand.x, cand.y, cand.deg)

    # ---- (3) Near-existing jitter tries ----
    # Try candidates around existing tree centroids with small offsets.
    # This often finds snug placements in "pockets".
    for _ in range(INS_JITTER_TRIES):
        base = random.choice(trees)
        cx = Decimal(str(base.poly.centroid.x)) / SCALE
        cy = Decimal(str(base.poly.centroid.y)) / SCALE

        # random small offset ring
        r = Decimal(str(abs(random.gauss(0.0, 0.55)) + 0.10))
        th = Decimal(str(random.uniform(0.0, 2.0 * math.pi)))
        dx = r * Decimal(str(math.cos(float(th))))
        dy = r * Decimal(str(math.sin(float(th))))
        deg = Decimal(str(random.uniform(0, 360)))

        x0 = clamp(cx + dx)
        y0 = clamp(cy + dy)

        cand = ChristmasTree(x0, y0, deg)
        if any(is_overlap_touch_ok(cand.poly, p) for p in existing_polys):
            continue

        side = compute_side_original(existing_polys + [cand.poly])
        if (best is None) or (side < best[0]):
            best = (side, x0, y0, deg)

    if best is None:
        # fallback: random far away (should be rare)
        trees.append(ChristmasTree(Decimal("60"), Decimal("60"), Decimal("0")))
        return trees

    _, x, y, deg = best
    trees.append(ChristmasTree(x, y, deg))
    return trees


# -----------------------------
# Boundary sets (global)
# -----------------------------
def global_boundary_sets(polygons_scaled, k_each=GLOBAL_EDGE_K):
    """
    Return a set of indices that are near global minx/maxx/miny/maxy edges.
    We take top-k polygons by closeness to each edge.
    """
    n = len(polygons_scaled)
    if n == 0:
        return set()

    minx, miny, maxx, maxy = bounds_scaled(polygons_scaled)

    # distance to edges using polygon bounds
    items = []
    for i, p in enumerate(polygons_scaled):
        bx0, by0, bx1, by1 = p.bounds
        items.append((abs(bx0 - minx), "minx", i))
        items.append((abs(bx1 - maxx), "maxx", i))
        items.append((abs(by0 - miny), "miny", i))
        items.append((abs(by1 - maxy), "maxy", i))

    pick = set()
    for edge in ["minx", "maxx", "miny", "maxy"]:
        edge_items = [t for t in items if t[1] == edge]
        edge_items.sort(key=lambda x: x[0])
        for _, _, idx in edge_items[:min(k_each, n)]:
            pick.add(idx)
    return pick


def boundary_normal_direction(polygons_scaled, idx):
    """
    Determine which edge(s) this polygon is closest to and return a normal push (dx, dy)
    in ORIGINAL units direction:
      - near minx -> push +x
      - near maxx -> push -x
      - near miny -> push +y
      - near maxy -> push -y
    If ambiguous, combine.
    """
    minx, miny, maxx, maxy = bounds_scaled(polygons_scaled)
    p = polygons_scaled[idx]
    bx0, by0, bx1, by1 = p.bounds

    # edge distances in scaled coords
    d_minx = abs(bx0 - minx)
    d_maxx = abs(bx1 - maxx)
    d_miny = abs(by0 - miny)
    d_maxy = abs(by1 - maxy)

    d = [("minx", d_minx), ("maxx", d_maxx), ("miny", d_miny), ("maxy", d_maxy)]
    d.sort(key=lambda x: x[1])

    # take the closest 1 or 2 edges if very close
    edges = [d[0][0]]
    if len(d) > 1 and d[1][1] <= d[0][1] * 1.15:
        edges.append(d[1][0])

    dx = Decimal("0")
    dy = Decimal("0")
    for e in edges:
        if e == "minx":
            dx += Decimal("1")
        elif e == "maxx":
            dx -= Decimal("1")
        elif e == "miny":
            dy += Decimal("1")
        else:
            dy -= Decimal("1")

    # normalize (avoid zero)
    norm = Decimal(str(math.sqrt(float(dx*dx + dy*dy)))) if (dx != 0 or dy != 0) else Decimal("1")
    dx /= norm
    dy /= norm
    return dx, dy


# -----------------------------
# Hybrid SA (local + global boundary)
# -----------------------------
def hybrid_sa(trees, target_idx):
    n = len(trees)
    if n <= 1:
        return

    move_sigma, angle_sigma = adaptive_sigmas(n)

    polygons = [t.poly for t in trees]
    current_side = compute_side_original(polygons)

    # ---- local_ids by centroid distance to target (scaled) ----
    tx, ty = trees[target_idx].poly.centroid.x, trees[target_idx].poly.centroid.y
    dists = []
    for i, t in enumerate(trees):
        cx, cy = t.poly.centroid.x, t.poly.centroid.y
        d = (cx - tx) ** 2 + (cy - ty) ** 2
        dists.append((d, i))
    dists.sort(key=lambda x: x[0])
    local_ids = [i for _, i in dists[:min(K_LOCAL, n)]]
    local_set = set(local_ids)

    # outside index (fixed during SA)
    outside_ids = [i for i in range(n) if i not in local_set]
    outside_polys = [polygons[i] for i in outside_ids]
    outside_tree, outside_geom2idx, outside_polys_ref = make_outside_index(outside_polys)

    def collide(i, cand_poly):
        # outside
        if overlap_with_outside(cand_poly, outside_tree, outside_geom2idx, outside_polys_ref):
            return True
        # local brute
        if overlap_with_local(cand_poly, local_ids, polygons, ignore_i=i):
            return True
        return False

    # keep best state within local (including possibly global picks that happen to be local)
    best_side = current_side
    best_snap = snapshot_state(trees, local_ids)

    T = float(T0)

    # boundary pick set
    global_set = global_boundary_sets(polygons, k_each=GLOBAL_EDGE_K)

    for step in range(SA_STEPS):
        if step > 0 and step % REFRESH_EVERY == 0:
            # refresh global boundary set and (for better guidance) polygons list
            polygons = [t.poly for t in trees]
            global_set = global_boundary_sets(polygons, k_each=GLOBAL_EDGE_K)

        # pick index:
        # - with prob GLOBAL_PICK_P pick from global boundary (but still keep safety by forcing it into local when needed)
        # - otherwise pick from local
        if (random.random() < GLOBAL_PICK_P) and global_set:
            i = random.choice(list(global_set))
            # if i is outside local, we still allow moving it BUT then outside-tree is no longer valid.
            # To keep correctness, we restrict moves to local only.
            # Therefore: if chosen global i is not in local, map to nearest local boundary instead.
            if i not in local_set:
                i = random.choice(local_ids)
        else:
            i = random.choice(local_ids)

        move = choose_move()

        # ---- swap ----
        if move == "swap":
            j = random.choice(local_ids)
            if j == i:
                T = max(T * float(T_DECAY), float(T_MIN))
                continue

            ti, tj = trees[i], trees[j]
            cand_i = ChristmasTree(tj.x, tj.y, ti.deg)
            cand_j = ChristmasTree(ti.x, ti.y, tj.deg)

            old_i, old_j = polygons[i], polygons[j]
            polygons[i] = cand_i.poly
            polygons[j] = cand_j.poly

            bad = collide(i, polygons[i]) or collide(j, polygons[j])
            if bad:
                polygons[i], polygons[j] = old_i, old_j
                T = max(T * float(T_DECAY), float(T_MIN))
                continue

            new_side = compute_side_original(polygons)
            delta = float(new_side - current_side)

            if step < WARMUP_STEPS:
                accept = (delta <= 0.0)
            else:
                accept = (delta <= 0.0) or (random.random() < math.exp(-delta / T))

            if accept:
                trees[i] = cand_i
                trees[j] = cand_j
                current_side = new_side
                if current_side < best_side:
                    best_side = current_side
                    best_snap = snapshot_state(trees, local_ids)
            else:
                polygons[i], polygons[j] = old_i, old_j

            T = max(T * float(T_DECAY), float(T_MIN))
            continue

        # ---- single tree move ----
        t = trees[i]

        dx = Decimal("0")
        dy = Decimal("0")
        da = Decimal("0")

        if move == "translate":
            dx = Decimal(str(random.gauss(0.0, float(move_sigma))))
            dy = Decimal(str(random.gauss(0.0, float(move_sigma))))

        elif move == "rotate":
            da = Decimal(str(random.gauss(0.0, float(angle_sigma))))

        elif move == "both":
            dx = Decimal(str(random.gauss(0.0, float(move_sigma))))
            dy = Decimal(str(random.gauss(0.0, float(move_sigma))))
            da = Decimal(str(random.gauss(0.0, float(angle_sigma))))

        else:  # boundary_push (most effective for score)
            nx, ny = boundary_normal_direction(polygons, i)
            push = Decimal(str(abs(random.gauss(0.0, float(move_sigma))) * 1.8 + 0.01))
            dx = nx * push
            dy = ny * push
            # small rotate to help slip
            da = Decimal(str(random.gauss(0.0, float(angle_sigma * 0.35))))

        new_x = clamp(t.x + dx)
        new_y = clamp(t.y + dy)
        new_deg = wrap_angle_deg(t.deg + da)

        cand = ChristmasTree(new_x, new_y, new_deg)

        old_poly = polygons[i]
        polygons[i] = cand.poly

        if collide(i, polygons[i]):
            polygons[i] = old_poly
            T = max(T * float(T_DECAY), float(T_MIN))
            continue

        new_side = compute_side_original(polygons)
        delta = float(new_side - current_side)

        if step < WARMUP_STEPS:
            accept = (delta <= 0.0)
        else:
            accept = (delta <= 0.0) or (random.random() < math.exp(-delta / T))

        if accept:
            trees[i] = cand
            current_side = new_side
            if current_side < best_side:
                best_side = current_side
                best_snap = snapshot_state(trees, local_ids)
        else:
            polygons[i] = old_poly

        T = max(T * float(T_DECAY), float(T_MIN))

    # restore best local state
    if best_snap is not None:
        restore_state(trees, best_snap)


# -----------------------------
# Build submission
# -----------------------------
def build_submission():
    ids = [f"{n:03d}_{t}" for n in range(1, 201) for t in range(n)]
    rows = []
    trees = []

    t0 = time.time()

    for n in tqdm(range(1, 201), desc="Building configs n=1..200"):
        # 1) Strong insertion (score-based)
        trees = insert_best_candidate(trees)

        # 2) Heavy hybrid SA around newly inserted tree
        hybrid_sa(trees, target_idx=len(trees) - 1)

        # 3) safety check occasionally (heavy but ok on gaming PC)
        if n <= 6 or (n % 15 == 0):
            assert_no_overlap_all(trees, where=f"(n={n})")

        # 4) append rows for this n
        for t in trees:
            rows.append([float(t.x), float(t.y), float(t.deg)])

    elapsed = time.time() - t0
    print(f"\nDone. Total time: {elapsed/60:.1f} min")

    sub = pd.DataFrame(rows, columns=["x", "y", "deg"])
    sub.insert(0, "id", ids)

    sub[["x", "y", "deg"]] = sub[["x", "y", "deg"]].astype(float).round(ROUND_DECIMALS).astype(str)
    sub["x"] = "s" + sub["x"]
    sub["y"] = "s" + sub["y"]
    sub["deg"] = "s" + sub["deg"]

    sub.to_csv(OUT_CSV, index=False)
    print(f"Saved: {OUT_CSV} rows={len(sub)}")
    print(sub.head(3))
    assert list(sub.columns) == ["id", "x", "y", "deg"]
    assert len(sub) == 20100
    assert sub["id"].iloc[0] == "001_0"
    assert sub["id"].iloc[-1] == "200_199"
    return sub


if __name__ == "__main__":
    if SEED is not None:
        random.seed(SEED)
        np.random.seed(SEED)

    submission = build_submission()

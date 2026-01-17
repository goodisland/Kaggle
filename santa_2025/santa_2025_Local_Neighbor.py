# Santa 2025 - Christmas Tree Packing (Score-focused Local SA) - Version 3.4
# ------------------------------------------------------------------------------------
# Fixes vs v3.3 (main reasons your v3.3 can score WORSE than Getting Started):
# ✅ (A) SCALE CONSISTENCY: use scale_factor=1e15 AND compute side in original units.
# ✅ (B) Collision rule matches Getting Started: overlap forbidden, touching allowed.
# ✅ (C) Greedy insertion optimizes "side" directly (not radius only).
# ✅ (D) SA temperature works in original-units delta, so acceptance is meaningful.
# ✅ (E) Adds "push-in" move for boundary trees (helps reduce bounding square blow-ups).
#
# Output: submission.csv with strict columns: id,x,y,deg (each has 's' prefix)

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
from shapely.ops import unary_union


# -----------------------------
# Global config (score-focused)
# -----------------------------
getcontext().prec = 28

# IMPORTANT: match Getting Started scale
scale_factor = Decimal("1e15")

XY_MIN, XY_MAX = Decimal("-100"), Decimal("100")

# Greedy placement trials (higher = better, slower)
GREEDY_TRIALS = 80

# Local SA
LOCAL_SA_STEPS = 2200
LOCAL_K_NEIGHBORS = 50
BOUNDARY_BIAS_P = 0.85

# Base move magnitudes (in ORIGINAL coordinate units)
MOVE_SIGMA = 0.050
ANGLE_SIGMA = 7.0

# SA schedule (delta is in original-units side length now)
T0 = 0.020
T_DECAY = 0.9992
T_MIN = 1e-6

# Warmup
WARMUP_STEPS = 400

# Boundary refresh
BOUNDARY_REFRESH_EVERY = 200

# Move type probabilities
P_ROTATE_ONLY    = 0.40
P_TRANSLATE_ONLY = 0.35
P_BOTH           = 0.15
P_SWAP           = 0.05
P_PUSH_IN        = 0.05  # new: boundary push-in

ROUND_DECIMALS = 6


# -----------------------------
# Utility helpers
# -----------------------------
def clamp_decimal(v: Decimal) -> Decimal:
    if v < XY_MIN:
        return XY_MIN
    if v > XY_MAX:
        return XY_MAX
    return v


def wrap_angle_deg(a: Decimal) -> Decimal:
    x = float(a) % 360.0
    return Decimal(str(x))


def choose_move_type():
    r = random.random()
    if r < P_ROTATE_ONLY:
        return "rotate"
    r -= P_ROTATE_ONLY
    if r < P_TRANSLATE_ONLY:
        return "translate"
    r -= P_TRANSLATE_ONLY
    if r < P_BOTH:
        return "both"
    r -= P_BOTH
    if r < P_SWAP:
        return "swap"
    return "push_in"


def compute_side_length_original(polygons):
    """
    Compute bounding square side length in ORIGINAL units.
    polygons are in scaled coordinates, so divide bounds by scale_factor.
    """
    minx = min(p.bounds[0] for p in polygons)
    miny = min(p.bounds[1] for p in polygons)
    maxx = max(p.bounds[2] for p in polygons)
    maxy = max(p.bounds[3] for p in polygons)

    w = (Decimal(str(maxx)) - Decimal(str(minx))) / scale_factor
    h = (Decimal(str(maxy)) - Decimal(str(miny))) / scale_factor
    return max(w, h)


def boundary_indices(polys, eps_ratio=0.02):
    """Indices of polygons close to global min/max edges (in scaled coords)."""
    if not polys:
        return []
    minx = min(p.bounds[0] for p in polys)
    miny = min(p.bounds[1] for p in polys)
    maxx = max(p.bounds[2] for p in polys)
    maxy = max(p.bounds[3] for p in polys)
    w = maxx - minx
    h = maxy - miny
    epsx = eps_ratio * w if w > 0 else 0.0
    epsy = eps_ratio * h if h > 0 else 0.0

    out = []
    for i, p in enumerate(polys):
        bx0, by0, bx1, by1 = p.bounds
        if (bx0 <= minx + epsx) or (bx1 >= maxx - epsx) or (by0 <= miny + epsy) or (by1 >= maxy - epsy):
            out.append(i)
    return out


def has_overlap_touch_ok(candidate, tree_index, polygons, ignore_idx=None):
    """
    Collision rule:
    - Overlap is forbidden
    - Touching is allowed
    Equivalent to: intersects AND NOT touches means invalid.
    Use STRtree to test only nearby polygons.
    """
    hits = tree_index.query(candidate)
    for j in hits:
        if ignore_idx is not None and j == ignore_idx:
            continue
        pj = polygons[j]
        if candidate.intersects(pj) and (not candidate.touches(pj)):
            return True
    return False


def assert_no_overlap(trees, where=""):
    polys = [t.polygon for t in trees]
    idx = STRtree(polys)
    for i, p in enumerate(polys):
        hits = idx.query(p)
        for j in hits:
            if j <= i:
                continue
            if p.intersects(polys[j]) and (not p.touches(polys[j])):
                raise RuntimeError(f"Overlap detected {where}: i={i}, j={j}")


def adaptive_sigmas(n: int):
    """
    Shrink step sizes as n grows (stability late-stage).
    """
    if n < 50:
        m = 1.00
    elif n < 100:
        m = 0.85
    elif n < 150:
        m = 0.70
    else:
        m = 0.60
    return MOVE_SIGMA * m, ANGLE_SIGMA * m


def snapshot_state(trees, idx_list):
    return [(i, trees[i].center_x, trees[i].center_y, trees[i].angle) for i in idx_list]


def restore_state(trees, snap):
    for i, x, y, a in snap:
        trees[i] = ChristmasTree(x, y, a)


def generate_weighted_angle():
    # Keep a mild bias like Getting Started (optional)
    while True:
        angle = random.uniform(0, 2 * math.pi)
        if random.uniform(0, 1) < abs(math.sin(2 * angle)):
            return angle


# -----------------------------
# Tree geometry (official)
# -----------------------------
class ChristmasTree:
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

    # Base polygon in ORIGINAL units
    base_polygon_original = Polygon([
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

    # Pre-scaled polygon (scaled coords)
    base_polygon_scaled = affinity.scale(
        base_polygon_original,
        xfact=float(scale_factor),
        yfact=float(scale_factor),
        origin=(0, 0),
    )

    def __init__(self, x=Decimal("0"), y=Decimal("0"), angle=Decimal("0")):
        self.center_x = Decimal(x)
        self.center_y = Decimal(y)
        self.angle = Decimal(angle)
        self.polygon = None
        self.rebuild_polygon()

    def rebuild_polygon(self):
        rotated = affinity.rotate(self.base_polygon_scaled, float(self.angle), origin=(0, 0))
        self.polygon = affinity.translate(
            rotated,
            xoff=float(self.center_x * scale_factor),
            yoff=float(self.center_y * scale_factor),
        )


# -----------------------------
# Greedy incremental placement (optimize side directly)
# -----------------------------
def add_one_tree_greedy(placed_trees):
    if not placed_trees:
        return [ChristmasTree()]

    polygons = [t.polygon for t in placed_trees]
    idx = STRtree(polygons)

    best = None  # (side, x, y, angle)

    # We'll try multiple rays + angles and choose min "side" after insertion
    for _ in range(GREEDY_TRIALS):
        theta = generate_weighted_angle()
        vx = Decimal(str(math.cos(theta)))
        vy = Decimal(str(math.sin(theta)))
        angle = Decimal(str(random.uniform(0, 360)))

        # Start far, push in
        radius = Decimal("25.0")
        step_in = Decimal("0.6")

        last_feasible = None

        while radius >= 0:
            px = radius * vx
            py = radius * vy

            cand = ChristmasTree(px, py, angle)

            if not has_overlap_touch_ok(cand.polygon, idx, polygons, ignore_idx=None):
                last_feasible = (px, py)
                radius -= step_in
            else:
                break

        if last_feasible is None:
            continue

        px, py = last_feasible
        cand = ChristmasTree(px, py, angle)

        # Evaluate resulting side (original units)
        side = compute_side_length_original(polygons + [cand.polygon])

        if (best is None) or (side < best[0]):
            best = (side, px, py, angle)

    if best is None:
        # fallback random search
        for _ in range(4000):
            px = Decimal(str(random.uniform(-10, 10)))
            py = Decimal(str(random.uniform(-10, 10)))
            angle = Decimal(str(random.uniform(0, 360)))
            cand = ChristmasTree(px, py, angle)
            if not has_overlap_touch_ok(cand.polygon, idx, polygons, ignore_idx=None):
                placed_trees.append(cand)
                return placed_trees
        placed_trees.append(ChristmasTree(Decimal("60"), Decimal("60"), Decimal("0")))
        return placed_trees

    _, px, py, angle = best
    placed_trees.append(ChristmasTree(px, py, angle))
    return placed_trees


# -----------------------------
# Local SA (stable + keep-best)
# -----------------------------
def local_sa_score_focused(trees, target_idx):
    n = len(trees)
    if n <= 1:
        return

    move_sigma, angle_sigma = adaptive_sigmas(n)

    polygons = [t.polygon for t in trees]
    current_side = compute_side_length_original(polygons)

    # --- Find K nearest neighbors to target ---
    tx, ty = trees[target_idx].polygon.centroid.x, trees[target_idx].polygon.centroid.y
    dists = []
    for i, t in enumerate(trees):
        cx, cy = t.polygon.centroid.x, t.polygon.centroid.y
        d = (cx - tx) ** 2 + (cy - ty) ** 2
        dists.append((d, i))
    dists.sort(key=lambda x: x[0])
    local_ids = [i for _, i in dists[:min(LOCAL_K_NEIGHBORS, n)]]
    local_set = set(local_ids)

    # Boundary ids (local-only)
    local_polys = [polygons[i] for i in local_ids]
    b_local = boundary_indices(local_polys, eps_ratio=0.02)
    boundary_ids = [local_ids[k] for k in b_local] if b_local else local_ids[:]

    # --- Split fixed(outside) vs moving(local) ---
    outside_ids = [i for i in range(n) if i not in local_set]
    outside_polys = [polygons[i] for i in outside_ids]
    outside_tree = STRtree(outside_polys) if outside_polys else None
    outside_geom2idx = {id(g): idx for idx, g in enumerate(outside_polys)} if outside_polys else None

    def overlap_with_outside(cand_poly):
        if outside_tree is None:
            return False
        hit_ids = _query_indices(outside_tree, cand_poly, outside_geom2idx)
        for k in hit_ids:
            p = outside_polys[k]
            if cand_poly.intersects(p) and (not cand_poly.touches(p)):
                return True
        return False

    def overlap_with_local(cand_poly, ignore_i):
        # brute-force over locals (<=50)
        for j in local_ids:
            if j == ignore_i:
                continue
            p = polygons[j]
            if cand_poly.intersects(p) and (not cand_poly.touches(p)):
                return True
        return False

    # keep-best
    best_side = current_side
    best_snap = snapshot_state(trees, local_ids)

    T = float(T0)

    for step in range(LOCAL_SA_STEPS):
        # periodic boundary refresh
        if (step > 0) and (step % BOUNDARY_REFRESH_EVERY == 0):
            local_polys = [polygons[i] for i in local_ids]
            b_local = boundary_indices(local_polys, eps_ratio=0.02)
            boundary_ids = [local_ids[k] for k in b_local] if b_local else local_ids[:]

        # pick index (boundary-biased)
        if random.random() < BOUNDARY_BIAS_P and boundary_ids:
            i = random.choice(boundary_ids)
        else:
            i = random.choice(local_ids)

        move_type = choose_move_type()

        # ---- swap ----
        if move_type == "swap":
            j = random.choice(local_ids)
            if j == i:
                T = max(T * float(T_DECAY), float(T_MIN))
                continue

            ti, tj = trees[i], trees[j]
            cand_i = ChristmasTree(tj.center_x, tj.center_y, ti.angle)
            cand_j = ChristmasTree(ti.center_x, ti.center_y, tj.angle)

            old_i, old_j = polygons[i], polygons[j]
            polygons[i] = cand_i.polygon
            polygons[j] = cand_j.polygon

            bad = (
                overlap_with_outside(polygons[i]) or overlap_with_outside(polygons[j]) or
                overlap_with_local(polygons[i], ignore_i=i) or overlap_with_local(polygons[j], ignore_i=j)
            )
            if bad:
                polygons[i], polygons[j] = old_i, old_j
                T = max(T * float(T_DECAY), float(T_MIN))
                continue

            new_side = compute_side_length_original(polygons)
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

        # ---- single-tree moves ----
        t = trees[i]
        dx = Decimal("0"); dy = Decimal("0"); da = Decimal("0")

        if move_type == "rotate":
            da = Decimal(str(random.gauss(0.0, angle_sigma)))
        elif move_type == "translate":
            dx = Decimal(str(random.gauss(0.0, move_sigma)))
            dy = Decimal(str(random.gauss(0.0, move_sigma)))
        elif move_type == "both":
            dx = Decimal(str(random.gauss(0.0, move_sigma)))
            dy = Decimal(str(random.gauss(0.0, move_sigma)))
            da = Decimal(str(random.gauss(0.0, angle_sigma)))
        else:  # push_in
            push = Decimal(str(abs(random.gauss(0.0, move_sigma)) * 1.3))
            ox, oy = t.center_x, t.center_y
            norm = Decimal(str(math.sqrt(float(ox*ox + oy*oy)))) if (ox != 0 or oy != 0) else Decimal("1")
            dx = -(ox / norm) * push
            dy = -(oy / norm) * push
            da = Decimal(str(random.gauss(0.0, angle_sigma * 0.4)))

        new_x = clamp_decimal(t.center_x + dx)
        new_y = clamp_decimal(t.center_y + dy)
        new_a = wrap_angle_deg(t.angle + da)

        cand = ChristmasTree(new_x, new_y, new_a)

        old_poly = polygons[i]
        polygons[i] = cand.polygon

        bad = (
            overlap_with_outside(polygons[i]) or
            overlap_with_local(polygons[i], ignore_i=i)
        )
        if bad:
            polygons[i] = old_poly
            T = max(T * float(T_DECAY), float(T_MIN))
            continue

        new_side = compute_side_length_original(polygons)
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

    # restore best
    if best_snap is not None:
        restore_state(trees, best_snap)

def _query_indices(tree, geom, geom2idx=None):
    hits = tree.query(geom)
    if len(hits) == 0:
        return []
    h0 = hits[0]
    # If Shapely returns indices (int)
    if isinstance(h0, (int, np.integer)):
        return list(hits)
    # If Shapely returns geometries
    if geom2idx is None:
        return []
    return [geom2idx[id(g)] for g in hits if id(g) in geom2idx]


# -----------------------------
# Main: build all n=1..200 and write submission
# -----------------------------
def build_submission():
    ids = [f"{n:03d}_{t}" for n in range(1, 201) for t in range(n)]
    tree_rows = []
    trees = []

    t_start = time.time()

    for n in tqdm(range(1, 201), desc="Building configs n=1..200"):
        trees = add_one_tree_greedy(trees)

        # local SA around newly added tree
        local_sa_score_focused(trees, target_idx=len(trees) - 1)

        # (optional) safety check occasionally
        if n <= 5 or (n % 25 == 0):
            assert_no_overlap(trees, where=f"(n={n})")

        for t in trees:
            tree_rows.append([float(t.center_x), float(t.center_y), float(t.angle)])

    elapsed = time.time() - t_start
    print(f"\nDone. Total time: {elapsed/60:.1f} minutes")

    cols = ["x", "y", "deg"]
    sub = pd.DataFrame(tree_rows, columns=cols)
    sub.insert(0, "id", ids)

    sub[cols] = sub[cols].astype(float).round(ROUND_DECIMALS).astype(str)
    for c in cols:
        sub[c] = "s" + sub[c]

    out_path = "submission.csv"
    sub.to_csv(out_path, index=False)

    print(f"Saved: {out_path} (rows={len(sub)})")
    print("Columns:", list(sub.columns))
    print(sub.head(3))

    assert list(sub.columns) == ["id", "x", "y", "deg"]
    assert len(sub) == 20100
    assert sub["id"].iloc[0] == "001_0"
    assert sub["id"].iloc[-1] == "200_199"

    return sub


submission = build_submission()
submission.head()

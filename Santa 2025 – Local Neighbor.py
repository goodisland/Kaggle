# Santa 2025 - Christmas Tree Packing (Score-focused Local SA) - Version 3.3 (KEEP-BEST + SMALL TRICKS)
# ---------------------------------------------------------------------------------------------------
# What changed vs your v3.2:
# ✅ (1) KEEP-BEST: Local SA keeps the best (lowest side) state and restores it at the end.
# ✅ (2) Warmup hill-climb: first WARMUP_STEPS accept only improvements (stabilizes early n).
# ✅ (3) Adaptive step size: MOVE/ANGLE sigma shrink as n grows (prevents late-stage blow-ups).
# ✅ (4) Periodic boundary refresh: boundary set is recomputed occasionally during SA.
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


# -----------------------------
# Global config (score-focused)
# -----------------------------
getcontext().prec = 28
scale_factor = Decimal("1e10")

# Competition constraints (as described in the rules)
XY_MIN, XY_MAX = Decimal("-100"), Decimal("100")

# Greedy placement trials (higher = better initial placement, slower)
GREEDY_TRIALS = 60

# Local SA (heavier)
LOCAL_SA_STEPS = 2500          # increase for better score (and slower runtime)
LOCAL_K_NEIGHBORS = 45         # local neighborhood size
BOUNDARY_BIAS_P = 0.80         # bias probability to pick boundary trees

# Move magnitudes (base; actual sigma is adapted by n)
MOVE_SIGMA = 0.045             # translation std in coordinate units
ANGLE_SIGMA = 6.5              # rotation std in degrees

# SA schedule (base; T0 can be effectively smaller via warmup hill-climb)
T0 = 0.10
T_DECAY = 0.9992
T_MIN = 1e-6

# Small tricks
KEEP_BEST_STATE = True         # ⭐ key fix: restore best local state after SA
WARMUP_STEPS = 450             # hill-climb steps (accept only delta<=0) then SA
BOUNDARY_REFRESH_EVERY = 200    # recompute boundary ids occasionally to stay relevant

# Move type probabilities (sum to 1.0)
P_ROTATE_ONLY    = 0.45
P_TRANSLATE_ONLY = 0.35
P_BOTH           = 0.15
P_SWAP           = 0.05

# Output formatting
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
    # keep within [0, 360)
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
    return "swap"


def compute_side_length(polygons):
    # Global bounding square side = max(width, height)
    minx = min(p.bounds[0] for p in polygons)
    miny = min(p.bounds[1] for p in polygons)
    maxx = max(p.bounds[2] for p in polygons)
    maxy = max(p.bounds[3] for p in polygons)
    return Decimal(str(max(maxx - minx, maxy - miny)))


def boundary_indices(polys, eps_ratio=0.02):
    # Return indices of polygons that touch near the global min/max edges.
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


def overlaps_exact(poly, all_polys, ignore_idx=None):
    # Robust overlap detection: check intersection area > 0 with any other polygon.
    # (No AABB shortcut.)
    for k, p in enumerate(all_polys):
        if ignore_idx is not None and k == ignore_idx:
            continue
        # Shapely: intersects() true even if just touching; we need overlap area > 0.
        if poly.intersects(p):
            inter = poly.intersection(p)
            if not inter.is_empty and inter.area > 0:
                return True
    return False


def assert_no_overlap(trees, where=""):
    polys = [t.polygon for t in trees]
    n = len(polys)
    for i in range(n):
        for j in range(i + 1, n):
            if polys[i].intersects(polys[j]):
                inter = polys[i].intersection(polys[j])
                if (not inter.is_empty) and (inter.area > 0):
                    raise RuntimeError(f"Overlap detected {where}: i={i}, j={j}, area={inter.area}")


def generate_weighted_angle():
    # simple uniform angle; you can customize to bias directions if desired
    return random.random() * 2.0 * math.pi


def adaptive_sigmas(n: int):
    """
    Shrink step sizes as n grows (late-stage: avoid expanding the bounding square).
    These multipliers are conservative and usually improve stability.
    """
    if n < 50:
        m = 1.00
    elif n < 100:
        m = 0.80
    elif n < 150:
        m = 0.65
    else:
        m = 0.55

    move = MOVE_SIGMA * m
    ang  = ANGLE_SIGMA * m
    return move, ang


def snapshot_state(trees, idx_list):
    # store (index, x, y, angle)
    return [(i, trees[i].center_x, trees[i].center_y, trees[i].angle) for i in idx_list]


def restore_state(trees, snap):
    # restore by re-creating objects (safe; guarantees polygon rebuilt)
    for i, x, y, a in snap:
        trees[i] = ChristmasTree(x, y, a)


# -----------------------------
# Tree geometry (official)
# -----------------------------
class ChristmasTree:
    """
    A single fixed-shape tree that can be translated and rotated.
    Geometry MUST match the official competition definition.
    """

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

    base_polygon = Polygon([
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

    def __init__(self, x=Decimal("0"), y=Decimal("0"), angle=Decimal("0")):
        self.center_x = Decimal(x)
        self.center_y = Decimal(y)
        self.angle = Decimal(angle)
        self.polygon = None
        self.rebuild_polygon()

    def rebuild_polygon(self):
        # Scale -> rotate -> translate (Shapely uses float internally)
        scaled = affinity.scale(
            self.base_polygon,
            xfact=float(scale_factor),
            yfact=float(scale_factor),
            origin=(0, 0)
        )
        rotated = affinity.rotate(scaled, float(self.angle), origin=(0, 0))
        self.polygon = affinity.translate(
            rotated,
            xoff=float(self.center_x * scale_factor),
            yoff=float(self.center_y * scale_factor),
        )


# -----------------------------
# Greedy incremental placement
# -----------------------------
def add_one_tree_greedy(placed_trees):
    if not placed_trees:
        return [ChristmasTree()]

    placed_polys = [t.polygon for t in placed_trees]

    best_px = None
    best_py = None
    best_angle = None
    best_radius = Decimal("Infinity")
    best_vx = Decimal("1")
    best_vy = Decimal("0")

    for _ in range(GREEDY_TRIALS):
        theta = generate_weighted_angle()
        vx = Decimal(str(math.cos(theta)))
        vy = Decimal(str(math.sin(theta)))

        angle = Decimal(str(random.uniform(0, 360)))

        radius = Decimal("20.0")
        step_in = Decimal("0.7")

        collision_found = False
        while radius >= 0:
            px = radius * vx
            py = radius * vy

            cand = ChristmasTree(px, py, angle)
            if not overlaps_exact(cand.polygon, placed_polys, ignore_idx=None):
                # feasible; try push further in
                radius -= step_in
                best_px, best_py, best_angle = px, py, angle
                best_radius = radius
                best_vx, best_vy = vx, vy
            else:
                collision_found = True
                break

        if not collision_found and best_px is not None:
            # already feasible at very small radius
            pass

    if best_px is None:
        # fallback (rare): random search
        for _ in range(2000):
            px = Decimal(str(random.uniform(-10, 10)))
            py = Decimal(str(random.uniform(-10, 10)))
            angle = Decimal(str(random.uniform(0, 360)))
            cand = ChristmasTree(px, py, angle)
            if not overlaps_exact(cand.polygon, placed_polys, ignore_idx=None):
                placed_trees.append(cand)
                return placed_trees
        # last resort: place far away
        placed_trees.append(ChristmasTree(Decimal("50"), Decimal("50"), Decimal("0")))
        return placed_trees

    placed_trees.append(ChristmasTree(best_px, best_py, best_angle))
    return placed_trees


# -----------------------------
# Local SA with KEEP-BEST + tricks
# -----------------------------
def local_sa_score_focused(trees, target_idx):
    """
    Local SA around the newly added tree:
      1) Choose K nearest neighbors (by centroid distance)
      2) Build boundary set within these locals
      3) Run SA steps with mixed move types
    Objective: minimize global bounding square side length.
    """
    n = len(trees)
    if n <= 1:
        return

    move_sigma, angle_sigma = adaptive_sigmas(n)

    polygons = [t.polygon for t in trees]
    current_side = compute_side_length(polygons)

    # --- Find K nearest neighbors to the target ---
    tx, ty = trees[target_idx].polygon.centroid.x, trees[target_idx].polygon.centroid.y
    dists = []
    for i, t in enumerate(trees):
        cx, cy = t.polygon.centroid.x, t.polygon.centroid.y
        d = (cx - tx) ** 2 + (cy - ty) ** 2
        dists.append((d, i))
    dists.sort(key=lambda x: x[0])
    local_ids = [i for _, i in dists[:min(LOCAL_K_NEIGHBORS, n)]]

    # initial boundary set (computed on local polygons)
    local_polys = [polygons[i] for i in local_ids]
    b_local = boundary_indices(local_polys, eps_ratio=0.02)
    boundary_ids = [local_ids[k] for k in b_local] if b_local else local_ids[:]

    # ✅ Keep best state (fix)
    if KEEP_BEST_STATE:
        best_side = current_side
        best_snap = snapshot_state(trees, local_ids)
    else:
        best_side = None
        best_snap = None

    # SA temperature
    T = float(T0)

    for step in range(LOCAL_SA_STEPS):
        # periodic boundary refresh (keeps boundary bias relevant)
        if (BOUNDARY_REFRESH_EVERY is not None) and (step > 0) and (step % BOUNDARY_REFRESH_EVERY == 0):
            polygons = [t.polygon for t in trees]  # refresh reference list
            local_polys = [polygons[i] for i in local_ids]
            b_local = boundary_indices(local_polys, eps_ratio=0.02)
            boundary_ids = [local_ids[k] for k in b_local] if b_local else local_ids[:]

        # Pick a tree index (boundary-biased)
        if random.random() < BOUNDARY_BIAS_P and boundary_ids:
            i = random.choice(boundary_ids)
        else:
            i = random.choice(local_ids)

        move_type = choose_move_type()

        # ---- Swap move (two-tree) ----
        if move_type == "swap":
            j = random.choice(local_ids)
            if j == i:
                continue

            ti, tj = trees[i], trees[j]

            # Swap positions, keep original angles
            cand_i = ChristmasTree(tj.center_x, tj.center_y, ti.angle)
            cand_j = ChristmasTree(ti.center_x, ti.center_y, tj.angle)

            old_i, old_j = polygons[i], polygons[j]
            polygons[i] = cand_i.polygon
            polygons[j] = cand_j.polygon

            # Robust collision checks
            if overlaps_exact(polygons[i], polygons, ignore_idx=i) or overlaps_exact(polygons[j], polygons, ignore_idx=j):
                polygons[i], polygons[j] = old_i, old_j
                T = max(T * float(T_DECAY), float(T_MIN))
                continue

            new_side = compute_side_length(polygons)
            delta = float(new_side - current_side)

            # warmup hill-climb: accept only improvements
            if step < WARMUP_STEPS:
                accept = (delta <= 0)
            else:
                accept = (delta <= 0) or (random.random() < math.exp(-delta / T))

            if accept:
                trees[i] = cand_i
                trees[j] = cand_j
                current_side = new_side

                if KEEP_BEST_STATE and current_side < best_side:
                    best_side = current_side
                    best_snap = snapshot_state(trees, local_ids)
            else:
                polygons[i], polygons[j] = old_i, old_j

            T = max(T * float(T_DECAY), float(T_MIN))
            continue

        # ---- Single-tree move ----
        t = trees[i]
        dx = Decimal("0"); dy = Decimal("0"); da = Decimal("0")

        if move_type == "rotate":
            da = Decimal(str(random.gauss(0.0, angle_sigma)))
        elif move_type == "translate":
            dx = Decimal(str(random.gauss(0.0, move_sigma)))
            dy = Decimal(str(random.gauss(0.0, move_sigma)))
        else:  # "both"
            dx = Decimal(str(random.gauss(0.0, move_sigma)))
            dy = Decimal(str(random.gauss(0.0, move_sigma)))
            da = Decimal(str(random.gauss(0.0, angle_sigma)))

        new_x = clamp_decimal(t.center_x + dx)
        new_y = clamp_decimal(t.center_y + dy)
        new_a = wrap_angle_deg(t.angle + da)

        cand = ChristmasTree(new_x, new_y, new_a)

        old_poly = polygons[i]
        polygons[i] = cand.polygon

        if overlaps_exact(polygons[i], polygons, ignore_idx=i):
            polygons[i] = old_poly
            T = max(T * float(T_DECAY), float(T_MIN))
            continue

        new_side = compute_side_length(polygons)
        delta = float(new_side - current_side)

        # warmup hill-climb: accept only improvements
        if step < WARMUP_STEPS:
            accept = (delta <= 0)
        else:
            accept = (delta <= 0) or (random.random() < math.exp(-delta / T))

        if accept:
            trees[i] = cand
            current_side = new_side

            if KEEP_BEST_STATE and current_side < best_side:
                best_side = current_side
                best_snap = snapshot_state(trees, local_ids)
        else:
            polygons[i] = old_poly

        T = max(T * float(T_DECAY), float(T_MIN))

    # ✅ IMPORTANT: restore best (prevents “SA ended worse than best”)
    if KEEP_BEST_STATE and best_snap is not None:
        restore_state(trees, best_snap)


# -----------------------------
# Main: build all n=1..200 and write submission
# -----------------------------
def build_submission():
    """
    Build all configurations (n=1..200), collect rows, and write submission.csv
    in the strict required format: id,x,y,deg with 's' prefix.
    """
    # Required submission ids (length = 1+2+...+200 = 20100)
    ids = [f"{n:03d}_{t}" for n in range(1, 201) for t in range(n)]

    tree_rows = []   # stores [x, y, deg] for all configurations concatenated
    trees = []

    t_start = time.time()

    for n in tqdm(range(1, 201), desc="Building configs n=1..200"):
        # 1) Add one tree using greedy radial push-in
        trees = add_one_tree_greedy(trees)

        # right after greedy (before SA)
        if n <= 3:
            assert_no_overlap(trees, where=f"(after greedy, n={n})")

        # 2) Heavy local SA around the newly added tree
        local_sa_score_focused(trees, target_idx=len(trees) - 1)

        # 3) Validate current configuration (robust)
        try:
            assert_no_overlap(trees, where=f"(n={n})")
        except RuntimeError:
            print(f"[DEBUG] Invalid configuration detected at n={n}.")
            raise

        # 4) Append this configuration's rows
        for t in trees:
            tree_rows.append([float(t.center_x), float(t.center_y), float(t.angle)])

    elapsed = time.time() - t_start
    print(f"\nDone. Total time: {elapsed/60:.1f} minutes")

    # ---- Build DataFrame with REQUIRED COLUMNS ----
    cols = ["x", "y", "deg"]
    sub = pd.DataFrame(tree_rows, columns=cols)
    sub.insert(0, "id", ids)  # IMPORTANT: 'id' must be a column

    # Round and convert to required string format with 's' prefix
    sub[cols] = sub[cols].astype(float).round(ROUND_DECIMALS).astype(str)
    for c in cols:
        sub[c] = "s" + sub[c]

    # Write exact header: id,x,y,deg and no extra index column
    out_path = "submission.csv"
    sub.to_csv(out_path, index=False)

    # Quick sanity checks (format)
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

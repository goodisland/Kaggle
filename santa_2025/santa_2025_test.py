#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_santa_bbox_local.py (Windows -> WSL auto runner, hardened)

ポイント:
- bbox3 が Linux バイナリでも Windows から自動で WSL 経由実行
- WSL 起動時に Windows の PATH/WSLENV/Conda 環境が混入して壊れるのを防ぐ
  -> wsl 呼び出し時に env を最小化 + WSL側で env -i を使ってクリーン実行

必須（グローバル変数で指定）:
- DONOR_CSV
- BBOX_EXE
- WORKDIR

依存:
pip install numpy pandas shapely scipy
"""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from decimal import Decimal, getcontext

from shapely import affinity
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree

from scipy.spatial import ConvexHull
from scipy.optimize import minimize_scalar


# =============================================================================
# 0) グローバル設定（ここだけ編集すれば動く）
# =============================================================================

DONOR_CSV = r"./santa-2025.csv"
BBOX_EXE  = r"./bbox3"
WORKDIR   = r"./work"

START_FROM_DONOR = True
MAX_HOURS = 7.0
PER_RUN_TIMEOUT_SEC = 1200

N_MIN, N_MAX, N_STEP = 500, 2000, 100
R_MIN, R_MAX, R_STEP = 10, 90, 10

FIX_EACH_RUN = True
FINAL_VALIDATE_OVERLAP = True

PRINT_SCORE_EACH_RUN = True
SCORE_EVERY_K = 1
SHUFFLE_PARAMS_EACH_EPOCH = True
KEEP_BEST_SUBMISSION = True
USE_BEST_AT_END = True

MAKE_ZIP = True
DEBUG_ONCE = False

# --- WSL 実行設定 ---
AUTO_USE_WSL = True
WSL_EXE = "wsl"
WSL_DISTRO = ""          # 例: "Ubuntu-22.04"（空なら既定）
WSL_SHELL = "bash"
WSL_PYTHON = "python3"
WSL_PIP_INSTALL = False  # ← まずは False 推奨（WSL自体の起動が安定してから True に）
WSL_DEPENDENCIES = ["numpy", "pandas", "shapely", "scipy"]

# WSL 呼び出し時に Windows 環境変数を極力渡さない（重要）
WSL_CLEAN_WINDOWS_ENV = True
WSL_CLEAN_KEYS = [
    "WSLENV", "WSLPATH", "PATH", "PYTHONPATH",
    "CONDA_PREFIX", "CONDA_DEFAULT_ENV", "CONDA_SHLVL",
    "VIRTUAL_ENV",
]


# =============================================================================
# ログ
# =============================================================================

def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def make_logger(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(log_path, "a", encoding="utf-8", buffering=1)

    def log(*args):
        s = " ".join(str(a) for a in args)
        line = f"[{_now()}] {s}"
        print(line, flush=True)
        f.write(line + "\n")
        f.flush()

    def close():
        try:
            f.flush()
            f.close()
        except Exception:
            pass

    return log, close


# =============================================================================
# WSL utilities
# =============================================================================

def is_windows() -> bool:
    return os.name == "nt"

def wsl_path(win_path: Path) -> str:
    p = win_path.resolve()
    drive = p.drive.rstrip(":").lower()
    if not drive:
        return str(p).replace("\\", "/")
    rest = str(p)[2:].lstrip("\\/")
    return f"/mnt/{drive}/" + rest.replace("\\", "/")

def _windows_env_for_wsl() -> dict:
    """
    wsl.exe に渡す Windows 側環境変数を最小化
    """
    env = os.environ.copy()
    if not WSL_CLEAN_WINDOWS_ENV:
        return env

    # まず “WSLENV” が地雷になることが多いので外す
    for k in WSL_CLEAN_KEYS:
        env.pop(k, None)

    # さらに PATH が長すぎ＆変換に失敗するケースが多いので、PATH を最小化
    # ここで wsl.exe 自体の起動に必要な最低限だけ残す（通常は不要だが保険）
    # env["PATH"] を消すのは危険な場合があるので空にする
    env["PATH"] = ""

    return env

def run_cmd(cmd: List[str], cwd: Path | None = None, timeout: int | None = None, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )

def wsl_base_cmd() -> List[str]:
    base = [WSL_EXE]
    if WSL_DISTRO:
        base += ["-d", WSL_DISTRO]
    return base

def wsl_run_bash(log, bash_cmd: str, timeout: int | None = None, cwd_win: Path | None = None) -> subprocess.CompletedProcess:
    """
    WSL上でコマンド実行。
    重要:
      - Windows環境変数がWSLに混入して翻訳失敗するのを防ぐため
        1) wsl.exeに渡す env を最小化
        2) WSL側では env -i で環境を初期化してから bash -lc を実行
    """
    prefix = ""
    if cwd_win is not None:
        prefix = f"cd {wsl_path(cwd_win)} && "

    # env -i で完全初期化し、最低限の環境だけ与える
    # HOME は root だと /root が普通。LANG も最低限。
    # PATH は /usr/bin:/bin 程度あれば十分。
    safe = f'env -i HOME="$HOME" LANG=C.UTF-8 PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" {WSL_SHELL} -lc "{prefix}{bash_cmd}"'

    cmd = wsl_base_cmd() + ["--exec", WSL_SHELL, "-lc", safe]
    env = _windows_env_for_wsl()

    res = run_cmd(cmd, timeout=timeout, env=env)

    log("WSL CMD:", " ".join(cmd))
    log("WSL returncode:", res.returncode)
    if res.stdout:
        log("WSL stdout:", res.stdout.strip()[:5000])
    if res.stderr:
        log("WSL stderr:", res.stderr.strip()[:5000])

    return res

def ensure_wsl_deps(log):
    if not WSL_PIP_INSTALL:
        return
    deps = " ".join(WSL_DEPENDENCIES)
    log("Try WSL pip install:", deps)
    wsl_run_bash(log, f'{WSL_PYTHON} -m pip --version || true', timeout=60)
    wsl_run_bash(log, f'{WSL_PYTHON} -m pip install -U {deps} || pip3 install -U {deps} || true', timeout=600)

def bbox3_is_windows_exe_runnable(log, bbox_exe: Path, workdir: Path) -> bool:
    try:
        res = run_cmd([str(bbox_exe)], cwd=workdir, timeout=5)
        log("bbox3 sanity (win) returncode:", res.returncode)
        if res.stdout:
            log("bbox3 sanity (win) stdout:", res.stdout.strip()[:2000])
        if res.stderr:
            log("bbox3 sanity (win) stderr:", res.stderr.strip()[:2000])
        return True
    except OSError as e:
        log("bbox3 sanity (win) OSError:", repr(e))
        return False
    except Exception as e:
        log("bbox3 sanity (win) Exception:", repr(e))
        log(traceback.format_exc())
        return False

def run_bbox3(log, use_wsl: bool, bbox_exe: Path, workdir: Path, n: int, r: int, timeout_sec: int) -> Tuple[int, str, str]:
    if use_wsl:
        bbox_wsl = wsl_path(bbox_exe)
        cmd = f"chmod +x {bbox_wsl} || true; {bbox_wsl} -n {n} -r {r}"
        res = wsl_run_bash(log, cmd, timeout=timeout_sec, cwd_win=workdir)
        return res.returncode, res.stdout or "", res.stderr or ""

    res = run_cmd([str(bbox_exe), "-n", str(n), "-r", str(r)], cwd=workdir, timeout=timeout_sec)
    return res.returncode, res.stdout or "", res.stderr or ""


# =============================================================================
# 1) fix_direction
# =============================================================================

getcontext().prec = 30
SCALE_FIX = Decimal("1")


class ChristmasTreeFix:
    def __init__(self, center_x="0", center_y="0", angle="0"):
        self.center_x = Decimal(str(center_x))
        self.center_y = Decimal(str(center_y))
        self.angle = Decimal(str(angle))

        trunk_w = Decimal("0.15")
        trunk_h = Decimal("0.2")
        base_w = Decimal("0.7")
        mid_w = Decimal("0.4")
        top_w = Decimal("0.25")
        tip_y = Decimal("0.8")
        tier_1_y = Decimal("0.5")
        tier_2_y = Decimal("0.25")
        base_y = Decimal("0.0")
        trunk_bottom_y = -trunk_h

        initial_polygon = Polygon(
            [
                (Decimal("0.0") * SCALE_FIX, tip_y * SCALE_FIX),
                (top_w / Decimal("2") * SCALE_FIX, tier_1_y * SCALE_FIX),
                (top_w / Decimal("4") * SCALE_FIX, tier_1_y * SCALE_FIX),
                (mid_w / Decimal("2") * SCALE_FIX, tier_2_y * SCALE_FIX),
                (mid_w / Decimal("4") * SCALE_FIX, tier_2_y * SCALE_FIX),
                (base_w / Decimal("2") * SCALE_FIX, base_y * SCALE_FIX),
                (trunk_w / Decimal("2") * SCALE_FIX, base_y * SCALE_FIX),
                (trunk_w / Decimal("2") * SCALE_FIX, trunk_bottom_y * SCALE_FIX),
                (-(trunk_w / Decimal("2")) * SCALE_FIX, trunk_bottom_y * SCALE_FIX),
                (-(trunk_w / Decimal("2")) * SCALE_FIX, base_y * SCALE_FIX),
                (-(base_w / Decimal("2")) * SCALE_FIX, base_y * SCALE_FIX),
                (-(mid_w / Decimal("4")) * SCALE_FIX, tier_2_y * SCALE_FIX),
                (-(mid_w / Decimal("2")) * SCALE_FIX, tier_2_y * SCALE_FIX),
                (-(top_w / Decimal("4")) * SCALE_FIX, tier_1_y * SCALE_FIX),
                (-(top_w / Decimal("2")) * SCALE_FIX, tier_1_y * SCALE_FIX),
            ]
        )
        rotated = affinity.rotate(initial_polygon, float(self.angle), origin=(0, 0))
        self.polygon = affinity.translate(rotated, xoff=float(self.center_x * SCALE_FIX), yoff=float(self.center_y * SCALE_FIX))

    def clone(self) -> "ChristmasTreeFix":
        return ChristmasTreeFix(str(self.center_x), str(self.center_y), str(self.angle))


def _side_length_for_trees_fix(trees: List[ChristmasTreeFix]) -> Decimal:
    bounds = unary_union([t.polygon for t in trees]).bounds
    return Decimal(max(bounds[2] - bounds[0], bounds[3] - bounds[1])) / SCALE_FIX


def _total_score_from_side_lengths(side: Dict[str, Decimal]) -> Decimal:
    score = Decimal("0")
    for k, v in side.items():
        score += v**2 / Decimal(k)
    return score


def _parse_submission_for_fix(csv_path: Path) -> Tuple[Dict[str, List[ChristmasTreeFix]], Dict[str, Decimal]]:
    df = pd.read_csv(csv_path)
    df["x"] = df["x"].astype(str).str.strip().str.lstrip("s")
    df["y"] = df["y"].astype(str).str.strip().str.lstrip("s")
    df["deg"] = df["deg"].astype(str).str.strip().str.lstrip("s")
    df[["group_id", "item_id"]] = df["id"].str.split("_", n=2, expand=True)

    dict_of_tree_list: Dict[str, List[ChristmasTreeFix]] = {}
    dict_of_side_length: Dict[str, Decimal] = {}

    for group_id, group_data in df.groupby("group_id"):
        trees = [ChristmasTreeFix(center_x=row["x"], center_y=row["y"], angle=row["deg"]) for _, row in group_data.iterrows()]
        dict_of_tree_list[group_id] = trees
        dict_of_side_length[group_id] = _side_length_for_trees_fix(trees)

    return dict_of_tree_list, dict_of_side_length


def _bbox_side_at_angle(angle_deg: float, points: np.ndarray) -> float:
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    rot_T = np.array([[c, s], [-s, c]])
    rotated = points.dot(rot_T)
    min_xy = np.min(rotated, axis=0)
    max_xy = np.max(rotated, axis=0)
    return float(max(max_xy[0] - min_xy[0], max_xy[1] - min_xy[1]))


def _optimize_rotation_angle(trees: List[ChristmasTreeFix]) -> Tuple[Decimal, float]:
    pts = []
    for t in trees:
        pts.extend(list(t.polygon.exterior.coords))
    pts = np.array(pts, dtype=np.float64)

    hull_pts = pts[ConvexHull(pts).vertices]
    initial_side = _bbox_side_at_angle(0.0, hull_pts)

    res = minimize_scalar(lambda a: _bbox_side_at_angle(a, hull_pts), bounds=(0.001, 89.999), method="bounded")
    found_angle = float(res.x)
    found_side = float(res.fun)

    if (initial_side - found_side) > 1e-8:
        return Decimal(found_side) / SCALE_FIX, found_angle
    return Decimal(initial_side) / SCALE_FIX, 0.0


def _apply_rotation_to_solution(trees: List[ChristmasTreeFix], angle_deg: float) -> List[ChristmasTreeFix]:
    if not trees or abs(angle_deg) < 1e-9:
        return [t.clone() for t in trees]

    b = [t.polygon.bounds for t in trees]
    min_x = min(x0 for x0, _, _, _ in b)
    min_y = min(y0 for _, y0, _, _ in b)
    max_x = max(x1 for _, _, x1, _ in b)
    max_y = max(y1 for _, _, _, y1 in b)
    center = np.array([(min_x + max_x) / 2.0, (min_y + max_y) / 2.0], dtype=np.float64)

    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)

    xy = np.array([[float(t.center_x), float(t.center_y)] for t in trees], dtype=np.float64)
    rotated = (xy - center).dot(rot.T) + center

    out: List[ChristmasTreeFix] = []
    for i, t in enumerate(trees):
        out.append(ChristmasTreeFix(Decimal(rotated[i, 0]), Decimal(rotated[i, 1]), Decimal(t.angle) + Decimal(str(angle_deg))))
    return out


def fix_direction(submission_csv: Path) -> Tuple[Decimal, Decimal]:
    dict_tree, dict_side = _parse_submission_for_fix(submission_csv)
    cur = _total_score_from_side_lengths(dict_side)

    for n in range(200, 2, -1):
        gid = f"{n:03d}"
        if gid not in dict_tree:
            continue
        trees = dict_tree[gid]
        best_side, best_angle = _optimize_rotation_angle(trees)
        if best_side < dict_side[gid]:
            dict_tree[gid] = _apply_rotation_to_solution(trees, best_angle)
            dict_side[gid] = best_side

    new = _total_score_from_side_lengths(dict_side)

    if (cur - new) > 0:
        rows = []
        for gid, trees in dict_tree.items():
            for item_id, t in enumerate(trees):
                rows.append({"id": f"{gid}_{item_id}", "x": f"s{t.center_x}", "y": f"s{t.center_y}", "deg": f"s{t.angle}"})
        pd.DataFrame(rows).to_csv(submission_csv, index=False)

    return cur, new


# =============================================================================
# 2) スコア＆重なり
# =============================================================================

SCALE_VALIDATE = Decimal("1e20")


class ChristmasTreeVal:
    def __init__(self, center_x="0", center_y="0", angle="0"):
        self.center_x = Decimal(str(center_x))
        self.center_y = Decimal(str(center_y))
        self.angle = Decimal(str(angle))

        trunk_w = Decimal("0.15")
        trunk_h = Decimal("0.2")
        base_w = Decimal("0.7")
        mid_w = Decimal("0.4")
        top_w = Decimal("0.25")
        tip_y = Decimal("0.8")
        tier_1_y = Decimal("0.5")
        tier_2_y = Decimal("0.25")
        base_y = Decimal("0.0")
        trunk_bottom_y = -trunk_h

        initial_polygon = Polygon(
            [
                (Decimal("0.0") * SCALE_VALIDATE, tip_y * SCALE_VALIDATE),
                (top_w / Decimal("2") * SCALE_VALIDATE, tier_1_y * SCALE_VALIDATE),
                (top_w / Decimal("4") * SCALE_VALIDATE, tier_1_y * SCALE_VALIDATE),
                (mid_w / Decimal("2") * SCALE_VALIDATE, tier_2_y * SCALE_VALIDATE),
                (mid_w / Decimal("4") * SCALE_VALIDATE, tier_2_y * SCALE_VALIDATE),
                (base_w / Decimal("2") * SCALE_VALIDATE, base_y * SCALE_VALIDATE),
                (trunk_w / Decimal("2") * SCALE_VALIDATE, base_y * SCALE_VALIDATE),
                (trunk_w / Decimal("2") * SCALE_VALIDATE, trunk_bottom_y * SCALE_VALIDATE),
                (-(trunk_w / Decimal("2")) * SCALE_VALIDATE, trunk_bottom_y * SCALE_VALIDATE),
                (-(trunk_w / Decimal("2")) * SCALE_VALIDATE, base_y * SCALE_VALIDATE),
                (-(base_w / Decimal("2")) * SCALE_VALIDATE, base_y * SCALE_VALIDATE),
                (-(mid_w / Decimal("4")) * SCALE_VALIDATE, tier_2_y * SCALE_VALIDATE),
                (-(mid_w / Decimal("2")) * SCALE_VALIDATE, tier_2_y * SCALE_VALIDATE),
                (-(top_w / Decimal("4")) * SCALE_VALIDATE, tier_1_y * SCALE_VALIDATE),
                (-(top_w / Decimal("2")) * SCALE_VALIDATE, tier_1_y * SCALE_VALIDATE),
            ]
        )
        rotated = affinity.rotate(initial_polygon, float(self.angle), origin=(0, 0))
        self.polygon = affinity.translate(rotated, xoff=float(self.center_x * SCALE_VALIDATE), yoff=float(self.center_y * SCALE_VALIDATE))


def _load_trees_for_n(df: pd.DataFrame, n: int) -> List[ChristmasTreeVal]:
    prefix = f"{n:03d}_"
    g = df[df["id"].astype(str).str.startswith(prefix)]
    out: List[ChristmasTreeVal] = []
    for _, row in g.iterrows():
        x = str(row["x"]).lstrip("s")
        y = str(row["y"]).lstrip("s")
        deg = str(row["deg"]).lstrip("s")
        if x and y and deg:
            out.append(ChristmasTreeVal(x, y, deg))
    return out


def _score_for_n(trees: List[ChristmasTreeVal], n: int) -> float:
    xys = np.concatenate([np.asarray(t.polygon.exterior.xy).T / float(SCALE_VALIDATE) for t in trees])
    min_x, min_y = xys.min(axis=0)
    max_x, max_y = xys.max(axis=0)
    side = max(max_x - min_x, max_y - min_y)
    return float(side**2 / n)


def _has_overlap(trees: List[ChristmasTreeVal]) -> bool:
    if len(trees) <= 1:
        return False
    polys = [t.polygon for t in trees]
    idx = STRtree(polys)
    for i, p in enumerate(polys):
        cand = idx.query(p)
        for j in cand:
            if j == i:
                continue
            if p.intersects(polys[j]) and not p.touches(polys[j]):
                return True
    return False


def score_submission_only(submission_csv: Path, max_n: int = 200) -> float:
    df = pd.read_csv(submission_csv)
    total = 0.0
    for n in range(1, max_n + 1):
        trees = _load_trees_for_n(df, n)
        if trees:
            total += _score_for_n(trees, n)
    return total


def score_and_validate_submission(submission_csv: Path, max_n: int = 200) -> dict:
    df = pd.read_csv(submission_csv)
    total = 0.0
    failed = []
    for n in range(1, max_n + 1):
        trees = _load_trees_for_n(df, n)
        if trees:
            total += _score_for_n(trees, n)
            if _has_overlap(trees):
                failed.append(n)
    return {"total_score": total, "failed_overlap_n": failed}


# =============================================================================
# 3) donor差し替え
# =============================================================================

def _load_groups_csv(filename: Path):
    groups = {}
    with open(filename, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            gid = row[0].split("_")[0]
            groups.setdefault(gid, []).append(row)
    return header, groups


def replace_group(target_file: Path, donor_file: Path, group_id: str, output_file: Path | None = None):
    if output_file is None:
        output_file = target_file

    header_t, groups_t = _load_groups_csv(target_file)
    _, groups_d = _load_groups_csv(donor_file)

    if group_id not in groups_d:
        raise ValueError(f"donorに group {group_id} がありません: {group_id}")

    groups_t[group_id] = groups_d[group_id]

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header_t)
        for g in sorted(groups_t.keys(), key=lambda x: int(x)):
            for row in groups_t[g]:
                w.writerow(row)


# =============================================================================
# 4) bbox3 ループ
# =============================================================================

@dataclass
class GridSpec:
    n_min: int
    n_max: int
    n_step: int
    r_min: int
    r_max: int
    r_step: int


def run_bbox_loop(log, use_wsl: bool, bbox_exe: Path, workdir: Path, max_hours: float, timeout_sec: int, grid: GridSpec):
    start = datetime.now()
    limit = timedelta(hours=max_hours)

    sub_csv = workdir / "submission.csv"
    (workdir / "bbox_sub").mkdir(exist_ok=True)

    params = [(n, r)
              for r in range(grid.r_min, grid.r_max + 1, grid.r_step)
              for n in range(grid.n_min, grid.n_max + 1, grid.n_step)]

    best_score = None
    best_path = workdir / "best_submission.csv"

    try:
        s0 = score_submission_only(sub_csv, 200)
        log(f"initial score={s0:.14f}")
        best_score = s0
        if KEEP_BEST_SUBMISSION:
            shutil.copy2(sub_csv, best_path)
            log("saved initial best_submission.csv")
    except Exception as e:
        log("WARN initial score calc failed:", e)
        log(traceback.format_exc())

    epoch = 0
    run_i = 0

    while datetime.now() - start < limit:
        epoch += 1
        if SHUFFLE_PARAMS_EACH_EPOCH:
            np.random.shuffle(params)

        for (n, r) in params:
            elapsed = datetime.now() - start
            if elapsed > limit:
                log(f"TIMEOUT total. elapsed={elapsed}")
                return

            run_i += 1
            log(f"RUN {run_i} epoch={epoch} elapsed={elapsed} n={n} r={r}")

            before_mtime = sub_csv.stat().st_mtime
            before_size = sub_csv.stat().st_size

            try:
                rc, out, err = run_bbox3(log, use_wsl, bbox_exe, workdir, n, r, timeout_sec)
                log("bbox3 rc=", rc)
                if out:
                    log("bbox3 stdout:", out.strip()[:2000])
                if err:
                    log("bbox3 stderr:", err.strip()[:2000])
            except subprocess.TimeoutExpired:
                log(f"WARN bbox3 timeout {timeout_sec}s n={n} r={r}")
                if DEBUG_ONCE:
                    return
                continue
            except Exception as e:
                log("ERROR bbox3 exception:", repr(e))
                log(traceback.format_exc())
                if DEBUG_ONCE:
                    return
                continue

            after_mtime = sub_csv.stat().st_mtime
            after_size = sub_csv.stat().st_size
            if after_mtime == before_mtime and after_size == before_size:
                log("WARN submission.csv not updated (mtime/size unchanged). bbox3即死/別出力/権限等の可能性。")

            snap = workdir / "bbox_sub" / f"submi-n{n}_r{r}_i{run_i}.csv"
            try:
                shutil.copy2(sub_csv, snap)
                log("saved snapshot:", snap.name)
            except Exception as e:
                log("WARN snapshot save failed:", e)

            if FIX_EACH_RUN:
                try:
                    a, b = fix_direction(sub_csv)
                    log(f"fix_direction: {float(a):.12f} -> {float(b):.12f}")
                except Exception as e:
                    log("WARN fix_direction failed:", e)

            if PRINT_SCORE_EACH_RUN and (run_i % SCORE_EVERY_K == 0):
                try:
                    s = score_submission_only(sub_csv, 200)
                    log(f"score={s:.14f}")
                    if best_score is None or s < best_score:
                        best_score = s
                        if KEEP_BEST_SUBMISSION:
                            shutil.copy2(sub_csv, best_path)
                        log(f"BEST UPDATED best={best_score:.14f}")
                except Exception as e:
                    log("WARN score calc failed:", e)

            if DEBUG_ONCE:
                return


# =============================================================================
# 5) ZIP
# =============================================================================

def make_zip(workdir: Path) -> Path:
    from zipfile import ZipFile, ZIP_DEFLATED
    ts = datetime.now().strftime("%Y-%m-%d-%H-%M")
    zip_path = workdir / f"kaggle_bbox_{ts}.zip"
    files = []
    files += list(workdir.glob("*.csv"))
    files += list(workdir.glob("*.log"))
    files += list((workdir / "bbox_sub").glob("*.csv"))
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED, compresslevel=9) as z:
        for p in files:
            z.write(p, arcname=str(p.relative_to(workdir)))
    return zip_path


# =============================================================================
# main
# =============================================================================

def main():
    workdir = Path(WORKDIR).expanduser().resolve()
    log_path = workdir / "run.log"
    log, close = make_logger(log_path)

    try:
        log("=== START run_santa_bbox_local.py ===")
        log("python:", sys.version.replace("\n", " "))
        log("cwd:", Path.cwd())
        log("WORKDIR:", workdir)
        log("DONOR_CSV:", DONOR_CSV)
        log("BBOX_EXE:", BBOX_EXE)

        donor_csv = Path(DONOR_CSV).expanduser().resolve()
        bbox_exe = Path(BBOX_EXE).expanduser().resolve()

        if not donor_csv.exists():
            log("FATAL DONOR_CSV not found:", donor_csv)
            return
        if not bbox_exe.exists():
            log("FATAL BBOX_EXE not found:", bbox_exe)
            return

        workdir.mkdir(parents=True, exist_ok=True)
        sub_csv = workdir / "submission.csv"

        if START_FROM_DONOR:
            shutil.copy2(donor_csv, sub_csv)
            log("copied donor ->", sub_csv)

        use_wsl = False
        if is_windows() and AUTO_USE_WSL:
            ok = bbox3_is_windows_exe_runnable(log, bbox_exe, workdir)
            if not ok:
                log("bbox3 is not runnable on Windows -> use WSL")
                use_wsl = True

        if use_wsl:
            # まず WSL が動くか最小コマンドで確認（ここが通らないなら WSL 自体が壊れてる）
            res = wsl_run_bash(log, "echo WSL_OK; uname -a; id; pwd", timeout=30, cwd_win=None)
            if res.returncode != 0:
                log("FATAL WSL bootstrap failed. WSL 자체が壊れている可能性。")
                log("Try running in PowerShell: wsl -e bash -lc \"echo ok\"")
                return

            ensure_wsl_deps(log)

            # 共有ドライブが見えているか確認
            wsl_run_bash(log, f"ls -la {wsl_path(workdir)} || true", timeout=30)

        grid = GridSpec(n_min=N_MIN, n_max=N_MAX, n_step=N_STEP, r_min=R_MIN, r_max=R_MAX, r_step=R_STEP)

        run_bbox_loop(log, use_wsl, bbox_exe, workdir, MAX_HOURS, PER_RUN_TIMEOUT_SEC, grid)

        best_path = workdir / "best_submission.csv"
        if USE_BEST_AT_END and KEEP_BEST_SUBMISSION and best_path.exists():
            shutil.copy2(best_path, sub_csv)
            log("use best_submission.csv as final:", best_path)

        try:
            s_final = score_submission_only(sub_csv, 200)
            log(f"FINAL score={s_final:.14f}")
        except Exception as e:
            log("WARN final score calc failed:", e)

        if FINAL_VALIDATE_OVERLAP:
            try:
                val = score_and_validate_submission(sub_csv, 200)
                log(f"FINAL validate score={val['total_score']:.14f} overlaps={val['failed_overlap_n']}")
                if val["failed_overlap_n"]:
                    for n in val["failed_overlap_n"]:
                        gid = f"{n:03d}"
                        replace_group(sub_csv, donor_csv, gid, output_file=sub_csv)
                        log("repaired group:", gid)
                    val2 = score_and_validate_submission(sub_csv, 200)
                    log(f"AFTER repair score={val2['total_score']:.14f} overlaps={val2['failed_overlap_n']}")
            except Exception as e:
                log("WARN validation failed:", e)

        if MAKE_ZIP:
            try:
                zp = make_zip(workdir)
                log("ZIP created:", zp)
            except Exception as e:
                log("WARN zip failed:", e)

        log("=== END ===")

    except Exception as e:
        log("FATAL unhandled exception:", repr(e))
        log(traceback.format_exc())
    finally:
        close()


if __name__ == "__main__":
    main()

# _nb4d_paths.py
#
# NB4D 共通スイープパスライブラリ (step1/step2/step3 共用)
#
# 【角度の定義】(theta_norm 0.0〜1.0 の正規化値)
#   0.000 = 正面
#   0.125 = 右斜め 45°
#   0.250 = 右横  90°
#   0.375 = 右後ろ 135°
#   0.500 = 後ろ  180°
#   0.625 = 左後ろ 225°
#   0.750 = 左横  270°
#   0.875 = 左斜め 315°
#
# 【パス関数の仕様】
#   入力 p : 進行度 float [0.0, 1.0]
#   出力   : theta_norm float [0.0, 1.0)  (0.0 = 正面, 右回り正)
#
#   time軸 (t_norm) は常に p に等しい（線形進行）と定義する。
#   step1/2 (Grid4DTraverse) は t_norm * (GRID_T-1) で実際の t に変換。
#   step3   (LTXVInterpolatorV3) は stage_idx が時間軸なので t_norm は無視。
#
# 【使い方】
#   from nodes._nb4d_paths import SWEEP_PATHS, theta_for_stage
#
#   # step3 用: stage_idx → theta_index (int)
#   theta_idx = theta_for_stage("orbit_right_half", stage_idx, n_stages, n_angles)
#
#   # step1/2 用: _get_path_func に追加するラッパー
#   theta_float = SWEEP_PATHS["orbit_right_half"](i / N) * grid_theta

import math

# ── Phase 1: 水平スイープパス群 ─────────────────────────────────────────────────

def _pendulum(p: float) -> float:
    """
    1. 振り子: 右斜め45°→正面→左斜め45°
    始まりは右45°、中間で正面を向き、最終で左45°
    """
    # theta: 0.125 → 0.0 → -0.125 (= 0.875)
    return (0.125 - 0.25 * p) % 1.0


def _orbit_right_half(p: float) -> float:
    """
    2. 右回り半周: 正面→右横→後ろ
    正面(0°) から右回りで後ろ(180°)まで
    """
    return p * 0.5


def _orbit_left_half(p: float) -> float:
    """
    3. 左回り半周: 正面→左横→後ろ
    正面(0°) から左回りで後ろ(180°)まで
    """
    return (1.0 - p * 0.5) % 1.0


def _orbit_right_full(p: float) -> float:
    """
    5. 右回り一周: 正面→右横→後ろ→左横→正面
    """
    return p % 1.0


def _orbit_left_full(p: float) -> float:
    """
    6. 左回り一周: 正面→左横→後ろ→右横→正面
    """
    return (1.0 - p) % 1.0


# ── Phase 2 placeholder (仰角グリッド実装後に追加) ────────────────────────────
# def _elevation_rise(p):   ...  # 4. 正面→真上→後ろ
# def _elevation_dive(p):   ...  # 7. 正面→真下→後ろ


# ── 公開辞書 ────────────────────────────────────────────────────────────────────

SWEEP_PATHS: dict = {
    "pendulum":         _pendulum,         # 1. 右45→正面→左45
    "orbit_right_half": _orbit_right_half, # 2. 正面→右横→後ろ
    "orbit_left_half":  _orbit_left_half,  # 3. 正面→左横→後ろ
    "orbit_right_full": _orbit_right_full, # 5. 正面→後ろ→正面（右回り）
    "orbit_left_full":  _orbit_left_full,  # 6. 正面→後ろ→正面（左回り）
}

SWEEP_PATH_NAMES = ["none"] + list(SWEEP_PATHS.keys())

# 日本語ラベル（UIヒント用）
SWEEP_PATH_LABELS = {
    "none":             "none (固定角度)",
    "pendulum":         "pendulum (右45→正面→左45)",
    "orbit_right_half": "orbit_right_half (正面→右横→後ろ)",
    "orbit_left_half":  "orbit_left_half (正面→左横→後ろ)",
    "orbit_right_full": "orbit_right_full (正面→後ろ→正面 右回り)",
    "orbit_left_full":  "orbit_left_full (正面→後ろ→正面 左回り)",
}


# ── ヘルパー関数 ────────────────────────────────────────────────────────────────

def theta_for_stage(
    path_name: str,
    stage_idx: int,
    n_stages: int,
    n_angles: int,
    fixed_angle: int = 0,
) -> int:
    """
    step3 (LTXVInterpolatorV3) 用:
    stage_idx → theta_index (int, 0 〜 n_angles-1)

    path_name="none" の場合は fixed_angle をそのまま返す。
    """
    if path_name == "none" or path_name not in SWEEP_PATHS:
        return int(fixed_angle) % n_angles

    p = stage_idx / max(n_stages - 1, 1)
    theta_norm = SWEEP_PATHS[path_name](p)
    return int(round(theta_norm * n_angles)) % n_angles


def sweep_lambda_for_grid(path_name: str, grid_t: float, grid_theta: int):
    """
    step1/2 (Grid4DTraverse) 用:
    _get_path_func に渡す (i, N) → (t, theta) ラムダを返す。

    既存の _get_path_func の funcs 辞書に追加するために使う。
    """
    fn = SWEEP_PATHS[path_name]

    def path_func(i, N):
        p = i / N
        t = p * (grid_t - 1)
        theta = fn(p) * grid_theta
        return (float(t), float(theta % grid_theta))

    return path_func

# _nb4d_paths.py
#
# NB4D 共通パスライブラリ (step1/step2/step3/PathNavigator 共用)
#
# 【格子空間の定義】
#   t軸: 時間 (0 〜 n_stages-1)
#   h軸: 水平角 (0 〜 grid_theta-1)  0=正面, grid_theta/4=右横, grid_theta/2=後ろ
#   v軸: 仰角  (0 〜 grid_elev-1)   デフォルト7段階: -30,-15,0,+15,+30,+45,+60°
#
# 【正規化値の定義】
#   t_norm, h_norm, v_norm: すべて [0.0, 1.0]
#   v_norm=0.0 → v=0 (-30°)
#   v_norm=2/6≈0.333 → v=2 (0°, 水平)  ← 「普通の目線」
#   v_norm=1.0 → v=6 (+60°, 見下ろし)
#
# 【スイープパス】(Grid4DTraverse / LTXVInterpolatorV3 用)
#   p (0→1) → (t_norm, h_norm, v_norm)
#   step3のsweepでは t_norm は無視（stage_idxが時間軸）
#
# 【ナビゲータープリセット】(LTXVPathNavigator 用)
#   名前 → [(t_norm, h_norm, v_norm), ...] のウェイポイントリスト
#   各ウェイポイント間でLTX-2が1クリップ生成

import math
import numpy as np

# ── 定数 ────────────────────────────────────────────────────────────────────────
V_HORIZONTAL = 2 / 6  # v_norm for 0° elevation (v=2 in 7-level grid)

# ── Phase 1: 水平スイープパス群（v固定） ────────────────────────────────────────

def _pendulum(p: float):
    return (p, (0.125 - 0.25 * p) % 1.0, V_HORIZONTAL)

def _orbit_right_half(p: float):
    return (p, p * 0.5, V_HORIZONTAL)

def _orbit_left_half(p: float):
    return (p, (1.0 - p * 0.5) % 1.0, V_HORIZONTAL)

def _orbit_right_full(p: float):
    return (p, p % 1.0, V_HORIZONTAL)

def _orbit_left_full(p: float):
    return (p, (1.0 - p) % 1.0, V_HORIZONTAL)

# ── 公開辞書: スイープパス ──────────────────────────────────────────────────────

SWEEP_PATHS: dict = {
    "pendulum":         _pendulum,
    "orbit_right_half": _orbit_right_half,
    "orbit_left_half":  _orbit_left_half,
    "orbit_right_full": _orbit_right_full,
    "orbit_left_full":  _orbit_left_full,
}

SWEEP_PATH_NAMES = ["none"] + list(SWEEP_PATHS.keys())

SWEEP_PATH_LABELS = {
    "none":             "none (固定角度)",
    "pendulum":         "pendulum (右45→正面→左45)",
    "orbit_right_half": "orbit_right_half (正面→右横→後ろ)",
    "orbit_left_half":  "orbit_left_half (正面→左横→後ろ)",
    "orbit_right_full": "orbit_right_full (正面→後ろ→正面 右回り)",
    "orbit_left_full":  "orbit_left_full (正面→後ろ→正面 左回り)",
}

# ── ナビゲータープリセット: ウェイポイントリスト ────────────────────────────────
#
# 各エントリ: (t_norm, h_norm, v_norm)
#   t_norm: 時間 [0,1]
#   h_norm: 水平角 [0,1)  0=正面, 0.25=右横, 0.5=後ろ, 0.75=左横
#   v_norm: 仰角 [0,1]   0=-30°, 0.333=0°(水平), 1.0=+60°
#
# ウェイポイント間は LTX-2 が1クリップ生成（隣接が推奨だが強制ではない）

_H = V_HORIZONTAL  # 0° elevation shorthand

NAVIGATOR_PRESETS: dict = {

    # 1. 右回り一周しながら時間進行（水平）
    "orbit_flat": [
        (0.0,   0.0,  _H),
        (0.25,  0.25, _H),
        (0.5,   0.5,  _H),
        (0.75,  0.75, _H),
        (1.0,   0.0,  _H),
    ],

    # 2. 右螺旋上昇: 正面・水平から右回りで見下ろし角度へ
    "spiral_ascend": [
        (0.0,   0.0,  _H),
        (0.33,  0.25, 0.5),
        (0.67,  0.5,  0.833),
        (1.0,   0.75, 1.0),
    ],

    # 3. 右螺旋下降: 見下ろしから右回りで水平へ
    "spiral_descend": [
        (0.0,   0.0,  1.0),
        (0.33,  0.25, 0.667),
        (0.67,  0.5,  0.333),
        (1.0,   0.75, _H),
    ],

    # 4. 真上からの鳥瞰→水平へ降下（正面固定）
    "top_survey": [
        (0.0,  0.0, 1.0),
        (0.5,  0.0, 0.667),
        (1.0,  0.0, _H),
    ],

    # 5. 仰角振り子（h固定、vが水平↔見下ろし↔水平）
    "pendulum_elevation": [
        (0.0,  0.0, _H),
        (0.5,  0.0, 1.0),
        (1.0,  0.0, _H),
    ],

    # 6. 時間往復（右回り軌道で時間が行って帰る）
    "time_loop": [
        (0.0,  0.0,  _H),
        (0.5,  0.25, _H),
        (1.0,  0.5,  _H),
        (0.5,  0.75, _H),
        (0.0,  0.0,  _H),
    ],

    # 7. 地面レベルから空へ（低仰角→高仰角、後ろへ回り込む）
    "ground_to_sky": [
        (0.0,  0.0, 0.0),
        (0.5,  0.25, 0.5),
        (1.0,  0.5, 1.0),
    ],
}

NAVIGATOR_PRESET_NAMES = ["custom_text"] + list(NAVIGATOR_PRESETS.keys())

NAVIGATOR_PRESET_LABELS = {
    "custom_text":          "custom_text (テキスト入力)",
    "orbit_flat":           "orbit_flat: 右回り一周・水平・時間進行",
    "spiral_ascend":        "spiral_ascend: 右螺旋上昇（水平→鳥瞰）",
    "spiral_descend":       "spiral_descend: 右螺旋下降（鳥瞰→水平）",
    "top_survey":           "top_survey: 真上→水平へ降下（正面固定）",
    "pendulum_elevation":   "pendulum_elevation: 仰角振り子（h固定）",
    "time_loop":            "time_loop: 時間往復（行って帰る）",
    "ground_to_sky":        "ground_to_sky: 地面→空へ上昇",
}

# ── ヘルパー関数 ────────────────────────────────────────────────────────────────

def theta_for_stage(
    path_name: str,
    stage_idx: int,
    n_stages: int,
    n_angles: int,
    fixed_angle: int = 0,
) -> int:
    """step3 (LTXVInterpolatorV3) 用: stage_idx → theta_index"""
    if path_name == "none" or path_name not in SWEEP_PATHS:
        return int(fixed_angle) % n_angles
    p = stage_idx / max(n_stages - 1, 1)
    _, h_norm, _ = SWEEP_PATHS[path_name](p)
    return int(round(h_norm * n_angles)) % n_angles


def sweep_lambda_for_grid(path_name: str, grid_t: float, grid_theta: int):
    """step1/2 (Grid4DTraverse) 用: (i,N)→(t,theta) ラムダ生成"""
    fn = SWEEP_PATHS[path_name]
    def path_func(i, N):
        p = i / N
        t_norm, h_norm, _ = fn(p)
        t = t_norm * (grid_t - 1)
        theta = h_norm * grid_theta
        return (float(t), float(theta % grid_theta))
    return path_func


def parse_path_text(text: str, n_stages: int, grid_theta: int, grid_elev: int) -> list:
    """
    テキスト入力をウェイポイントリスト [(t_idx, h_idx, v_idx), ...] に変換。

    サポート形式:
      整数インデックス: "0, 0, 2"
      正規化float:     "0.0, 0.0, 0.333"
      コメント行: # で始まる行は無視
    """
    waypoints = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.replace(";", ",").split(",")]
        if len(parts) < 3:
            continue
        try:
            vals = [float(p) for p in parts[:3]]
        except ValueError:
            continue

        # float か int かを判定（すべて 0-1 の範囲なら正規化値として扱う）
        if all(0.0 <= v <= 1.0 for v in vals) and any("." in p for p in parts[:3]):
            # 正規化値 → インデックス変換
            t_idx = int(round(vals[0] * (n_stages - 1)))
            h_idx = int(round(vals[1] * (grid_theta - 1))) % grid_theta
            v_idx = int(round(vals[2] * (grid_elev - 1)))
        else:
            t_idx = int(vals[0])
            h_idx = int(vals[1]) % grid_theta
            v_idx = int(vals[2])

        t_idx = max(0, min(t_idx, n_stages - 1))
        v_idx = max(0, min(v_idx, grid_elev - 1))
        waypoints.append((t_idx, h_idx, v_idx))

    return waypoints


def preset_to_waypoints(preset_name: str, n_stages: int, grid_theta: int, grid_elev: int) -> list:
    """プリセット名 → [(t_idx, h_idx, v_idx), ...] に変換"""
    if preset_name not in NAVIGATOR_PRESETS:
        return []
    result = []
    for t_norm, h_norm, v_norm in NAVIGATOR_PRESETS[preset_name]:
        t_idx = int(round(t_norm * (n_stages - 1)))
        h_idx = int(round(h_norm * (grid_theta - 1))) % grid_theta
        v_idx = int(round(v_norm * (grid_elev - 1)))
        t_idx = max(0, min(t_idx, n_stages - 1))
        v_idx = max(0, min(v_idx, grid_elev - 1))
        result.append((t_idx, h_idx, v_idx))
    return result


def check_adjacency(waypoints: list) -> list:
    """
    隣接制約チェック。
    各ステップで |dt|>1 or |dh_min|>1 or |dv|>1 の場合に警告メッセージを返す。
    dh_min: 水平は循環するので min(|dh|, grid_theta-|dh|) を使う想定だが、
            ここでは単純に raw diff を返す（呼び出し側で grid_theta を考慮）。
    """
    warnings = []
    for i in range(len(waypoints) - 1):
        t0, h0, v0 = waypoints[i]
        t1, h1, v1 = waypoints[i + 1]
        dt = abs(t1 - t0)
        dh = abs(h1 - h0)
        dv = abs(v1 - v0)
        if dt > 1 or dh > 1 or dv > 1:
            warnings.append(
                f"  ステップ{i}→{i+1}: (t={t0},h={h0},v={v0})→(t={t1},h={h1},v={v1})"
                f"  Δ=({dt},{dh},{dv}) > 1 → 補間品質低下の可能性"
            )
    return warnings

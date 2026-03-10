#!/usr/bin/env python3
"""
render_bg_animation.py
======================
4Dタイムラプスの走査パスに同期した背景パン動画を生成する。

カメラがtheta回転するとき、背景画像も同じ方向に少しスクロールすることで
「カメラが本当に飛んでいる」感が出る（視差効果）。

PowerDirectorでの使い方:
  1. export_transparent_video.py → transparent_bee_spiral_in_*.webm （透明トマト）
  2. render_bg_animation.py      → bg_bee_spiral_in_*.mp4           （同期背景）
  3. PowerDirector:
       Track 1 (下): bg_*.mp4
       Track 2 (上): transparent_*.webm
     → 自動で同期する（同じframes/fpsで生成しているため）

パラメータ:
  --parallax    背景のパン量（0.0=固定, 0.2=20%スクロール, 1.0=背景1周）
                ミツバチ視点なら 0.15〜0.3 が自然
  --zoom_range  ズーム幅（0.0=ズームなし, 0.1=10%ズームイン/アウト）
                「花に近づく」感を出す

使い方:
  python render_bg_animation.py \
    --keyframes_dir "D:/NB4D_test/tulip/grid_keyframes/run_XXXXXXXX" \
    --bg_image "D:/NB4D_test/tulip/bg_garden.jpg" \
    --path bee_spiral_in \
    --frames 720 \
    --parallax 0.2 \
    --zoom_range 0.15 \
    --output_dir "D:/NB4D_test/tulip/output"
"""

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

GRID_T = 120

FFMPEG_CANDIDATES = [
    r"C:\Users\fareg\AppData\Local\FFmpeg\bin\ffmpeg.exe",
    "ffmpeg",
]


def get_ffmpeg():
    for p in FFMPEG_CANDIDATES:
        try:
            subprocess.run([p, "-version"], capture_output=True, check=True)
            return p
        except Exception:
            pass
    raise RuntimeError("ffmpeg が見つかりません")


def _make_paths():
    rng     = np.random.default_rng(42)
    _jitter = rng.uniform(-0.15, 0.15, 2000)

    # ── 猫視点パス群 ────────────────────────────────────────────────────────────

    def cat_double_arc(i, N, GT):
        p = i / N
        t = float(np.clip(p * (GRID_T - 1), 0, GRID_T - 1))
        if p < 0.5:
            theta = p / 0.5 * (GT * 0.5)
        else:
            theta = (1.0 - (p - 0.5) / 0.5) * (GT * 0.5)
        return (t, float(theta))

    def cat_young_linger(i, N, GT):
        p = i / N
        if p < 0.7:
            t = float(np.clip(p / 0.7 * (GRID_T - 1) * 0.4, 0, GRID_T - 1))
        else:
            t = float(np.clip((GRID_T - 1) * 0.4 + (p - 0.7) / 0.3 * (GRID_T - 1) * 0.6, 0, GRID_T - 1))
        theta = float(p * GT % GT)
        return (t, theta)

    def cat_waltz(i, N, GT):
        p = i / N
        t = float(np.clip(p * (GRID_T - 1), 0, GRID_T - 1))
        center = p * GT * 0.25
        swing  = GT * 0.10 * np.sin(2 * np.pi * p * 3)
        theta  = float((center + swing) % GT)
        return (t, theta)

    def bee_hover(i, N, GT):
        p     = i / N
        t     = float(np.clip(p * (GRID_T - 1) + (GRID_T - 1) * 0.06 * np.sin(2 * np.pi * i * 7 / N), 0, GRID_T - 1))
        theta = ((i * GT / N) + _jitter[i % len(_jitter)] * GT) % GT
        return (t, theta)

    def bee_spiral_in(i, N, GT):
        p = i / N
        t = float(np.clip(p * (GRID_T - 1), 0, GRID_T - 1))
        theta = (p / 0.7) * GT * 3.0 % GT if p < 0.7 else ((p - 0.7) / 0.3) * GT % GT
        return (t, float(theta))

    def bee_inspect(i, N, GT):
        p     = i / N
        t     = float(np.clip(p * (GRID_T - 1), 0, GRID_T - 1))
        phase = (p * 4) % 1.0
        if   phase < 0.3: theta = phase / 0.3 * 0.15
        elif phase < 0.6: theta = 0.15 + (phase - 0.3) / 0.3 * 0.5
        elif phase < 0.8: theta = 0.65 + (phase - 0.6) / 0.2 * 0.2
        else:             theta = 0.85 + (phase - 0.8) / 0.2 * 0.15
        return (t, float((theta * GT) % GT))

    return {
        "diagonal":      lambda i, N, GT: (i * (GRID_T - 1) / N, i * GT / N % GT),
        "spiral_2x":     lambda i, N, GT: (i * (GRID_T - 1) / N, i * GT * 2 / N % GT),
        "orbit_green":   lambda i, N, GT: (0.0, i * GT / N % GT),
        "orbit_ripe":    lambda i, N, GT: (GRID_T - 1, i * GT / N % GT),
        "ripen_front":   lambda i, N, GT: (i * (GRID_T - 1) / N, 0.0),
        "reverse_diag":  lambda i, N, GT: ((GRID_T - 1) - i * (GRID_T - 1) / N, i * GT / N % GT),
        "time_wave":     lambda i, N, GT: (
            (GRID_T - 1) / 2 * (1 + np.sin(4 * np.pi * i / N)),
            i * GT / N % GT,
        ),
        "zoom_in_time":  lambda i, N, GT: (
            i * (GRID_T - 1) / N,
            i * GT / N % GT if i < N * 0.8 else (GRID_T - 1) * 0.8 + (i - N * 0.8) * 0.2 * (GRID_T - 1) / (N * 0.2),
        ),
        "bee_hover":       bee_hover,
        "bee_spiral_in":   bee_spiral_in,
        "bee_inspect":     bee_inspect,
        # ── 猫視点 7パス ─────────────────────────────────────────────────────────
        "cat_passthrough":  lambda i, N, GT: (
            float(np.clip(i * (GRID_T - 1) / N, 0, GRID_T - 1)),
            float(i / N * (GT * 0.5)),
        ),
        "cat_reverse_pass": lambda i, N, GT: (
            float(np.clip(i * (GRID_T - 1) / N, 0, GRID_T - 1)),
            float((1.0 - i / N) * (GT * 0.5)),
        ),
        "cat_circle_age":   lambda i, N, GT: (
            float(np.clip(i * (GRID_T - 1) / N, 0, GRID_T - 1)),
            float(i * GT / N % GT),
        ),
        "cat_double_arc":   cat_double_arc,
        "cat_young_linger": cat_young_linger,
        "cat_side_age":     lambda i, N, GT: (
            float(np.clip(i * (GRID_T - 1) / N, 0, GRID_T - 1)),
            float(GT * 0.25),
        ),
        "cat_waltz":        cat_waltz,
    }

PATHS = _make_paths()


def render_bg_frame(
    bg_rgb: np.ndarray,
    out_w: int,
    out_h: int,
    theta_norm: float,   # 0.0〜1.0（0=正面, 1.0=1周）
    t_norm: float,       # 0.0〜1.0（0=最初のstage, 1.0=最後のstage）
    parallax: float,
    zoom_range: float,
) -> np.ndarray:
    """
    背景画像をパン・ズームして out_w × out_h のフレームを返す。

    theta_norm → 左右パン（カメラ回転と同期）
    t_norm     → ズーム（成長に連れてカメラが少し近づく感）
    parallax   → パン量（背景幅に対する比率）
    zoom_range → ズーム変動幅
    """
    H_bg, W_bg = bg_rgb.shape[:2]

    # ズーム率: t=0 → (1+zoom_range), t=1 → 1.0（近づくほど引き）
    zoom = 1.0 + zoom_range * (1.0 - t_norm)

    # クロップサイズ
    crop_w = int(out_w / zoom)
    crop_h = int(out_h / zoom)

    # パン: theta_norm に比例して横方向にスクロール
    # 背景がちょうど1周するよりずっと少ない量で自然な視差感
    max_pan = int(W_bg * parallax)
    pan_x   = int(theta_norm * max_pan)

    # タイル対応: 背景幅が足りない場合は横に繰り返す
    if W_bg < crop_w + max_pan + 10:
        bg_tiled = np.tile(bg_rgb, (1, 3, 1))  # 3倍横に並べる
        W_bg_t   = bg_tiled.shape[1]
        offset_x = W_bg  # 中央タイルから始める
    else:
        bg_tiled = bg_rgb
        W_bg_t   = W_bg
        offset_x = 0

    # クロップ位置（中央 + パン）
    cx = W_bg_t // 2 + pan_x - offset_x
    cy = bg_tiled.shape[0] // 2

    x0 = max(0, cx - crop_w // 2)
    y0 = max(0, cy - crop_h // 2)
    x1 = min(W_bg_t, x0 + crop_w)
    y1 = min(bg_tiled.shape[0], y0 + crop_h)

    cropped = bg_tiled[y0:y1, x0:x1]

    # out_w × out_h にリサイズ
    frame = cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
    return frame


def main():
    ap = argparse.ArgumentParser(description="走査パスに同期した背景パン動画を生成")
    ap.add_argument("--keyframes_dir", required=True, help="grid_meta.json があるディレクトリ")
    ap.add_argument("--bg_image",      required=True, help="背景画像（JPG/PNG）")
    ap.add_argument("--path",    default="diagonal", choices=list(PATHS.keys()))
    ap.add_argument("--frames",  type=int,   default=480,  help="総フレーム数（透明動画と一致させること）")
    ap.add_argument("--fps",     type=int,   default=24)
    ap.add_argument("--out_w",   type=int,   default=1280, help="出力幅")
    ap.add_argument("--out_h",   type=int,   default=720,  help="出力高さ")
    ap.add_argument("--parallax",   type=float, default=0.2,
                    help="背景パン量（0=固定, 0.2=20%%スクロール）ミツバチ: 0.15〜0.3")
    ap.add_argument("--zoom_range", type=float, default=0.10,
                    help="ズーム変動幅（0=なし, 0.1=10%%変動）")
    ap.add_argument("--output_dir",  default="")
    ap.add_argument("--output_name", default="")
    args = ap.parse_args()

    kf_dir    = Path(args.keyframes_dir)
    meta_path = kf_dir / "grid_meta.json"
    if not meta_path.exists():
        sys.exit(f"[ERROR] grid_meta.json が見つかりません: {meta_path}")

    with open(meta_path) as f:
        meta = json.load(f)
    grid_theta = meta["grid_theta"]

    bg_path = Path(args.bg_image)
    if not bg_path.exists():
        sys.exit(f"[ERROR] 背景画像が見つかりません: {bg_path}")
    bg_rgb = np.array(Image.open(str(bg_path)).convert("RGB"), dtype=np.uint8)
    print(f"[INFO] 背景: {bg_rgb.shape[1]}×{bg_rgb.shape[0]}  parallax={args.parallax}  zoom={args.zoom_range}")

    out_dir = Path(args.output_dir) if args.output_dir else kf_dir.parent.parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = args.output_name.strip() or f"bg_{args.path}_{ts}.mp4"
    if not fname.endswith(".mp4"):
        fname += ".mp4"
    video_path = out_dir / fname

    path_func = PATHS[args.path]
    N = args.frames

    try:
        ffmpeg = get_ffmpeg()
    except RuntimeError as e:
        sys.exit(f"[ERROR] {e}")

    print(f"[INFO] {N}フレーム生成中...")

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)

        for i in range(N):
            t_raw, theta_raw = path_func(i, N, grid_theta)
            t_norm     = float(t_raw) / (GRID_T - 1)          # 0〜1
            theta_norm = float(theta_raw) / grid_theta         # 0〜1

            frame = render_bg_frame(
                bg_rgb, args.out_w, args.out_h,
                theta_norm, t_norm,
                args.parallax, args.zoom_range,
            )
            Image.fromarray(frame).save(str(tmp / f"frame_{i:05d}.png"))

            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{N}")

        print("[INFO] MP4エンコード中...")
        cmd = [
            ffmpeg, "-y",
            "-framerate", str(args.fps),
            "-i", str(tmp / "frame_%05d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
            str(video_path),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)

    if r.returncode != 0:
        sys.exit(f"[ffmpeg ERROR] {r.stderr[-600:]}")

    print(f"\n[完了] {video_path}")
    print(f"  PowerDirector Track 1 (下) にこのファイルを配置")
    print(f"  Track 2 (上) に transparent_*.webm を重ねてください")


if __name__ == "__main__":
    main()

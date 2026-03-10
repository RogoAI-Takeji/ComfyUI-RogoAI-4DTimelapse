#!/usr/bin/env python3
"""
composite_4d_video.py
====================
4Dタイムラプスグリッドを走査しながら、背景画像に合成してMP4を生成する。

アルファ抽出:
  グリッドフレームはグレー背景（bg_gray=0.12 → RGB≈31,31,31）でレンダリング済み。
  黒白二重レンダリング不要のシンプルな方法:
    1. グレー背景色を読み込み（grid_meta.json の render_params.bg_gray）
    2. 背景と同色の画素をマスクで抜く（HSV彩度 + 輝度閾値）
    3. erode/dilate でエッジを整える
    4. 背景画像に alpha合成

使い方:
  python composite_4d_video.py \
    --keyframes_dir "D:/NB4D_test/tomato/grid_keyframes/run_20260307_103932" \
    --bg_image "D:/NB4D_test/tomato/bg_farm.jpg" \
    --path diagonal \
    --frames 480 \
    --output_dir "D:/NB4D_test/tomato/output" \
    [--plate_x 0.5] [--plate_y 0.65] [--plate_scale 0.55] \
    [--fps 24] [--output_name ""]

plate_x/y: 背景画像上のオブジェクト中心（0〜1の相対座標）
plate_scale: 背景画像の短辺を1.0としたときのオブジェクトの幅
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

# ─── パス定義（grid4d_nodes.py と共有） ─────────────────────────────────────

GRID_T     = 120
FFMPEG_CANDIDATES = [
    r"C:\Users\fareg\AppData\Local\FFmpeg\bin\ffmpeg.exe",
    "ffmpeg",
]

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
        p = i / N
        t = float(np.clip(p * (GRID_T - 1) + (GRID_T - 1) * 0.06 * np.sin(2 * np.pi * i * 7 / N), 0, GRID_T - 1))
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
        if   phase < 0.3: theta = phase / 0.3 * (GT * 0.15)
        elif phase < 0.6: theta = GT * 0.15 + (phase - 0.3) / 0.3 * (GT * 0.5)
        elif phase < 0.8: theta = GT * 0.65 + (phase - 0.6) / 0.2 * (GT * 0.2)
        else:             theta = GT * 0.85 + (phase - 0.8) / 0.2 * (GT * 0.15)
        return (t, float(theta % GT))

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


# ─── グリッドトラバーサー ─────────────────────────────────────────────────────

class GridTraverser:
    def __init__(self, kf_dir: Path, n_stages: int, grid_theta: int):
        self.kf_dir     = kf_dir
        self.n_stages   = n_stages
        self.grid_theta = grid_theta
        self.kf_times   = np.linspace(0, GRID_T - 1, n_stages)
        self._cache     = {}

    def _load(self, stage: int, theta: int) -> np.ndarray:
        key = (stage, theta)
        if key not in self._cache:
            if len(self._cache) > 400:
                self._cache.pop(next(iter(self._cache)))
            p = self.kf_dir / f"stage_{stage:02d}" / f"angle_{theta:03d}.png"
            self._cache[key] = np.array(Image.open(str(p)).convert("RGB"), dtype=np.uint8)
        return self._cache[key]

    def get_frame(self, t: float, theta: float) -> np.ndarray:
        t     = float(np.clip(t, 0, GRID_T - 1))
        theta = float(theta % self.grid_theta)
        tidx  = int(round(theta)) % self.grid_theta

        kft = self.kf_times
        if t <= kft[0]:
            return self._load(0, tidx)
        if t >= kft[-1]:
            return self._load(self.n_stages - 1, tidx)

        ka    = int(np.searchsorted(kft, t) - 1)
        kb    = ka + 1
        alpha = (t - kft[ka]) / (kft[kb] - kft[ka])

        img_a = self._load(ka, tidx)
        img_b = self._load(kb, tidx)
        return _interpolate(img_a, img_b, float(alpha))


def _interpolate(img_a: np.ndarray, img_b: np.ndarray, alpha: float) -> np.ndarray:
    if alpha <= 0.0:
        return img_a
    if alpha >= 1.0:
        return img_b

    ga = cv2.cvtColor(img_a, cv2.COLOR_RGB2GRAY)
    gb = cv2.cvtColor(img_b, cv2.COLOR_RGB2GRAY)

    flow_ab = cv2.calcOpticalFlowFarneback(
        ga, gb, None, 0.5, 5, 25, 10, 7, 1.5, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    flow_ba = cv2.calcOpticalFlowFarneback(
        gb, ga, None, 0.5, 5, 25, 10, 7, 1.5, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

    mse = float(np.mean((ga.astype(float) - gb.astype(float)) ** 2))

    H, W = img_a.shape[:2]
    map_x_ab = np.arange(W, dtype=np.float32)[None, :] + (alpha * flow_ab[..., 0])
    map_y_ab = np.arange(H, dtype=np.float32)[:, None] + (alpha * flow_ab[..., 1])
    map_x_ba = np.arange(W, dtype=np.float32)[None, :] + ((1 - alpha) * flow_ba[..., 0])
    map_y_ba = np.arange(H, dtype=np.float32)[:, None] + ((1 - alpha) * flow_ba[..., 1])

    warp_a = cv2.remap(img_a.astype(np.float32), map_x_ab, map_y_ab,
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    warp_b = cv2.remap(img_b.astype(np.float32), map_x_ba, map_y_ba,
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    flow_blend = (1 - alpha) * warp_a + alpha * warp_b

    if mse > 2000:
        dissolve = img_a.astype(np.float32) * (1 - alpha) + img_b.astype(np.float32) * alpha
        result   = flow_blend * 0.60 + dissolve * 0.40
    else:
        result = flow_blend

    return np.clip(result, 0, 255).astype(np.uint8)


# ─── アルファ抽出 ─────────────────────────────────────────────────────────────

def extract_alpha(frame_rgb: np.ndarray, bg_gray: float, threshold: int = 18) -> np.ndarray:
    """
    グレー背景色からオブジェクトのアルファマスクを生成。

    bg_gray: 0.0〜1.0 の float（grid_meta.json の render_params.bg_gray）
    threshold: 背景色からの許容距離（色空間 L2 距離）
    戻り値: uint8 マスク（0=背景, 255=前景）
    """
    bg_val = int(round(bg_gray * 255))
    bg_rgb = np.array([bg_val, bg_val, bg_val], dtype=np.float32)

    diff = frame_rgb.astype(np.float32) - bg_rgb
    dist = np.linalg.norm(diff, axis=2)  # H×W

    mask = (dist > threshold).astype(np.uint8) * 255

    # ノイズ除去 + エッジ拡大
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # ソフトエッジ（ぼかし）
    mask = cv2.GaussianBlur(mask, (7, 7), 0)

    return mask


# ─── 背景合成 ─────────────────────────────────────────────────────────────────

def pan_bg(
    bg_rgb: np.ndarray,
    out_w: int, out_h: int,
    theta_norm: float,
    t_norm: float,
    parallax: float,
    zoom_range: float,
) -> np.ndarray:
    """背景をパン・ズームして out_w × out_h で返す"""
    H_bg, W_bg = bg_rgb.shape[:2]
    zoom   = 1.0 + zoom_range * (1.0 - t_norm)
    crop_w = int(out_w / zoom)
    crop_h = int(out_h / zoom)

    max_pan = int(W_bg * parallax)
    pan_x   = int(theta_norm * max_pan)

    if W_bg < crop_w + max_pan + 10:
        bg_tiled = np.tile(bg_rgb, (1, 3, 1))
        offset_x = W_bg
    else:
        bg_tiled = bg_rgb
        offset_x = 0

    W_bg_t = bg_tiled.shape[1]
    cx = W_bg_t // 2 + pan_x - offset_x
    cy = bg_tiled.shape[0] // 2
    x0 = max(0, cx - crop_w // 2)
    y0 = max(0, cy - crop_h // 2)
    x1 = min(W_bg_t, x0 + crop_w)
    y1 = min(bg_tiled.shape[0], y0 + crop_h)

    cropped = bg_tiled[y0:y1, x0:x1]
    return cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)


def composite_on_bg(
    frame_rgb: np.ndarray,
    alpha_mask: np.ndarray,
    bg_frame: np.ndarray,   # すでにパン済みの背景（out_w × out_h）
    plate_cx: float,
    plate_cy: float,
    plate_scale: float,
) -> np.ndarray:
    H_bg, W_bg = bg_frame.shape[:2]
    H_f,  W_f  = frame_rgb.shape[:2]

    target_w = int(min(W_bg, H_bg) * plate_scale)
    target_h = int(target_w * H_f / W_f)

    frame_rs = cv2.resize(frame_rgb,  (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    alpha_rs = cv2.resize(alpha_mask, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    cx = int(W_bg * plate_cx)
    cy = int(H_bg * plate_cy)
    x0 = cx - target_w // 2
    y0 = cy - target_h // 2
    x1 = x0 + target_w
    y1 = y0 + target_h

    sx0 = max(0, -x0);   sy0 = max(0, -y0)
    ex0 = max(0, x0);    ey0 = max(0, y0)
    sx1 = target_w - max(0, x1 - W_bg)
    sy1 = target_h - max(0, y1 - H_bg)
    ex1 = min(W_bg, x1); ey1 = min(H_bg, y1)

    result  = bg_frame.copy().astype(np.float32)
    fg_crop = frame_rs[sy0:sy1, sx0:sx1].astype(np.float32)
    al_crop = alpha_rs[sy0:sy1, sx0:sx1].astype(np.float32) / 255.0
    al_crop = al_crop[:, :, np.newaxis]

    roi = result[ey0:ey1, ex0:ex1]
    result[ey0:ey1, ex0:ex1] = roi * (1.0 - al_crop) + fg_crop * al_crop

    return np.clip(result, 0, 255).astype(np.uint8)


# ─── ffmpeg ───────────────────────────────────────────────────────────────────

def get_ffmpeg():
    for p in FFMPEG_CANDIDATES:
        try:
            subprocess.run([p, "-version"], capture_output=True, check=True)
            return p
        except Exception:
            pass
    raise RuntimeError("ffmpeg が見つかりません")


# ─── メイン ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="4D グリッドタイムラプスを背景画像に合成して MP4 生成")
    ap.add_argument("--keyframes_dir", required=True,  help="grid_meta.json があるグリッドディレクトリ")
    ap.add_argument("--bg_image",      required=True,  help="背景画像（JPG/PNG）")
    ap.add_argument("--path",          default="diagonal",
                    choices=list(PATHS.keys()), help="走査パス")
    ap.add_argument("--frames",        type=int,   default=480,  help="総フレーム数")
    ap.add_argument("--fps",           type=int,   default=24,   help="fps")
    ap.add_argument("--plate_x",       type=float, default=0.5,  help="配置中心X（0〜1）")
    ap.add_argument("--plate_y",       type=float, default=0.65, help="配置中心Y（0〜1）")
    ap.add_argument("--plate_scale",   type=float, default=0.45, help="背景短辺×scaleがオブジェクト幅")
    ap.add_argument("--alpha_thresh",  type=int,   default=18,   help="背景除去閾値（距離）")
    ap.add_argument("--parallax",      type=float, default=0.0,
                    help="背景パン量（0=固定, 0.2=20%%スクロール）ミツバチ視点: 0.15〜0.3")
    ap.add_argument("--zoom_range",    type=float, default=0.0,
                    help="ズーム変動幅（0=なし, 0.1=10%%変動）")
    ap.add_argument("--output_dir",    default="",               help="MP4出力先（空=keyframes_dirの親）")
    ap.add_argument("--output_name",   default="",               help="ファイル名（空=自動）")
    args = ap.parse_args()

    kf_dir    = Path(args.keyframes_dir)
    meta_path = kf_dir / "grid_meta.json"
    if not meta_path.exists():
        sys.exit(f"[ERROR] grid_meta.json が見つかりません: {meta_path}")

    with open(meta_path) as f:
        meta = json.load(f)

    n_stages  = meta["n_stages"]
    grid_theta = meta["grid_theta"]
    bg_gray   = meta.get("render_params", {}).get("bg_gray", 0.12)

    print(f"[INFO] グリッド: {n_stages}stages × {grid_theta}angles")
    print(f"[INFO] bg_gray = {bg_gray}")

    # 背景画像
    bg_path = Path(args.bg_image)
    if not bg_path.exists():
        sys.exit(f"[ERROR] 背景画像が見つかりません: {bg_path}")
    bg_rgb = np.array(Image.open(str(bg_path)).convert("RGB"), dtype=np.uint8)
    print(f"[INFO] 背景画像: {bg_rgb.shape[1]}×{bg_rgb.shape[0]} px")

    # 出力先
    out_dir = Path(args.output_dir) if args.output_dir else kf_dir.parent.parent / "output_composite"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = args.output_name.strip() or f"composite_{args.path}_{ts}.mp4"
    if not fname.endswith(".mp4"):
        fname += ".mp4"
    video_path = out_dir / fname

    # トラバーサー・パス関数
    traverser = GridTraverser(kf_dir, n_stages, grid_theta)
    path_func = PATHS[args.path]
    N = args.frames

    ffmpeg = get_ffmpeg()

    out_w = bg_rgb.shape[1]
    out_h = bg_rgb.shape[0]
    print(f"[INFO] フレーム生成開始: {N}フレーム, パス={args.path}")
    print(f"[INFO] parallax={args.parallax}  zoom_range={args.zoom_range}")

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)

        for i in range(N):
            t, theta = path_func(i, N, grid_theta)
            frame    = traverser.get_frame(float(t), float(theta))

            # 背景パン（parallax=0 のとき bg_rgb そのまま）
            t_norm     = float(t) / (GRID_T - 1)
            theta_norm = float(theta) / grid_theta
            bg_frame   = pan_bg(bg_rgb, out_w, out_h,
                                theta_norm, t_norm,
                                args.parallax, args.zoom_range)

            alpha    = extract_alpha(frame, bg_gray, threshold=args.alpha_thresh)
            composed = composite_on_bg(
                frame, alpha, bg_frame,
                args.plate_x, args.plate_y, args.plate_scale,
            )

            Image.fromarray(composed).save(str(tmp / f"frame_{i:05d}.png"))

            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{N} フレーム完了")

        print(f"[INFO] ffmpeg でMP4生成中...")
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

    duration = N / args.fps
    print(f"\n[完了] {video_path}")
    print(f"  {N}フレーム = {duration:.1f}秒 @ {args.fps}fps")
    print(f"  配置: center=({args.plate_x}, {args.plate_y}), scale={args.plate_scale}")


if __name__ == "__main__":
    main()

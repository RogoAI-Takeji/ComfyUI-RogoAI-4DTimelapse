#!/usr/bin/env python3
"""
export_transparent_video.py
===========================
4Dグリッドを走査して「透明背景動画」を出力する。
PowerDirector / DaVinci Resolve / Premiere などの外部エディターに持ち込める。

出力フォーマット:
  --format webm   : WebM (VP9 + alpha) ← PowerDirector推奨
  --format png    : PNG連番 (RGBA)      ← 最も互換性が高い
  --format mov    : MOV (ProRes 4444)  ← DaVinci Resolve / Premiere

使い方:
  python export_transparent_video.py \
    --keyframes_dir "D:/NB4D_test/tomato/grid_keyframes/run_20260307_103932" \
    --path diagonal \
    --frames 480 \
    --format webm \
    --output_dir "D:/NB4D_test/tomato/output_transparent"

alpha_thresh:
  グレー背景の除去感度。上げると背景がよく抜ける（抜けすぎには注意）。
  デフォルト 18。トマトの場合 15〜25 が目安。
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

# ─── パス定義 ─────────────────────────────────────────────────────────────────

GRID_T = 120

def _make_paths():
    rng = np.random.default_rng(42)
    _jitter = rng.uniform(-0.15, 0.15, 2000)

    # ── 猫視点パス群 ────────────────────────────────────────────────────────────

    def cat_zoom_face(i, N, GT):
        """
        stage_01の顔クローズアップから始まり、ズームアウトしながら全身へ。
        その後 stage_01→stage_08 の成長を正面付近の小さな揺れで見せる。
        Phase 1 (0→0.30): stage_01固定、ズームアウト中（theta≈0）
        Phase 2 (0.30→1.0): stage_01→stage_08、正面で微細に揺れながら成長
        """
        p = i / N
        if p < 0.30:
            t = 0.0
        else:
            t = float(np.clip((p - 0.30) / 0.70 * (GRID_T - 1), 0, GRID_T - 1))
        swing = GT * 0.04 * np.sin(2 * np.pi * p * 1.5)
        theta = float(swing % GT)
        return (t, theta)

    def cat_double_arc(i, N, GT):
        """正面→背面→正面と往復しながら幼→老"""
        p = i / N
        t = float(np.clip(p * (GRID_T - 1), 0, GRID_T - 1))
        if p < 0.5:
            theta = p / 0.5 * (GT * 0.5)        # 0° → 180°
        else:
            theta = (1.0 - (p - 0.5) / 0.5) * (GT * 0.5)  # 180° → 0°
        return (t, float(theta))

    def cat_young_linger(i, N, GT):
        """若い時期を70%のフレームで丁寧に見せ、後半30%で老齢へ急速移行+1周"""
        p = i / N
        if p < 0.7:
            # 前半70%フレームで前半40%の成長を表示（子猫をじっくり）
            t = float(np.clip(p / 0.7 * (GRID_T - 1) * 0.4, 0, GRID_T - 1))
        else:
            # 後半30%フレームで残り60%の成長を早送り
            t = float(np.clip((GRID_T - 1) * 0.4 + (p - 0.7) / 0.3 * (GRID_T - 1) * 0.6, 0, GRID_T - 1))
        theta = float(p * GT % GT)  # 同時に1周
        return (t, theta)

    def cat_waltz(i, N, GT):
        """正面付近で左右にゆっくり揺れながら幼→老（ワルツ）"""
        p = i / N
        t = float(np.clip(p * (GRID_T - 1), 0, GRID_T - 1))
        # 中心角がゆっくり前面→横へ移動、その周りを3周期で揺れる
        center = p * GT * 0.25             # 0° → 90°へ緩やかに移動
        swing  = GT * 0.10 * np.sin(2 * np.pi * p * 3)  # ±~0.1GT の揺れ
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
        if p < 0.7:
            theta = (p / 0.7) * GT * 3.0 % GT
        else:
            theta = ((p - 0.7) / 0.3) * GT % GT
        return (t, float(theta))

    def bee_inspect(i, N, GT):
        p     = i / N
        t     = float(np.clip(p * (GRID_T - 1), 0, GRID_T - 1))
        phase = (p * 4) % 1.0
        if phase < 0.3:
            theta = phase / 0.3 * (GT * 0.15)
        elif phase < 0.6:
            theta = GT * 0.15 + (phase - 0.3) / 0.3 * (GT * 0.5)
        elif phase < 0.8:
            theta = GT * 0.65 + (phase - 0.6) / 0.2 * (GT * 0.2)
        else:
            theta = GT * 0.85 + (phase - 0.8) / 0.2 * (GT * 0.15)
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
        # ── 猫視点 7+1パス ───────────────────────────────────────────────────────
        # 0. 顔クローズアップから全身ズームアウト→成長（zoom_start/zoom_framesと組み合わせ）
        "cat_zoom_face":    cat_zoom_face,
        # 1. 正面(0°)→背面(180°)に通り過ぎながら幼→老
        "cat_passthrough":  lambda i, N, GT: (
            float(np.clip(i * (GRID_T - 1) / N, 0, GRID_T - 1)),
            float(i / N * (GT * 0.5)),
        ),
        # 2. 背面(180°)→正面(0°)に通り過ぎながら幼→老（逆方向）
        "cat_reverse_pass": lambda i, N, GT: (
            float(np.clip(i * (GRID_T - 1) / N, 0, GRID_T - 1)),
            float((1.0 - i / N) * (GT * 0.5)),
        ),
        # 3. 1周(360°)しながら幼→老（猫版diagonal）
        "cat_circle_age":   lambda i, N, GT: (
            float(np.clip(i * (GRID_T - 1) / N, 0, GRID_T - 1)),
            float(i * GT / N % GT),
        ),
        # 4. 正面→背面→正面と往復しながら幼→老
        "cat_double_arc":   cat_double_arc,
        # 5. 若い時期を70%のフレームで丁寧に見せ、急速に老化+1周
        "cat_young_linger": cat_young_linger,
        # 6. 横顔(90°=GT/4)固定で幼→老の変遷を見せる
        "cat_side_age":     lambda i, N, GT: (
            float(np.clip(i * (GRID_T - 1) / N, 0, GRID_T - 1)),
            float(GT * 0.25),
        ),
        # 7. 正面付近で左右にゆっくり揺れながら幼→老（ワルツ）
        "cat_waltz":        cat_waltz,
    }

PATHS = _make_paths()

FFMPEG_CANDIDATES = [
    r"C:\Users\fareg\AppData\Local\FFmpeg\bin\ffmpeg.exe",
    "ffmpeg",
]


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
        return _interpolate(self._load(ka, tidx), self._load(kb, tidx), float(alpha))


def _interpolate(img_a: np.ndarray, img_b: np.ndarray, alpha: float) -> np.ndarray:
    if alpha <= 0.0:
        return img_a
    if alpha >= 1.0:
        return img_b

    ga = cv2.cvtColor(img_a, cv2.COLOR_RGB2GRAY)
    gb = cv2.cvtColor(img_b, cv2.COLOR_RGB2GRAY)

    # DIS（Dense Inverse Search）— Farnebackより小さな動きに強く滑らか
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    flow_ab = dis.calc(ga, gb, None)
    flow_ba = dis.calc(gb, ga, None)

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

    blend = (1 - alpha) * warp_a + alpha * warp_b
    if mse > 2000:
        dissolve = img_a.astype(np.float32) * (1 - alpha) + img_b.astype(np.float32) * alpha
        blend    = blend * 0.60 + dissolve * 0.40

    return np.clip(blend, 0, 255).astype(np.uint8)


# ─── アルファ抽出 ─────────────────────────────────────────────────────────────

def extract_alpha(frame_rgb: np.ndarray, bg_gray: float, threshold: int) -> np.ndarray:
    """グレー単色背景からアルファマスクを生成（0=透明, 255=不透明）"""
    bg_val = int(round(bg_gray * 255))
    bg_arr = np.array([bg_val, bg_val, bg_val], dtype=np.float32)
    dist   = np.linalg.norm(frame_rgb.astype(np.float32) - bg_arr, axis=2)
    mask   = (dist > threshold).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask   = cv2.GaussianBlur(mask, (7, 7), 0)
    return mask


def to_rgba(frame_rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """RGB + alpha マスク → RGBA uint8"""
    rgba = np.dstack([frame_rgb, alpha])
    return rgba.astype(np.uint8)


def apply_zoom(rgba: np.ndarray, zoom: float) -> np.ndarray:
    """
    RGBA フレームに中央クロップ＋拡大でズームを適用。
    zoom=1.0: 変化なし, zoom=2.5: 中央を2.5倍に拡大（顔クローズアップ）
    """
    if zoom <= 1.001:
        return rgba
    H, W = rgba.shape[:2]
    crop_w = max(1, int(W / zoom))
    crop_h = max(1, int(H / zoom))
    x0 = (W - crop_w) // 2
    y0 = (H - crop_h) // 2
    cropped = rgba[y0:y0 + crop_h, x0:x0 + crop_w]
    # RGBA の 4ch まとめて LANCZOS4 でリサイズ
    return cv2.resize(cropped, (W, H), interpolation=cv2.INTER_LANCZOS4)


def _zoom_curve(i: int, zoom_start: float, zoom_frames: int) -> float:
    """
    フレーム i でのズーム倍率を返す。
    zoom_frames フレームかけて zoom_start → 1.0 に線形補間。
    """
    if zoom_frames <= 0 or zoom_start <= 1.0:
        return 1.0
    if i >= zoom_frames:
        return 1.0
    frac = i / zoom_frames
    return zoom_start + (1.0 - zoom_start) * frac  # zoom_start → 1.0


# ─── ffmpeg ───────────────────────────────────────────────────────────────────

def get_ffmpeg():
    for p in FFMPEG_CANDIDATES:
        try:
            subprocess.run([p, "-version"], capture_output=True, check=True)
            return p
        except Exception:
            pass
    raise RuntimeError("ffmpeg が見つかりません")


def encode_video(tmp_dir: Path, video_path: Path, fps: int, fmt: str, ffmpeg: str):
    """PNG連番（RGBA）→ 指定フォーマットで動画エンコード"""
    if fmt == "webm":
        cmd = [
            ffmpeg, "-y",
            "-framerate", str(fps),
            "-i", str(tmp_dir / "frame_%05d.png"),
            "-c:v", "libvpx-vp9",
            "-pix_fmt", "yuva420p",   # alpha対応ピクセルフォーマット
            "-b:v", "0", "-crf", "20",
            "-auto-alt-ref", "0",     # alpha使用時は必須
            str(video_path),
        ]
    elif fmt == "mov":
        cmd = [
            ffmpeg, "-y",
            "-framerate", str(fps),
            "-i", str(tmp_dir / "frame_%05d.png"),
            "-c:v", "prores_ks",
            "-profile:v", "4444",     # ProRes 4444: alpha対応
            "-pix_fmt", "yuva444p10le",
            str(video_path),
        ]
    else:
        raise ValueError(f"未対応フォーマット: {fmt}")

    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg エラー: {r.stderr[-600:]}")


# ─── メイン ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="4Dグリッドから透明背景動画を出力")
    ap.add_argument("--keyframes_dir", required=True,  help="grid_meta.json があるグリッドディレクトリ")
    ap.add_argument("--path",          default="diagonal", choices=list(PATHS.keys()))
    ap.add_argument("--frames",        type=int,   default=480,   help="総フレーム数")
    ap.add_argument("--fps",           type=int,   default=24,    help="fps")
    ap.add_argument("--format",        default="webm",
                    choices=["webm", "png", "mov"],
                    help="出力フォーマット: webm=PowerDirector用, png=PNG連番, mov=ProRes4444")
    ap.add_argument("--alpha_thresh",  type=int,   default=18,    help="背景除去閾値（推奨: 15〜25）")
    ap.add_argument("--zoom_start",   type=float, default=1.0,
                    help="開始ズーム倍率（1.0=等倍, 2.5=顔クローズアップ）cat_zoom_face用")
    ap.add_argument("--zoom_frames",  type=int,   default=0,
                    help="ズームアウトにかけるフレーム数（0=ズームなし, 例: 120=5秒@24fps）")
    ap.add_argument("--output_dir",    default="",                help="出力先（空=keyframes_dir/../../output_transparent）")
    ap.add_argument("--output_name",   default="",                help="ファイル名（空=自動）")
    args = ap.parse_args()

    kf_dir    = Path(args.keyframes_dir)
    meta_path = kf_dir / "grid_meta.json"
    if not meta_path.exists():
        sys.exit(f"[ERROR] grid_meta.json が見つかりません: {meta_path}")

    with open(meta_path) as f:
        meta = json.load(f)

    n_stages   = meta["n_stages"]
    grid_theta = meta["grid_theta"]
    bg_gray    = meta.get("render_params", {}).get("bg_gray", 0.12)

    print(f"[INFO] グリッド: {n_stages}stages × {grid_theta}angles")
    print(f"[INFO] bg_gray = {bg_gray}  alpha_thresh = {args.alpha_thresh}")

    out_dir = Path(args.output_dir) if args.output_dir else kf_dir.parent.parent / "output_transparent"
    out_dir.mkdir(parents=True, exist_ok=True)

    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    ext_map = {"webm": ".webm", "png": "", "mov": ".mov"}

    if args.format == "png":
        # PNG連番はサブフォルダに保存
        png_dir = out_dir / (args.output_name.strip() or f"transparent_{args.path}_{ts}")
        png_dir.mkdir(parents=True, exist_ok=True)
    else:
        fname = args.output_name.strip() or f"transparent_{args.path}_{ts}{ext_map[args.format]}"
        if not fname.endswith(ext_map[args.format]):
            fname += ext_map[args.format]
        video_path = out_dir / fname

    traverser = GridTraverser(kf_dir, n_stages, grid_theta)
    path_func = PATHS[args.path]
    N = args.frames

    use_zoom = args.zoom_start > 1.0 and args.zoom_frames > 0
    if use_zoom:
        print(f"[INFO] ズーム: {args.zoom_start}x → 1.0x  ({args.zoom_frames}フレームでズームアウト)")
    print(f"[INFO] {N}フレーム生成開始, パス={args.path}, 形式={args.format}")

    if args.format == "png":
        # PNG連番: 直接書き出し
        for i in range(N):
            t, theta = path_func(i, N, grid_theta)
            frame    = traverser.get_frame(float(t), float(theta))
            alpha    = extract_alpha(frame, bg_gray, args.alpha_thresh)
            rgba     = to_rgba(frame, alpha)
            if use_zoom:
                zoom = _zoom_curve(i, args.zoom_start, args.zoom_frames)
                rgba = apply_zoom(rgba, zoom)
            Image.fromarray(rgba, mode="RGBA").save(str(png_dir / f"frame_{i:05d}.png"))
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{N}")

        print(f"\n[完了] PNG連番: {png_dir}")
        print(f"  PowerDirectorでこのフォルダを「画像シーケンス」として読み込んでください。")

    else:
        # 動画フォーマット: 一時フォルダ経由
        try:
            ffmpeg = get_ffmpeg()
        except RuntimeError as e:
            sys.exit(f"[ERROR] {e}")

        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)

            for i in range(N):
                t, theta = path_func(i, N, grid_theta)
                frame    = traverser.get_frame(float(t), float(theta))
                alpha    = extract_alpha(frame, bg_gray, args.alpha_thresh)
                rgba     = to_rgba(frame, alpha)
                if use_zoom:
                    zoom = _zoom_curve(i, args.zoom_start, args.zoom_frames)
                    rgba = apply_zoom(rgba, zoom)
                Image.fromarray(rgba, mode="RGBA").save(str(tmp / f"frame_{i:05d}.png"))
                if (i + 1) % 50 == 0:
                    print(f"  {i+1}/{N}")

            print(f"[INFO] エンコード中: {args.format.upper()} ...")
            try:
                encode_video(tmp, video_path, args.fps, args.format, ffmpeg)
            except RuntimeError as e:
                sys.exit(str(e))

        duration = N / args.fps
        print(f"\n[完了] {video_path}")
        print(f"  {N}フレーム = {duration:.1f}秒 @ {args.fps}fps")
        print(f"  PowerDirectorに透明動画として読み込み可能です。")


if __name__ == "__main__":
    main()

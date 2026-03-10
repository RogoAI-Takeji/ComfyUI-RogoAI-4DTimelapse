#!/usr/bin/env python3
"""
remove_bg_video.py
==================
既存の動画ファイル（MP4など）からグレー背景を除去し、
透明背景動画（WebM/PNG連番）に変換する。

使い方:
  python remove_bg_video.py \
    --input "D:/NB4D_test/tomato/output/4d_diagonal_20260307_103932.mp4" \
    --format webm \
    --output_dir "D:/NB4D_test/tomato/output_transparent"

オプション:
  --bg_gray    背景グレー値 0.0-1.0 (デフォルト 0.12 = pyrender デフォルト)
  --thresh     除去感度 (デフォルト 18、上げると抜けやすい)
  --format     webm / png (デフォルト webm)
"""

import argparse
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

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


def extract_alpha(frame_rgb: np.ndarray, bg_gray: float, threshold: int) -> np.ndarray:
    bg_val = int(round(bg_gray * 255))
    bg_arr = np.array([bg_val, bg_val, bg_val], dtype=np.float32)
    dist   = np.linalg.norm(frame_rgb.astype(np.float32) - bg_arr, axis=2)
    mask   = (dist > threshold).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask   = cv2.GaussianBlur(mask, (7, 7), 0)
    return mask


def main():
    ap = argparse.ArgumentParser(description="動画のグレー背景を透明化")
    ap.add_argument("--input",      required=True, help="入力動画ファイル（MP4など）")
    ap.add_argument("--bg_gray",    type=float, default=0.12, help="背景グレー値 0.0-1.0")
    ap.add_argument("--thresh",     type=int,   default=18,   help="除去感度（15-25）")
    ap.add_argument("--format",     default="webm", choices=["webm", "png"], help="出力形式")
    ap.add_argument("--output_dir", default="", help="出力先（空=入力ファイルと同じフォルダ）")
    ap.add_argument("--output_name", default="", help="ファイル名（空=自動）")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        sys.exit(f"[ERROR] 入力ファイルが見つかりません: {in_path}")

    out_dir = Path(args.output_dir) if args.output_dir else in_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem  = args.output_name.strip() or f"{in_path.stem}_transparent_{ts}"

    cap = cv2.VideoCapture(str(in_path))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 24.0
    print(f"[INFO] {in_path.name}  {total}フレーム @ {fps:.1f}fps")
    print(f"[INFO] bg_gray={args.bg_gray}  thresh={args.thresh}  形式={args.format}")

    if args.format == "png":
        png_dir = out_dir / stem
        png_dir.mkdir(parents=True, exist_ok=True)
        i = 0
        while True:
            ret, bgr = cap.read()
            if not ret:
                break
            rgb   = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            alpha = extract_alpha(rgb, args.bg_gray, args.thresh)
            rgba  = np.dstack([rgb, alpha]).astype(np.uint8)
            Image.fromarray(rgba, mode="RGBA").save(str(png_dir / f"frame_{i:05d}.png"))
            i += 1
            if i % 50 == 0:
                print(f"  {i}/{total}")
        cap.release()
        print(f"\n[完了] PNG連番: {png_dir}")

    else:  # webm
        video_path = out_dir / f"{stem}.webm"
        ffmpeg = get_ffmpeg()

        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            i = 0
            while True:
                ret, bgr = cap.read()
                if not ret:
                    break
                rgb   = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                alpha = extract_alpha(rgb, args.bg_gray, args.thresh)
                rgba  = np.dstack([rgb, alpha]).astype(np.uint8)
                Image.fromarray(rgba, mode="RGBA").save(str(tmp / f"frame_{i:05d}.png"))
                i += 1
                if i % 50 == 0:
                    print(f"  {i}/{total}")
            cap.release()

            print("[INFO] WebMエンコード中...")
            cmd = [
                ffmpeg, "-y",
                "-framerate", str(int(round(fps))),
                "-i", str(tmp / "frame_%05d.png"),
                "-c:v", "libvpx-vp9",
                "-pix_fmt", "yuva420p",
                "-b:v", "0", "-crf", "20",
                "-auto-alt-ref", "0",
                str(video_path),
            ]
            r = subprocess.run(cmd, capture_output=True, text=True)

        if r.returncode != 0:
            sys.exit(f"[ffmpeg ERROR] {r.stderr[-600:]}")

        print(f"\n[完了] {video_path}")
        print("PowerDirectorにそのままドラッグできます。")


if __name__ == "__main__":
    main()

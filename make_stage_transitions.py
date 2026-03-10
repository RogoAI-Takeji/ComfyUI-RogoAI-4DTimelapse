# make_stage_transitions.py
#
# ルリシジミ 9態ライフサイクル - ステージ間タイムラプス遷移動画生成
#
# アルゴリズム:
#   各ステージ画像ペアの SSIM を計算し、変化量に応じて自動選択
#     - 変化小 (SSIM >= 0.55) : 双方向光学フローワープ（方法A）
#     - 変化大 (SSIM <  0.55) : クロスディゾルブ + 双方向フローワープ（方法A+B）
#
#   双方向フローワープ (FILM相当をOpenCVで実装):
#     t=0.0: stage_N の画像
#     t=0.5: warp(A, 0.5*flow_AB)*(0.5) + warp(B, 0.5*flow_BA)*(0.5)
#     t=1.0: stage_N+1 の画像
#     ← 単方向ワープよりワープ歪みが均等化される
#
# 使い方:
#   python make_stage_transitions.py
#   python make_stage_transitions.py --stages_dir D:/path/to/9stages --fps 24 --duration 2.0
#
# 出力:
#   OUTPUT_DIR/transitions/transition_01_02.mp4  (stage1→stage2)
#   OUTPUT_DIR/transitions/transition_02_03.mp4  ...
#   OUTPUT_DIR/concat_list.txt  (ffmpeg concat 用)
#
# 前提: pip install opencv-python pillow numpy scikit-image

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    print("scikit-image が見つかりません: pip install scikit-image")
    sys.exit(1)

# ─── 設定 ────────────────────────────────────────────────────────────────────

STAGES_DIR = Path(
    r"C:\Users\fareg\Desktop\老後AI\work_folder\老後画像AI_シリーズ"
    r"\015_comfyui_nanobanana\making_movie\Celastrina _argiolus\9stages"
)
OUTPUT_DIR  = Path(r"D:\NB4D_test\transitions")
ORBIT_DIR   = Path(r"D:\NB4D_test\orbit_mesh")   # 各stage のorbit動画があるディレクトリ

STAGE_NAMES = [
    "stage_01.png", "stage_02.png", "stage_03.png",
    "stage_04.png", "stage_05.png", "stage_06.png",
    "stage_07.png", "stage_08.png", "stage_09.png",
]

RENDER_W = 1280
RENDER_H = 720

FFMPEG_CANDIDATES = [
    r"C:\Users\fareg\AppData\Local\FFmpeg\bin\ffmpeg.exe",
    "ffmpeg",
]

# ─── ffmpeg ──────────────────────────────────────────────────────────────────

def get_ffmpeg():
    for p in FFMPEG_CANDIDATES:
        try:
            subprocess.run([p, "-version"], capture_output=True, check=True)
            return p
        except Exception:
            pass
    raise RuntimeError("ffmpeg が見つかりません")


# ─── 画像読み込み ─────────────────────────────────────────────────────────────

def load_stage(path: Path, W: int, H: int) -> np.ndarray:
    img = Image.open(str(path)).convert("RGB").resize((W, H), Image.LANCZOS)
    return np.array(img, dtype=np.uint8)


# ─── SSIM で変化量を計算 ───────────────────────────────────────────────────────

def calc_ssim(img_a: np.ndarray, img_b: np.ndarray) -> float:
    ga = cv2.cvtColor(img_a, cv2.COLOR_RGB2GRAY)
    gb = cv2.cvtColor(img_b, cv2.COLOR_RGB2GRAY)
    score, _ = ssim(ga, gb, full=True)
    return float(score)


# ─── 双方向フローワープ ───────────────────────────────────────────────────────

def _warp_by_flow(img: np.ndarray, flow: np.ndarray) -> np.ndarray:
    H, W = img.shape[:2]
    map_x = (np.arange(W, dtype=np.float32)[None, :] + flow[..., 0])
    map_y = (np.arange(H, dtype=np.float32)[:, None] + flow[..., 1])
    return cv2.remap(
        img.astype(np.float32), map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _calc_flow(src_gray: np.ndarray, dst_gray: np.ndarray) -> np.ndarray:
    return cv2.calcOpticalFlowFarneback(
        src_gray, dst_gray,
        None,
        pyr_scale=0.5, levels=5, winsize=25,
        iterations=10, poly_n=7, poly_sigma=1.5,
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
    )


def make_transition_frames(
    img_a: np.ndarray,
    img_b: np.ndarray,
    n_frames: int,
    ssim_score: float,
) -> list:
    """
    img_a → img_b の遷移フレームを生成 (img_a, img_b 自体は含まない)
    n_frames: 中間フレーム数
    """
    ga = cv2.cvtColor(img_a, cv2.COLOR_RGB2GRAY)
    gb = cv2.cvtColor(img_b, cv2.COLOR_RGB2GRAY)

    flow_ab = _calc_flow(ga, gb)   # A→B 方向フロー
    flow_ba = _calc_flow(gb, ga)   # B→A 方向フロー（逆方向）

    frames = []
    for i in range(n_frames):
        t = (i + 1) / (n_frames + 1)   # 0 < t < 1

        # 双方向ワープ
        warp_a = _warp_by_flow(img_a.astype(np.float32), t * flow_ab)
        warp_b = _warp_by_flow(img_b.astype(np.float32), (1.0 - t) * flow_ba)

        if ssim_score < 0.55:
            # 変化大: クロスディゾルブ + 双方向ワープをブレンド
            # ワープ結果に加え、単純ディゾルブも混合してアーティファクトを目立たせない
            dissolve = img_a.astype(np.float32) * (1.0 - t) + img_b.astype(np.float32) * t
            flow_blend = warp_a * (1.0 - t) + warp_b * t
            # ワープ 60% + ディゾルブ 40% でアーティファクト緩和
            frame = flow_blend * 0.60 + dissolve * 0.40
        else:
            # 変化小: 双方向ワープのみ
            frame = warp_a * (1.0 - t) + warp_b * t

        frame = np.clip(frame, 0, 255).astype(np.uint8)
        frames.append(frame)

    return frames


# ─── 遷移動画をMP4に保存 ──────────────────────────────────────────────────────

def frames_to_mp4(
    frames_all: list,
    output_path: Path,
    fps: int,
    ffmpeg: str,
) -> None:
    """frames_all: list of np.ndarray (H,W,3) uint8"""
    import tempfile, os

    tmp_dir = Path(tempfile.mkdtemp())
    for i, f in enumerate(frames_all):
        Image.fromarray(f).save(str(tmp_dir / f"frame_{i:05d}.png"))

    cmd = [
        ffmpeg, "-y",
        "-framerate", str(fps),
        "-i", str(tmp_dir / "frame_%05d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        str(output_path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  [ffmpeg ERROR] {r.stderr[-300:]}")

    # 一時ファイル削除
    for p in tmp_dir.iterdir():
        p.unlink()
    tmp_dir.rmdir()


# ─── メイン ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ステージ間遷移動画生成")
    parser.add_argument("--stages_dir", type=str, default=str(STAGES_DIR))
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--orbit_dir",  type=str, default=str(ORBIT_DIR),
                        help="各stageのorbit動画があるディレクトリ (concat用)")
    parser.add_argument("--fps",      type=int,   default=24)
    parser.add_argument("--duration", type=float, default=2.0,
                        help="遷移の長さ（秒）。デフォルト 2.0s")
    parser.add_argument("--width",  type=int, default=RENDER_W)
    parser.add_argument("--height", type=int, default=RENDER_H)
    parser.add_argument("--concat", action="store_true",
                        help="orbit動画と遷移動画を結合して最終動画を生成")
    args = parser.parse_args()

    stages_dir = Path(args.stages_dir)
    output_dir = Path(args.output_dir)
    trans_dir  = output_dir / "transitions"
    trans_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg = get_ffmpeg()
    n_frames = int(args.fps * args.duration)
    W, H = args.width, args.height

    print("=" * 62)
    print("  make_stage_transitions.py")
    print(f"  遷移フレーム数: {n_frames}  ({args.duration}s @ {args.fps}fps)")
    print(f"  解像度: {W}x{H}")
    print("=" * 62)

    # ステージ画像を読み込む
    stage_imgs = []
    for name in STAGE_NAMES:
        p = stages_dir / name
        if not p.exists():
            print(f"  [警告] {p} が見つかりません → スキップ")
            stage_imgs.append(None)
        else:
            stage_imgs.append(load_stage(p, W, H))
            print(f"  読み込み: {name}")

    # 各ペアで遷移動画を生成
    transition_paths = []
    for i in range(len(STAGE_NAMES) - 1):
        img_a = stage_imgs[i]
        img_b = stage_imgs[i + 1]
        if img_a is None or img_b is None:
            transition_paths.append(None)
            continue

        label_a = STAGE_NAMES[i].replace(".png", "")
        label_b = STAGE_NAMES[i + 1].replace(".png", "")
        out_path = trans_dir / f"transition_{i+1:02d}_{i+2:02d}.mp4"

        score = calc_ssim(img_a, img_b)
        method = "双方向フローのみ" if score >= 0.55 else "クロスディゾルブ+双方向フロー"
        print(f"\n  [{label_a} → {label_b}]  SSIM={score:.3f}  手法={method}")

        frames = make_transition_frames(img_a, img_b, n_frames, score)

        # 前後のステージ画像を先頭・末尾に追加（つなぎ目を滑らかに）
        all_frames = [img_a] + frames + [img_b]
        frames_to_mp4(all_frames, out_path, args.fps, ffmpeg)
        print(f"  保存: {out_path.name}")
        transition_paths.append(out_path)

    # concat リスト生成（orbit動画 + 遷移動画の交互結合用）
    concat_txt = output_dir / "concat_list.txt"
    orbit_dir  = Path(args.orbit_dir)
    orbit_glob_names = [
        f"stage_{i+1:02d}_orbit.mp4" for i in range(len(STAGE_NAMES))
    ]

    with open(str(concat_txt), "w", encoding="utf-8") as f:
        for i, stage_name in enumerate(STAGE_NAMES):
            orbit_name = orbit_glob_names[i]
            orbit_path = orbit_dir / orbit_name
            if orbit_path.exists():
                f.write(f"file '{orbit_path}'\n")
            else:
                print(f"  [情報] orbit動画なし: {orbit_name} (スキップ)")

            if i < len(transition_paths) and transition_paths[i] is not None:
                f.write(f"file '{transition_paths[i]}'\n")

    print(f"\n  concat リスト生成: {concat_txt}")

    # --concat オプション: 最終動画を結合
    if args.concat:
        final_path = output_dir / "lifecycle_4D.mp4"
        cmd = [
            ffmpeg, "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_txt),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
            str(final_path),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            print(f"\n  最終動画完成: {final_path}")
        else:
            print(f"\n  [ffmpeg ERROR] {r.stderr[-300:]}")

    print(f"\n{'=' * 62}")
    print("  完了。次のステップ:")
    print(f"  1. 各stageのorbit動画を {orbit_dir} に配置")
    print(f"     ファイル名: stage_01_orbit.mp4, stage_02_orbit.mp4 ...")
    print(f"  2. python make_stage_transitions.py --concat")
    print(f"     → {output_dir}/lifecycle_4D.mp4 が完成")
    print("=" * 62)


if __name__ == "__main__":
    main()

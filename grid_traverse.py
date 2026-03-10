# grid_traverse.py
#
# 4D グリッド STEP 2: グリッド上の任意パスで動画生成
#
# グリッド構造:
#   縦軸 (T=120): 時間軸 - トマトの熟成（緑→赤）
#   横軸 (Θ=120): 空間軸 - カメラ角度（0°〜357°、3°刻み）
#
#   各セル [t, θ] = 「熟成度 t でカメラ角度 θ°から見たフレーム」
#
# パスとは: グリッド上の座標列 (t_0,θ_0), (t_1,θ_1), ..., (t_N,θ_N)
#   → 自由な時空間軌跡で動画が作れる
#
# 使い方:
#   python grid_traverse.py --path diagonal
#   python grid_traverse.py --path spiral_2x --frames 240
#   python grid_traverse.py --list_paths

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# ─── 設定 ────────────────────────────────────────────────────────────────────

KEYFRAMES_DIR = Path(r"D:\NB4D_test\tomato\grid_keyframes")
OUTPUT_DIR    = Path(r"D:\NB4D_test\tomato\output")

GRID_T        = 120    # 時間軸サイズ
GRID_THETA    = 120    # 空間軸サイズ
FPS           = 24

FFMPEG_CANDIDATES = [
    r"C:\Users\fareg\AppData\Local\FFmpeg\bin\ffmpeg.exe",
    "ffmpeg",
]

# ─── 事前定義パス ─────────────────────────────────────────────────────────────
#
# path関数のシグネチャ: (i: int, N: int) -> (t: float, theta: float)
#   i     : 出力フレームインデックス (0 〜 N-1)
#   N     : 総フレーム数
#   t     : 時間軸座標 (0.0 〜 GRID_T-1)
#   theta : 角度軸座標 (0.0 〜 GRID_THETA-1)

PATHS = {
    # ── 基本パス ──────────────────────────────────────────────
    "orbit_green": {
        "func":  lambda i, N: (0.0, i * GRID_THETA / N % GRID_THETA),
        "frames": 120,
        "desc":  "緑のトマトの周りを1周orbit",
    },
    "orbit_ripe": {
        "func":  lambda i, N: (GRID_T - 1, i * GRID_THETA / N % GRID_THETA),
        "frames": 120,
        "desc":  "完熟トマトの周りを1周orbit",
    },
    "ripen_front": {
        "func":  lambda i, N: (i * (GRID_T - 1) / N, 0.0),
        "frames": 120,
        "desc":  "正面固定でどんどん熟れる",
    },
    # ── 時空間複合パス ────────────────────────────────────────
    "diagonal": {
        "func":  lambda i, N: (
            i * (GRID_T - 1) / N,
            i * GRID_THETA / N % GRID_THETA,
        ),
        "frames": 120,
        "desc":  "回りながら同時に熟れる（斜め移動）",
    },
    "spiral_2x": {
        "func":  lambda i, N: (
            i * (GRID_T - 1) / N,
            i * GRID_THETA * 2 / N % GRID_THETA,
        ),
        "frames": 240,
        "desc":  "2周しながら熟れる（螺旋）",
    },
    "reverse_diag": {
        "func":  lambda i, N: (
            (GRID_T - 1) - i * (GRID_T - 1) / N,
            i * GRID_THETA / N % GRID_THETA,
        ),
        "frames": 120,
        "desc":  "過去（完熟）から現在（緑）へ戻りながら回る",
    },
    # ── ドラマティックパス ────────────────────────────────────
    "time_wave": {
        "func":  lambda i, N: (
            (GRID_T - 1) / 2 * (1 + np.sin(4 * np.pi * i / N)),
            i * GRID_THETA / N % GRID_THETA,
        ),
        "frames": 240,
        "desc":  "時間を往復しながらorbit（時間の波）",
    },
    "zoom_in_time": {
        "func":  lambda i, N: (
            i * (GRID_T - 1) / N,
            i * GRID_THETA / N % GRID_THETA
            if i < N * 0.8
            else (GRID_T - 1) * 0.8 + (i - N * 0.8) * 0.2 * (GRID_T - 1) / (N * 0.2),
        ),
        "frames": 240,
        "desc":  "前半は斜め移動、後半はゆっくり完熟に近づく",
    },
}

# ─── ユーティリティ ────────────────────────────────────────────────────────────

def get_ffmpeg():
    for p in FFMPEG_CANDIDATES:
        try:
            subprocess.run([p, "-version"], capture_output=True, check=True)
            return p
        except Exception:
            pass
    raise RuntimeError("ffmpeg が見つかりません")


def load_meta(keyframes_dir: Path) -> dict:
    meta_path = keyframes_dir / "grid_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"grid_meta.json が見つかりません: {meta_path}")
    with open(meta_path) as f:
        return json.load(f)


def load_frame(keyframes_dir: Path, stage_idx: int, theta_idx: int) -> np.ndarray:
    path = keyframes_dir / f"stage_{stage_idx:02d}" / f"angle_{theta_idx:03d}.png"
    return np.array(Image.open(str(path)).convert("RGB"), dtype=np.uint8)


def warp_by_flow(img: np.ndarray, flow: np.ndarray) -> np.ndarray:
    H, W = img.shape[:2]
    map_x = np.arange(W, dtype=np.float32)[None, :] + flow[..., 0]
    map_y = np.arange(H, dtype=np.float32)[:, None] + flow[..., 1]
    return cv2.remap(
        img.astype(np.float32), map_x, map_y,
        cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )


def interpolate_between(img_a: np.ndarray, img_b: np.ndarray,
                         alpha: float) -> np.ndarray:
    """双方向光学フローで img_a(alpha=0) → img_b(alpha=1) を補間"""
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

    ssim_val = float(np.mean((ga.astype(float) - gb.astype(float)) ** 2))
    is_large_change = ssim_val > 2000   # MSEで判定

    warp_a = warp_by_flow(img_a.astype(np.float32), alpha * flow_ab)
    warp_b = warp_by_flow(img_b.astype(np.float32), (1 - alpha) * flow_ba)
    flow_blend = (1 - alpha) * warp_a + alpha * warp_b

    if is_large_change:
        dissolve = img_a.astype(np.float32) * (1 - alpha) + img_b.astype(np.float32) * alpha
        result = flow_blend * 0.60 + dissolve * 0.40
    else:
        result = flow_blend

    return np.clip(result, 0, 255).astype(np.uint8)


# ─── グリッドトラバーサル ──────────────────────────────────────────────────────

class GridTraverser:
    def __init__(self, keyframes_dir: Path):
        self.kf_dir = keyframes_dir
        self.meta   = load_meta(keyframes_dir)
        self.n_stages = self.meta["n_stages"]

        # キーフレームの時間軸上の位置 [0, GRID_T-1] に等間隔配置
        self.kf_times = np.linspace(0, GRID_T - 1, self.n_stages)

        # フローキャッシュ: {(stage_a, stage_b, theta): (flow_ab, flow_ba)}
        self._flow_cache = {}
        self._img_cache  = {}   # {(stage, theta): img}

    def _get_img(self, stage: int, theta: int) -> np.ndarray:
        key = (stage, theta)
        if key not in self._img_cache:
            # キャッシュが大きくなりすぎないよう古いものを削除
            if len(self._img_cache) > 300:
                oldest = next(iter(self._img_cache))
                del self._img_cache[oldest]
            self._img_cache[key] = load_frame(self.kf_dir, stage, theta)
        return self._img_cache[key]

    def get_frame(self, t: float, theta: float) -> np.ndarray:
        """
        連続座標 (t, theta) に対応するフレームを返す
        t     : [0, GRID_T-1]
        theta : [0, GRID_THETA-1]
        """
        t     = float(np.clip(t,     0, GRID_T - 1))
        theta = float(theta % GRID_THETA)

        # 角度: 最近傍整数インデックス
        theta_idx = int(round(theta)) % GRID_THETA

        # 時間: 前後のキーフレームを見つけてフロー補間
        kf_times = self.kf_times

        if t <= kf_times[0]:
            return self._get_img(0, theta_idx)
        if t >= kf_times[-1]:
            return self._get_img(self.n_stages - 1, theta_idx)

        # t を挟む 2つのキーフレームを探す
        ka = int(np.searchsorted(kf_times, t) - 1)
        kb = ka + 1
        t_a, t_b = kf_times[ka], kf_times[kb]
        alpha = (t - t_a) / (t_b - t_a)

        img_a = self._get_img(ka, theta_idx)
        img_b = self._get_img(kb, theta_idx)

        return interpolate_between(img_a, img_b, alpha)


# ─── 動画生成 ─────────────────────────────────────────────────────────────────

def generate_video(path_name: str, custom_func=None, custom_frames: int = None,
                   output_path: Path = None, fps: int = FPS,
                   keyframes_dir: Path = None, output_dir: Path = None):
    keyframes_dir = keyframes_dir or KEYFRAMES_DIR
    output_dir    = output_dir    or OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg = get_ffmpeg()

    if custom_func is not None:
        path_func   = custom_func
        total_frames = custom_frames or 120
        desc        = "カスタムパス"
    else:
        if path_name not in PATHS:
            print(f"[ERROR] 不明なパス: {path_name}")
            print(f"  利用可能: {list(PATHS.keys())}")
            sys.exit(1)
        p            = PATHS[path_name]
        path_func    = p["func"]
        total_frames = custom_frames or p["frames"]
        desc         = p["desc"]

    if output_path is None:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"4d_{path_name}_{ts}.mp4"

    print("=" * 62)
    print(f"  grid_traverse.py  パス: {path_name}")
    print(f"  {desc}")
    print(f"  フレーム数: {total_frames}  ({total_frames/fps:.1f}s @ {fps}fps)")
    print("=" * 62)

    traverser = GridTraverser(keyframes_dir)

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp_dir = Path(tmp_str)

        for i in range(total_frames):
            t, theta = path_func(i, total_frames)
            frame    = traverser.get_frame(float(t), float(theta))
            Image.fromarray(frame).save(str(tmp_dir / f"frame_{i:05d}.png"))

            if i % fps == 0:
                pct = i / total_frames * 100
                print(f"  {i:4d}/{total_frames}  ({pct:.0f}%)  "
                      f"t={float(t):.1f}  θ={float(theta)*360/GRID_THETA:.0f}°")

        cmd = [
            ffmpeg, "-y",
            "-framerate", str(fps),
            "-i", str(tmp_dir / "frame_%05d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
            str(output_path),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  [ffmpeg ERROR] {r.stderr[-400:]}")
        else:
            print(f"\n  完成: {output_path}")

    return output_path


# ─── メイン ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="4Dグリッド 任意パス動画生成")
    parser.add_argument("--path",           default="diagonal",
                        help="パス名（--list_paths で一覧表示）")
    parser.add_argument("--frames",         type=int, default=None,
                        help="総フレーム数（省略時はパスのデフォルト）")
    parser.add_argument("--fps",            type=int, default=FPS)
    parser.add_argument("--output",         default=None,
                        help="出力MP4パス（省略時は自動命名）")
    parser.add_argument("--keyframes_dir",  default=None,
                        help="キーフレームグリッドのディレクトリ（省略時はスクリプト内のデフォルト）")
    parser.add_argument("--output_dir",     default=None,
                        help="MP4出力先ディレクトリ（省略時はスクリプト内のデフォルト）")
    parser.add_argument("--list_paths",     action="store_true",
                        help="利用可能なパス一覧を表示して終了")
    parser.add_argument("--all_paths",      action="store_true",
                        help="全パスの動画を一括生成")
    args = parser.parse_args()

    if args.list_paths:
        print("\n利用可能なパス一覧:")
        print(f"  {'名前':<20} {'フレーム':>7}   説明")
        print("  " + "-" * 55)
        for name, p in PATHS.items():
            print(f"  {name:<20} {p['frames']:>7}   {p['desc']}")
        print()
        return

    kf_dir     = Path(args.keyframes_dir) if args.keyframes_dir else None
    out_dir    = Path(args.output_dir)    if args.output_dir    else None
    output     = Path(args.output)        if args.output        else None

    if args.all_paths:
        for name in PATHS:
            generate_video(
                path_name=name,
                custom_frames=args.frames,
                fps=args.fps,
                keyframes_dir=kf_dir,
                output_dir=out_dir,
            )
        return

    generate_video(
        path_name=args.path,
        custom_frames=args.frames,
        fps=args.fps,
        output_path=output,
        keyframes_dir=kf_dir,
        output_dir=out_dir,
    )


if __name__ == "__main__":
    main()

"""
NanoBanana 4D - パララックス方式テストスクリプト（9態対応）
=========================================================
各態の静止画 + Depth Map → パララックスワープ → FFmpeg 遷移 → MP4

使い方（ComfyUI ルートから）:
  python custom_nodes\\ComfyUI-RogoAI-NanoBanana\\run_4d_parallax_test.py

または直接:
  python D:\\...\\run_4d_parallax_test.py [--stages 9] [--midas] [--reuse]

オプション:
  --stages N  : テストする態の数 (default: 9)
  --midas     : MiDaS 深度推定を使用（要インターネット初回のみ）
  --reuse     : 既存のフレームと動画を再利用（生成済みのものをスキップ）

入力画像（省略可）:
  D:\\NB4D_test\\input\\stage_01.png 〜 stage_09.png
  ない場合はグラデーションのテスト画像を自動生成します。

ディレクトリ構造:
  D:\\NB4D_test\\
    input/          ← 入力画像（任意）
    depth/          ← Depth Map キャッシュ
    stages/stage01/ ← 各態のフレーム PNG
    videos/         ← 各態の MP4
    output/         ← 最終動画

ドラマ的弧線（Celastrina argiolus 9態）:
  ①卵    静かな始まり・命の予感
  ②孵化   最初の動き・誕生の瞬間
  ③若齢   小さな命の躍動
  ④中期   成長・共生・食の営み
  ⑤終齢   円熟・静かな予感
  ⑥蛹    暗闇・内なる変容        ← どん底（最長・最暗）
  ⑦羽化   光の爆発・劇的な誕生    ← 転換点（最ダイナミック）
  ⑧吸蜜   植物との円環・充足      ← 静かな高揚
  ⑨飛翔   大空へ・解放・余韻      ← 開放的なエンド（最長）
"""

import argparse
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# ── グローバル設定 ────────────────────────────────────────────────────────────
FPS    = 24
WIDTH  = 1920
HEIGHT = 1080

# ── 態ごとの個別設定（ドラマ的弧線に対応） ──────────────────────────────────
# (秒数, 水平最大シフト, 垂直シフト, zoom_in, zoom_out, 感情メモ)
STAGE_CONFIGS = [
    #  sec  sx     sy     zi     zo     感情
    (  8, 0.030, 0.005, 1.10, 1.05, "静かな始まり・命の予感"),       # ① 卵
    (  8, 0.040, 0.010, 1.15, 1.05, "最初の動き・誕生の瞬間"),       # ② 孵化
    (  8, 0.045, 0.010, 1.15, 1.05, "小さな命の躍動"),               # ③ 若齢
    (  8, 0.050, 0.012, 1.18, 1.05, "成長・共生・食の営み"),         # ④ 中期
    ( 10, 0.030, 0.008, 1.12, 1.05, "円熟・静かな予感"),             # ⑤ 終齢
    ( 12, 0.020, 0.005, 1.08, 1.03, "暗闇・内なる変容（どん底）"),   # ⑥ 蛹
    ( 12, 0.060, 0.015, 1.20, 1.08, "光の爆発・劇的な誕生（転換）"), # ⑦ 羽化
    (  8, 0.035, 0.008, 1.13, 1.05, "植物との円環・充足"),           # ⑧ 吸蜜
    ( 14, 0.040, 0.030, 1.15, 1.05, "大空へ・解放・余韻"),           # ⑨ 飛翔（上昇）
]

OUTPUT_ROOT = Path(r"D:\NB4D_test")
INPUT_DIR   = OUTPUT_ROOT / "input"
DEPTH_DIR   = OUTPUT_ROOT / "depth"
STAGES_DIR  = OUTPUT_ROOT / "stages"
VIDEO_DIR   = OUTPUT_ROOT / "videos"
FINAL_DIR   = OUTPUT_ROOT / "output"

FFMPEG_CANDIDATES = [
    r"C:\Users\fareg\AppData\Local\FFmpeg\bin\ffmpeg.exe",
    "ffmpeg",
]

# ── 遷移エフェクト（ドラマ的弧線に対応） ──────────────────────────────────────
# (xfade_transition, duration_sec, 演出コメント)
TRANSITION_MAP = [
    ("fadewhite", 1.5, "①→②: ゆっくり白フラッシュ・誕生の光"),
    ("dissolve",  1.0, "②→③: ディゾルブ・活発な成長"),
    ("fade",      1.0, "③→④: フェード・緑の充実"),
    ("fadeblack", 1.5, "④→⑤: 暗転・活動が落ち着く"),
    ("fadeblack", 2.0, "⑤→⑥: ゆっくり暗転・静寂へ"),  # どん底への入口
    ("fadewhite", 0.5, "⑥→⑦: 瞬間的白フラッシュ・光の爆発"),  # 転換点！
    ("dissolve",  1.0, "⑦→⑧: やわらかディゾルブ・充足"),
    ("slideup",   1.5, "⑧→⑨: 上昇スライド・空への解放"),
]

# ── テスト用合成画像 ─────────────────────────────────────────────────────────

def create_test_image(stage_idx: int, width: int, height: int) -> Image.Image:
    """ヤマトシジミの各態を模した簡易テスト画像を生成する。"""
    configs = [
        # (bg_color, fg_color, shape,            label)
        ((30, 100, 30), (200, 220, 140), "egg",              "Stage 1: 卵 (Egg)"),
        ((40, 120, 40), (170, 200, 110), "caterpillar_small","Stage 2: 孵化 (Hatching)"),
        ((20,  90, 20), (150, 185,  85), "caterpillar_large","Stage 3: 若齢幼虫 (Young Larva)"),
        ((90,  65, 30), (165, 135,  70), "pupa",             "Stage 4: 蛹 (Pupa)"),
    ]

    cfg = configs[stage_idx % len(configs)]
    bg_color, fg_color, shape, label = cfg

    img  = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    # 背景: 葉脈風テクスチャ
    dim_bg = tuple(max(0, c - 18) for c in bg_color)
    for i in range(0, width, 70):
        draw.line([(i, 0), (i + height // 3, height)], fill=dim_bg, width=2)
    for j in range(0, height, 90):
        draw.line([(0, j), (width, j + 50)], fill=dim_bg, width=1)

    cx, cy = width // 2, height // 2

    if shape == "egg":
        r = min(width, height) // 11
        draw.ellipse([cx - r, cy - r // 2, cx + r, cy + r // 2], fill=fg_color)
        # 光沢
        draw.ellipse([cx - r // 3, cy - r // 4,
                      cx + r // 6, cy + r // 6], fill=(230, 240, 200))

    elif shape == "caterpillar_small":
        r = min(width, height) // 18
        for i in range(5):
            x = cx + int((i - 2) * r * 1.6)
            shade = tuple(min(255, c + i * 4) for c in fg_color)
            draw.ellipse([x - r, cy - r // 2, x + r, cy + r // 2], fill=shade)

    elif shape == "caterpillar_large":
        r = min(width, height) // 12
        for i in range(8):
            x = cx + int((i - 3.5) * r * 1.45)
            shade = tuple(min(255, c + i * 3) for c in fg_color)
            draw.ellipse([x - r, cy - int(r * 0.7), x + r, cy + int(r * 0.7)],
                         fill=shade)

    elif shape == "pupa":
        rx, ry = width // 8, height // 5
        draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=fg_color)
        # 絹糸帯
        draw.line([(cx - rx // 2, cy - ry),
                   (cx - rx // 2, cy - ry - 50)],
                  fill=(220, 205, 155), width=4)

    # ラベルテキスト
    try:
        from PIL import ImageFont
        font = ImageFont.load_default()
        draw.text((40, 40), label, fill=(255, 255, 200), font=font)
        draw.text((40, 65), f"frame test  {width}x{height}", fill=(200, 200, 160), font=font)
    except Exception:
        pass

    # 軽くぼかして自然な質感に
    img = img.filter(ImageFilter.GaussianBlur(radius=2))
    return img


# ── 深度推定 ─────────────────────────────────────────────────────────────────

def estimate_depth_synthetic(image_pil: Image.Image) -> np.ndarray:
    """合成深度マップ（MiDaS 不使用、高速フォールバック）。"""
    H, W = image_pil.height, image_pil.width
    img_np = np.array(image_pil.convert("RGB")).astype(np.float32) / 255.0
    gray   = (0.299 * img_np[:, :, 0]
              + 0.587 * img_np[:, :, 1]
              + 0.114 * img_np[:, :, 2])

    y, x  = np.mgrid[0:H, 0:W].astype(np.float32)
    dx    = (x - W / 2) / (W / 2 + 1e-8)
    dy    = (y - H / 2) / (H / 2 + 1e-8)
    cw    = np.clip(1.0 - np.sqrt(dx**2 + dy**2) * 0.85, 0.0, 1.0)

    depth = 0.35 * (1.0 - gray) + 0.65 * cw
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

    blur_r = max(H, W) // 50
    dp = Image.fromarray((depth * 255).astype(np.uint8))
    dp = dp.filter(ImageFilter.GaussianBlur(radius=blur_r))
    return np.array(dp).astype(np.float32) / 255.0


def estimate_depth_midas(image_pil: Image.Image) -> np.ndarray:
    """MiDaS_small で深度推定。失敗時は合成深度にフォールバック。"""
    try:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"    MiDaS_small を {device} で実行中...")

        midas      = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
        midas      = midas.to(device).eval()
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        transform  = transforms.small_transform

        img_u8     = np.array(image_pil.convert("RGB"))
        batch      = transform(img_u8).to(device)

        with torch.no_grad():
            pred = midas(batch)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1),
                size=img_u8.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth = pred.cpu().numpy().astype(np.float32)
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        print(f"    MiDaS 完了")
        return depth

    except Exception as e:
        print(f"    MiDaS 失敗 ({e}) → 合成深度にフォールバック")
        return estimate_depth_synthetic(image_pil)


# ── カメラ動作 ───────────────────────────────────────────────────────────────

def get_camera_params(frame_idx: int, total_frames: int,
                      max_shift_x: float, max_shift_y: float,
                      zoom_in: float, zoom_out: float,
                      stage_idx: int = 0) -> tuple:
    """3幕構成のカメラパラメータを返す (scale, shift_x_frac, shift_y_frac)。
    ⑨飛翔のみ shift_y をマイナス（上昇）にする特殊処理あり。
    """
    t = frame_idx / max(1, total_frames - 1)

    if t < 0.25:        # イン: ズームイン
        act_t   = t / 0.25
        scale   = 1.0 + (zoom_in - 1.0) * act_t
        shift_x = 0.0
        shift_y = 0.0
    elif t < 0.75:      # 見せ場: 左→右パン
        act_t   = (t - 0.25) / 0.5
        scale   = zoom_in
        shift_x = (-1.0 + 2.0 * act_t) * max_shift_x
        # ⑨飛翔: 上昇パン（shift_y がマイナス=上方向へ）
        shift_y = -act_t * max_shift_y if stage_idx == 8 else 0.0
    else:               # アウト: わずかにズームアウト
        act_t   = (t - 0.75) / 0.25
        scale   = zoom_in - (zoom_in - zoom_out) * act_t
        shift_x = max_shift_x
        shift_y = -max_shift_y if stage_idx == 8 else 0.0

    return scale, shift_x, shift_y


# ── パララックスワープ ────────────────────────────────────────────────────────

def warp_frame(image_np: np.ndarray, depth_np: np.ndarray,
               shift_x_frac: float, shift_y_frac: float,
               scale: float) -> np.ndarray:
    """パララックスワープ（geometrically-consistent）。
    image_np : H,W,3  float32  0-1
    depth_np : H,W    float32  0-1  (1=手前, 0=奥)
    """
    H, W       = image_np.shape[:2]
    shift_x_px = shift_x_frac * W
    shift_y_px = shift_y_frac * H
    cx, cy     = W / 2.0, H / 2.0

    dst_y, dst_x = np.mgrid[0:H, 0:W].astype(np.float32)

    # ソース座標 = ズームアンドゥ + 深度依存パララックスシフト
    src_x = (dst_x - cx) / scale + cx + shift_x_px * depth_np
    src_y = (dst_y - cy) / scale + cy + shift_y_px * depth_np

    try:
        import cv2
        img_u8 = (np.clip(image_np, 0, 1) * 255).astype(np.uint8)
        warped = cv2.remap(img_u8, src_x, src_y,
                           cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
        return warped.astype(np.float32) / 255.0

    except ImportError:
        # cv2 なし: numpy 最近傍（高速・低品質）
        src_xi = np.clip(src_x.astype(np.int32), 0, W - 1)
        src_yi = np.clip(src_y.astype(np.int32), 0, H - 1)
        return image_np[src_yi, src_xi, :]


# ── FFmpeg ────────────────────────────────────────────────────────────────────

def get_ffmpeg() -> str:
    for p in FFMPEG_CANDIDATES:
        try:
            subprocess.run([p, "-version"], capture_output=True, check=True)
            return p
        except Exception:
            continue
    raise RuntimeError("ffmpeg が見つかりません。PATH を確認してください。")


def render_stage_video(stage_dir: Path, video_path: Path, ffmpeg: str) -> None:
    """連番 PNG → MP4"""
    cmd = [
        ffmpeg, "-y",
        "-framerate", str(FPS),
        "-i", str(stage_dir / "frame_%05d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg エラー:\n{result.stderr[-400:]}")


def concat_with_transitions(video_paths: list, output_path: Path,
                             ffmpeg: str, configs: list) -> None:
    """xfade フィルターで遷移付き連結。失敗時は単純連結にフォールバック。"""
    num_stages = len(video_paths)
    if num_stages == 1:
        shutil.copy(video_paths[0], output_path)
        return

    inputs = []
    for vp in video_paths:
        inputs += ["-i", str(vp)]

    # offset 計算: 各クリップが開始する累積時刻 - 遷移overlap の合計
    # offset_N = sum(stage_secs[0..N]) - sum(trans_secs[0..N-1])
    filter_parts  = []
    current_label = "[0:v]"
    cumulative    = 0.0

    for i in range(num_stages - 1):
        stage_sec  = configs[i][0]
        trans_rec  = TRANSITION_MAP[i] if i < len(TRANSITION_MAP) else ("fade", 1.0, "")
        trans_name, trans_dur, comment = trans_rec

        cumulative += stage_sec
        offset      = cumulative - trans_dur * (i + 1)

        is_last   = (i == num_stages - 2)
        out_label = "[vout]" if is_last else f"[v{i + 1:02d}]"

        print(f"    遷移 態{i+1}→態{i+2}: [{trans_name} {trans_dur}s]  "
              f"offset={offset:.1f}s  {comment}")

        filter_parts.append(
            f"{current_label}[{i + 1}:v]"
            f"xfade=transition={trans_name}"
            f":duration={trans_dur}:offset={offset:.3f}"
            f"{out_label}"
        )
        current_label = out_label

    filter_complex = ";".join(filter_parts)

    cmd = [ffmpeg, "-y"] + inputs + [
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [警告] xfade 失敗: {result.stderr[-200:]}")
        print(f"  [フォールバック] 単純連結を試みます...")
        _simple_concat(video_paths, output_path, ffmpeg)
    else:
        print(f"  xfade 連結成功")


def _simple_concat(video_paths: list, output_path: Path, ffmpeg: str) -> None:
    """xfade が使えない場合の単純連結。"""
    list_file = output_path.parent / "_concat_list.txt"
    with open(list_file, "w", encoding="utf-8") as f:
        for vp in video_paths:
            safe = str(vp).replace("\\", "/")
            f.write(f"file '{safe}'\n")
    cmd = [
        ffmpeg, "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"単純連結も失敗:\n{result.stderr[-300:]}")
    print(f"  単純連結成功（遷移エフェクトなし）")


# ── メインパイプライン ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NanoBanana 4D - パララックステスト")
    parser.add_argument("--stages", type=int, default=9,
                        help="テストする態の数 (default: 9)")
    parser.add_argument("--midas",  action="store_true",
                        help="MiDaS 深度推定を使用")
    parser.add_argument("--reuse",  action="store_true",
                        help="既存のフレーム・動画をスキップ（差分のみ再生成）")
    args = parser.parse_args()

    num_stages = min(args.stages, len(STAGE_CONFIGS))
    use_midas  = args.midas
    reuse      = args.reuse

    # 使用する態の設定を取得
    configs = STAGE_CONFIGS[:num_stages]
    stage_secs  = [c[0] for c in configs]
    trans_secs  = [TRANSITION_MAP[i][1] for i in range(num_stages - 1)]
    total_dur   = sum(stage_secs) - sum(trans_secs)

    print("=" * 62)
    print("  NanoBanana 4D  パララックス テストパイプライン")
    print(f"  態数: {num_stages}  |  {FPS}fps  |  {WIDTH}×{HEIGHT}")
    print(f"  深度推定: {'MiDaS' if use_midas else '合成'}")
    print(f"  スキップ: {'有効' if reuse else '無効'}")
    print(f"  合計尺: ~{total_dur:.0f}s")
    print()
    for i, cfg in enumerate(configs):
        label = cfg[5]
        print(f"  ⑰態{i+1:02d} ({cfg[0]}s) {label}")
    print("=" * 62)

    # ── 環境チェック ──────────────────────────────────────────────────
    ffmpeg = get_ffmpeg()
    print(f"\n[OK] ffmpeg: {ffmpeg}")

    try:
        import cv2
        print(f"[OK] opencv-python: {cv2.__version__}")
    except ImportError:
        print("[警告] opencv-python が見つかりません。numpy fallback を使用（低速）。")

    # ── ディレクトリ準備 ─────────────────────────────────────────────
    for d in [INPUT_DIR, DEPTH_DIR, STAGES_DIR, VIDEO_DIR, FINAL_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────
    # Step 1: 入力画像の準備
    # ─────────────────────────────────────────────────────────────────
    print(f"\n[Step 1] 入力画像の確認・準備 ({num_stages} 態)")

    stage_images: list[Image.Image] = []
    for i in range(num_stages):
        found = None
        for fname in [f"stage_{i + 1:02d}.png", f"stage{i + 1:02d}.png",
                      f"stage{i + 1}.png"]:
            p = INPUT_DIR / fname
            if p.exists():
                found = p
                break

        if found:
            print(f"  [態{i + 1}] 読み込み: {found.name}")
            img = Image.open(found).convert("RGB").resize(
                (WIDTH, HEIGHT), Image.LANCZOS)
        else:
            print(f"  [態{i + 1}] テスト画像を生成します")
            img = create_test_image(i, WIDTH, HEIGHT)
            save_p = INPUT_DIR / f"stage_{i + 1:02d}_generated.png"
            img.save(save_p)
            print(f"    保存: {save_p.name}")

        stage_images.append(img)

    # ─────────────────────────────────────────────────────────────────
    # Step 2: Depth Map 生成
    # ─────────────────────────────────────────────────────────────────
    print(f"\n[Step 2] Depth Map 生成")

    stage_depths: list[np.ndarray] = []
    for i, img in enumerate(stage_images):
        npy_path = DEPTH_DIR / f"depth_{i + 1:02d}.npy"
        png_path = DEPTH_DIR / f"depth_{i + 1:02d}.png"

        if reuse and npy_path.exists():
            print(f"  [態{i + 1}] キャッシュ使用: {npy_path.name}")
            depth = np.load(str(npy_path))
        else:
            print(f"  [態{i + 1}] 深度推定中...")
            depth = (estimate_depth_midas(img) if use_midas
                     else estimate_depth_synthetic(img))
            np.save(str(npy_path), depth)
            # 可視化 PNG も保存（白=手前）
            Image.fromarray((depth * 255).astype(np.uint8)).save(str(png_path))
            print(f"    保存: {npy_path.name}")

        stage_depths.append(depth)

    # ─────────────────────────────────────────────────────────────────
    # Step 3: パララックスフレーム生成
    # ─────────────────────────────────────────────────────────────────
    total_frames_all = sum(FPS * c[0] for c in configs)
    print(f"\n[Step 3] パララックスフレーム生成"
          f" (合計 {total_frames_all} フレーム / {num_stages} 態・態ごとに可変)")

    stage_video_paths: list[Path] = []

    for i, (img, depth) in enumerate(zip(stage_images, stage_depths)):
        cfg         = configs[i]
        sec, sx_max, sy_max, zi, zo, label = cfg
        frames      = FPS * sec

        stage_dir   = STAGES_DIR / f"stage{i + 1:02d}"
        video_path  = VIDEO_DIR  / f"stage{i + 1:02d}.mp4"
        stage_dir.mkdir(exist_ok=True)

        if reuse and video_path.exists():
            print(f"  [態{i + 1}] 既存動画を使用: {video_path.name}")
            stage_video_paths.append(video_path)
            continue

        img_np = np.array(img).astype(np.float32) / 255.0
        print(f"  [態{i + 1}] {label} ({sec}s / {frames}f)")

        for fi in range(frames):
            frame_path = stage_dir / f"frame_{fi:05d}.png"
            if reuse and frame_path.exists():
                continue

            scale, sx, sy = get_camera_params(fi, frames,
                                               sx_max, sy_max, zi, zo,
                                               stage_idx=i)
            warped = warp_frame(img_np, depth, sx, sy, scale)
            Image.fromarray(
                (np.clip(warped, 0, 1) * 255).astype(np.uint8)
            ).save(str(frame_path))

            # 進捗表示（各幕の最初のフレームのみ）
            act = ("イン  " if fi < frames // 4 else
                   "見せ場" if fi < frames * 3 // 4 else
                   "アウト")
            if fi % (FPS * 2) == 0:  # 2秒ごと
                print(f"    [{act}] f{fi:03d}/{frames}"
                      f"  scale={scale:.3f}  shift_x={sx:+.4f}")

        print(f"    MP4 生成中...")
        render_stage_video(stage_dir, video_path, ffmpeg)
        print(f"    完了: {video_path.name}")
        stage_video_paths.append(video_path)

    # ─────────────────────────────────────────────────────────────────
    # Step 4: 遷移エフェクト付き連結
    # ─────────────────────────────────────────────────────────────────
    print(f"\n[Step 4] 遷移エフェクト付き連結 ({num_stages - 1} 遷移)")

    final_path = FINAL_DIR / f"4d_parallax_{num_stages}stages.mp4"
    concat_with_transitions(stage_video_paths, final_path, ffmpeg, configs)

    # ── 完了 ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 62}")
    print(f"  完成！")
    print(f"  合計尺: ~{total_dur:.0f} 秒  ({FPS}fps · {WIDTH}×{HEIGHT})")
    print(f"  → {final_path}")
    print(f"\n  Depth Map (確認用):")
    for i in range(num_stages):
        print(f"    {DEPTH_DIR / f'depth_{i + 1:02d}.png'}")
    print(f"{'=' * 62}")


if __name__ == "__main__":
    main()

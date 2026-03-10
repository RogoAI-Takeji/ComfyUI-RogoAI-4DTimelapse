# NanoBanana 4D - タイムラプス v2（stage1〜3 試作）
#
# 解決する問題:
#   (1) タイムラプスがない問題
#       -> 各態に時刻サイクル（夜明け->正午->夕暮れ->夜）の色温度変化を適用
#       -> 任意: stage_XX_end.png があれば光学フローモーフィングを重ねる
#   (2) 3D カメラ問題（今回はパララックス継続）
#       -> モーフィング後に parallax warp を重ねる
#
# 使い方（PowerShell）:
#   $PY = "D:/Python_VENV/for_comfy_ltx2_260115/Data/Packages/ComfyUI_for_LTX2/venv/Scripts/python.exe"
#   & $PY run_4d_timelapse_v2.py --stages 3
#
# オプション:
#   --stages N   : テスト態数 (default: 3)
#   --midas      : MiDaS 深度推定
#   --reuse      : 既存フレームをスキップ
#   --cycles F   : タイムラプスサイクル数 (default: 2.0)
#   --no-morph   : 光学フローモーフィングを無効化
#
# ドラマ的弧線（stage1〜3）:
#   (1)卵    静かな始まり・命の予感
#   (2)孵化  最初の動き・誕生の瞬間
#   (3)若齢  小さな命の躍動

import argparse
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# ── グローバル設定 ─────────────────────────────────────────────────────────
FPS    = 24
WIDTH  = 1920
HEIGHT = 1080

# ── 態ごとの個別設定（stage1〜9、今回は1〜3を使用） ─────────────────────
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

# ── 遷移エフェクト ────────────────────────────────────────────────────────
TRANSITION_MAP = [
    ("fadewhite", 1.5, "①→②: ゆっくり白フラッシュ・誕生の光"),
    ("dissolve",  1.0, "②→③: ディゾルブ・活発な成長"),
    ("fade",      1.0, "③→④: フェード・緑の充実"),
    ("fadeblack", 1.5, "④→⑤: 暗転・活動が落ち着く"),
    ("fadeblack", 2.0, "⑤→⑥: ゆっくり暗転・静寂へ"),
    ("fadewhite", 0.5, "⑥→⑦: 瞬間的白フラッシュ・光の爆発"),
    ("dissolve",  1.0, "⑦→⑧: やわらかディゾルブ・充足"),
    ("slideup",   1.5, "⑧→⑨: 上昇スライド・空への解放"),
]

OUTPUT_ROOT = Path(r"D:\NB4D_test")
INPUT_DIR   = Path(r"C:\Users\fareg\Desktop\老後AI\work_folder\老後画像AI_シリーズ\015_comfyui_nanobanana\making_movie\Celastrina _argiolus\9stages")
DEPTH_DIR   = OUTPUT_ROOT / "depth"
STAGES_DIR  = OUTPUT_ROOT / "stages_v2"
VIDEO_DIR   = OUTPUT_ROOT / "videos_v2"
FINAL_DIR   = OUTPUT_ROOT / "output_v2"

FFMPEG_CANDIDATES = [
    r"C:\Users\fareg\AppData\Local\FFmpeg\bin\ffmpeg.exe",
    "ffmpeg",
]


# ══════════════════════════════════════════════════════════════════════════════
# ① タイムラプス：時刻サイクル（色温度）アニメーション
# ══════════════════════════════════════════════════════════════════════════════

def _time_of_day_lut(phase: float) -> tuple:
    """
    phase 0.0〜1.0 → (明度係数, R調整, G調整, B調整)
    1サイクル = 夜明け→正午→夕暮れ→深夜
    """
    # キーフレーム: (phase, brightness, R_gain, G_gain, B_gain)
    keyframes = [
        (0.00, 0.55,  1.10, 0.92, 0.78),  # 夜明け前（青みがかった暗め）
        (0.15, 0.80,  1.15, 1.00, 0.85),  # 夜明け（暖色）
        (0.30, 1.00,  1.00, 1.00, 1.00),  # 正午（ニュートラル）
        (0.55, 0.85,  1.12, 0.98, 0.82),  # 夕方（暖色・オレンジ）
        (0.70, 0.65,  1.08, 0.90, 0.80),  # 日没直後（暖かい薄暗さ）
        (0.85, 0.40,  0.88, 0.90, 1.05),  # 深夜（青みがかった暗闇）
        (1.00, 0.55,  1.10, 0.92, 0.78),  # 夜明け前（ループ）
    ]

    # phase が属する区間を探す
    for i in range(len(keyframes) - 1):
        p0, b0, r0, g0, bl0 = keyframes[i]
        p1, b1, r1, g1, bl1 = keyframes[i + 1]
        if p0 <= phase <= p1:
            t = (phase - p0) / (p1 - p0 + 1e-9)
            # イーズイン/アウト
            t = t * t * (3.0 - 2.0 * t)
            bri  = b0  + (b1  - b0)  * t
            rg   = r0  + (r1  - r0)  * t
            gg   = g0  + (g1  - g0)  * t
            bg   = bl0 + (bl1 - bl0) * t
            return bri, rg, gg, bg

    return keyframes[-1][1:]


def apply_time_color(image_np: np.ndarray, phase: float) -> np.ndarray:
    """
    image_np : H,W,3  float32  0〜1
    phase    : 0.0〜1.0 (タイムラプス1サイクル内の位置)
    """
    bri, rg, gg, bg = _time_of_day_lut(phase)
    result = image_np.copy()
    result[:, :, 0] = np.clip(result[:, :, 0] * rg * bri, 0, 1)
    result[:, :, 1] = np.clip(result[:, :, 1] * gg * bri, 0, 1)
    result[:, :, 2] = np.clip(result[:, :, 2] * bg * bri, 0, 1)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# ① タイムラプス：光学フローモーフィング
# ══════════════════════════════════════════════════════════════════════════════

def _optical_flow_warp(img_np: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """
    img_np : H,W,3  float32  0〜1
    flow   : H,W,2  float32  ピクセル変位 (dx, dy)
    """
    try:
        import cv2
        H, W = img_np.shape[:2]
        dst_y, dst_x = np.mgrid[0:H, 0:W].astype(np.float32)
        map_x = dst_x + flow[:, :, 0]
        map_y = dst_y + flow[:, :, 1]
        img_u8 = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
        warped = cv2.remap(img_u8, map_x, map_y,
                           cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return warped.astype(np.float32) / 255.0
    except ImportError:
        return img_np


def compute_optical_flow(src_np: np.ndarray, dst_np: np.ndarray) -> np.ndarray:
    """
    src_np, dst_np : H,W,3  float32  0〜1
    return         : H,W,2  float32  (dx, dy) src→dst フロー
    """
    try:
        import cv2
        src_u8 = (np.clip(src_np, 0, 1) * 255).astype(np.uint8)
        dst_u8 = (np.clip(dst_np, 0, 1) * 255).astype(np.uint8)
        src_g  = cv2.cvtColor(src_u8, cv2.COLOR_RGB2GRAY)
        dst_g  = cv2.cvtColor(dst_u8, cv2.COLOR_RGB2GRAY)
        flow   = cv2.calcOpticalFlowFarneback(
            src_g, dst_g, None,
            pyr_scale=0.5, levels=5, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        print(f"    光学フロー計算完了 (max_disp={np.abs(flow).max():.1f}px)")
        return flow
    except ImportError:
        print("    [警告] cv2 なし。光学フローをスキップ（単純クロスディゾルブに切替）")
        return None


def morph_frame(src_np: np.ndarray, dst_np: np.ndarray,
                flow: np.ndarray, alpha: float) -> np.ndarray:
    """
    src_np, dst_np : H,W,3  float32  0〜1
    flow           : H,W,2  (src→dst フロー) または None
    alpha          : 0.0=src → 1.0=dst
    """
    if flow is None:
        # フォールバック: 単純クロスディゾルブ
        return src_np * (1.0 - alpha) + dst_np * alpha

    # 前進ワープ + 後進ワープのブレンド（FILM-style）
    fwd = _optical_flow_warp(src_np,  flow *  alpha)
    bwd = _optical_flow_warp(dst_np, -flow * (1.0 - alpha))
    return fwd * (1.0 - alpha) + bwd * alpha


# ══════════════════════════════════════════════════════════════════════════════
# ② パララックスワープ（既存システム）
# ══════════════════════════════════════════════════════════════════════════════

def estimate_depth_synthetic(image_pil: Image.Image) -> np.ndarray:
    H, W   = image_pil.height, image_pil.width
    img_np = np.array(image_pil.convert("RGB")).astype(np.float32) / 255.0
    gray   = (0.299 * img_np[:,:,0]
              + 0.587 * img_np[:,:,1]
              + 0.114 * img_np[:,:,2])
    y, x   = np.mgrid[0:H, 0:W].astype(np.float32)
    dx     = (x - W / 2) / (W / 2 + 1e-8)
    dy     = (y - H / 2) / (H / 2 + 1e-8)
    cw     = np.clip(1.0 - np.sqrt(dx**2 + dy**2) * 0.85, 0.0, 1.0)
    depth  = 0.35 * (1.0 - gray) + 0.65 * cw
    depth  = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    blur_r = max(H, W) // 50
    dp     = Image.fromarray((depth * 255).astype(np.uint8))
    dp     = dp.filter(ImageFilter.GaussianBlur(radius=blur_r))
    return np.array(dp).astype(np.float32) / 255.0


def estimate_depth_midas(image_pil: Image.Image) -> np.ndarray:
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
                pred.unsqueeze(1), size=img_u8.shape[:2],
                mode="bicubic", align_corners=False).squeeze()
        depth = pred.cpu().numpy().astype(np.float32)
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        print(f"    MiDaS 完了")
        return depth
    except Exception as e:
        print(f"    MiDaS 失敗 ({e}) → 合成深度にフォールバック")
        return estimate_depth_synthetic(image_pil)


def get_camera_params(frame_idx: int, total_frames: int,
                      max_shift_x: float, max_shift_y: float,
                      zoom_in: float, zoom_out: float,
                      stage_idx: int = 0) -> tuple:
    t = frame_idx / max(1, total_frames - 1)
    if t < 0.25:
        act_t   = t / 0.25
        scale   = 1.0 + (zoom_in - 1.0) * act_t
        shift_x = 0.0
        shift_y = 0.0
    elif t < 0.75:
        act_t   = (t - 0.25) / 0.5
        scale   = zoom_in
        shift_x = (-1.0 + 2.0 * act_t) * max_shift_x
        shift_y = -act_t * max_shift_y if stage_idx == 8 else 0.0
    else:
        act_t   = (t - 0.75) / 0.25
        scale   = zoom_in - (zoom_in - zoom_out) * act_t
        shift_x = max_shift_x
        shift_y = -max_shift_y if stage_idx == 8 else 0.0
    return scale, shift_x, shift_y


def warp_frame(image_np: np.ndarray, depth_np: np.ndarray,
               shift_x_frac: float, shift_y_frac: float,
               scale: float) -> np.ndarray:
    H, W       = image_np.shape[:2]
    shift_x_px = shift_x_frac * W
    shift_y_px = shift_y_frac * H
    cx, cy     = W / 2.0, H / 2.0
    dst_y, dst_x = np.mgrid[0:H, 0:W].astype(np.float32)
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
        src_xi = np.clip(src_x.astype(np.int32), 0, W - 1)
        src_yi = np.clip(src_y.astype(np.int32), 0, H - 1)
        return image_np[src_yi, src_xi, :]


# ══════════════════════════════════════════════════════════════════════════════
# テスト用合成画像
# ══════════════════════════════════════════════════════════════════════════════

def create_test_image(stage_idx: int, width: int, height: int) -> Image.Image:
    cfgs = [
        ((30, 100, 30), (200, 220, 140), "egg",              "Stage 1: 卵 (Egg)"),
        ((40, 120, 40), (170, 200, 110), "caterpillar_small","Stage 2: 孵化 (Hatching)"),
        ((20,  90, 20), (150, 185,  85), "caterpillar_large","Stage 3: 若齢 (Young Larva)"),
        ((90,  65, 30), (165, 135,  70), "pupa",             "Stage 4: 蛹 (Pupa)"),
    ]
    cfg = cfgs[stage_idx % len(cfgs)]
    bg_color, fg_color, shape, label = cfg
    img  = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)
    dim_bg = tuple(max(0, c - 18) for c in bg_color)
    for i in range(0, width, 70):
        draw.line([(i, 0), (i + height // 3, height)], fill=dim_bg, width=2)
    for j in range(0, height, 90):
        draw.line([(0, j), (width, j + 50)], fill=dim_bg, width=1)
    cx, cy = width // 2, height // 2
    if shape == "egg":
        r = min(width, height) // 11
        draw.ellipse([cx - r, cy - r // 2, cx + r, cy + r // 2], fill=fg_color)
        draw.ellipse([cx - r // 3, cy - r // 4,
                      cx + r // 6, cy + r // 6], fill=(230, 240, 200))
    elif shape == "caterpillar_small":
        r = min(width, height) // 18
        for k in range(5):
            x = cx + int((k - 2) * r * 1.6)
            shade = tuple(min(255, c + k * 4) for c in fg_color)
            draw.ellipse([x - r, cy - r // 2, x + r, cy + r // 2], fill=shade)
    elif shape == "caterpillar_large":
        r = min(width, height) // 12
        for k in range(8):
            x = cx + int((k - 3.5) * r * 1.45)
            shade = tuple(min(255, c + k * 3) for c in fg_color)
            draw.ellipse([x - r, cy - int(r * 0.7), x + r, cy + int(r * 0.7)], fill=shade)
    elif shape == "pupa":
        rx, ry = width // 8, height // 5
        draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=fg_color)
        draw.line([(cx - rx // 2, cy - ry), (cx - rx // 2, cy - ry - 50)],
                  fill=(220, 205, 155), width=4)
    try:
        from PIL import ImageFont
        font = ImageFont.load_default()
        draw.text((40, 40), label, fill=(255, 255, 200), font=font)
    except Exception:
        pass
    return img.filter(ImageFilter.GaussianBlur(radius=2))


# ══════════════════════════════════════════════════════════════════════════════
# FFmpeg
# ══════════════════════════════════════════════════════════════════════════════

def get_ffmpeg() -> str:
    for p in FFMPEG_CANDIDATES:
        try:
            subprocess.run([p, "-version"], capture_output=True, check=True)
            return p
        except Exception:
            continue
    raise RuntimeError("ffmpeg が見つかりません。PATH を確認してください。")


def render_stage_video(stage_dir: Path, video_path: Path, ffmpeg: str) -> None:
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
    num_stages = len(video_paths)
    if num_stages == 1:
        shutil.copy(str(video_paths[0]), str(output_path))
        return

    inputs = []
    for vp in video_paths:
        inputs += ["-i", str(vp)]

    filter_parts  = []
    current_label = "[0:v]"
    cumulative    = 0.0

    for i in range(num_stages - 1):
        stage_sec  = configs[i][0]
        trans_rec  = TRANSITION_MAP[i] if i < len(TRANSITION_MAP) else ("fade", 1.0, "")
        trans_name, trans_dur, comment = trans_rec
        cumulative += stage_sec
        offset      = cumulative - trans_dur * (i + 1)
        is_last     = (i == num_stages - 2)
        out_label   = "[vout]" if is_last else f"[v{i + 1:02d}]"
        print(f"    遷移 態{i+1}→態{i+2}: [{trans_name} {trans_dur}s]  "
              f"offset={offset:.1f}s  {comment}")
        filter_parts.append(
            f"{current_label}[{i + 1}:v]"
            f"xfade=transition={trans_name}"
            f":duration={trans_dur}:offset={offset:.3f}"
            f"{out_label}"
        )
        current_label = out_label

    cmd = [ffmpeg, "-y"] + inputs + [
        "-filter_complex", ";".join(filter_parts),
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
    list_file = output_path.parent / "_concat_list.txt"
    with open(str(list_file), "w", encoding="utf-8") as f:
        for vp in video_paths:
            safe = str(vp).replace("\\", "/")
            f.write(f"file '{safe}'\n")
    cmd = [ffmpeg, "-y", "-f", "concat", "-safe", "0",
           "-i", str(list_file), "-c", "copy", str(output_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"単純連結も失敗:\n{result.stderr[-300:]}")
    print(f"  単純連結成功（遷移エフェクトなし）")


# ══════════════════════════════════════════════════════════════════════════════
# メインパイプライン
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="NanoBanana 4D v2 - タイムラプス＋パララックス")
    parser.add_argument("--stages", type=int, default=3,
                        help="テストする態の数 (default: 3)")
    parser.add_argument("--midas",    action="store_true",
                        help="MiDaS 深度推定を使用")
    parser.add_argument("--reuse",    action="store_true",
                        help="既存のフレーム・動画をスキップ")
    parser.add_argument("--cycles",   type=float, default=2.0,
                        help="タイムラプスサイクル数 (default: 2.0 = 昼夜2往復)")
    parser.add_argument("--no-morph", action="store_true",
                        help="光学フローモーフィングを無効化")
    args = parser.parse_args()

    num_stages    = min(args.stages, len(STAGE_CONFIGS))
    use_midas     = args.midas
    reuse         = args.reuse
    tl_cycles     = args.cycles
    use_morph     = not args.no_morph

    configs    = STAGE_CONFIGS[:num_stages]
    stage_secs = [c[0] for c in configs]
    trans_secs = [TRANSITION_MAP[i][1] for i in range(num_stages - 1)]
    total_dur  = sum(stage_secs) - sum(trans_secs)

    print("=" * 66)
    print("  NanoBanana 4D v2  タイムラプス＋パララックス試作")
    print(f"  態数: {num_stages}  |  {FPS}fps  |  {WIDTH}×{HEIGHT}")
    print(f"  タイムラプスサイクル数: {tl_cycles:.1f}  |  深度: {'MiDaS' if use_midas else '合成'}")
    print(f"  光学フローモーフィング: {'有効' if use_morph else '無効'}")
    print(f"  合計尺: ~{total_dur:.0f}s")
    print()
    print("  ─ タイムラプス設計 ──────────────────────────────────────────")
    print("  各フレームに「時刻サイクル（夜明け→正午→夕暮れ→深夜）」の")
    print("  色温度変化を重ねて、時間の経過を視覚的に表現します。")
    print("  stage_XX_end.png があれば光学フローで形状変化も加えます。")
    print()
    for i, cfg in enumerate(configs):
        print(f"  態{i+1:02d} ({cfg[0]}s) {cfg[5]}")
    print("=" * 66)

    # ── 環境チェック ─────────────────────────────────────────────────
    ffmpeg = get_ffmpeg()
    print(f"\n[OK] ffmpeg: {ffmpeg}")
    try:
        import cv2
        print(f"[OK] opencv-python: {cv2.__version__}")
    except ImportError:
        print("[警告] cv2 なし → 光学フロー無効・numpy fallback")

    for d in [INPUT_DIR, DEPTH_DIR, STAGES_DIR, VIDEO_DIR, FINAL_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────
    # Step 1: 入力画像の準備（start + end）
    # ──────────────────────────────────────────────────────────────────
    print(f"\n[Step 1] 入力画像の確認・準備 ({num_stages} 態)")

    stage_starts: list[Image.Image] = []
    stage_ends:   list[Image.Image] = []

    for i in range(num_stages):
        # --- 始点画像 ---
        found_start = None
        for fname in [f"stage_{i+1:02d}.png", f"stage{i+1:02d}.png",
                      f"stage{i+1}.png"]:
            p = INPUT_DIR / fname
            if p.exists():
                found_start = p
                break

        if found_start:
            print(f"  [態{i+1}] 始点: {found_start.name}")
            img_s = Image.open(str(found_start)).convert("RGB").resize(
                (WIDTH, HEIGHT), Image.LANCZOS)
        else:
            print(f"  [態{i+1}] 始点: テスト画像を生成")
            img_s = create_test_image(i, WIDTH, HEIGHT)
            img_s.save(str(INPUT_DIR / f"stage_{i+1:02d}_gen_start.png"))

        stage_starts.append(img_s)

        # --- 終点画像（任意）---
        found_end = None
        for fname in [f"stage_{i+1:02d}_end.png", f"stage{i+1:02d}_end.png"]:
            p = INPUT_DIR / fname
            if p.exists():
                found_end = p
                break

        if found_end and use_morph:
            print(f"  [態{i+1}] 終点: {found_end.name}  → 光学フローモーフィング有効")
            img_e = Image.open(str(found_end)).convert("RGB").resize(
                (WIDTH, HEIGHT), Image.LANCZOS)
            stage_ends.append(img_e)
        else:
            if use_morph:
                print(f"  [態{i+1}] 終点: なし → 色温度サイクルのみ（モーフィングなし）")
            stage_ends.append(None)

    # ──────────────────────────────────────────────────────────────────
    # Step 2: Depth Map 生成（始点画像から）
    # ──────────────────────────────────────────────────────────────────
    print(f"\n[Step 2] Depth Map 生成")

    stage_depths: list[np.ndarray] = []
    for i, img in enumerate(stage_starts):
        npy_path = DEPTH_DIR / f"depth_{i+1:02d}.npy"
        png_path = DEPTH_DIR / f"depth_{i+1:02d}.png"

        if reuse and npy_path.exists():
            print(f"  [態{i+1}] キャッシュ使用: {npy_path.name}")
            depth = np.load(str(npy_path))
        else:
            print(f"  [態{i+1}] 深度推定中...")
            depth = (estimate_depth_midas(img) if use_midas
                     else estimate_depth_synthetic(img))
            np.save(str(npy_path), depth)
            Image.fromarray((depth * 255).astype(np.uint8)).save(str(png_path))
            print(f"    保存: {npy_path.name}")

        stage_depths.append(depth)

    # ──────────────────────────────────────────────────────────────────
    # Step 3: 光学フロー計算（終点画像がある態のみ）
    # ──────────────────────────────────────────────────────────────────
    print(f"\n[Step 3] 光学フロー計算")

    stage_flows = []
    for i, (img_s, img_e) in enumerate(zip(stage_starts, stage_ends)):
        if img_e is None:
            stage_flows.append(None)
            print(f"  [態{i+1}] スキップ（終点画像なし）")
            continue

        flow_path = DEPTH_DIR / f"flow_{i+1:02d}.npy"
        if reuse and flow_path.exists():
            print(f"  [態{i+1}] キャッシュ使用: {flow_path.name}")
            flow = np.load(str(flow_path))
        else:
            print(f"  [態{i+1}] 光学フロー計算中...")
            src_np = np.array(img_s).astype(np.float32) / 255.0
            dst_np = np.array(img_e).astype(np.float32) / 255.0
            flow   = compute_optical_flow(src_np, dst_np)
            if flow is not None:
                np.save(str(flow_path), flow)
                print(f"    保存: {flow_path.name}")

        stage_flows.append(flow)

    # ──────────────────────────────────────────────────────────────────
    # Step 4: フレーム生成（タイムラプス色温度 + モーフィング + パララックス）
    # ──────────────────────────────────────────────────────────────────
    total_frames_all = sum(FPS * c[0] for c in configs)
    print(f"\n[Step 4] フレーム生成 (合計 {total_frames_all} フレーム)")
    print(f"  パイプライン: 時刻色温度 → 光学フローモーフィング → パララックスワープ")

    stage_video_paths: list[Path] = []

    for i in range(num_stages):
        cfg            = configs[i]
        sec, sx_max, sy_max, zi, zo, label = cfg
        frames         = FPS * sec

        img_s_np = np.array(stage_starts[i]).astype(np.float32) / 255.0
        img_e_np = (np.array(stage_ends[i]).astype(np.float32) / 255.0
                    if stage_ends[i] is not None else None)
        depth    = stage_depths[i]
        flow     = stage_flows[i] if i < len(stage_flows) else None

        stage_dir  = STAGES_DIR / f"stage{i+1:02d}"
        video_path = VIDEO_DIR  / f"stage{i+1:02d}.mp4"
        stage_dir.mkdir(exist_ok=True)

        if reuse and video_path.exists():
            print(f"  [態{i+1}] 既存動画を使用: {video_path.name}")
            stage_video_paths.append(video_path)
            continue

        print(f"\n  [態{i+1}] {label} ({sec}s / {frames}f)")
        morph_mode = ("光学フロー" if (img_e_np is not None and flow is not None)
                      else "クロスディゾルブ" if img_e_np is not None
                      else "色温度のみ")
        print(f"    タイムラプス方式: {morph_mode}")

        for fi in range(frames):
            frame_path = stage_dir / f"frame_{fi:05d}.png"
            if reuse and frame_path.exists():
                continue

            # ① タイムラプス位相（サイクル数に応じて0〜1を繰り返す）
            t_stage   = fi / max(1, frames - 1)          # 0.0〜1.0（態内の位置）
            tl_phase  = (t_stage * tl_cycles) % 1.0      # 時刻サイクル位相

            # ② モーフィング位相（始点→終点に向かって単調増加）
            alpha = t_stage  # 態の始め=始点, 態の終わり=終点

            # ③ ベースフレーム生成（モーフィング）
            if img_e_np is not None:
                base_np = morph_frame(img_s_np, img_e_np, flow, alpha)
            else:
                base_np = img_s_np.copy()

            # ④ 時刻色温度を適用
            base_np = apply_time_color(base_np, tl_phase)

            # ⑤ パララックスワープを適用
            scale, sx, sy = get_camera_params(fi, frames,
                                               sx_max, sy_max, zi, zo,
                                               stage_idx=i)
            result_np = warp_frame(base_np, depth, sx, sy, scale)

            # ⑥ 保存
            Image.fromarray(
                (np.clip(result_np, 0, 1) * 255).astype(np.uint8)
            ).save(str(frame_path))

            # 進捗表示（4秒ごと）
            if fi % (FPS * 4) == 0:
                act = ("イン  " if fi < frames // 4 else
                       "見せ場" if fi < frames * 3 // 4 else
                       "アウト")
                hour = tl_phase * 24
                print(f"    [{act}] f{fi:03d}/{frames}"
                      f"  α={alpha:.2f}  時刻≈{hour:.0f}h"
                      f"  scale={scale:.3f}  sx={sx:+.4f}")

        print(f"    MP4 生成中...")
        render_stage_video(stage_dir, video_path, ffmpeg)
        print(f"    完了: {video_path.name}")
        stage_video_paths.append(video_path)

    # ──────────────────────────────────────────────────────────────────
    # Step 5: 遷移エフェクト付き連結
    # ──────────────────────────────────────────────────────────────────
    print(f"\n[Step 5] 遷移エフェクト付き連結 ({num_stages - 1} 遷移)")

    final_path = FINAL_DIR / f"4d_timelapse_v2_{num_stages}stages.mp4"
    concat_with_transitions(stage_video_paths, final_path, ffmpeg, configs)

    print(f"\n{'=' * 66}")
    print(f"  完成！")
    print(f"  合計尺: ~{total_dur:.0f} 秒  ({FPS}fps · {WIDTH}×{HEIGHT})")
    print(f"  → {final_path}")
    print()
    print(f"  【タイムラプス効果】")
    print(f"    - 各態: {tl_cycles:.0f}サイクルの昼夜変化（色温度）")
    print(f"    - 形状変化: stage_XX_end.png があれば光学フローモーフィング")
    print()
    print(f"  【次のステップ】")
    print(f"    各態の終点画像 (stage_01_end.png など) を用意すると")
    print(f"    形状変化も含んだより本格的なタイムラプスになります。")
    print(f"{'=' * 66}")


if __name__ == "__main__":
    main()

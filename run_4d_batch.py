"""
NanoBanana 4D - スタンドアロン バッチ生成スクリプト
==================================================
ComfyUI のワークフローを使わずに、直接 Gemini API を叩いて
100枚の連番画像を生成し、FFmpeg で動画を作成します。

使い方:
  cd D:\Python_VENV\for_comfy_ltx2_260115\Data\Packages\ComfyUI_for_LTX2
  python custom_nodes\ComfyUI-RogoAI-NanoBanana\run_4d_batch.py
"""

import os
import io
import json
import math
import time
import subprocess
from pathlib import Path
from PIL import Image

# ── 設定 ─────────────────────────────────────────────────────────────
TOTAL_FRAMES = 300      # 100から300に増加（30fpsで10秒）
FPS = 30
REVOLUTIONS = 1.5       # カメラの回転数
START_ELEVATION = 0.0
END_ELEVATION = 45.0
MODEL = "gemini-3.1-flash-image-preview"

# 出力先 (ASCII のみのパスにすること！)
OUTPUT_DIR = Path(r"D:\NB4D_output")
FRAMES_DIR = OUTPUT_DIR / "frames"
VIDEO_PATH = OUTPUT_DIR / "ruri_shijimi_4d_timelapse.mp4"

# APIキーファイル
API_KEY_FILE = Path(r"D:\Python_VENV\for_comfy_ltx2_260115\Data\Packages\ComfyUI_for_LTX2\models\gemini_api_key.txt")

# アンカー画像フォルダ（あれば使う、なければなしで生成）
ANCHOR_DIR = None  # 例: Path(r"D:\NB4D_output\anchors")

# ── ライフステージ定義 ───────────────────────────────────────────────
STAGES = [
    {"t": 0.00, "stage": "egg",         "desc": "A tiny, pale-green, lens-shaped egg of Celastrina argiolus (Holly Blue butterfly) laid on a flower bud of a host plant, extreme macro photography"},
    {"t": 0.15, "stage": "larva_early", "desc": "A very small, slug-like pale green caterpillar of Celastrina argiolus, camouflaged on a young flower bud"},
    {"t": 0.35, "stage": "larva_late",  "desc": "A mature slug-like green caterpillar of Celastrina argiolus (Pale Grass Blue), smooth textured, resting on a green leaf"},
    {"t": 0.50, "stage": "pupa",        "desc": "A small, rounded, brownish-grey pupa of Celastrina argiolus, attached to a stem with a silk girdle, macro photography"},
    {"t": 0.65, "stage": "eclosion",    "desc": "A small blue butterfly, Celastrina argiolus, emerging from its greyish pupa. The wings are wet and small, showing soft silvery-blue underside with black spots. No orange, no black veins like monarch."},
    {"t": 0.80, "stage": "adult_open",  "desc": "A beautiful Holly Blue butterfly (Celastrina argiolus) with vibrant sky-blue wings fully open, delicate black borders, resting on a flower"},
    {"t": 1.00, "stage": "flying",      "desc": "A small sky-blue butterfly, Celastrina argiolus, fluttering around green foliage in soft sunlight"},
]

# ── ヘルパー関数 ─────────────────────────────────────────────────────

def load_api_key():
    """APIキーを読み込む"""
    if API_KEY_FILE.exists():
        key = API_KEY_FILE.read_text(encoding="utf-8").strip()
        if key:
            print(f"[NB4D] APIキーを {API_KEY_FILE.name} から読み込みました。")
            return key
    raise RuntimeError(f"[NB4D] APIキーが見つかりません: {API_KEY_FILE}")

def get_stage_prompt(t):
    """時間 t に対応するプロンプトを取得する"""
    stages = sorted(STAGES, key=lambda x: x["t"])
    
    prev_s = stages[0]
    next_s = stages[-1]
    
    for i in range(len(stages) - 1):
        if stages[i]["t"] <= t <= stages[i+1]["t"]:
            prev_s = stages[i]
            next_s = stages[i+1]
            break
    
    if prev_s["t"] == next_s["t"]:
        return prev_s["desc"], prev_s["stage"]
    
    local_t = (t - prev_s["t"]) / (next_s["t"] - prev_s["t"])
    
    if local_t < 0.3:
        return prev_s["desc"], prev_s["stage"]
    elif local_t > 0.7:
        return next_s["desc"], next_s["stage"]
    else:
        combined = f"Between {prev_s['desc']} and {next_s['desc']}, focusing on {next_s['stage']} stage"
        return combined, f"{prev_s['stage']}_to_{next_s['stage']}"

def get_camera_desc(frame_index, total_frames):
    """フレーム番号からカメラの角度の説明を取得する"""
    t = frame_index / max(1, total_frames - 1)
    
    theta = (t * 360 * REVOLUTIONS) % 360
    phi = START_ELEVATION + (END_ELEVATION - START_ELEVATION) * t
    
    # 左右
    lr = "center"
    if 15 < theta <= 165: lr = "right side"
    elif 195 < theta <= 345: lr = "left side"
    
    # 上下
    ud = "eye level"
    if phi > 20: ud = "high angle (looking down)"
    elif phi < -20: ud = "low angle (looking up)"
    
    # 距離
    dist = "medium shot"
    if t < 0.2: dist = "extreme close up"
    elif t < 0.5: dist = "close up"
    
    return lr, ud, dist

def compose_prompt(stage_prompt, lr, ud, dist):
    """最終プロンプトを組み立てる"""
    return (
        f"{stage_prompt}, "
        f"seen from the {lr} at a {ud}, "
        f"{dist}, "
        f"photorealistic, 8k, highly detailed, nature documentary"
    )

def load_anchor_for_stage(t):
    """アンカー画像を読み込む（設定されている場合）"""
    if ANCHOR_DIR is None or not Path(ANCHOR_DIR).exists():
        return None
    
    # t に最も近いアンカーを選ぶ
    anchor_files = sorted(Path(ANCHOR_DIR).glob("*.png"))
    if not anchor_files:
        anchor_files = sorted(Path(ANCHOR_DIR).glob("*.jpg"))
    if not anchor_files:
        return None
    
    num_anchors = len(anchor_files)
    idx = min(int(t * num_anchors), num_anchors - 1)
    
    print(f"  [アンカー] {anchor_files[idx].name}")
    return Image.open(anchor_files[idx])

def pil_to_bytes(img: Image.Image) -> bytes:
    """PIL → PNG bytes"""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def generate_one_frame(client, genai_types, prompt, ref_image=None):
    """Gemini Flash で1枚生成する"""
    if ref_image:
        response_modalities = ["TEXT", "IMAGE"]
    else:
        response_modalities = ["IMAGE"]
    
    config = genai_types.GenerateContentConfig(
        response_modalities=response_modalities,
    )
    
    if ref_image:
        parts = [genai_types.Part(text=prompt)]
        parts.append(
            genai_types.Part(
                inline_data=genai_types.Blob(
                    mime_type="image/png",
                    data=pil_to_bytes(ref_image),
                )
            )
        )
        contents = [genai_types.Content(role="user", parts=parts)]
    else:
        contents = prompt
    
    response = client.models.generate_content(
        model=MODEL,
        contents=contents,
        config=config,
    )
    
    # レスポンスから画像を抽出
    if response.candidates:
        for part in response.candidates[0].content.parts:
            if hasattr(part, "inline_data") and part.inline_data is not None:
                return Image.open(io.BytesIO(part.inline_data.data))
    
    return None

def get_ffmpeg_path():
    """ffmpegの実行パスを特定する"""
    candidates = [
        r"C:\Users\fareg\AppData\Local\FFmpeg\bin\ffmpeg.exe",
        "ffmpeg", # PATHにある場合
    ]
    
    for path in candidates:
        try:
            subprocess.run([path, "-version"], capture_output=True, check=True)
            return path
        except:
            continue
    return None

def assemble_video():
    """FFmpeg で連番画像を動画に変換する"""
    output_path = str(VIDEO_PATH)
    
    ffmpeg_cmd = get_ffmpeg_path()
    if not ffmpeg_cmd:
        tried = r"C:\Users\fareg\AppData\Local\FFmpeg\bin\ffmpeg.exe"
        print(f"[NB4D] ❌ エラー: ffmpeg が見つかりません。({tried} または PATH)")
        return

    # 欠損があっても最後まで動画にするため、ファイルリストを作成する
    frame_files = sorted(FRAMES_DIR.glob("frame_*.png"))
    if not frame_files:
        print("[NB4D] ❌ フレーム画像が見つかりません。")
        return

    list_path = OUTPUT_DIR / "ffmpeg_list.txt"
    with open(list_path, "w", encoding="utf-8") as f:
        for fp in frame_files:
            # FFmpegのconcatデマクサ形式: file 'path/to/file'
            # パス内のバックスラッシュをエスケープするかスラッシュに変換
            p = str(fp.absolute()).replace("\\", "/")
            f.write(f"file '{p}'\n")
            f.write(f"duration {1.0/FPS}\n")
    
    # 最後のフレームにもdurationが必要な場合があるため、リストの最後に再度記述するか
    # または通常の順次読み込みモードに戻す（ただし欠損は飛ばしたい）
    
    # より確実な方法: 欠損ファイルをコピーして埋める (88番など)
    print(f"\n[NB4D] 欠損チェックと補完...")
    for i in range(TOTAL_FRAMES):
        target = FRAMES_DIR / f"frame_{i:05d}.png"
        if not target.exists():
            # 直前のフレームを探す
            prev_idx = i - 1
            while prev_idx >= 0:
                prev_path = FRAMES_DIR / f"frame_{prev_idx:05d}.png"
                if prev_path.exists():
                    import shutil
                    shutil.copy(prev_path, target)
                    print(f"  ⚠ {target.name} が欠損しているため {prev_path.name} で補完しました。")
                    break
                prev_idx -= 1

    input_pattern = str(FRAMES_DIR / "frame_%05d.png")
    cmd = [
        ffmpeg_cmd, '-y',
        '-framerate', str(FPS),
        '-i', input_pattern,
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', '18',
        output_path,
    ]
    
    print(f"\n[NB4D] 動画生成中... {output_path}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"[NB4D] ✅ 動画が完成しました！ → {output_path}")
    else:
        print(f"[NB4D] ❌ FFmpegエラー: {result.stderr[:500]}")

# ── メイン ───────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  NanoBanana 4D - ルリシジミ立体タイムラプス生成")
    print("=" * 60)
    
    # google-genai をインポート
    try:
        from google import genai
        from google.genai import types as genai_types
    except ImportError:
        print("[NB4D] エラー: google-genai がインストールされていません。")
        print("  pip install google-genai>=1.0.0")
        return
    
    # APIキー読み込み
    api_key = load_api_key()
    client = genai.Client(api_key=api_key)
    
    # 出力ディレクトリを作成
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    
    # 既存のフレームをチェックし、欠損を埋める
    print(f"\n[NB4D] 生成・欠損チェック開始: {TOTAL_FRAMES} フレーム | {FPS}fps")
    print(f"[NB4D] 出力先: {FRAMES_DIR}\n")
    
    prev_image = None
    last_primary_stage = None
    break_counter = 0
    
    for frame_idx in range(TOTAL_FRAMES):
        save_path = FRAMES_DIR / f"frame_{frame_idx:05d}.png"
        
        # 既に存在するかチェック
        if save_path.exists():
            try:
                prev_image = Image.open(save_path)
                # ファイルから現在のステージを特定して記録しておく
                t = frame_idx / max(1, TOTAL_FRAMES - 1)
                _, stage_name = get_stage_prompt(t)
                last_primary_stage = stage_name
                continue
            except:
                pass

        # 欠損している場合は生成
        t = frame_idx / max(1, TOTAL_FRAMES - 1)
        
        # プロンプト生成
        stage_prompt, stage_name = get_stage_prompt(t)
        lr, ud, dist = get_camera_desc(frame_idx, TOTAL_FRAMES)
        final_prompt = compose_prompt(stage_prompt, lr, ud, dist)
        
        # ステージ遷移の検知（一貫性が強すぎて変化できないのを防ぐ）
        is_break_frame = False
        if last_primary_stage and stage_name != last_primary_stage:
            print(f"[{frame_idx + 1:3d}/{TOTAL_FRAMES}] ✨ ステージ遷移検知: {last_primary_stage} -> {stage_name}")
            break_counter = 5 # 遷移開始から5フレームはリファレンスを弱める/外す
        
        last_primary_stage = stage_name
        
        print(f"[{frame_idx + 1:3d}/{TOTAL_FRAMES}] 🔄 生成: t={t:.3f} | {stage_name} | {lr}, {ud} {'(Ref Break)' if break_counter > 0 else ''}")
        
        # リファレンス画像を決定
        ref_image = None
        if break_counter > 0:
            # 遷移中はアンカーのみ（あれば）、またはリファレンスなしにする
            anchor = load_anchor_for_stage(t)
            if anchor:
                ref_image = anchor
            break_counter -= 1
        else:
            anchor = load_anchor_for_stage(t)
            if anchor:
                ref_image = anchor
            elif prev_image:
                ref_image = prev_image
        
        # 生成（リトライ付き）
        max_retries = 3
        result_image = None
        for attempt in range(max_retries):
            try:
                result_image = generate_one_frame(client, genai_types, final_prompt, ref_image)
                if result_image:
                    break
                else:
                    print(f"  ⚠ 画像なし（フィルター除外の可能性）。リトライ {attempt + 1}/{max_retries}")
            except Exception as e:
                print(f"  ⚠ エラー: {e}")
                if attempt < max_retries - 1:
                    wait_sec = 5 * (attempt + 1)
                    print(f"  {wait_sec} 秒後にリトライ...")
                    time.sleep(wait_sec)
        
        if result_image:
            result_image.save(save_path)
            prev_image = result_image
            print(f"  ✅ 保存: {save_path.name} ({result_image.size[0]}x{result_image.size[1]})")
        else:
            print(f"  ❌ フレーム {frame_idx} の生成に失敗しました。スキップします。")
        
        # API レート制限対策（1秒待ち）
        time.sleep(1)
    
    # 動画生成
    print(f"\n[NB4D] {TOTAL_FRAMES} フレームの生成が完了！")
    assemble_video()
    
    print("\n" + "=" * 60)
    print("  🦋 完成！ルリシジミ立体タイムラプスの動画が作成されました！")
    print(f"  → {VIDEO_PATH}")
    print("=" * 60)

if __name__ == "__main__":
    main()

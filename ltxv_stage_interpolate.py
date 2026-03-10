#!/usr/bin/env python3
"""
ltxv_stage_interpolate.py
=========================
LTX-2 (img2vid) を ComfyUI API 経由で呼び出し、
4D グリッドの stage 間を AI 補間して滑らかな動画を生成する。

処理フロー:
  stage_00/angle_000.png → LTX-2 img2vid → 49フレーム
  stage_01/angle_000.png → LTX-2 img2vid → 49フレーム
  ...
  stage_07/angle_000.png → LTX-2 img2vid → 49フレーム
  → 全クリップを連結 → ズームイン/アウト効果 → RGBA PNG連番出力

前提:
  ComfyUI が http://127.0.0.1:8188 で起動中であること

使い方:
  $PY = "D:/Python_VENV/for_comfy_ltx2_260115/Data/Packages/ComfyUI_for_LTX2/venv/Scripts/python.exe"
  & $PY ltxv_stage_interpolate.py `
      --keyframes_dir "D:/NB4D_test/cat/grid_keyframes/run_20260308_071601" `
      --output_dir    "D:/NB4D_test/cat/output_ltxv" `
      --zoom_start    2.5 `
      --zoom_frames   72
"""

import argparse
import json
import sys
import time
import urllib.request
import urllib.error
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# ─── 定数 ─────────────────────────────────────────────────────────────────────

COMFYUI_URL  = "http://127.0.0.1:8188"
GRID_T       = 120
ANGLE_FRONT  = 0       # 正面アングル番号
CLIP_FRAMES  = 49      # 1クリップのフレーム数 (8n+1 推奨: 25/49/97)
RENDER_W     = 768
RENDER_H     = 512
BG_GRAY      = 0.12
ALPHA_THRESH = 18
OVERLAP      = 1       # クリップ間の重複フレーム

# LTX-2 モデルファイル名（ComfyUI の models フォルダ内の相対パス）
# DiT: GGUF Q6_K（ユーザー指定「GGUF優先」）
MODEL_GGUF      = r"ltx2\ltx-2-19b-distilled_Q6_K.gguf"
MODEL_VAE       = r"ltx2\LTX2_video_vae_bf16.safetensors"
# テキストエンコーダー: DualCLIPLoader (type=ltxv)
#   Gemma 3 12B FP4混合（OOMリスクの低い軽量版）+ embeddings connector
GEMMA_FP4       = r"ltx2\gemma_3_12B_it_fp4_mixed.safetensors"
EMBED_CONNECTOR = r"ltx2\ltx-2-19b-embeddings_connector_distill_bf16.safetensors"

POSITIVE_PROMPT = (
    "cat grooming, washing face with paw, natural fur movement, "
    "soft lighting, gentle animation, photorealistic, high quality"
)
NEGATIVE_PROMPT = (
    "blurry, distorted, low quality, morphing artifacts, "
    "color shift, background change, double exposure, ugly"
)


# ─── ComfyUI API ───────────────────────────────────────────────────────────────

def _api(path: str, data: dict | None = None) -> dict:
    """ComfyUI REST API 呼び出しユーティリティ"""
    url = COMFYUI_URL + path
    if data is None:
        req = urllib.request.Request(url)
    else:
        body = json.dumps(data).encode("utf-8")
        req  = urllib.request.Request(url, data=body,
                                       headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} {e.reason}: {body[:1000]}") from e


def upload_image(img_path: Path) -> str:
    """
    PNG ファイルを ComfyUI にアップロードし、ComfyUI 側のファイル名を返す。
    multipart/form-data で /upload/image に POST する。
    """
    import io, mimetypes
    boundary = "----FormBoundary" + uuid.uuid4().hex
    body = io.BytesIO()

    def write(s: str):
        body.write(s.encode("utf-8"))

    write(f"--{boundary}\r\n")
    write(f'Content-Disposition: form-data; name="image"; filename="{img_path.name}"\r\n')
    write("Content-Type: image/png\r\n\r\n")
    body.write(img_path.read_bytes())
    write(f"\r\n--{boundary}\r\n")
    write('Content-Disposition: form-data; name="overwrite"\r\n\r\n')
    write("true")
    write(f"\r\n--{boundary}--\r\n")

    req = urllib.request.Request(
        COMFYUI_URL + "/upload/image",
        data   = body.getvalue(),
        headers= {"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method = "POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        result = json.load(r)
    return result["name"]  # ComfyUI 側のファイル名


def build_workflow(
    image_name: str,
    positive:   str,
    negative:   str,
    num_frames: int,
    cfg:        float,
    steps:      int,
    seed:       int,
) -> dict:
    """
    ComfyUI API 形式のワークフロー JSON を構築して返す。

    ノード構成:
      0: UnetLoaderGGUF          → MODEL  (GGUF Q6_K)
      1: DualCLIPLoader          → CLIP   (Gemma FP4 + embeddings connector, type=ltxv)
      2: VAELoader               → VAE
      3: CLIPTextEncode (pos)    → CONDITIONING_pos
      4: CLIPTextEncode (neg)    → CONDITIONING_neg
      5: LTXVConditioning        → CONDITIONING_pos2, CONDITIONING_neg2
      6: LoadImage               → IMAGE
      7: LTXVImgToVideo          → CONDITIONING_pos3, CONDITIONING_neg3, LATENT
      8: CFGGuider               → GUIDER
      9: KSamplerSelect          → SAMPLER
     10: LTXVScheduler           → SIGMAS
     11: RandomNoise             → NOISE
     12: SamplerCustomAdvanced   → LATENT_out
     13: VAEDecode               → IMAGE_out
     14: SaveImage               → (出力)
    """
    return {
        "0": {
            "class_type": "UnetLoaderGGUF",
            "inputs": {"unet_name": MODEL_GGUF},
        },
        "1": {
            # DualCLIPLoader type=ltxv: Gemma FP4 (OOMリスク低) + embeddings connector
            "class_type": "DualCLIPLoader",
            "inputs": {
                "clip_name1": GEMMA_FP4,
                "clip_name2": EMBED_CONNECTOR,
                "type":       "ltxv",
                "device":     "default",
            },
        },
        "2": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": MODEL_VAE},
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": positive, "clip": ["1", 0]},
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative, "clip": ["1", 0]},
        },
        "5": {
            "class_type": "LTXVConditioning",
            "inputs": {
                "positive":   ["3", 0],
                "negative":   ["4", 0],
                "frame_rate": 24,
            },
        },
        "6": {
            "class_type": "LoadImage",
            "inputs": {"image": image_name},
        },
        "7": {
            "class_type": "LTXVImgToVideo",
            "inputs": {
                "positive":   ["5", 0],
                "negative":   ["5", 1],
                "vae":        ["2", 0],
                "image":      ["6", 0],
                "width":      RENDER_W,
                "height":     RENDER_H,
                "length":     num_frames,
                "batch_size": 1,
                "strength":   1.0,
            },
        },
        "8": {
            "class_type": "CFGGuider",
            "inputs": {
                "model":    ["0", 0],   # UnetLoaderGGUF → MODEL
                "positive": ["7", 0],
                "negative": ["7", 1],
                "cfg":      cfg,
            },
        },
        "9": {
            "class_type": "KSamplerSelect",
            "inputs": {"sampler_name": "euler"},
        },
        "10": {
            "class_type": "LTXVScheduler",
            "inputs": {
                "steps":      steps,
                "max_shift":  2.05,
                "base_shift": 0.95,
                "stretch":    True,
                "terminal":   0.1,
            },
        },
        "11": {
            "class_type": "RandomNoise",
            "inputs": {"noise_seed": seed},
        },
        "12": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise":         ["11", 0],
                "guider":        ["8",  0],
                "sampler":       ["9",  0],
                "sigmas":        ["10", 0],
                "latent_image":  ["7",  2],
            },
        },
        "13": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["12", 0],
                "vae":     ["2",  0],
            },
        },
        "14": {
            "class_type": "SaveImage",
            "inputs": {
                "images":          ["13", 0],
                "filename_prefix": "ltxv_stage_",
            },
        },
    }


def queue_prompt(workflow: dict) -> str:
    """ワークフローを ComfyUI のキューに追加し、prompt_id を返す"""
    payload = {"prompt": workflow, "client_id": uuid.uuid4().hex}
    result  = _api("/prompt", payload)
    return result["prompt_id"]


def wait_for_completion(prompt_id: str, timeout: int = 300) -> dict:
    """
    prompt_id の完了を polling で待機し、history エントリを返す。
    timeout 秒以内に完了しなければ例外を送出。
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        hist = _api(f"/history/{prompt_id}")
        if prompt_id in hist:
            return hist[prompt_id]
        time.sleep(2)
    raise TimeoutError(f"ComfyUI が {timeout}秒以内に完了しませんでした")


def fetch_output_images(history_entry: dict) -> list[Image.Image]:
    """
    history エントリから SaveImage の出力ファイル名を取得し、
    ComfyUI から画像を順番にダウンロードして PIL Image のリストで返す。
    """
    # デバッグ: status と outputs を表示
    status = history_entry.get("status", {})
    status_str = status.get("status_str", "unknown")
    if status_str != "success":
        msgs = status.get("messages", [])
        print(f"[DEBUG] ComfyUI status={status_str}  messages={msgs}")
    outputs = history_entry.get("outputs", {})
    if not outputs:
        print(f"[DEBUG] outputs が空です。status={status_str}")
        # エラー詳細を出力
        for k, v in history_entry.items():
            if k not in ("prompt",):
                print(f"[DEBUG]   {k}: {str(v)[:300]}")

    images = []
    for node_id, node_output in outputs.items():
        for img_meta in node_output.get("images", []):
            fname     = img_meta["filename"]
            subfolder = img_meta.get("subfolder", "")
            ftype     = img_meta.get("type", "output")
            url = (
                f"{COMFYUI_URL}/view?"
                f"filename={urllib.request.quote(fname)}"
                f"&subfolder={urllib.request.quote(subfolder)}"
                f"&type={ftype}"
            )
            print(f"[DEBUG] DL node={node_id} file={fname}")
            with urllib.request.urlopen(url, timeout=30) as r:
                images.append(Image.open(r).copy())
    return images


# ─── ユーティリティ ─────────────────────────────────────────────────────────────

def extract_alpha(rgb: np.ndarray, bg_gray: float, thresh: int) -> np.ndarray:
    bg_val = int(round(bg_gray * 255))
    bg_arr = np.array([bg_val, bg_val, bg_val], dtype=np.float32)
    dist   = np.linalg.norm(rgb.astype(np.float32) - bg_arr, axis=2)
    mask   = (dist > thresh).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask   = cv2.GaussianBlur(mask, (7, 7), 0)
    return mask


def apply_zoom(rgba: np.ndarray, zoom: float) -> np.ndarray:
    if zoom <= 1.001:
        return rgba
    H, W = rgba.shape[:2]
    cw = max(1, int(W / zoom))
    ch = max(1, int(H / zoom))
    x0 = (W - cw) // 2
    y0 = (H - ch) // 2
    return cv2.resize(rgba[y0:y0+ch, x0:x0+cw], (W, H), interpolation=cv2.INTER_LANCZOS4)


def zoom_factor(i: int, zoom_start: float, zoom_frames: int) -> float:
    if zoom_frames <= 0 or zoom_start <= 1.0 or i >= zoom_frames:
        return 1.0
    return zoom_start + (1.0 - zoom_start) * (i / zoom_frames)


# ─── メイン ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="LTX-2 img2vid で stage間を AI補間")
    ap.add_argument("--keyframes_dir", required=True,
                    help="grid_meta.json があるグリッドディレクトリ")
    ap.add_argument("--output_dir",    default="",
                    help="出力先（空=keyframes_dir/../output_ltxv）")
    ap.add_argument("--positive",      default=POSITIVE_PROMPT)
    ap.add_argument("--negative",      default=NEGATIVE_PROMPT)
    ap.add_argument("--clip_frames",   type=int,   default=CLIP_FRAMES,
                    help="1クリップのフレーム数（25/49/97）")
    ap.add_argument("--cfg",           type=float, default=3.5)
    ap.add_argument("--steps",         type=int,   default=30)
    ap.add_argument("--angle",         type=int,   default=ANGLE_FRONT,
                    help="固定カメラ角度番号（angle_start/end 未指定時に使用）")
    ap.add_argument("--angle_start",   type=int,   default=-1,
                    help="開始角度番号（stage_00で使用。例: 6=右横90度）")
    ap.add_argument("--angle_end",     type=int,   default=-1,
                    help="終了角度番号（stage_last で使用。例: 0=正面）")
    ap.add_argument("--zoom_start",    type=float, default=1.0,
                    help="開始ズーム倍率（2.5 = 顔クローズアップ）")
    ap.add_argument("--zoom_frames",   type=int,   default=72,
                    help="ズームアウトにかけるフレーム数（72 = 3秒@24fps）")
    ap.add_argument("--alpha_thresh",  type=int,   default=ALPHA_THRESH)
    ap.add_argument("--seed",          type=int,   default=42)
    ap.add_argument("--fps",           type=int,   default=24)
    ap.add_argument("--timeout",       type=int,   default=600,
                    help="1クリップあたりの生成タイムアウト秒数")
    ap.add_argument("--prompts_json",  default="",
                    help="stage別プロンプトJSON: {\"0\":\"prompt0\",\"1\":\"prompt1\",...}")
    args = ap.parse_args()

    # ── stage別プロンプト読み込み ─────────────────────────────────────────────
    stage_prompts: dict[int, str] = {}
    if args.prompts_json:
        pjson = Path(args.prompts_json)
        if not pjson.exists():
            sys.exit(f"[ERROR] prompts_json が見つかりません: {pjson}")
        with open(pjson, encoding="utf-8") as f:
            raw = json.load(f)
        stage_prompts = {int(k): v for k, v in raw.items()}
        print(f"[INFO] stage別プロンプト読み込み: {len(stage_prompts)} stages")

    # ── ComfyUI 接続確認 ──────────────────────────────────────────────────────
    try:
        _api("/system_stats")
        print(f"[INFO] ComfyUI 接続確認: {COMFYUI_URL}")
    except Exception:
        sys.exit(
            f"[ERROR] ComfyUI に接続できません: {COMFYUI_URL}\n"
            "ComfyUI を起動してから再実行してください。"
        )

    # ── グリッドメタ読み込み ──────────────────────────────────────────────────
    kf_dir    = Path(args.keyframes_dir)
    meta_path = kf_dir / "grid_meta.json"
    if not meta_path.exists():
        sys.exit(f"[ERROR] grid_meta.json が見つかりません: {meta_path}")

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    n_stages = meta["n_stages"]
    n_angles = meta.get("grid_theta", 24)   # 猫=24, チューリップ=120 など
    bg_gray  = meta.get("render_params", {}).get("bg_gray", BG_GRAY)
    print(f"[INFO] グリッド: {n_stages} stages × {n_angles} angles  bg_gray={bg_gray}")

    # ── 出力ディレクトリ ──────────────────────────────────────────────────────
    out_dir = Path(args.output_dir) if args.output_dir else kf_dir.parent.parent / "output_ltxv"
    # ── 角度スイープの設定 ────────────────────────────────────────────────────
    use_angle_sweep = (args.angle_start >= 0 and args.angle_end >= 0)
    if use_angle_sweep:
        angle_mode = f"sweep_{args.angle_start:03d}_to_{args.angle_end:03d}"
    else:
        angle_mode = f"angle_{args.angle:03d}"

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    png_dir = out_dir / f"ltxv_{angle_mode}_{ts}"
    png_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] 出力: {png_dir}")

    # ── 一時ディレクトリ（ComfyUI アップロード用）────────────────────────────
    import tempfile
    tmp_dir = Path(tempfile.mkdtemp(prefix="ltxv_upload_"))

    all_frames: list[Image.Image] = []

    # ── stage ループ ──────────────────────────────────────────────────────────
    for stage_idx in range(n_stages):
        # 1. stage 画像を取得（角度スイープ or 固定）
        if use_angle_sweep and n_stages > 1:
            t = stage_idx / (n_stages - 1)
            angle = round(args.angle_start + (args.angle_end - args.angle_start) * t) % n_angles
        else:
            angle = args.angle % n_angles
        deg = angle * 360 / n_angles
        src_path = kf_dir / f"stage_{stage_idx:02d}" / f"angle_{angle:03d}.png"
        print(f"[INFO] stage_{stage_idx:02d}: angle={angle:03d} ({deg:.0f}°)")
        if not src_path.exists():
            sys.exit(f"[ERROR] 画像が見つかりません: {src_path}")

        # 768x512 にリサイズして一時保存
        img_pil = Image.open(str(src_path)).convert("RGB").resize(
            (RENDER_W, RENDER_H), Image.LANCZOS
        )
        tmp_png = tmp_dir / f"stage_{stage_idx:02d}.png"
        img_pil.save(str(tmp_png))

        # 2. ComfyUI にアップロード
        print(f"\n[INFO] stage_{stage_idx:02d}: 画像アップロード中...")
        comfy_name = upload_image(tmp_png)

        # 3. ワークフロー構築＆キュー投入
        pos = stage_prompts.get(stage_idx, args.positive)
        if stage_idx in stage_prompts:
            print(f"[INFO] stage_{stage_idx:02d}: 専用プロンプト使用")
        wf = build_workflow(
            image_name = comfy_name,
            positive   = pos,
            negative   = args.negative,
            num_frames = args.clip_frames,
            cfg        = args.cfg,
            steps      = args.steps,
            seed       = args.seed + stage_idx,
        )
        prompt_id = queue_prompt(wf)
        print(f"[INFO] stage_{stage_idx:02d}: 生成中... (prompt_id={prompt_id})")

        # 4. 完了待機
        hist  = wait_for_completion(prompt_id, timeout=args.timeout)
        frames = fetch_output_images(hist)

        if not frames:
            print(f"[WARN] stage_{stage_idx:02d}: 出力画像が取得できませんでした。スキップします。")
            continue

        print(f"[INFO] stage_{stage_idx:02d}: {len(frames)}フレーム取得完了")

        # 5. クリップ連結（先頭クリップは全部、以降は overlap を除く）
        if stage_idx == 0:
            all_frames.extend(frames)
        else:
            all_frames.extend(frames[OVERLAP:])

        print(f"  累計フレーム: {len(all_frames)}")

    # ── PNG 連番出力（アルファ抽出 + ズーム）────────────────────────────────
    N = len(all_frames)
    if N == 0:
        sys.exit("[ERROR] 有効なフレームが生成されませんでした。")

    print(f"\n[INFO] {N}フレームを RGBA PNG で保存中...")
    for i, pil_img in enumerate(all_frames):
        # リサイズ（出力が RENDER_W×RENDER_H でない場合の保険）
        pil_img = pil_img.resize((RENDER_W, RENDER_H), Image.LANCZOS)

        rgb   = np.array(pil_img.convert("RGB"), dtype=np.uint8)
        alpha = extract_alpha(rgb, bg_gray, args.alpha_thresh)
        rgba  = np.dstack([rgb, alpha]).astype(np.uint8)

        # ズーム効果
        z = zoom_factor(i, args.zoom_start, args.zoom_frames)
        if z > 1.001:
            rgba = apply_zoom(rgba, z)

        Image.fromarray(rgba, mode="RGBA").save(str(png_dir / f"frame_{i:05d}.png"))
        if (i + 1) % 50 == 0 or i == N - 1:
            print(f"  {i+1}/{N}  zoom={z:.2f}")

    # クリーンアップ
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    duration = N / args.fps
    print(f"\n[完了] {png_dir}")
    print(f"  {N}フレーム = {duration:.1f}秒 @ {args.fps}fps")
    print(f"  ズーム: {args.zoom_start}x → 1.0x (最初の {args.zoom_frames}フレーム = {args.zoom_frames/args.fps:.1f}秒)")
    print()
    print("PowerDirector での使い方:")
    print("  このフォルダを「画像シーケンス」として読み込んでください（RGBA PNG）。")


if __name__ == "__main__":
    main()

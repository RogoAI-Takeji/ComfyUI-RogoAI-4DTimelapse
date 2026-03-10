# ltxv_interpolator.py
#
# ComfyUI カスタムノード: LTXV2 ステージ補間
#
# Node: NB4D_LTXVStageInterpolator
#   4D グリッドの各 stage 画像を LTX-Video 2 (img2vid) で AI 補間し
#   RGBA PNG 連番として出力する。
#
# カテゴリ: NanoBanana/4DGrid

import io
import json
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


# ─── ComfyUI API ──────────────────────────────────────────────────────────────

def _api(base_url: str, path: str, data: dict | None = None) -> dict:
    """ComfyUI REST API 呼び出しユーティリティ"""
    url = base_url.rstrip("/") + path
    if data is None:
        req = urllib.request.Request(url)
    else:
        body = json.dumps(data).encode("utf-8")
        req  = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json"}
        )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} {e.reason}: {body[:1000]}") from e


def upload_image(base_url: str, img_path: Path) -> str:
    """
    PNG ファイルを ComfyUI にアップロードし、ComfyUI 側のファイル名を返す。
    multipart/form-data で /upload/image に POST する。
    """
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
        base_url.rstrip("/") + "/upload/image",
        data    = body.getvalue(),
        headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method  = "POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        result = json.load(r)
    return result["name"]


def build_workflow(
    image_name:       str,
    positive:         str,
    negative:         str,
    num_frames:       int,
    cfg:              float,
    steps:            int,
    seed:             int,
    model_gguf:       str,
    model_vae:        str,
    gemma_fp4:        str,
    embed_connector:  str,
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
            "inputs": {"unet_name": model_gguf},
        },
        "1": {
            "class_type": "DualCLIPLoader",
            "inputs": {
                "clip_name1": gemma_fp4,
                "clip_name2": embed_connector,
                "type":       "ltxv",
                "device":     "default",
            },
        },
        "2": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": model_vae},
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
                "width":      768,
                "height":     512,
                "length":     num_frames,
                "batch_size": 1,
                "strength":   1.0,
            },
        },
        "8": {
            "class_type": "CFGGuider",
            "inputs": {
                "model":    ["0", 0],
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
                "noise":        ["11", 0],
                "guider":       ["8",  0],
                "sampler":      ["9",  0],
                "sigmas":       ["10", 0],
                "latent_image": ["7",  2],
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


def queue_prompt(base_url: str, workflow: dict) -> str:
    """ワークフローを ComfyUI のキューに追加し、prompt_id を返す"""
    payload = {"prompt": workflow, "client_id": uuid.uuid4().hex}
    result  = _api(base_url, "/prompt", payload)
    return result["prompt_id"]


def wait_for_completion(base_url: str, prompt_id: str, timeout: int = 600) -> dict:
    """
    prompt_id の完了を polling で待機し、history エントリを返す。
    timeout 秒以内に完了しなければ例外を送出。
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        hist = _api(base_url, f"/history/{prompt_id}")
        if prompt_id in hist:
            return hist[prompt_id]
        time.sleep(2)
    raise TimeoutError(f"ComfyUI が {timeout}秒以内に完了しませんでした (prompt_id={prompt_id})")


def fetch_output_images(base_url: str, history_entry: dict) -> list:
    """
    history エントリから SaveImage の出力ファイル名を取得し、
    ComfyUI から画像を順番にダウンロードして PIL Image のリストで返す。
    """
    status     = history_entry.get("status", {})
    status_str = status.get("status_str", "unknown")
    if status_str != "success":
        msgs = status.get("messages", [])
        print(f"[WARN] ComfyUI status={status_str}  messages={msgs}")

    outputs = history_entry.get("outputs", {})
    images  = []
    for node_id, node_output in outputs.items():
        for img_meta in node_output.get("images", []):
            fname     = img_meta["filename"]
            subfolder = img_meta.get("subfolder", "")
            ftype     = img_meta.get("type", "output")
            url = (
                f"{base_url.rstrip('/')}/view?"
                f"filename={urllib.request.quote(fname)}"
                f"&subfolder={urllib.request.quote(subfolder)}"
                f"&type={ftype}"
            )
            print(f"[DEBUG] DL node={node_id} file={fname}")
            with urllib.request.urlopen(url, timeout=30) as r:
                images.append(Image.open(r).copy())
    return images


# ─── ユーティリティ ────────────────────────────────────────────────────────────

def extract_alpha(rgb: np.ndarray, bg_gray: float, thresh: int) -> np.ndarray:
    """グレー背景からアルファチャンネルを抽出する"""
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
    """ズームイン効果を適用する（中央トリミング→リサイズ）"""
    if zoom <= 1.001:
        return rgba
    H, W = rgba.shape[:2]
    cw = max(1, int(W / zoom))
    ch = max(1, int(H / zoom))
    x0 = (W - cw) // 2
    y0 = (H - ch) // 2
    return cv2.resize(rgba[y0:y0+ch, x0:x0+cw], (W, H), interpolation=cv2.INTER_LANCZOS4)


def zoom_factor(i: int, zoom_start: float, zoom_frames: int) -> float:
    """フレームインデックスに対するズーム係数を計算する"""
    if zoom_frames <= 0 or zoom_start <= 1.0 or i >= zoom_frames:
        return 1.0
    return zoom_start + (1.0 - zoom_start) * (i / zoom_frames)


# ─── ノード本体 ────────────────────────────────────────────────────────────────

class NB4D_LTXVStageInterpolator:
    """
    4D グリッドの各 stage 画像を LTX-Video 2 (img2vid) で AI 補間し
    RGBA PNG 連番として出力する ComfyUI カスタムノード。

    ComfyUI 内から同じ ComfyUI の REST API（localhost:8188）に
    queue_prompt を投げる方式。LTX-Video 2 の重いモデルを
    ComfyUI 側で共有できるため VRAM 効率が良い。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "keyframes_dir":   ("STRING", {
                    "default": r"D:\NB4D_test\grid_keyframes\run_20260101_000000"
                }),
                "positive_prompt": ("STRING", {
                    "multiline": True,
                    "default": "natural movement, smooth animation, photorealistic, high quality"
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "blurry, distorted, low quality, morphing artifacts, color shift"
                }),
                "clip_frames": ("INT", {
                    "default": 49, "min": 25, "max": 97, "step": 24,
                    "tooltip": "1ステージのフレーム数。8n+1推奨: 25/49/97"
                }),
                "angle": ("INT", {
                    "default": 0, "min": 0, "max": 359
                }),
                "angle_start": ("INT", {
                    "default": -1, "min": -1, "max": 359,
                    "tooltip": "-1=固定角度。0以上=スイープ開始角度"
                }),
                "angle_end": ("INT", {
                    "default": -1, "min": -1, "max": 359,
                    "tooltip": "-1=固定角度。0以上=スイープ終了角度"
                }),
                "cfg": ("FLOAT", {
                    "default": 3.5, "min": 1.0, "max": 10.0, "step": 0.1
                }),
                "steps": ("INT", {
                    "default": 30, "min": 5, "max": 50
                }),
                "seed": ("INT", {
                    "default": 42, "min": 0, "max": 2147483647
                }),
                "fps": ("INT", {
                    "default": 24, "min": 12, "max": 60
                }),
                "zoom_start": ("FLOAT", {
                    "default": 1.0, "min": 1.0, "max": 5.0, "step": 0.1,
                    "tooltip": "1.0=ズームなし。2.5=クローズアップ開始"
                }),
                "zoom_frames": ("INT", {
                    "default": 72, "min": 0, "max": 240,
                    "tooltip": "ズームアウトにかけるフレーム数。72=3秒@24fps"
                }),
                "alpha_thresh": ("INT", {
                    "default": 18, "min": 1, "max": 80
                }),
                "output_dir": ("STRING", {
                    "default": r"D:\NB4D_test\output_ltxv"
                }),
                "comfyui_url": ("STRING", {
                    "default": "http://127.0.0.1:8188"
                }),
                "model_gguf": ("STRING", {
                    "default": r"ltx2\ltx-2-19b-distilled_Q6_K.gguf"
                }),
                "model_vae": ("STRING", {
                    "default": r"ltx2\LTX2_video_vae_bf16.safetensors"
                }),
                "gemma_fp4": ("STRING", {
                    "default": r"ltx2\gemma_3_12B_it_fp4_mixed.safetensors"
                }),
                "embed_connector": ("STRING", {
                    "default": r"ltx2\ltx-2-19b-embeddings_connector_distill_bf16.safetensors"
                }),
            }
        }

    RETURN_TYPES  = ("STRING", "INT", "STRING")
    RETURN_NAMES  = ("output_dir", "frame_count", "status")
    FUNCTION      = "interpolate"
    OUTPUT_NODE   = True
    CATEGORY      = "NanoBanana/4DGrid"

    def interpolate(
        self,
        keyframes_dir,
        positive_prompt,
        negative_prompt,
        clip_frames,
        angle,
        angle_start,
        angle_end,
        cfg,
        steps,
        seed,
        fps,
        zoom_start,
        zoom_frames,
        alpha_thresh,
        output_dir,
        comfyui_url,
        model_gguf,
        model_vae,
        gemma_fp4,
        embed_connector,
    ):
        # ── 1. grid_meta.json 読み込み ─────────────────────────────────────────
        kf_dir    = Path(keyframes_dir)
        meta_path = kf_dir / "grid_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"grid_meta.json が見つかりません: {meta_path}")

        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

        n_stages = meta["n_stages"]
        n_angles = meta.get("grid_theta", 24)
        bg_gray  = meta.get("render_params", {}).get("bg_gray", 0.12)
        print(f"[INFO] グリッド: {n_stages} stages × {n_angles} angles  bg_gray={bg_gray}")

        # ── 2. 出力ディレクトリ作成 ────────────────────────────────────────────
        use_sweep = (angle_start >= 0 and angle_end >= 0)
        if use_sweep:
            angle_mode = f"sweep_{angle_start:03d}_to_{angle_end:03d}"
        else:
            angle_mode = f"angle_{angle:03d}"

        ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(output_dir)
        png_dir = out_dir / f"ltxv_{angle_mode}_{ts}"
        png_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] 出力先: {png_dir}")

        # ── 3. ComfyUI 接続確認 ────────────────────────────────────────────────
        try:
            _api(comfyui_url, "/system_stats")
            print(f"[INFO] ComfyUI 接続確認: {comfyui_url}")
        except Exception as e:
            raise RuntimeError(
                f"ComfyUI に接続できません: {comfyui_url}\n"
                "ComfyUI を起動してから再実行してください。\n"
                f"詳細: {e}"
            )

        # ── 4. 一時ディレクトリ（アップロード用） ─────────────────────────────
        tmp_dir    = Path(tempfile.mkdtemp(prefix="ltxv_upload_"))
        all_frames = []

        try:
            # ── 5. stage ループ ────────────────────────────────────────────────
            for stage_idx in range(n_stages):

                # 角度計算（スイープ or 固定）
                if use_sweep and n_stages > 1:
                    t         = stage_idx / (n_stages - 1)
                    cur_angle = round(angle_start + (angle_end - angle_start) * t) % n_angles
                else:
                    cur_angle = angle % n_angles

                deg = cur_angle * 360 / n_angles
                print(f"[INFO] stage_{stage_idx:02d}: angle={cur_angle:03d} ({deg:.0f}°)")

                # 画像取得・768×512 にリサイズ
                src = kf_dir / f"stage_{stage_idx:02d}" / f"angle_{cur_angle:03d}.png"
                if not src.exists():
                    print(f"[WARN] 画像が見つかりません: {src}  スキップします。")
                    continue

                img = Image.open(str(src)).convert("RGB").resize((768, 512), Image.LANCZOS)
                tmp_png = tmp_dir / f"stage_{stage_idx:02d}.png"
                img.save(str(tmp_png))

                # アップロード
                print(f"[INFO] stage_{stage_idx:02d}: 画像アップロード中...")
                comfy_name = upload_image(comfyui_url, tmp_png)

                # ワークフロー構築＆キュー投入
                wf = build_workflow(
                    image_name      = comfy_name,
                    positive        = positive_prompt,
                    negative        = negative_prompt,
                    num_frames      = clip_frames,
                    cfg             = cfg,
                    steps           = steps,
                    seed            = seed + stage_idx,
                    model_gguf      = model_gguf,
                    model_vae       = model_vae,
                    gemma_fp4       = gemma_fp4,
                    embed_connector = embed_connector,
                )
                prompt_id = queue_prompt(comfyui_url, wf)
                print(f"[INFO] stage_{stage_idx:02d}: 生成中... (prompt_id={prompt_id})")

                # 完了待機
                hist   = wait_for_completion(comfyui_url, prompt_id, timeout=600)
                frames = fetch_output_images(comfyui_url, hist)

                if not frames:
                    print(f"[WARN] stage_{stage_idx:02d}: 出力画像が取得できませんでした。スキップします。")
                    continue

                print(f"[INFO] stage_{stage_idx:02d}: {len(frames)}フレーム取得完了")

                # クリップ連結（先頭クリップは全部、以降は overlap=1 を除く）
                if stage_idx == 0:
                    all_frames.extend(frames)
                else:
                    all_frames.extend(frames[1:])

                print(f"  累計フレーム: {len(all_frames)}")

        finally:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

        if not all_frames:
            raise RuntimeError("有効なフレームが生成されませんでした。")

        # ── 6. RGBA PNG 保存（アルファ抽出 + ズーム） ─────────────────────────
        N = len(all_frames)
        print(f"\n[INFO] {N}フレームを RGBA PNG で保存中...")

        for i, pil_img in enumerate(all_frames):
            pil_img = pil_img.resize((768, 512), Image.LANCZOS)
            rgb     = np.array(pil_img.convert("RGB"), dtype=np.uint8)
            alpha   = extract_alpha(rgb, bg_gray, alpha_thresh)
            rgba    = np.dstack([rgb, alpha]).astype(np.uint8)

            z = zoom_factor(i, zoom_start, zoom_frames)
            if z > 1.001:
                rgba = apply_zoom(rgba, z)

            Image.fromarray(rgba, mode="RGBA").save(str(png_dir / f"frame_{i:05d}.png"))
            if (i + 1) % 50 == 0 or i == N - 1:
                print(f"  {i+1}/{N}  zoom={z:.2f}")

        duration = N / fps
        status   = (
            f"完了: {N}フレーム = {duration:.1f}秒 @ {fps}fps → {png_dir}"
        )
        print(f"\n[完了] {status}")

        return (str(png_dir), N, status)


# ─── 登録 ─────────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "NB4D_LTXVStageInterpolator": NB4D_LTXVStageInterpolator,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "NB4D_LTXVStageInterpolator": "NB4D LTXV Stage Interpolator (AI補間)",
}

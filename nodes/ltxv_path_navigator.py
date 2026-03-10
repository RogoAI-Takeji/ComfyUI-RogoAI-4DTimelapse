# ltxv_path_navigator.py
#
# ComfyUI カスタムノード: LTXV2 時空間パスナビゲーター
#
# Node: NB4D_LTXVPathNavigator
#
# 概念:
#   (t, h, v) = (時間, 水平角, 仰角) の3次元格子を自由に移動するパスを定義し、
#   隣接するウェイポイント間を LTX-2 で AI 補間して動画を生成する。
#
#   格子:
#     t: 0 〜 n_stages-1  (GLB keyframes)
#     h: 0 〜 grid_theta-1 (水平角, 0=正面, 右回り正)
#     v: 0 〜 grid_elev-1  (仰角, 0=-30°, 2=0°水平, 6=+60°)
#
#   パス指定:
#     プリセット COMBO: 7種類のプリセット
#     カスタムテキスト: "t, h, v" 形式で1行1ウェイポイント（整数または正規化float）
#
#   隣接制約:
#     推奨は各ステップで |Δt|≤1, |Δh|≤1, |Δv|≤1
#     違反時は警告を出すが実行は続行（LTX-2が補間を試みる）
#
# カテゴリ: NanoBanana/4DGrid

import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from nodes._nb4d_paths import (
    NAVIGATOR_PRESET_NAMES,
    NAVIGATOR_PRESET_LABELS,
    preset_to_waypoints,
    parse_path_text,
    check_adjacency,
)


# ─── ヘルパー関数 ────────────────────────────────────────────────────────────────

def _get_node_class(class_name: str):
    import importlib
    m = importlib.import_module("nodes")
    cls = m.NODE_CLASS_MAPPINGS.get(class_name)
    if cls is None:
        raise RuntimeError(f"ノードクラス '{class_name}' が登録されていません。")
    return cls


def pil_to_tensor(pil_img: Image.Image):
    rgb = np.array(pil_img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(rgb).unsqueeze(0)


def tensor_to_pils(tensor) -> list:
    arr = tensor.cpu().float().numpy()
    return [
        Image.fromarray((arr[i] * 255.0).clip(0, 255).astype(np.uint8), "RGB")
        for i in range(arr.shape[0])
    ]


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


def load_keyframe(kf_dir: Path, t: int, h: int, v: int, grid_elev: int, size=(768, 512)) -> Image.Image:
    """(t, h, v) に対応するキーフレーム画像を読み込む"""
    if grid_elev > 1:
        path = kf_dir / f"stage_{t:02d}" / f"elev_{v:02d}" / f"angle_{h:03d}.png"
    else:
        path = kf_dir / f"stage_{t:02d}" / f"angle_{h:03d}.png"
    if not path.exists():
        raise FileNotFoundError(f"キーフレームが見つかりません: {path}")
    return Image.open(str(path)).convert("RGB").resize(size, Image.LANCZOS)


# ─── ノード本体 ────────────────────────────────────────────────────────────────────

class NB4D_LTXVPathNavigator:
    """
    (t, h, v) 時空間格子を自由に飛び回るパスを定義し、
    ウェイポイント間を LTX-2 で AI 補間して動画を生成する。

    格子:
      t = 時間 (keyframe stages)
      h = 水平角 (grid_theta 分割)
      v = 仰角   (grid_elev 分割, 推奨7段階: -30°〜+60°)

    パス = ウェイポイントのシーケンス。隣接(Δ=1)が推奨だが強制ではない。
    各セグメント (waypoint[i] → waypoint[i+1]) でLTX-2が1クリップを生成。
    """

    @classmethod
    def INPUT_TYPES(cls):
        preset_tooltip = "\n".join(
            f"  {k}: {v}" for k, v in NAVIGATOR_PRESET_LABELS.items()
        )
        return {
            "required": {
                "keyframes_dir": ("STRING", {
                    "default": r"D:\NB4D_test\grid_keyframes\run_20260101_000000"
                }),
                "path_preset": (NAVIGATOR_PRESET_NAMES, {
                    "default": "orbit_flat",
                    "tooltip": f"パスプリセット。custom_textの場合はpath_customを使用。\n{preset_tooltip}",
                }),
                "path_custom": ("STRING", {
                    "multiline": True,
                    "default": (
                        "# (t_idx, h_idx, v_idx) を1行1ウェイポイントで入力\n"
                        "# 整数インデックスまたは正規化float(0.0-1.0)を使用\n"
                        "# path_preset='custom_text' のときのみ有効\n"
                        "0, 0, 2\n"
                        "1, 3, 2\n"
                        "2, 6, 2\n"
                        "3, 9, 3\n"
                        "4, 12, 3\n"
                    ),
                    "tooltip": "path_preset='custom_text' のときに使用するウェイポイントリスト",
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
                    "tooltip": "1セグメントのフレーム数 (8n+1推奨: 25/49/97)"
                }),
                "cfg": ("FLOAT", {"default": 3.5, "min": 1.0, "max": 10.0, "step": 0.1}),
                "steps": ("INT",  {"default": 30,  "min": 5,   "max": 50}),
                "seed":  ("INT",  {"default": 42,  "min": 0,   "max": 2147483647}),
                "fps":   ("INT",  {"default": 24,  "min": 12,  "max": 60}),
                "zoom_start":  ("FLOAT", {"default": 1.0, "min": 1.0, "max": 5.0, "step": 0.1}),
                "zoom_frames": ("INT",   {"default": 0,   "min": 0,   "max": 240}),
                "alpha_thresh": ("INT",  {"default": 18,  "min": 1,   "max": 80}),
                "output_dir":  ("STRING", {"default": r"D:\NB4D_test\output_ltxv"}),
                "comfyui_url": ("STRING", {"default": "http://127.0.0.1:8188",
                                           "tooltip": "未使用（直接Python呼び出し）"}),
                "model_gguf":       ("STRING", {"default": r"ltx2\ltx-2-19b-distilled_Q6_K.gguf"}),
                "model_vae":        ("STRING", {"default": r"ltx2\LTX2_video_vae_bf16.safetensors"}),
                "gemma_fp4":        ("STRING", {"default": r"ltx2\gemma_3_12B_it_fp4_mixed.safetensors"}),
                "embed_connector":  ("STRING", {"default": r"ltx2\ltx-2-19b-embeddings_connector_distill_bf16.safetensors"}),
            }
        }

    RETURN_TYPES  = ("STRING", "INT", "STRING")
    RETURN_NAMES  = ("output_dir", "frame_count", "status")
    FUNCTION      = "navigate"
    OUTPUT_NODE   = True
    CATEGORY      = "NanoBanana/4DGrid"

    def navigate(
        self,
        keyframes_dir,
        path_preset,
        path_custom,
        positive_prompt,
        negative_prompt,
        clip_frames,
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

        n_stages  = meta["n_stages"]
        grid_theta = meta.get("grid_theta", 24)
        grid_elev  = meta.get("grid_elev", 1)
        bg_gray    = meta.get("render_params", {}).get("bg_gray", 0.12)
        elev_angles = meta.get("elev_angles") or []

        print(f"[INFO] グリッド: t={n_stages} × h={grid_theta} × v={grid_elev}")
        if elev_angles:
            print(f"[INFO] 仰角一覧: {[f'{e:.0f}°' for e in elev_angles]}")

        # ── 2. ウェイポイント解決 ──────────────────────────────────────────────
        if path_preset == "custom_text":
            waypoints = parse_path_text(path_custom, n_stages, grid_theta, grid_elev)
            if not waypoints:
                raise ValueError("path_customのウェイポイントが解析できませんでした。")
        else:
            waypoints = preset_to_waypoints(path_preset, n_stages, grid_theta, grid_elev)
            if not waypoints:
                raise ValueError(f"プリセット '{path_preset}' が見つかりません。")

        print(f"[INFO] ウェイポイント数: {len(waypoints)}")
        for i, wp in enumerate(waypoints):
            t, h, v = wp
            elev_str = f" ({elev_angles[v]:.0f}°)" if v < len(elev_angles) else ""
            deg_h = h * 360.0 / grid_theta
            print(f"  [{i:02d}] t={t}, h={h}({deg_h:.0f}°), v={v}{elev_str}")

        # 隣接チェック（警告のみ）
        warnings = check_adjacency(waypoints)
        if warnings:
            print("[WARN] 隣接制約違反（品質低下の可能性）:")
            for w in warnings:
                print(w)

        n_segments = len(waypoints) - 1
        if n_segments < 1:
            raise ValueError("ウェイポイントは最低2点必要です。")

        # ── 3. 出力ディレクトリ ────────────────────────────────────────────────
        ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(output_dir)
        png_dir = out_dir / f"ltxv_nav_{path_preset}_{ts}"
        png_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] 出力先: {png_dir}")

        # ── 4. ノードクラス取得 ─────────────────────────────────────────────────
        print("[INFO] ノードクラス取得中...")
        UnetLoaderGGUF_cls   = _get_node_class("UnetLoaderGGUF")
        DualCLIPLoader_cls   = _get_node_class("DualCLIPLoader")
        VAELoader_cls        = _get_node_class("VAELoader")
        CLIPTextEncode_cls   = _get_node_class("CLIPTextEncode")
        LTXVConditioning_cls = _get_node_class("LTXVConditioning")
        LTXVImgToVideo_cls   = _get_node_class("LTXVImgToVideo")
        LTXVPreprocess_cls   = _get_node_class("LTXVPreprocess")
        LTXVAddGuide_cls     = _get_node_class("LTXVAddGuide")
        CFGGuider_cls        = _get_node_class("CFGGuider")
        KSamplerSelect_cls   = _get_node_class("KSamplerSelect")
        LTXVScheduler_cls    = _get_node_class("LTXVScheduler")
        RandomNoise_cls      = _get_node_class("RandomNoise")
        SamplerCustomAdv_cls = _get_node_class("SamplerCustomAdvanced")
        VAEDecode_cls        = _get_node_class("VAEDecode")

        # ── 5. モデル読み込み ─────────────────────────────────────────────────
        print("[INFO] モデル読み込み中... (unet)")
        (model,) = UnetLoaderGGUF_cls().load_unet(unet_name=model_gguf)
        print("[INFO] モデル読み込み中... (clip)")
        (clip,) = DualCLIPLoader_cls().load_clip(
            clip_name1=gemma_fp4, clip_name2=embed_connector,
            type="ltxv", device="default",
        )
        print("[INFO] モデル読み込み中... (vae)")
        (vae,) = VAELoader_cls().load_vae(vae_name=model_vae)

        # ── 6. テキストエンコード ─────────────────────────────────────────────
        print("[INFO] テキストエンコード中...")
        (pos_cond,) = CLIPTextEncode_cls().encode(text=positive_prompt, clip=clip)
        (neg_cond,) = CLIPTextEncode_cls().encode(text=negative_prompt, clip=clip)
        ltxv_pos, ltxv_neg = LTXVConditioning_cls().execute(
            positive=pos_cond, negative=neg_cond, frame_rate=24
        )

        # ── 7. サンプラー / スケジューラー ────────────────────────────────────
        (sampler,) = KSamplerSelect_cls().execute(sampler_name="euler")
        (sigmas,)  = LTXVScheduler_cls().execute(
            steps=steps, max_shift=2.05, base_shift=0.95,
            stretch=True, terminal=0.1,
        )

        # ── 8. セグメントループ ────────────────────────────────────────────────
        all_frames = []

        try:
            from comfy.utils import ProgressBar
            pbar = ProgressBar(n_segments)
        except Exception:
            pbar = None

        for seg_idx in range(n_segments):
            t0, h0, v0 = waypoints[seg_idx]
            t1, h1, v1 = waypoints[seg_idx + 1]

            elev0_str = f"({elev_angles[v0]:.0f}°)" if v0 < len(elev_angles) else ""
            elev1_str = f"({elev_angles[v1]:.0f}°)" if v1 < len(elev_angles) else ""
            print(f"\n[SEG {seg_idx+1}/{n_segments}] "
                  f"(t={t0},h={h0},v={v0}{elev0_str}) → "
                  f"(t={t1},h={h1},v={v1}{elev1_str})")

            # キーフレーム読み込み
            start_img    = load_keyframe(kf_dir, t0, h0, v0, grid_elev)
            start_tensor = pil_to_tensor(start_img)
            end_img      = load_keyframe(kf_dir, t1, h1, v1, grid_elev)
            end_tensor   = pil_to_tensor(end_img)

            # LTXVImgToVideo: start → latent
            img2vid_pos, img2vid_neg, img2vid_lat = LTXVImgToVideo_cls().execute(
                positive=ltxv_pos, negative=ltxv_neg,
                image=start_tensor, vae=vae,
                width=768, height=512, length=clip_frames,
                batch_size=1, strength=1.0,
            )

            # LTXVAddGuide: 先頭 + 終端ガイド
            (preprocessed_start,) = LTXVPreprocess_cls().execute(
                image=start_tensor, img_compression=35
            )
            guide_pos1, guide_neg1, guide_lat1 = LTXVAddGuide_cls().execute(
                positive=img2vid_pos, negative=img2vid_neg,
                vae=vae, latent=img2vid_lat,
                image=preprocessed_start, frame_idx=0, strength=1.0,
            )
            (preprocessed_end,) = LTXVPreprocess_cls().execute(
                image=end_tensor, img_compression=35
            )
            guide_pos2, guide_neg2, guide_lat2 = LTXVAddGuide_cls().execute(
                positive=guide_pos1, negative=guide_neg1,
                vae=vae, latent=guide_lat1,
                image=preprocessed_end, frame_idx=-1, strength=0.9,
            )

            # サンプリング
            (guider,) = CFGGuider_cls().execute(
                model=model, positive=guide_pos2, negative=guide_neg2, cfg=cfg
            )
            (noise,) = RandomNoise_cls().execute(noise_seed=seed + seg_idx)
            output_latent, _ = SamplerCustomAdv_cls().execute(
                noise=noise, guider=guider, sampler=sampler,
                sigmas=sigmas, latent_image=guide_lat2,
            )

            # VAEデコード
            (image_tensor,) = VAEDecode_cls().decode(vae=vae, samples=output_latent)
            frames = tensor_to_pils(image_tensor)

            if not frames:
                print(f"[WARN] セグメント{seg_idx}: フレーム取得失敗。スキップします。")
                if pbar:
                    pbar.update(1)
                continue

            print(f"  → {len(frames)}フレーム生成")

            # クリップ連結（先頭以外は overlap 1 フレームを除く）
            if seg_idx == 0:
                all_frames.extend(frames)
            else:
                all_frames.extend(frames[1:])

            print(f"  累計フレーム: {len(all_frames)}")
            if pbar:
                pbar.update(1)

        if not all_frames:
            raise RuntimeError("有効なフレームが生成されませんでした。")

        # ── 9. RGBA PNG 保存 ────────────────────────────────────────────────────
        N = len(all_frames)
        print(f"\n[INFO] {N}フレームを RGBA PNG で保存中...")

        for i, pil_img in enumerate(all_frames):
            pil_img = pil_img.resize((768, 512), Image.LANCZOS)
            rgb     = np.array(pil_img.convert("RGB"), dtype=np.uint8)
            alpha   = extract_alpha(rgb, bg_gray, alpha_thresh)
            rgba    = np.dstack([rgb, alpha]).astype(np.uint8)

            if zoom_frames > 0 and zoom_start > 1.0:
                z = zoom_start + (1.0 - zoom_start) * min(i / zoom_frames, 1.0)
                if z > 1.001:
                    rgba = apply_zoom(rgba, z)

            Image.fromarray(rgba, mode="RGBA").save(str(png_dir / f"frame_{i:05d}.png"))
            if (i + 1) % 50 == 0 or i == N - 1:
                print(f"  {i+1}/{N}")

        duration = N / fps
        wp_str   = " → ".join(f"({t},{h},{v})" for t, h, v in waypoints)
        status   = (
            f"完了: {N}フレーム = {duration:.1f}秒 @ {fps}fps\n"
            f"  パス: {wp_str}\n"
            f"  → {png_dir}"
        )
        print(f"\n[完了] {status}")
        return (str(png_dir), N, status)


# ─── 登録 ─────────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "NB4D_LTXVPathNavigator": NB4D_LTXVPathNavigator,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "NB4D_LTXVPathNavigator": "NB4D LTXV Path Navigator (時空間ナビゲーター)",
}

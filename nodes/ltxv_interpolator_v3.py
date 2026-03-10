# ltxv_interpolator_v3.py
#
# ComfyUI カスタムノード: LTXV2 ステージ補間 v3
#
# Node: NB4D_LTXVStageInterpolatorV3
#
# v2 からの変更点:
#   - angle_start / angle_end を廃止
#   - sweep_path COMBO を追加（_nb4d_paths.py の共通パスライブラリを使用）
#   - step1/step2 (Grid4DTraverse) と同じパス名で統一的に角度を制御可能
#   - fixed_angle: sweep_path="none" のときの固定角度（v2 の angle に相当）
#
# v2 との互換性:
#   v2 ノード (NB4D_LTXVStageInterpolatorV2) はそのまま残っているため
#   既存ワークフローは変更不要。
#
# カテゴリ: NanoBanana/4DGrid

import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from ._nb4d_paths import SWEEP_PATH_NAMES, SWEEP_PATH_LABELS, theta_for_stage


# ─── ヘルパー関数（v2 と共通） ───────────────────────────────────────────────

def _get_node_class(class_name: str):
    import importlib
    m = importlib.import_module("nodes")
    cls = m.NODE_CLASS_MAPPINGS.get(class_name)
    if cls is None:
        raise RuntimeError(
            f"ノードクラス '{class_name}' が登録されていません。"
        )
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


def zoom_factor(i: int, zoom_start: float, zoom_frames: int) -> float:
    if zoom_frames <= 0 or zoom_start <= 1.0 or i >= zoom_frames:
        return 1.0
    return zoom_start + (1.0 - zoom_start) * (i / zoom_frames)


# ─── ノード本体 ────────────────────────────────────────────────────────────────

class NB4D_LTXVStageInterpolatorV3:
    """
    4D グリッドの各 stage 画像を LTX-Video 2 (img2vid) で AI 補間し
    RGBA PNG 連番として出力する ComfyUI カスタムノード（v3）。

    v3 の変更点:
      sweep_path COMBO により、step1/step2 と同じパス名で
      角度スイープを制御できる。
      _nb4d_paths.py の共通パスライブラリを使用。

    sweep_path="none" の場合は fixed_angle で固定角度（v2 の angle と同等）。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "keyframes_dir": ("STRING", {
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
                # ── v3: sweep_path + fixed_angle (angle_start/end を廃止) ──────
                "sweep_path": (SWEEP_PATH_NAMES, {
                    "default": "none",
                    "tooltip": (
                        "none=固定角度。それ以外は時間進行に合わせて角度スイープ。\n"
                        + "\n".join(
                            f"  {k}: {v}"
                            for k, v in SWEEP_PATH_LABELS.items()
                        )
                    )
                }),
                "fixed_angle": ("INT", {
                    "default": 0, "min": 0, "max": 359,
                    "tooltip": "sweep_path='none' のときの固定角度インデックス（0〜grid_theta-1）"
                }),
                # ──────────────────────────────────────────────────────────────
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
                    "tooltip": "1.0=ズームなし"
                }),
                "zoom_frames": ("INT", {
                    "default": 72, "min": 0, "max": 240
                }),
                "alpha_thresh": ("INT", {
                    "default": 18, "min": 1, "max": 80
                }),
                "use_end_frame": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "ONにすると keyframe(n+1) を終端ガイドとして注入。"
                }),
                "end_frame_strength": ("FLOAT", {
                    "default": 0.9, "min": 0.5, "max": 1.0, "step": 0.05
                }),
                "output_dir": ("STRING", {
                    "default": r"D:\NB4D_test\output_ltxv"
                }),
                "comfyui_url": ("STRING", {
                    "default": "http://127.0.0.1:8188",
                    "tooltip": "v2.1以降は未使用（直接Python呼び出しに変更）"
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
    CATEGORY      = "RogoAI/4DTimelapse"

    def interpolate(
        self,
        keyframes_dir,
        positive_prompt,
        negative_prompt,
        clip_frames,
        sweep_path,
        fixed_angle,
        cfg,
        steps,
        seed,
        fps,
        zoom_start,
        zoom_frames,
        alpha_thresh,
        use_end_frame,
        end_frame_strength,
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
        mode_str = f"sweep_path={sweep_path}  end_frame={'ON' if use_end_frame else 'OFF'}"
        print(f"[INFO] グリッド: {n_stages} stages × {n_angles} angles  bg_gray={bg_gray}")
        print(f"[INFO] モード: {mode_str}")

        try:
            from comfy.utils import ProgressBar
            pbar = ProgressBar(n_stages)
        except Exception:
            pbar = None

        # ── 2. 出力ディレクトリ作成 ────────────────────────────────────────────
        sweep_tag = sweep_path if sweep_path != "none" else f"fixed{fixed_angle:03d}"
        ef_tag    = f"_ef{end_frame_strength:.2f}" if use_end_frame else "_noef"
        ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir   = Path(output_dir)
        png_dir   = out_dir / f"ltxv_v3_{sweep_tag}{ef_tag}_{ts}"
        png_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] 出力先: {png_dir}")

        # ── 3. ノードクラス取得 ─────────────────────────────────────────────────
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

        # ── 4. モデル読み込み ─────────────────────────────────────────────────
        print("[INFO] モデル読み込み中... (unet)")
        (model,) = UnetLoaderGGUF_cls().load_unet(unet_name=model_gguf)
        print("[INFO] モデル読み込み中... (clip)")
        (clip,) = DualCLIPLoader_cls().load_clip(
            clip_name1=gemma_fp4,
            clip_name2=embed_connector,
            type="ltxv",
            device="default",
        )
        print("[INFO] モデル読み込み中... (vae)")
        (vae,) = VAELoader_cls().load_vae(vae_name=model_vae)

        # ── 5. テキストエンコード ─────────────────────────────────────────────
        print("[INFO] テキストエンコード中...")
        (pos_cond,) = CLIPTextEncode_cls().encode(text=positive_prompt, clip=clip)
        (neg_cond,) = CLIPTextEncode_cls().encode(text=negative_prompt, clip=clip)
        ltxv_pos, ltxv_neg = LTXVConditioning_cls().execute(
            positive=pos_cond, negative=neg_cond, frame_rate=24
        )

        # ── 6. サンプラー / スケジューラー初期化 ─────────────────────────────
        (sampler,) = KSamplerSelect_cls().execute(sampler_name="euler")
        (sigmas,)  = LTXVScheduler_cls().execute(
            steps=steps, max_shift=2.05, base_shift=0.95,
            stretch=True, terminal=0.1,
        )

        # ── 7. stage ループ ─────────────────────────────────────────────────────
        all_frames = []

        for stage_idx in range(n_stages):

            # ── v3: 共通パスライブラリで角度を決定 ───────────────────────────
            cur_angle = theta_for_stage(
                sweep_path, stage_idx, n_stages, n_angles, fixed_angle
            )
            deg = cur_angle * 360.0 / n_angles
            print(f"\n[INFO] stage_{stage_idx:02d}: angle={cur_angle:03d} ({deg:.0f}°)  path={sweep_path}")

            # 現 stage 画像 → テンソル
            src = kf_dir / f"stage_{stage_idx:02d}" / f"angle_{cur_angle:03d}.png"
            if not src.exists():
                print(f"[WARN] 画像が見つかりません: {src}  スキップします。")
                if pbar is not None:
                    pbar.update(1)
                continue

            start_img    = Image.open(str(src)).convert("RGB").resize((768, 512), Image.LANCZOS)
            start_tensor = pil_to_tensor(start_img)

            # 終端フレーム（次 stage の角度で取得）
            end_tensor    = None
            is_last_stage = (stage_idx == n_stages - 1)

            if use_end_frame and not is_last_stage:
                next_angle = theta_for_stage(
                    sweep_path, stage_idx + 1, n_stages, n_angles, fixed_angle
                )
                next_src = kf_dir / f"stage_{stage_idx+1:02d}" / f"angle_{next_angle:03d}.png"
                if next_src.exists():
                    next_img   = Image.open(str(next_src)).convert("RGB").resize((768, 512), Image.LANCZOS)
                    end_tensor = pil_to_tensor(next_img)
                else:
                    print(f"[WARN] 終端フレーム画像が見つかりません: {next_src}")
            elif is_last_stage and use_end_frame:
                print(f"[INFO] stage_{stage_idx:02d}: 最終 stage のため end_frame なしで生成します。")

            # ── LTXVImgToVideo ──────────────────────────────────────────────
            ef_log = f"end_frame={end_tensor is not None}"
            pct_start = stage_idx / n_stages * 100
            print(f"[STEP3 {stage_idx+1}/{n_stages} ({pct_start:.0f}%)] 生成中... ({ef_log})")

            img2vid_pos, img2vid_neg, img2vid_lat = LTXVImgToVideo_cls().execute(
                positive=ltxv_pos,
                negative=ltxv_neg,
                image=start_tensor,
                vae=vae,
                width=768,
                height=512,
                length=clip_frames,
                batch_size=1,
                strength=1.0,
            )

            pos_final = img2vid_pos
            neg_final = img2vid_neg
            lat_final = img2vid_lat

            # ── LTXVAddGuide チェーン ────────────────────────────────────────
            if end_tensor is not None:
                (preprocessed_start,) = LTXVPreprocess_cls().execute(
                    image=start_tensor, img_compression=35
                )
                guide_pos1, guide_neg1, guide_lat1 = LTXVAddGuide_cls().execute(
                    positive=pos_final, negative=neg_final,
                    vae=vae, latent=lat_final,
                    image=preprocessed_start, frame_idx=0, strength=1.0,
                )
                (preprocessed_end,) = LTXVPreprocess_cls().execute(
                    image=end_tensor, img_compression=35
                )
                guide_pos2, guide_neg2, guide_lat2 = LTXVAddGuide_cls().execute(
                    positive=guide_pos1, negative=guide_neg1,
                    vae=vae, latent=guide_lat1,
                    image=preprocessed_end, frame_idx=-1, strength=end_frame_strength,
                )
                pos_final = guide_pos2
                neg_final = guide_neg2
                lat_final = guide_lat2

            # ── サンプリング ─────────────────────────────────────────────────
            (guider,) = CFGGuider_cls().execute(
                model=model, positive=pos_final, negative=neg_final, cfg=cfg
            )
            (noise,) = RandomNoise_cls().execute(noise_seed=seed + stage_idx)
            output_latent, _ = SamplerCustomAdv_cls().execute(
                noise=noise, guider=guider, sampler=sampler,
                sigmas=sigmas, latent_image=lat_final,
            )

            # ── VAE デコード ─────────────────────────────────────────────────
            (image_tensor,) = VAEDecode_cls().decode(vae=vae, samples=output_latent)
            frames = tensor_to_pils(image_tensor)

            if not frames:
                print(f"[WARN] stage_{stage_idx:02d}: フレームが取得できませんでした。スキップします。")
                if pbar is not None:
                    pbar.update(1)
                continue

            pct_done = (stage_idx + 1) / n_stages * 100
            print(f"[STEP3 {stage_idx+1}/{n_stages} ({pct_done:.0f}%)] 完了: {len(frames)}フレーム")

            if stage_idx == 0:
                all_frames.extend(frames)
            else:
                all_frames.extend(frames[1:])

            print(f"  累計フレーム: {len(all_frames)}")
            if pbar is not None:
                pbar.update(1)

        if not all_frames:
            raise RuntimeError("有効なフレームが生成されませんでした。")

        # ── 8. RGBA PNG 保存 ────────────────────────────────────────────────────
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
        status   = f"完了: {N}フレーム = {duration:.1f}秒 @ {fps}fps  {mode_str} → {png_dir}"
        print(f"\n[完了] {status}")

        return (str(png_dir), N, status)


# ─── 登録 ─────────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "RogoAI_NB4D_LTXVStageInterpolatorV3": NB4D_LTXVStageInterpolatorV3,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RogoAI_NB4D_LTXVStageInterpolatorV3": "RogoAI 4D: LTXV Stage Interpolator V3",
}

"""
NB4D Parallax Renderer ノード群
================================
カメラワーク一貫性の問題をパララックスワープで解決。
1枚の静止画 + Depth Map から幾何学的に一貫したカメラ動作を生成する。

ノード:
  - NB4D_DepthEstimator:     画像から Depth Map を生成
  - NB4D_ParallaxRenderer:   Depth Map を使ってパララックスワープ（3幕構成）
  - NB4D_StageAnchorManager: 態ごとの参照画像と否定プロンプトを管理（ハルシネーション対策）
"""

import os
import json
import numpy as np
import torch
from PIL import Image, ImageFilter

# ── 態固有の否定プロンプトデフォルト（ヤマトシジミ8態） ─────────────────
DEFAULT_STAGE_EXCLUSIONS = {
    "0": "",
    "1": "no intact egg, no unhatched egg",
    "2": "no egg, no hatching pose, fully elongated slug-like body",
    "3": "no egg, no tiny larva, prominent body segments visible",
    "4": "no moving larva, stationary prepupa, girdle silk visible",
    "5": "no larval segments, smooth chrysalis surface, no legs",
    "6": "no pupa casing, no closed chrysalis, fully formed wings even if wet",
    "7": "no pupa, no static pose, wings spread in flight",
}


# ── 深度推定ヘルパー ─────────────────────────────────────────────────────

def _estimate_depth_synthetic(image_np: np.ndarray) -> np.ndarray:
    """合成深度マップ（MiDaS 不使用フォールバック）
    中心重み + 輝度の逆数で簡易深度を生成。
    """
    H, W = image_np.shape[:2]
    img = image_np / 255.0 if image_np.max() > 1.5 else image_np.copy()

    gray = (0.299 * img[:, :, 0]
            + 0.587 * img[:, :, 1]
            + 0.114 * img[:, :, 2])

    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    dx = (x - W / 2) / (W / 2 + 1e-8)
    dy = (y - H / 2) / (H / 2 + 1e-8)
    center_weight = np.clip(1.0 - np.sqrt(dx ** 2 + dy ** 2) * 0.85, 0.0, 1.0)

    depth = 0.35 * (1.0 - gray) + 0.65 * center_weight
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

    # スムージング
    blur_r = max(H, W) // 50
    depth_pil = Image.fromarray((depth * 255).astype(np.uint8))
    depth_pil = depth_pil.filter(ImageFilter.GaussianBlur(radius=blur_r))
    return np.array(depth_pil).astype(np.float32) / 255.0


def _estimate_depth_midas(image_np: np.ndarray, model_type: str = "MiDaS_small") -> np.ndarray:
    """MiDaS で深度推定。失敗時は合成深度にフォールバック。"""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        midas = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
        midas = midas.to(device).eval()
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        transform = (transforms.small_transform
                     if model_type == "MiDaS_small"
                     else transforms.dpt_transform)

        img_u8 = (image_np * 255).astype(np.uint8) if image_np.max() <= 1.5 else image_np.astype(np.uint8)
        batch = transform(img_u8).to(device)

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
        return depth

    except Exception as e:
        print(f"[NB4D Depth] MiDaS 失敗: {e} → 合成深度にフォールバック")
        return _estimate_depth_synthetic(image_np)


# ── カメラ動作計算 ────────────────────────────────────────────────────────

def _get_camera_params(frame_idx: int, total_frames: int,
                       max_shift_x: float, max_shift_y: float,
                       zoom_in: float, zoom_out: float) -> tuple:
    """3幕構成のカメラパラメータを計算して返す。
    Returns: (scale, shift_x_frac, shift_y_frac)
      - scale: ズーム倍率（>1 でズームイン）
      - shift_x_frac: 水平シフト（画像幅の割合、正=右、負=左）
      - shift_y_frac: 垂直シフト（画像高の割合）
    """
    t = frame_idx / max(1, total_frames - 1)  # 0.0〜1.0

    if t < 0.25:        # イン: ズームイン（シフトなし）
        act_t   = t / 0.25
        scale   = 1.0 + (zoom_in - 1.0) * act_t
        shift_x = 0.0
        shift_y = 0.0
    elif t < 0.75:      # 見せ場: 左→右 水平パン
        act_t   = (t - 0.25) / 0.5
        scale   = zoom_in
        shift_x = (-1.0 + 2.0 * act_t) * max_shift_x  # -max→+max
        shift_y = 0.0
    else:               # アウト: わずかにズームアウト・パン保持
        act_t   = (t - 0.75) / 0.25
        scale   = zoom_in - (zoom_in - zoom_out) * act_t
        shift_x = max_shift_x
        shift_y = 0.0

    return scale, shift_x, shift_y


# ── パララックスワープ ────────────────────────────────────────────────────

def _warp_parallax(image_np: np.ndarray, depth_np: np.ndarray,
                   shift_x_frac: float, shift_y_frac: float,
                   scale: float, fill_mode: str = "edge_extend") -> np.ndarray:
    """パララックスワープのコア関数。
    image_np : H,W,3  float32  0-1
    depth_np : H,W    float32  0-1  (1=手前/foreground, 0=奥/background)
    shift_x_frac: 正＝カメラ右移動（前景が左にシフト）
    scale       : >1 でズームイン
    """
    H, W = image_np.shape[:2]
    shift_x_px = shift_x_frac * W
    shift_y_px = shift_y_frac * H
    cx, cy = W / 2.0, H / 2.0

    # 出力ピクセル座標グリッド
    dst_y, dst_x = np.mgrid[0:H, 0:W].astype(np.float32)

    # ソース座標 = ズームアンドゥ + 深度依存パララックスシフト
    # (深度が大きい=手前の画素ほど大きくシフト = 背景より速く動く)
    src_x = (dst_x - cx) / scale + cx + shift_x_px * depth_np
    src_y = (dst_y - cy) / scale + cy + shift_y_px * depth_np

    try:
        import cv2
        border = (cv2.BORDER_REPLICATE if fill_mode == "edge_extend"
                  else cv2.BORDER_REFLECT_101)
        img_u8 = (np.clip(image_np, 0, 1) * 255).astype(np.uint8)
        warped = cv2.remap(img_u8, src_x, src_y,
                           cv2.INTER_LINEAR, borderMode=border)
        return warped.astype(np.float32) / 255.0

    except ImportError:
        # cv2 なし: numpy 最近傍サンプリング（高速・低品質）
        src_xi = np.clip(src_x.astype(np.int32), 0, W - 1)
        src_yi = np.clip(src_y.astype(np.int32), 0, H - 1)
        return image_np[src_yi, src_xi, :]


# ═══════════════════════════════════════════════════════════════════════
# ComfyUI ノード定義
# ═══════════════════════════════════════════════════════════════════════

class NB4D_DepthEstimator:
    """画像から Depth Map を生成する (MiDaS / 合成 選択可)

    出力:
      depth_image: グレースケールDepth (IMAGE, 表示・確認用)
      depth_mask : 0-1 float32 マスク (MASK, ParallaxRenderer へ直接接続)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":        ("IMAGE",),
                "method":       (["synthetic", "midas_small", "midas_dpt"],
                                 {"default": "midas_small"}),
                "invert":       ("BOOLEAN", {"default": False,
                                 "tooltip": "近距離=白にする場合 True (MiDaS出力は通常 近=明るい)"}),
                "blur_radius":  ("INT", {"default": 10, "min": 0, "max": 100,
                                 "tooltip": "深度マップをスムージングするブラー半径"}),
            }
        }

    RETURN_TYPES  = ("IMAGE", "MASK")
    RETURN_NAMES  = ("depth_image", "depth_mask")
    FUNCTION      = "estimate"
    CATEGORY      = "RogoAI/4D"

    def estimate(self, image, method, invert, blur_radius):
        img_np = image[0].cpu().numpy()   # H,W,3  float32  0-1

        if method == "synthetic":
            depth = _estimate_depth_synthetic(img_np)
        elif method == "midas_small":
            depth = _estimate_depth_midas(img_np, "MiDaS_small")
        else:
            depth = _estimate_depth_midas(img_np, "DPT_Large")

        if invert:
            depth = 1.0 - depth

        if blur_radius > 0:
            dp = Image.fromarray((depth * 255).astype(np.uint8))
            dp = dp.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            depth = np.array(dp).astype(np.float32) / 255.0

        # depth_image: (1,H,W,3) RGB グレースケール（表示用）
        depth_rgb = np.stack([depth, depth, depth], axis=-1)
        depth_tensor = torch.from_numpy(depth_rgb).unsqueeze(0)

        # depth_mask: (1,H,W) 0-1
        depth_mask = torch.from_numpy(depth).unsqueeze(0)

        return (depth_tensor, depth_mask)


class NB4D_ParallaxRenderer:
    """Depth Map を使った幾何学的パララックスワープ

    仕様書 §4.4 の「左→右 45° 水平回転」を Depth ベースのパラックスで実現。
    全態で同一パラメータを使うことでカメラワークを数学的に完全一致させる。

    3幕構成（フレームインデックスに基づき自動分岐）:
      イン  (0〜25%)  : ズームイン (1.0 → zoom_in)
      見せ場(25〜75%) : 左→右 水平パン (-max_shift_x → +max_shift_x)
      アウト(75〜100%): わずかにズームアウト (zoom_in → zoom_out)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":         ("IMAGE",),
                "depth_mask":    ("MASK",),
                "frame_index":   ("INT", {"default": 0,   "min": 0, "max": 10000,
                                  "forceInput": True}),
                "total_frames":  ("INT", {"default": 192, "min": 1, "max": 10000,
                                  "forceInput": True}),
                "max_shift_x":   ("FLOAT", {"default": 0.04, "min": 0.0, "max": 0.3,
                                  "step": 0.005,
                                  "tooltip": "水平最大シフト（画像幅の割合）"}),
                "max_shift_y":   ("FLOAT", {"default": 0.01, "min": 0.0, "max": 0.1,
                                  "step": 0.005}),
                "zoom_in":       ("FLOAT", {"default": 1.15, "min": 1.0, "max": 2.0,
                                  "step": 0.01, "tooltip": "イン時のズーム倍率"}),
                "zoom_out":      ("FLOAT", {"default": 1.05, "min": 1.0, "max": 2.0,
                                  "step": 0.01, "tooltip": "アウト時のズーム倍率"}),
                "fill_mode":     (["edge_extend", "mirror"],
                                  {"default": "edge_extend",
                                   "tooltip": "シフトで露出した端の補完方式"}),
            }
        }

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("warped_frame",)
    FUNCTION      = "render"
    CATEGORY      = "RogoAI/4D"

    def render(self, image, depth_mask, frame_index, total_frames,
               max_shift_x, max_shift_y, zoom_in, zoom_out, fill_mode):

        img_np   = image[0].cpu().numpy()       # H,W,3  float32  0-1
        depth_np = depth_mask[0].cpu().numpy()  # H,W    float32  0-1

        scale, shift_x, shift_y = _get_camera_params(
            frame_index, total_frames,
            max_shift_x, max_shift_y,
            zoom_in, zoom_out,
        )

        warped = _warp_parallax(img_np, depth_np, shift_x, shift_y,
                                scale, fill_mode)

        return (torch.from_numpy(warped).unsqueeze(0),)


class NB4D_StageAnchorManager:
    """態ごとの参照画像と否定プロンプトを管理する

    ハルシネーション対策の2本柱:
      1. 実写科学参照画像 (ref_science)    → Gemini の subject reference に渡す
      2. 前態の確定済み生成画像 (ref_prev) → スタイル継続性の参照
      3. 態固有の否定プロンプト            → PromptComposer の negative に追加

    anchor_dir に stage_01.png〜stage_08.png を置くと自動ロードされる。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stage_index":   ("INT", {"default": 0, "min": 0, "max": 19,
                                  "tooltip": "0-indexed 態番号"}),
                "total_stages":  ("INT", {"default": 8, "min": 1, "max": 20}),
                "frame_index":   ("INT", {"default": 0, "min": 0, "max": 10000,
                                  "forceInput": True}),
                "total_frames":  ("INT", {"default": 192, "min": 1, "max": 10000,
                                  "forceInput": True}),
                "anchor_dir":    ("STRING", {"default": "",
                                  "tooltip": "実写参照画像フォルダ (stage_01.png 等)"}),
                "stage_exclusions_json": ("STRING", {
                    "multiline": True,
                    "default": json.dumps(DEFAULT_STAGE_EXCLUSIONS,
                                          indent=2, ensure_ascii=False),
                }),
            },
            "optional": {
                "previous_stage_image": ("IMAGE",),
            },
        }

    RETURN_TYPES  = ("IMAGE", "IMAGE", "STRING", "BOOLEAN")
    RETURN_NAMES  = ("ref_science", "ref_prev_stage",
                     "negative_supplement", "is_stage_start")
    FUNCTION      = "manage"
    CATEGORY      = "RogoAI/4D"

    def manage(self, stage_index, total_stages, frame_index, total_frames,
               anchor_dir, stage_exclusions_json, previous_stage_image=None):

        is_stage_start = (frame_index == 0)

        # 否定プロンプト取得
        try:
            excl = json.loads(stage_exclusions_json)
            negative = excl.get(str(stage_index),
                                excl.get(stage_index,
                                         DEFAULT_STAGE_EXCLUSIONS.get(
                                             str(stage_index), "")))
        except Exception:
            negative = DEFAULT_STAGE_EXCLUSIONS.get(str(stage_index), "")

        # 実写参照画像のロード
        ref_science = None
        if anchor_dir and os.path.isdir(anchor_dir):
            for fname in [
                f"stage_{stage_index + 1:02d}.png",
                f"stage{stage_index + 1:02d}.png",
                f"anchor_{stage_index:02d}.png",
                f"ref_{stage_index + 1:02d}.png",
            ]:
                fpath = os.path.join(anchor_dir, fname)
                if os.path.exists(fpath):
                    try:
                        arr = np.array(Image.open(fpath).convert("RGB"),
                                       dtype=np.float32) / 255.0
                        ref_science = torch.from_numpy(arr).unsqueeze(0)
                        break
                    except Exception:
                        pass

        return (ref_science, previous_stage_image, negative, is_stage_start)


# ── ノード登録 ────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "NB4D_DepthEstimator":    NB4D_DepthEstimator,
    "NB4D_ParallaxRenderer":  NB4D_ParallaxRenderer,
    "NB4D_StageAnchorManager": NB4D_StageAnchorManager,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NB4D_DepthEstimator":    "NB4D Depth Estimator (深度推定)",
    "NB4D_ParallaxRenderer":  "NB4D Parallax Renderer (パララックス)",
    "NB4D_StageAnchorManager": "NB4D Stage Anchor Manager (態アンカー管理)",
}

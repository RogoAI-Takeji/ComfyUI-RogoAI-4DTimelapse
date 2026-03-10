"""
GeminiImageGenerator Node
=========================
Google Gemini / Imagen 3 API でテキスト・参考画像から画像を生成する ComfyUI ノード。

対応モデル:
  - gemini-2.0-flash-exp-image-generation  ← 無料枠あり（デフォルト）
  - imagen-3.0-generate-002                ← 高品質（有料）
  - imagen-3.0-fast-generate-001           ← 高速・低コスト（有料）

API 経路の違い:
  - Gemini Flash : generate_content() + response_modalities=["IMAGE"]
  - Imagen 3     : generate_images() + GenerateImagesConfig

API key は Google AI Studio で取得:
  https://aistudio.google.com/app/apikey

Dependency:
  pip install google-genai>=1.0.0
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# ── google-genai SDK ──────────────────────────────────────────────────────────
try:
    from google import genai
    from google.genai import types as genai_types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    genai = None
    genai_types = None


# ── 定数 ──────────────────────────────────────────────────────────────────────

# 先頭がデフォルト値になる（ドロップダウン表示用）
ALL_MODELS = [
    "NanoBanana 2 (gemini-3.1-flash-image-preview)",
    "NanoBanana Pro (gemini-3.0-pro-exp-image-generation)",
    "NanoBanana (gemini-2.0-flash-exp-image-generation)",
    "Imagen 4.0 (imagen-4.0-generate-001)",
    "Imagen 4.0 Fast (imagen-4.0-fast-generate-001)",
    "Imagen 4.0 Ultra (imagen-4.0-ultra-generate-001)",
]

# Imagen 3 系モデルの識別（それ以外は Gemini Flash 経路）
_IMAGEN_PREFIXES = ("imagen-",)

ASPECT_RATIOS = ["1:1", "3:4", "4:3", "16:9", "9:16"]

SAFETY_FILTERS = [
    "BLOCK_LOW_AND_ABOVE",
    "BLOCK_MEDIUM_AND_ABOVE",
    "BLOCK_ONLY_HIGH",
    "BLOCK_NONE",
]

REFERENCE_MODES = [
    "none",     # テキストのみ
    "style",    # スタイル転写（Imagen 3 のみ）
    "subject",  # 被写体参照（Imagen 3 のみ）
]

PERSON_GENERATION = [
    "ALLOW_ADULT",
    "DONT_ALLOW",
    "ALLOW_ALL",
]

# API key 永続保存先: ComfyUI の models フォルダ直下
_KEY_FILE = Path(__file__).parent.parent.parent.parent / "models" / "gemini_api_key.txt"

# モジュールロード時に1回だけ読み込む（INPUT_TYPES()は複数回呼ばれるためキャッシュする）
_CACHED_API_KEY: str | None = None


# ── ヘルパー ──────────────────────────────────────────────────────────────────

def _is_imagen(model: str) -> bool:
    """モデル名が Imagen 3 系かどうかを判定する。"""
    return any(model.startswith(p) for p in _IMAGEN_PREFIXES)


def _load_saved_key() -> str:
    """models/gemini_api_key.txt から保存済み API key を読み込む（モジュール内キャッシュ付き）。"""
    global _CACHED_API_KEY
    if _CACHED_API_KEY is not None:
        return _CACHED_API_KEY
    try:
        if _KEY_FILE.exists():
            key = _KEY_FILE.read_text(encoding="utf-8").strip()
            if key:
                print(f"[NanaBanana] API key を {_KEY_FILE} から読み込みました。")
            _CACHED_API_KEY = key
            return key
    except Exception as e:
        print(f"[NanaBanana] API key 読み込み失敗: {e}")
    _CACHED_API_KEY = ""
    return ""


def _save_key(key: str) -> None:
    """API key を models/gemini_api_key.txt に保存し、キャッシュも更新する。"""
    global _CACHED_API_KEY
    try:
        _KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
        _KEY_FILE.write_text(key.strip(), encoding="utf-8")
        _CACHED_API_KEY = key.strip()
        print(f"[NanaBanana] API key を {_KEY_FILE} に保存しました。")
    except Exception as e:
        print(f"[NanaBanana] API key 保存失敗: {e}")


def _pil_to_comfy(pil_img: Image.Image) -> torch.Tensor:
    """PIL Image → ComfyUI IMAGE tensor (1, H, W, C) float32 0-1"""
    arr = np.array(pil_img.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _comfy_to_pil(tensor: torch.Tensor) -> Image.Image:
    """ComfyUI IMAGE tensor (B, H, W, C) → PIL Image（先頭1枚）"""
    arr = (tensor[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _pil_to_bytes(pil_img: Image.Image, fmt: str = "PNG") -> bytes:
    """PIL Image → bytes（API 送信用）"""
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return buf.getvalue()


def _extract_images_from_content_response(response) -> list[Image.Image]:
    """
    generate_content() のレスポンスから画像パーツを抽出して PIL リストで返す。
    Gemini Flash 画像生成用。
    """
    images = []
    for candidate in response.candidates:
        for part in candidate.content.parts:
            if part.inline_data and part.inline_data.mime_type.startswith("image/"):
                images.append(Image.open(io.BytesIO(part.inline_data.data)))
    return images


# ── ノード ────────────────────────────────────────────────────────────────────

class GeminiImageGenerator:
    """
    NanoBanana 2 (3.1 Flash) / NanoBanana Pro (3.0 Pro) / Imagen 4.0 を使用して
    画像を生成するオールインワン型ノードです。
    参考画像を最大3枚まで入力でき、強力なマルチモーダル指示が可能です。
    """

    @classmethod
    def INPUT_TYPES(cls):
        saved_key = _load_saved_key()
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "画像生成プロンプト（英語推奨）",
                        "tooltip": "Gemini / Imagen への生成指示テキスト。英語推奨。",
                    },
                ),
                "model": (
                    ALL_MODELS,
                    {
                        "default": ALL_MODELS[0],
                        "tooltip": "NanoBanana (3.1/3.0/2.0) / Imagen 4",
                    },
                ),
                "aspect_ratio": (
                    ASPECT_RATIOS,
                    {
                        "default": "1:1",
                        "tooltip": "Imagen 3 のみ有効。Gemini Flash は無視されます。",
                    },
                ),
                "num_images": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 4,
                        "step": 1,
                        "tooltip": (
                            "生成枚数。\n"
                            "Imagen 3: 1回のリクエストで最大4枚\n"
                            "Gemini Flash: 1枚ずつ複数回リクエスト"
                        ),
                    },
                ),
                 "api_key": (
                    "STRING",
                    {
                        "default": "",
                        "password": True,
                        "hide_value": True,
                        "placeholder": "Enter API Key here (Required for first time)",
                        "tooltip": (
                            "Google AI API key。\n"
                            "空欄にすると models/gemini_api_key.txt の保存済みキーを使用します。\n"
                            "取得: https://aistudio.google.com/app/apikey"
                        ),
                    },
                ),
                "save_key": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "保存する",
                        "label_off": "保存しない",
                        "tooltip": "True にすると models/gemini_api_key.txt に保存（次回から自動読み込み）",
                    },
                ),
            },
            "optional": {
                "negative_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "避けたい要素（例: blurry, low quality, text）",
                        "tooltip": "Imagen 3 のみ有効。Gemini Flash は無視されます。",
                    },
                ),
                "reference_image_1": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "参考画像1（省略可）。\n"
                            "Imagen 4: style / subject モードで使用\n"
                            "Gemini Flash: プロンプトと一緒にマルチモーダル入力として送信"
                        ),
                    },
                ),
                "reference_image_2": (
                    "IMAGE",
                    {
                        "tooltip": "参考画像2（省略可）。",
                    },
                ),
                "reference_image_3": (
                    "IMAGE",
                    {
                        "tooltip": "参考画像3（省略可）。",
                    },
                ),
                "reference_mode": (
                    REFERENCE_MODES,
                    {
                        "default": "none",
                        "tooltip": (
                            "none = テキストのみ\n"
                            "style = 参考画像のスタイルを転写（Imagen 3 のみ）\n"
                            "subject = 参考画像の被写体を使用（Imagen 3 のみ）\n"
                            "※ Gemini Flash は none 以外でも画像をマルチモーダル入力として使用"
                        ),
                    },
                ),
                "safety_filter": (
                    SAFETY_FILTERS,
                    {
                        "default": "BLOCK_LOW_AND_ABOVE",
                        "tooltip": "セーフティフィルターレベル（Imagen のみ有効）",
                    },
                ),
                "person_generation": (
                    PERSON_GENERATION,
                    {
                        "default": "ALLOW_ADULT",
                        "tooltip": "人物生成許可レベル（Imagen 3 のみ有効）",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "generate"
    CATEGORY = "RogoAI"
    DESCRIPTION = "NanoBanana 2 (3.1 Flash) / Pro (3.0 Pro) / Imagen 4.0 による画像生成ノード"

    # ── (A) Gemini Flash 経路 ─────────────────────────────────────────────────

    def _generate_flash(
        self,
        client,
        prompt: str,
        model: str,
        num_images: int,
        ref_pils: list[Image.Image],
    ) -> list[Image.Image]:
        """
        gemini-2.0-flash-exp-image-generation 用。
        generate_content() に response_modalities=["IMAGE"] を指定して呼ぶ。
        num_images 分だけ繰り返し呼び出す（1回1枚）。
        参考画像がある場合はマルチモーダル入力として付加する。
        """
        if ref_pils:
            response_modalities = ["TEXT", "IMAGE"]
        else:
            response_modalities = ["IMAGE"]

        config = genai_types.GenerateContentConfig(
            response_modalities=response_modalities,
        )

        pil_images = []
        for i in range(num_images):
            if ref_pils:
                parts = [genai_types.Part(text=prompt)]
                for rp in ref_pils:
                    parts.append(
                        genai_types.Part(
                            inline_data=genai_types.Blob(
                                mime_type="image/png",
                                data=_pil_to_bytes(rp),
                            )
                        )
                    )
                contents = [genai_types.Content(role="user", parts=parts)]
            else:
                contents = prompt

            print(f"[NanaBanana] Gemini Flash リクエスト {i + 1}/{num_images}...")
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            extracted = _extract_images_from_content_response(response)
            if extracted:
                pil_images.extend(extracted)
            else:
                print(f"[NanaBanana] リクエスト {i + 1}: 画像が返ってきませんでした（フィルター除外の可能性）")

        return pil_images

    # ── (B) Imagen 3 経路 ─────────────────────────────────────────────────────

    def _generate_imagen(
        self,
        client,
        prompt: str,
        model: str,
        num_images: int,
        aspect_ratio: str,
        negative_prompt: str,
        safety_filter: str,
        person_generation: str,
        ref_pils: list[Image.Image],
        reference_mode: str,
    ) -> list[Image.Image]:
        """
        Imagen 4 用。
        参考画像なし → generate_images()
        参考画像あり → edit_image() (style / subject)
        """
        pil_images = []

        # Imagen 4.0 API enforces BLOCK_LOW_AND_ABOVE minimum
        imagen_supported_filters = ["BLOCK_LOW_AND_ABOVE"]
        if safety_filter not in imagen_supported_filters:
            print(f"[NanaBanana] Warning: Imagen models only support BLOCK_LOW_AND_ABOVE. Overriding {safety_filter} to BLOCK_LOW_AND_ABOVE.")
            safety_filter = "BLOCK_LOW_AND_ABOVE"

        if not ref_pils or reference_mode == "none":
            # テキストのみ
            config = genai_types.GenerateImagesConfig(
                number_of_images=num_images,
                aspect_ratio=aspect_ratio,
                negative_prompt=negative_prompt.strip() or None,
                safety_filter_level=safety_filter,
                person_generation=person_generation,
            )
            print(
                f"[NanaBanana] Imagen generate_images | model={model} "
                f"ratio={aspect_ratio} n={num_images}"
            )
            response = client.models.generate_images(
                model=model,
                prompt=prompt,
                config=config,
            )
            for gen_img in response.generated_images:
                pil_images.append(
                    Image.open(io.BytesIO(gen_img.image.image_bytes))
                )

        else:
            # 参考画像あり (style / subject)
            ref_objs = []
            for rp in ref_pils:
                ref_bytes = _pil_to_bytes(rp)
                if reference_mode == "style":
                    ref_objs.append(
                        genai_types.StyleReferenceImage(
                            reference_image=genai_types.Image(image_bytes=ref_bytes),
                            config=genai_types.StyleReferenceConfig(
                                style_description=prompt,
                            ),
                        )
                    )
                else:  # "subject"
                    ref_objs.append(
                        genai_types.SubjectReferenceImage(
                            reference_image=genai_types.Image(image_bytes=ref_bytes),
                            config=genai_types.SubjectReferenceConfig(
                                subject_description=prompt,
                            ),
                        )
                    )

            edit_config = genai_types.EditImageConfig(
                number_of_images=num_images,
                safety_filter_level=safety_filter,
                person_generation=person_generation,
            )
            print(
                f"[NanaBanana] Imagen edit_image | model={model} "
                f"mode={reference_mode} refs={len(ref_pils)} n={num_images}"
            )
            response = client.models.edit_image(
                model=model,
                prompt=prompt,
                reference_images=ref_objs,
                config=edit_config,
            )
            for gen_img in response.generated_images:
                pil_images.append(
                    Image.open(io.BytesIO(gen_img.image.image_bytes))
                )

        return pil_images

    # ── メイン ────────────────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        model: str,
        aspect_ratio: str,
        num_images: int,
        api_key: str,
        save_key: bool,
        negative_prompt: str = "",
        reference_image_1=None,
        reference_image_2=None,
        reference_image_3=None,
        reference_mode: str = "none",
        safety_filter: str = "BLOCK_LOW_AND_ABOVE",
        person_generation: str = "ALLOW_ADULT",
    ):
        # ── 0. 依存チェック ──────────────────────────────────────────
        if not GENAI_AVAILABLE:
            raise RuntimeError(
                "[NanaBanana] google-genai が未インストールです。\n"
                "ComfyUI の Python 環境で以下を実行してください:\n"
                "  pip install google-genai>=1.0.0"
            )

        # ── 1. API key 処理 ─────────────────────────────────────────
        key = api_key.strip()
        if not key:
            key = _load_saved_key()
            if key:
                print("[NanaBanana] 保存済みの API key を使用します。")
        
        if not key:
            raise ValueError(
                "[NanaBanana] API key が未入力です。\n"
                "https://aistudio.google.com/app/apikey で取得してください。"
            )

        if save_key and api_key.strip():
            _save_key(key)

        # ── 2. クライアント初期化 ────────────────────────────────────
        client = genai.Client(api_key=key)

        # ── 2.5 モデル名のパース (Alias対応) ─────────────────────────
        actual_model = model
        if "(" in model and model.endswith(")"):
            actual_model = model.split("(")[-1].rstrip(")")

        # ── 3. 参考画像の準備 ────────────────────────────────────────
        ref_pils = []
        for ri in [reference_image_1, reference_image_2, reference_image_3]:
            if ri is not None:
                p = _comfy_to_pil(ri)
                ref_pils.append(p)
                print(f"[NanaBanana] 参考画像 size={p.size}")

        # ── 4. モデル経路の分岐 ──────────────────────────────────────
        try:
            if _is_imagen(actual_model):
                pil_images = self._generate_imagen(
                    client, prompt, actual_model, num_images,
                    aspect_ratio, negative_prompt, safety_filter,
                    person_generation, ref_pils, reference_mode,
                )
            else:
                # Gemini Flash / Pro 経路
                pil_images = self._generate_flash(
                    client, prompt, actual_model, num_images, ref_pils,
                )
        except Exception as e:
            raise RuntimeError(f"[NanaBanana] Gemini API エラー: {e}") from e

        # ── 5. 生成結果チェック ──────────────────────────────────────
        if not pil_images:
            raise RuntimeError(
                "[NanaBanana] 画像が生成されませんでした。\n"
                "・プロンプトを変更してみてください\n"
                "・Imagen 3 の場合: safety_filter を下げてみてください\n"
                "・無料枠の場合: 1日のリクエスト上限に達した可能性があります"
            )

        # ── 6. ComfyUI テンソルに変換・バッチ結合 (N,H,W,C) ─────────
        if len(pil_images) > 1:
            base_size = pil_images[0].size
            for i in range(1, len(pil_images)):
                if pil_images[i].size != base_size:
                    print(f"[NanaBanana] Resize {pil_images[i].size} -> {base_size}")
                    pil_images[i] = pil_images[i].resize(base_size, Image.Resampling.LANCZOS)

        tensors = [_pil_to_comfy(img) for img in pil_images]
        output = torch.cat(tensors, dim=0)

        print(
            f"[NanaBanana] 生成完了 {len(pil_images)} 枚 | "
            f"shape={tuple(output.shape)}"
        )
        return (output,)


# ── 登録 ──────────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "RogoAI_GeminiImageGenerator": GeminiImageGenerator,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RogoAI_GeminiImageGenerator": "RogoAI Gemini Image Generator",
}

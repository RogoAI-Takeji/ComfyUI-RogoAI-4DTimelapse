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

# 先頭がデフォルト値になる
ALL_MODELS = [
    "gemini-2.0-flash-exp-image-generation",  # 無料枠あり（デフォルト）
    "imagen-4.0-generate-001",                # 高品質（有料）
    "imagen-4.0-fast-generate-001",           # 高速・低コスト（有料）
    "imagen-4.0-ultra-generate-001",
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


# ── ヘルパー ──────────────────────────────────────────────────────────────────

def _is_imagen(model: str) -> bool:
    """モデル名が Imagen 3 系かどうかを判定する。"""
    return any(model.startswith(p) for p in _IMAGEN_PREFIXES)


def _load_saved_key() -> str:
    """models/gemini_api_key.txt から保存済み API key を読み込む。"""
    try:
        if _KEY_FILE.exists():
            key = _KEY_FILE.read_text(encoding="utf-8").strip()
            if key:
                print(f"[NanaBanana] API key を {_KEY_FILE} から読み込みました。")
            return key
    except Exception as e:
        print(f"[NanaBanana] API key 読み込み失敗: {e}")
    return ""


def _save_key(key: str) -> None:
    """API key を models/gemini_api_key.txt に保存する。"""
    try:
        _KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
        _KEY_FILE.write_text(key.strip(), encoding="utf-8")
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
    Google Gemini / Imagen 3 API でテキストプロンプトと参考画像から画像を生成します。

    gemini-2.0-flash-exp-image-generation は無料枠あり（毎日リセット）。
    Imagen 3 は有料（高品質・参考画像モード対応）。
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
                    [
                        "gemini-2.0-flash-exp-image-generation",
                        "imagen-4.0-generate-001",
                        "imagen-4.0-fast-generate-001",
                        "imagen-4.0-ultra-generate-001"
                    ],
                    {
                        "default": "gemini-2.0-flash-exp-image-generation",
                        "tooltip": "Gemini: 無料/マルチモーダル, Imagen 4: 有料/高品質"
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
                        "default": saved_key,
                        "password": True,
                        "tooltip": (
                            "Google AI API key。\n"
                            "取得: https://aistudio.google.com/app/apikey\n"
                            "save_key=True で自動保存できます。"
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
                "reference_image": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "参考画像（省略可）。\n"
                            "Imagen 3: style / subject モードで使用\n"
                            "Gemini Flash: プロンプトと一緒にマルチモーダル入力として送信"
                        ),
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
    DESCRIPTION = "Google Gemini Flash（無料）/ Imagen 3（有料）で画像を生成します"

    # ── (A) Gemini Flash 経路 ─────────────────────────────────────────────────

    def _generate_flash(
        self,
        client,
        prompt: str,
        model: str,
        num_images: int,
        ref_pil: Image.Image | None,
    ) -> list[Image.Image]:
        """
        gemini-2.0-flash-exp-image-generation 用。
        generate_content() に response_modalities=["IMAGE"] を指定して呼ぶ。
        num_images 分だけ繰り返し呼び出す（1回1枚）。
        参考画像がある場合はマルチモーダル入力として付加する。
        """
        # 参考画像がある場合は TEXT も出力モダリティに含める必要がある
        # （Gemini Flash は multimodal 入力時に IMAGE だけだと 400 エラー）
        if ref_pil is not None:
            response_modalities = ["TEXT", "IMAGE"]
        else:
            response_modalities = ["IMAGE"]

        config = genai_types.GenerateContentConfig(
            response_modalities=response_modalities,
        )

        pil_images = []
        for i in range(num_images):
            if ref_pil is not None:
                # マルチモーダル: テキスト + 参考画像
                # role="user" は必須フィールド
                contents = [
                    genai_types.Content(
                        role="user",
                        parts=[
                            genai_types.Part(text=prompt),
                            genai_types.Part(
                                inline_data=genai_types.Blob(
                                    mime_type="image/png",
                                    data=_pil_to_bytes(ref_pil),
                                )
                            ),
                        ]
                    )
                ]
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
        ref_pil: Image.Image | None,
        reference_mode: str,
    ) -> list[Image.Image]:
        """
        Imagen 3 用。
        参考画像なし → generate_images()
        参考画像あり → edit_image() (style / subject)
        """
        pil_images = []

        # Imagen 4.0 API enforces BLOCK_LOW_AND_ABOVE minimum
        imagen_supported_filters = ["BLOCK_LOW_AND_ABOVE"]
        if safety_filter not in imagen_supported_filters:
            print(f"[NanaBanana] Warning: Imagen models only support BLOCK_LOW_AND_ABOVE. Overriding {safety_filter} to BLOCK_LOW_AND_ABOVE.")
            safety_filter = "BLOCK_LOW_AND_ABOVE"

        if ref_pil is None or reference_mode == "none":
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
            ref_bytes = _pil_to_bytes(ref_pil)
            if reference_mode == "style":
                ref_obj = genai_types.StyleReferenceImage(
                    reference_image=genai_types.Image(image_bytes=ref_bytes),
                    config=genai_types.StyleReferenceConfig(
                        style_description=prompt,
                    ),
                )
            else:  # "subject"
                ref_obj = genai_types.SubjectReferenceImage(
                    reference_image=genai_types.Image(image_bytes=ref_bytes),
                    config=genai_types.SubjectReferenceConfig(
                        subject_description=prompt,
                    ),
                )

            edit_config = genai_types.EditImageConfig(
                number_of_images=num_images,
                safety_filter_level=safety_filter,
                person_generation=person_generation,
            )
            print(
                f"[NanaBanana] Imagen edit_image | model={model} "
                f"mode={reference_mode} n={num_images}"
            )
            response = client.models.edit_image(
                model=model,
                prompt=prompt,
                reference_images=[ref_obj],
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
        reference_image=None,
        reference_mode: str = "none",
        safety_filter: str = "BLOCK_MOST",
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
        key = api_key.strip() or _load_saved_key()
        if not key:
            raise ValueError(
                "[NanaBanana] API key が未入力です。\n"
                "https://aistudio.google.com/app/apikey で取得してください。"
            )
        if save_key:
            _save_key(key)

        # ── 2. クライアント初期化 ────────────────────────────────────
        client = genai.Client(api_key=key)

        # ── 3. 参考画像の準備 ────────────────────────────────────────
        ref_pil = None
        if reference_image is not None:
            ref_pil = _comfy_to_pil(reference_image)
            print(f"[NanaBanana] 参考画像 size={ref_pil.size}")

        # ── 4. モデル経路の分岐 ──────────────────────────────────────
        try:
            if _is_imagen(model):
                pil_images = self._generate_imagen(
                    client, prompt, model, num_images,
                    aspect_ratio, negative_prompt, safety_filter,
                    person_generation, ref_pil, reference_mode,
                )
            else:
                # Gemini Flash（無料枠）
                pil_images = self._generate_flash(
                    client, prompt, model, num_images, ref_pil,
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
        tensors = [_pil_to_comfy(img) for img in pil_images]
        output = torch.cat(tensors, dim=0)

        print(
            f"[NanaBanana] 生成完了 {len(pil_images)} 枚 | "
            f"shape={tuple(output.shape)}"
        )
        return (output,)


# ── 登録 ──────────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "GeminiImageGenerator": GeminiImageGenerator,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiImageGenerator": "RogoAI Gemini Image Generator",
}

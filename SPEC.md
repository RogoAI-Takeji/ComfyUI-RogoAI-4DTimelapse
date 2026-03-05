# ComfyUI-RogoAI-NanoBanana — 実装仕様書

**作成日**: 2026-03-05
**目的**: セッション引き継ぎ用。この仕様書だけで実装を完結できる粒度で記述する。

---

## 概要

Google Gemini API（Imagen 3 / Gemini 2.0 Flash）を使って、ComfyUI 上でテキストプロンプト・参考画像から画像を生成するカスタムノードパッケージ。

**パッケージ名**: `ComfyUI-RogoAI-NanoBanana`
**配置先**: `d:\Python_VENV\for_comfy_ltx2_260115\Data\Packages\ComfyUI_for_LTX2\custom_nodes\ComfyUI-RogoAI-NanoBanana\`
**カテゴリ**: `RogoAI`

---

## ディレクトリ構成（作成すべきファイル全て）

```
ComfyUI-RogoAI-NanoBanana/
├── __init__.py               ← NODE_CLASS_MAPPINGS をエクスポート
├── pyproject.toml            ← パッケージメタデータ
├── README.md                 ← 使い方（任意）
└── nodes/
    ├── __init__.py           ← 各ノードを集約
    └── gemini_image_gen.py   ← メインノード実装
```

---

## 必須 pip パッケージ

```
google-genai>=1.0.0
Pillow>=9.0
```

`google-genai` は Google の最新 SDK（`google-generativeai` の後継）。
ComfyUI 環境に未インストールの場合は `pip install google-genai` が必要。

---

## pyproject.toml

```toml
[project]
name = "comfyui-rogo-nanobanana"
version = "1.0.0"
description = "ComfyUI nodes for Gemini / Imagen image generation via Google AI API"
license = { text = "MIT" }
requires-python = ">=3.10"
dependencies = ["google-genai>=1.0.0"]

[tool.comfy]
PublisherId = "rogoai"
DisplayName = "NanoBanana - Gemini Image Generator"
Icon = ""
```

---

## トップレベル `__init__.py`

```python
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
```

---

## `nodes/__init__.py`

```python
from .gemini_image_gen import (
    NODE_CLASS_MAPPINGS as GEN_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as GEN_DISPLAY,
)

NODE_CLASS_MAPPINGS = {**GEN_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS = {**GEN_DISPLAY}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
```

---

## `nodes/gemini_image_gen.py` — 実装仕様

### 依存インポート

```python
from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

# google-genai SDK (pip install google-genai)
try:
    from google import genai
    from google.genai import types as genai_types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
```

### 定数

```python
# モデルリスト（先頭がデフォルト）
ALL_MODELS = [
    "gemini-2.0-flash-exp-image-generation",  # 無料枠あり（デフォルト）← generate_content() 経路
    "imagen-3.0-generate-002",                # 高品質（有料）← generate_images() 経路
    "imagen-3.0-fast-generate-001",           # 高速・低コスト（有料）← generate_images() 経路
]

# Imagen 3 系の識別（それ以外は Gemini Flash 経路）
_IMAGEN_PREFIXES = ("imagen-",)
def _is_imagen(model: str) -> bool:
    return any(model.startswith(p) for p in _IMAGEN_PREFIXES)

# アスペクト比
ASPECT_RATIOS = ["1:1", "3:4", "4:3", "16:9", "9:16"]

# セーフティフィルター
SAFETY_FILTERS = [
    "BLOCK_MOST",
    "BLOCK_SOME",
    "BLOCK_FEW",
    "BLOCK_ONLY_HIGH",
]

# 参考画像モード
REFERENCE_MODES = [
    "none",          # テキストのみ
    "style",         # スタイル転写
    "subject",       # 被写体参照
]

# API key 保存パス（ComfyUI の models フォルダ直下）
_KEY_FILE = Path(__file__).parent.parent.parent.parent / "models" / "gemini_api_key.txt"
# 実際のパス例: .../ComfyUI_for_LTX2/models/gemini_api_key.txt
```

### ヘルパー関数

```python
def _load_saved_key() -> str:
    """models/gemini_api_key.txt からキーを読み込む（なければ空文字）"""
    try:
        if _KEY_FILE.exists():
            return _KEY_FILE.read_text(encoding="utf-8").strip()
    except Exception:
        pass
    return ""


def _save_key(key: str) -> None:
    """API key をファイルに保存する"""
    try:
        _KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
        _KEY_FILE.write_text(key.strip(), encoding="utf-8")
    except Exception as e:
        print(f"[NanoBanana] API key 保存失敗: {e}")


def _pil_to_comfy(pil_img: Image.Image) -> torch.Tensor:
    """PIL Image → ComfyUI IMAGE tensor (B,H,W,C) float32 0-1"""
    arr = np.array(pil_img.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _comfy_to_pil(tensor: torch.Tensor) -> Image.Image:
    """ComfyUI IMAGE tensor (B,H,W,C) → PIL Image（バッチ先頭1枚）"""
    arr = (tensor[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)
```

### ノードクラス `GeminiImageGenerator`

#### INPUT_TYPES

```python
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
                },
            ),
            "model": (IMAGEN_MODELS, {"default": IMAGEN_MODELS[0]}),
            "aspect_ratio": (ASPECT_RATIOS, {"default": "1:1"}),
            "num_images": (
                "INT",
                {"default": 1, "min": 1, "max": 4, "step": 1,
                 "tooltip": "生成枚数（Imagen 3 の上限は 4）"},
            ),
            "api_key": (
                "STRING",
                {
                    "default": saved_key,
                    "password": True,        # ← パスワード表示
                    "tooltip": "Google AI Studio の API key。入力後 save_key=True で保存可能",
                },
            ),
            "save_key": (
                "BOOLEAN",
                {
                    "default": False,
                    "label_on": "保存する",
                    "label_off": "保存しない",
                    "tooltip": "True にすると models/gemini_api_key.txt に API key を保存します",
                },
            ),
        },
        "optional": {
            "negative_prompt": (
                "STRING",
                {
                    "multiline": True,
                    "default": "",
                    "placeholder": "避けたい要素（例: blurry, low quality）",
                },
            ),
            "reference_image": ("IMAGE", {"tooltip": "参考画像（省略可）"}),
            "reference_mode": (REFERENCE_MODES, {"default": "none"}),
            "safety_filter": (SAFETY_FILTERS, {"default": "BLOCK_MOST"}),
            "seed": (
                "INT",
                {
                    "default": -1,
                    "min": -1,
                    "max": 2147483647,
                    "tooltip": "-1 でランダム（Imagen 3 はシード非対応のため参考値）",
                },
            ),
        },
    }
```

#### クラス属性

```python
RETURN_TYPES = ("IMAGE",)
RETURN_NAMES = ("images",)
FUNCTION = "generate"
CATEGORY = "RogoAI"
DESCRIPTION = "Google Gemini / Imagen 3 API で画像を生成します"
```

#### `generate()` メソッド — ロジック詳細

```python
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
    seed: int = -1,
):
    # ── 0. 依存チェック ──────────────────────────────────────────────
    if not GENAI_AVAILABLE:
        raise RuntimeError(
            "[NanoBanana] google-genai が未インストールです。\n"
            "pip install google-genai を実行してください。"
        )

    # ── 1. API key 処理 ─────────────────────────────────────────────
    key = api_key.strip()
    if not key:
        key = _load_saved_key()
    if not key:
        raise ValueError("[NanoBanana] API key が未入力です。")
    if save_key:
        _save_key(key)
        print("[NanoBanana] API key を保存しました。")

    # ── 2. クライアント初期化 ────────────────────────────────────────
    client = genai.Client(api_key=key)

    # ── 3. 参考画像の準備 ────────────────────────────────────────────
    ref_pil = None
    if reference_image is not None and reference_mode != "none":
        ref_pil = _comfy_to_pil(reference_image)

    # ── 4. 画像生成リクエスト ────────────────────────────────────────
    #
    # (A) 参考画像なし → generate_images()
    # (B) 参考画像あり → edit_image() / subject_reference
    #
    # generate_images() のパラメータ:
    #   model          : モデル名
    #   prompt         : テキストプロンプト
    #   config         : GenerateImagesConfig
    #
    # GenerateImagesConfig のフィールド:
    #   number_of_images        : int (1-4)
    #   aspect_ratio            : str "1:1" etc.
    #   negative_prompt         : str
    #   safety_filter_level     : "BLOCK_MOST" | "BLOCK_SOME" | "BLOCK_FEW" | "BLOCK_ONLY_HIGH"
    #   person_generation       : "DONT_ALLOW" | "ALLOW_ADULT" | "ALLOW_ALL"
    #
    # edit_image() のパラメータ（参考画像使用時）:
    #   model          : モデル名
    #   prompt         : テキストプロンプト
    #   reference_images : list of ReferenceImage
    #   config         : EditImageConfig

    pil_images = []

    try:
        if ref_pil is None or reference_mode == "none":
            # ── テキストのみ生成 ────────────────────────────────────
            config = genai_types.GenerateImagesConfig(
                number_of_images=num_images,
                aspect_ratio=aspect_ratio,
                negative_prompt=negative_prompt if negative_prompt.strip() else None,
                safety_filter_level=safety_filter,
                person_generation="ALLOW_ADULT",
            )
            response = client.models.generate_images(
                model=model,
                prompt=prompt,
                config=config,
            )
            for img_resp in response.generated_images:
                pil_images.append(Image.open(io.BytesIO(img_resp.image.image_bytes)))

        else:
            # ── 参考画像あり ────────────────────────────────────────
            # reference_mode: "style" → StyleReferenceImage
            #                 "subject" → SubjectReferenceImage
            if reference_mode == "style":
                ref_config = genai_types.StyleReferenceConfig(
                    style_description=prompt,
                )
                ref_obj = genai_types.StyleReferenceImage(
                    reference_image=genai_types.Image(
                        image_bytes=_pil_to_bytes(ref_pil)
                    ),
                    config=ref_config,
                )
            else:  # "subject"
                ref_obj = genai_types.SubjectReferenceImage(
                    reference_image=genai_types.Image(
                        image_bytes=_pil_to_bytes(ref_pil)
                    ),
                    config=genai_types.SubjectReferenceConfig(
                        subject_description=prompt,
                    ),
                )

            edit_config = genai_types.EditImageConfig(
                number_of_images=num_images,
                safety_filter_level=safety_filter,
                person_generation="ALLOW_ADULT",
            )
            response = client.models.edit_image(
                model=model,
                prompt=prompt,
                reference_images=[ref_obj],
                config=edit_config,
            )
            for img_resp in response.generated_images:
                pil_images.append(Image.open(io.BytesIO(img_resp.image.image_bytes)))

    except Exception as e:
        raise RuntimeError(f"[NanoBanana] Gemini API エラー: {e}") from e

    if not pil_images:
        raise RuntimeError("[NanoBanana] 画像が生成されませんでした（セーフティフィルターで除外された可能性）")

    # ── 5. ComfyUI テンソルに変換してバッチ化 ───────────────────────
    # 各画像を (1,H,W,C) にして torch.cat で (N,H,W,C) に結合
    tensors = [_pil_to_comfy(img) for img in pil_images]
    output = torch.cat(tensors, dim=0)

    print(f"[NanoBanana] 生成完了 {len(pil_images)} 枚 | shape={output.shape}")
    return (output,)
```

#### ヘルパー（`generate()` 内で使用）

```python
def _pil_to_bytes(pil_img: Image.Image, fmt: str = "PNG") -> bytes:
    """PIL Image → bytes"""
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return buf.getvalue()
```

---

## NODE_CLASS_MAPPINGS（`gemini_image_gen.py` 末尾）

```python
NODE_CLASS_MAPPINGS = {
    "GeminiImageGenerator": GeminiImageGenerator,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiImageGenerator": "RogoAI Gemini Image Generator",
}
```

---

## ComfyUI パターン（既存コードとの整合性）

このプロジェクトの他のノードは以下のパターンに従っている：

| 項目 | パターン |
|------|---------|
| クラス API | Classic `NODE_CLASS_MAPPINGS`（`IO.ComfyNode` は SaveAudio のみ） |
| カテゴリ | `"RogoAI"` |
| 画像変換 | `(B,H,W,C)` float32 0-1 の torch.Tensor |
| エラー | `raise RuntimeError(f"[NodeName] メッセージ")` |
| ログ | `print(f"[NodeName] メッセージ")` |
| パッケージ登録 | `nodes/__init__.py` で集約、top-level `__init__.py` で再エクスポート |

---

## 実装チェックリスト

- [ ] `ComfyUI-RogoAI-NanoBanana/__init__.py` 作成
- [ ] `ComfyUI-RogoAI-NanoBanana/pyproject.toml` 作成
- [ ] `ComfyUI-RogoAI-NanoBanana/nodes/__init__.py` 作成
- [ ] `ComfyUI-RogoAI-NanoBanana/nodes/gemini_image_gen.py` 作成
  - [ ] インポート + GENAI_AVAILABLE フラグ
  - [ ] 定数（IMAGEN_MODELS, ASPECT_RATIOS, SAFETY_FILTERS, REFERENCE_MODES）
  - [ ] `_KEY_FILE` パス（`models/gemini_api_key.txt`）
  - [ ] `_load_saved_key()` / `_save_key()`
  - [ ] `_pil_to_comfy()` / `_comfy_to_pil()` / `_pil_to_bytes()`
  - [ ] `GeminiImageGenerator.INPUT_TYPES()` — 上記仕様通り（`password: True` 必須）
  - [ ] `GeminiImageGenerator.generate()` — テキストのみ分岐 / 参考画像分岐
  - [ ] `NODE_CLASS_MAPPINGS` / `NODE_DISPLAY_NAME_MAPPINGS`

---

## API 仕様の注意点（google-genai SDK v1.x）

### Gemini Flash 経路（無料枠）

```python
# generate_content() + response_modalities=["IMAGE"]
config = genai_types.GenerateContentConfig(response_modalities=["IMAGE"])

# テキストのみ
response = client.models.generate_content(
    model="gemini-2.0-flash-exp-image-generation",
    contents=prompt,
    config=config,
)
# マルチモーダル（参考画像付き）
contents = [genai_types.Content(parts=[
    genai_types.Part(text=prompt),
    genai_types.Part(inline_data=genai_types.Blob(mime_type="image/png", data=img_bytes)),
])]
response = client.models.generate_content(model=model, contents=contents, config=config)

# 画像抽出
for candidate in response.candidates:
    for part in candidate.content.parts:
        if part.inline_data and part.inline_data.mime_type.startswith("image/"):
            img = Image.open(io.BytesIO(part.inline_data.data))
```

**注意**: Gemini Flash は1リクエスト1枚。num_images 分だけループして呼び出す。
aspect_ratio / negative_prompt / safety_filter は無効（Imagen 3 のみ）。

### Imagen 3 経路（有料）

```python
# クライアント初期化
client = genai.Client(api_key="YOUR_KEY")

# テキスト→画像（Imagen 3）
response = client.models.generate_images(
    model="imagen-3.0-generate-002",
    prompt="...",
    config=genai_types.GenerateImagesConfig(
        number_of_images=1,      # 1-4
        aspect_ratio="1:1",      # "1:1","3:4","4:3","16:9","9:16"
        negative_prompt="...",
        safety_filter_level="BLOCK_MOST",
        person_generation="ALLOW_ADULT",
    ),
)

# レスポンスから画像取得
for generated in response.generated_images:
    image_bytes = generated.image.image_bytes  # bytes
    pil_img = Image.open(io.BytesIO(image_bytes))

# 参考画像あり編集
response = client.models.edit_image(
    model="imagen-3.0-generate-002",
    prompt="...",
    reference_images=[
        genai_types.StyleReferenceImage(
            reference_image=genai_types.Image(image_bytes=...),
            config=genai_types.StyleReferenceConfig(style_description="..."),
        )
    ],
    config=genai_types.EditImageConfig(
        number_of_images=1,
        safety_filter_level="BLOCK_MOST",
    ),
)
```

---

## ワークフロー JSON への追加（任意）

ノード登録後、workflow json にノードを追加する場合のテンプレート：

```json
{
  "id": 99,
  "type": "GeminiImageGenerator",
  "pos": [100, 100],
  "size": [400, 400],
  "flags": {},
  "order": 1,
  "mode": 0,
  "inputs": [
    {"name": "reference_image", "type": "IMAGE", "shape": 7, "link": null}
  ],
  "outputs": [
    {"name": "images", "type": "IMAGE", "links": []}
  ],
  "properties": {"Node name for S&R": "GeminiImageGenerator"},
  "widgets_values": [
    "",           // prompt
    "imagen-3.0-generate-002",  // model
    "1:1",        // aspect_ratio
    1,            // num_images
    "",           // api_key (空 or 保存済みキー)
    false,        // save_key
    "",           // negative_prompt
    "none",       // reference_mode
    "BLOCK_MOST", // safety_filter
    -1            // seed
  ]
}
```

---

## Google AI Studio — API key 取得先

`https://aistudio.google.com/app/apikey`

無料枠: Imagen 3 は有料（従量課金）。Gemini 2.0 Flash は無料枠あり。

---

*このファイルは Claude Code (claude-sonnet-4-6) が 2026-03-05 に生成した引き継ぎ仕様書です。*

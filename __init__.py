"""
ComfyUI-RogoAI-NanoBanana
==========================
Google Gemini / Imagen 3 API を使った画像生成カスタムノード。

Nodes:
  - Gemini Image Generator : テキスト/参考画像 → 画像生成（Imagen 3 / Gemini 2.0）

Usage:
  Google AI Studio (https://aistudio.google.com/app/apikey) で API key を取得し、
  ノードの api_key 欄に入力してください。
  save_key=True にすると models/gemini_api_key.txt に保存されます。

Dependencies (pip install):
  google-genai>=1.0.0
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

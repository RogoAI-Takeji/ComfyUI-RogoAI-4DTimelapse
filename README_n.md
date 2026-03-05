# RogoAI Gemini Image Generator — 使い方ガイド

ComfyUI で Google Gemini / Imagen 3 API を使って画像を生成するカスタムノードです。

---

## 1. インストール

### 1-1. このフォルダの配置を確認

```
ComfyUI_for_LTX2/custom_nodes/ComfyUI-RogoAI-NanoBanana/
```
にあれば OK です。

### 1-2. 必要な Python パッケージをインストール

ComfyUI の Python 環境で以下を実行してください。

**Windows（ComfyUI フォルダから）:**
```bat
python_embeded\python.exe -m pip install google-genai>=1.0.0
```

**仮想環境を使っている場合:**
```bash
pip install google-genai>=1.0.0
```

### 1-3. ComfyUI を再起動

再起動後、ノードメニューの **RogoAI** カテゴリに
**`RogoAI Gemini Image Generator`** が現れます。

---

## 2. API key の取得

1. ブラウザで [Google AI Studio](https://aistudio.google.com/app/apikey) を開く
2. **「APIキーを作成」** をクリック
3. 表示されたキー（`AIzaSy...` で始まる文字列）をコピーする

> **無料枠について**
> `gemini-2.0-flash-exp-image-generation` は無料で使えます。
> 上限は1日あたり数十〜数百回程度（太平洋時間の深夜にリセット）。
> 個人用途では十分な量です。

---

## 3. ノードの使い方

### 基本的な接続例

```
[LoadImage] ──────────────────────────────────────────┐
                                                       ↓
[テキスト] → [GeminiImageGenerator] → [IMAGE] → [PreviewImage]
               ↑
          api_key 入力
```

### 各入力パラメータの説明

| パラメータ | 説明 |
|-----------|------|
| **prompt** | 生成したい画像の説明。英語推奨。例: `"A serene Japanese garden at sunset, cherry blossoms falling"` |
| **model** | 使用するモデル（下表参照） |
| **aspect_ratio** | 画像のアスペクト比。`1:1` / `3:4` / `4:3` / `16:9` / `9:16` |
| **num_images** | 生成枚数（1〜4枚） |
| **api_key** | Google AI API key（パスワード表示） |
| **save_key** | `保存する` にすると次回から自動入力 |
| **negative_prompt** | 避けたい要素。例: `"blurry, low quality, text"` ※Imagen 3 のみ有効 |
| **reference_image** | 参考画像（省略可） |
| **reference_mode** | 参考画像の使い方（下表参照） |
| **safety_filter** | セーフティレベル ※Imagen 3 のみ有効 |
| **person_generation** | 人物生成の許可レベル ※Imagen 3 のみ有効 |

### モデルの選択

| モデル | 料金 | 品質 | 参考画像 | 備考 |
|--------|------|------|----------|------|
| `gemini-2.0-flash-exp-image-generation` | **無料** | 普通〜良好 | ◯（マルチモーダル） | **デフォルト** |
| `imagen-3.0-generate-002` | 有料 | 高品質 | ◯（style/subject） | 最高品質 |
| `imagen-3.0-fast-generate-001` | 有料（低コスト） | 良好 | ◯（style/subject） | 高速版 |

### reference_mode の選択

| モード | 効果 | 対応モデル |
|--------|------|------------|
| `none` | テキストのみで生成 | 全モデル |
| `style` | 参考画像のスタイル・雰囲気を転写 | Imagen 3 のみ |
| `subject` | 参考画像の被写体・人物を維持 | Imagen 3 のみ |

> **Gemini Flash + 参考画像:**
> `style` / `subject` モードを選んでも Imagen 3 専用機能は使えませんが、
> 参考画像をマルチモーダル入力として自動付加します（AIが画像を参照して生成）。

---

## 4. API key の自動保存・自動読み込み

1. `api_key` 欄に API key を貼り付ける
2. `save_key` を **「保存する」** に変更してから実行
3. `ComfyUI_for_LTX2/models/gemini_api_key.txt` に保存されます
4. 次回以降、`api_key` 欄は自動的に読み込まれます（手入力不要）

> **セキュリティ注意:**
> `models/gemini_api_key.txt` はプレーンテキストです。
> 第三者にアクセスさせないよう管理してください。

---

## 5. よくある問題と対処

### 画像が生成されない

- **セーフティフィルターに引っかかっている**: プロンプトを変更するか、`safety_filter` を `BLOCK_SOME` / `BLOCK_FEW` に下げる
- **無料枠の上限**: 翌日の太平洋時間深夜まで待つ（または有料モデルに切り替える）
- **プロンプトが日本語のみ**: 英語プロンプトに変更してみる

### `google-genai が未インストールです` エラー

```bat
python_embeded\python.exe -m pip install google-genai>=1.0.0
```
を実行して ComfyUI を再起動。

### `API key が未入力です` エラー

- `api_key` 欄に Google AI Studio のキーを入力する
- または `save_key=保存する` で先に保存しておく

### Imagen 3 で `API エラー` が出る

- Imagen 3 は有料モデルです。Google Cloud の請求設定が必要です
- 無料で使いたい場合は `gemini-2.0-flash-exp-image-generation` を選択

---

## 6. 使用例

### 例1: シンプルなテキスト→画像

```
prompt: "A futuristic Tokyo cityscape at night, neon lights, rain reflections,
         cinematic photography, 8K"
model: gemini-2.0-flash-exp-image-generation
aspect_ratio: 16:9
num_images: 1
```

### 例2: 参考画像のスタイルを使って生成（Imagen 3）

```
reference_image: [好きな絵のスタイル画像を接続]
reference_mode: style
prompt: "A young woman walking in a park, spring morning"
model: imagen-3.0-generate-002
aspect_ratio: 3:4
num_images: 4
```

### 例3: ラップ動画のサムネイル画像を生成（RapSong との連携）

```
RapSongConfig → RapLyricsGenerator → [lyrics テキストを参照]
                                       ↓
prompt: "Album cover art for a Japanese rap song, dark urban aesthetic,
         Tokyo streets at night"
model: gemini-2.0-flash-exp-image-generation
aspect_ratio: 1:1
num_images: 2
```

---

## 7. 出力

- **images**: `IMAGE` 型テンソル（ComfyUI 標準形式）
- 複数枚生成した場合はバッチ（N枚）として出力されます
- `PreviewImage`、`SaveImage`、KJNodes の各ノードと直接接続できます

---

*RogoAI NanaBanana v1.0.0 — powered by Google Gemini API*

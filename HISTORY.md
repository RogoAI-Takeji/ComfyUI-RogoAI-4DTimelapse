# ComfyUI-RogoAI-NanoBanana 開発履歴

**記録担当**: Claude Sonnet 4.6 (claude-sonnet-4-6)
**最終更新**: 2026-03-10

---

## フェーズ1: Gemini画像生成ノード (2026-03-05)

### 概要
Google Gemini / Imagen 3 API を ComfyUI から使えるようにするカスタムノードを実装。

### 実装内容
- `nodes/gemini_image_gen.py`: `GeminiImageGenerator` ノード
  - `gemini-2.0-flash-exp-image-generation` (無料枠)
  - `imagen-3.0-generate-002` / `imagen-3.0-fast-generate-001` (有料)
  - テキスト→画像 / スタイル転写 / 被写体参照
  - API key の暗号化表示・自動保存 (`models/gemini_api_key.txt`)
- `pyproject.toml`: パッケージメタデータ
- `SPEC.md`: 実装仕様書

### 関連ドキュメント
- `SPEC.md` (旧仕様書)
- `README_n.md`

---

## フェーズ2: 4Dグリッドシステム基盤 (2026-03-07〜08)

### 概要
GLB（3Dモデルファイル）を時間軸・空間軸のグリッドでレンダリングし、
光学フロー補間で4Dタイムラプス動画を生成するシステムの実装。

### 実装内容

#### 4DGrid系ノード (`nodes/grid4d_nodes.py`)
- `Grid4DRenderKeyframes`: GLBフォルダ → キーフレームグリッド
  - pyrender で各角度のレンダリング画像を生成
  - `stage_XX/angle_YYY.png` 形式で保存
  - `grid_meta.json` に設定を記録
- `Grid4DTraverse`: キーフレームグリッド → MP4
  - 18種類の走査パス（diagonal, spiral, bee視点, cat視点等）
  - OpenCV 光学フロー補間
  - ffmpeg で MP4 生成
- `Grid4DComposite`: 4D走査 + 背景合成
- `Grid4DInfo`: グリッド情報確認ユーティリティ

#### NB4D系ノード (`nodes/volumetric_timelapse_nodes.py`)
- `NB4D TimePlanner`: 時間軸プロンプト生成
- `NB4D CameraPath`: カメラパス計算
- `NB4D PromptComposer`: プロンプト合成
- `NB4D ConsistencyEngine`: 一貫性維持
- `NB4D VideoAssembler`: 動画生成

#### パララックス系 (`nodes/parallax_renderer.py`)
- `NB4D DepthEstimator`: 深度推定
- `NB4D ParallaxRenderer`: パララックスワープ
- `NB4D StageAnchorManager`: アンカー管理

### 課題・発見事項
- ComfyUI の `prompt_worker` は**単一スレッド**であるため、
  ノード内からAPIで内部ジョブをキューするとデッドロックが発生する
  → V2 で解決

---

## フェーズ3: LTX-2 AI補間 V1→V2 (2026-03-08〜09)

### 概要
GLBグリッドの各ステージ間を LTX-Video 2 でAI補間して高品質動画を生成。

### V1 (API呼び出し方式) — ltxv_interpolator.py

**実装**: ComfyUI REST API (`/prompt`, `/history/{id}`) を経由してジョブ投入
**問題**: デッドロック → V2 で廃止

### V2 (直接Python呼び出し方式) — ltxv_interpolator_v2.py

**実装**: `nodes.NODE_CLASS_MAPPINGS` から直接クラスをインスタンス化して呼び出す

```
モデル読み込み (ループ外で1回):
  UnetLoaderGGUF / DualCLIPLoader / VAELoader

テキストエンコード (1回):
  CLIPTextEncode × 2 → LTXVConditioning

サンプラー初期化 (1回):
  KSamplerSelect(euler) / LTXVScheduler

ステージループ:
  LTXVImgToVideo → LTXVPreprocess × 2 → LTXVAddGuide × 2
  → CFGGuider → RandomNoise → SamplerCustomAdvanced → VAEDecode
```

**解決事項**:
- デッドロック完全解消
- angle_start / angle_end でスイープ角度を指定可能
- use_end_frame / end_frame_strength で終端ガイドを制御

**LTXVAddGuide API 確認**:
- `LTXVPreprocess.execute(image, img_compression=35)` → IMAGE (NOT LATENT)
- `LTXVAddGuide.execute(positive, negative, vae, latent, image, frame_idx, strength)`
  - `latent`: LTXVImgToVideo から
  - `image`: LTXVPreprocess から

---

## フェーズ4: 共通パスライブラリ + 仰角軸追加 (2026-03-09)

### 概要
step1/step2/step3 で共通利用できるパスライブラリを作成。
水平角スイープをV3に追加し、仰角軸 (v) を格子に追加。

### 実装内容

#### `nodes/_nb4d_paths.py` (新規)
step1/step2/step3 すべてから `from ._nb4d_paths import` で使用する共通ライブラリ。

```python
SWEEP_PATHS = {
    "pendulum", "orbit_right_half", "orbit_left_half",
    "orbit_right_full", "orbit_left_full"
}
NAVIGATOR_PRESETS = {
    "orbit_flat", "spiral_ascend", "spiral_descend",
    "top_survey", "pendulum_elevation", "time_loop", "ground_to_sky"
}
```

ヘルパー関数:
- `theta_for_stage()`: V3 用
- `sweep_lambda_for_grid()`: step1/step2 用
- `preset_to_waypoints()`: PathNavigator 用
- `parse_path_text()`: カスタムテキスト解析
- `check_adjacency()`: 隣接制約チェック

#### `nodes/ltxv_interpolator_v3.py` (新規)
- V2 の `angle_start/end` を廃止
- `sweep_path` COMBO を追加（_nb4d_paths.py の共通パス）
- `fixed_angle` パラメータ（sweep_path="none" 時の固定角度）

#### `Grid4DRenderKeyframes` に `grid_elev` 追加
- `grid_elev=7` で仰角7段階 (-30°〜+60°) に対応
- 出力: `stage_XX/elev_YY/angle_ZZZ.png` 形式

#### `Grid4DTraverse` に sweep パス追加
- `sweep_pendulum` / `sweep_orbit_*` 5種を追加（計23種）

#### `workflow_4d_youtube_02.json` (新規)
- step1: `grid_elev=7, elev_start=-30, elev_end=60`
- step3: `NB4D_LTXVPathNavigator`

### バグ修正
- **絶対importエラー**: `from nodes._nb4d_paths import` → `from ._nb4d_paths import`
  - `grid4d_nodes.py`, `ltxv_interpolator_v3.py`, `ltxv_path_navigator.py` の3ファイル

---

## フェーズ5: PathNavigator実装 (2026-03-10)

### 概要
(t, h, v) の3次元格子を自由に飛び回る「時空間パスナビゲーター」を実装。

### 実装内容

#### `nodes/ltxv_path_navigator.py` (新規)
- `NB4D_LTXVPathNavigator` ノード
- ウェイポイントリスト形式でパスを定義
- プリセット8種 + カスタムテキスト入力
- 各ウェイポイント間でLTX-2が1クリップを生成
- `use_end_frame` + `end_frame_strength` を追加（V3と同等）

#### 概念的ブレークスルー
ユーザーとの議論で確認した事項:
- **時間方向の制約は不要**: LTX-2は時間が逆行する方向も補間可能
- **(t,h,v) 隣接ベクトル移動**: 各軸を±1ずつ移動する「お隣移動」で全格子点に到達可能
- **終端フレームガイドが重要**: end_frame（次ウェイポイント画像）を注入することで
  補間精度が大幅に向上

### 追加ドキュメント
- `HANDOVER_2026-03-10.md`: YouTube + GitHub 引き継ぎ書
- `SPEC_4D_LATTICE.md`: 4D格子システム最新仕様書
- `HISTORY.md`: このファイル

---

## 未解決・今後のタスク

### YouTube動画作成
- [ ] GLBファイル群の準備（Hunyuan3D環境で生成）
- [ ] 比較動画の生成（複数パス）
- [ ] PathNavigator動画の生成
- [ ] ffmpegで最終編集・結合
- [ ] サムネイル作成
- [ ] YouTube アップロード

### GitHub公開
- [ ] `README.md` 英語版の作成
- [ ] `pyproject.toml` の更新（4DGrid系依存関係追加）
- [ ] `.gitignore` の作成
- [ ] `LICENSE` ファイルの追加
- [ ] git init + initial commit + push

### 機能拡張（オプション）
- [ ] PathNavigator: パスアニメーションのプレビュー機能
- [ ] Grid4DRenderKeyframes: 並列レンダリング（複数ステージを同時処理）
- [ ] 背景合成 (Grid4DComposite) の PathNavigator対応
- [ ] 日本語チュートリアル動画の字幕ファイル作成

---

## 主要なgitコミット

```
de2dca66  feat(rogoai): add line_density + tight duration calculation
4acd956f  Rename RogoAI custom nodes to unify naming convention
35007de2  Refactor: Move RapSong specific nodes to ComfyUI-RogoAI-RapSong
827dc41e  Restore: Fix node parameter order
15b20351  Refactor: Remove duplicate widgets from RapLyricsGenerator
```

*(NanoBanana固有のコミットは上記とは別ブランチで管理)*

---

*ComfyUI-RogoAI-NanoBanana 開発履歴 — Claude Sonnet 4.6 記録 — 2026-03-10*

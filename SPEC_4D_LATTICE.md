# ComfyUI-RogoAI-NanoBanana — 4D格子ナビゲーション システム仕様書

**バージョン**: 2.0.0
**作成日**: 2026-03-10
**前バージョン**: SPEC.md (Gemini画像生成ノード仕様、旧)

---

## 1. システム概念

### 4次元時空間格子

```
次元        軸名   記号   範囲               説明
──────────────────────────────────────────────────────────────
時間        time    t     0 〜 n_stages-1    GLBキーフレームの番号
水平角      theta   h     0 〜 grid_theta-1  カメラ水平角 (0=正面, 右回り正)
仰角        elev    v     0 〜 grid_elev-1   カメラ仰角
```

### 仰角 (v) の7段階定義（デフォルト）

```
v=0: -30° (ローアングル / アリの視点)
v=1: -15°
v=2:   0° (水平 / 目線高さ) ← V_HORIZONTAL = 2/6 ≈ 0.333
v=3: +15°
v=4: +30°
v=5: +45°
v=6: +60° (ハイアングル / 鳥の視点)
```

### 隣接制約（推奨）

LTX-2 の補間品質を保つため、各ステップで:
```
|Δt| ≤ 1, |Δh| ≤ 1, |Δv| ≤ 1
```
違反しても動作するが、差が大きいほど補間品質が低下する。

---

## 2. ノード一覧（4DGrid系）

### カテゴリ: `NanoBanana/4DGrid`

| ノードクラス名 | 表示名 | ファイル | 役割 |
|-------------|--------|---------|------|
| `Grid4DRenderKeyframes` | Grid4D Render Keyframes | grid4d_nodes.py | STEP1: GLB→グリッド |
| `Grid4DTraverse` | Grid4D Traverse | grid4d_nodes.py | STEP2: 光学フロー走査 |
| `Grid4DComposite` | Grid4D Composite | grid4d_nodes.py | STEP2: 背景合成 |
| `Grid4DInfo` | Grid4D Info | grid4d_nodes.py | 情報確認 |
| `NB4D_LTXVStageInterpolatorV2` | NB4D LTXV Stage Interpolator V2 | ltxv_interpolator_v2.py | STEP3: LTX-2補間 |
| `NB4D_LTXVStageInterpolatorV3` | NB4D LTXV Stage Interpolator V3 (共通スイープパス) | ltxv_interpolator_v3.py | STEP3: LTX-2 + sweep |
| `NB4D_LTXVPathNavigator` | NB4D LTXV Path Navigator (時空間ナビゲーター) | ltxv_path_navigator.py | PathNavigator |

---

## 3. Grid4DRenderKeyframes — STEP1 仕様

### INPUT_TYPES

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|---------|------|
| `glb_dir` | STRING | `D:\NB4D_test\tomato\glb` | GLBファイルのディレクトリ |
| `keyframes_base_dir` | STRING | `D:\NB4D_test\tomato\grid_keyframes` | 出力親ディレクトリ |
| `run_name` | STRING | "" | 試行名（空=タイムスタンプ自動） |
| `grid_theta` | INT | 120 | 水平角分割数 (8〜360, step=8) |
| `grid_elev` | INT | 1 | 仰角段階数 (1=旧互換, 7=フル3D) |
| `elev_start` | FLOAT | -30.0 | grid_elev>1 の最小仰角（度）|
| `elev_end` | FLOAT | 60.0 | grid_elev>1 の最大仰角（度）|
| `cam_dist` | FLOAT | 2.5 | カメラ距離 |
| `render_w` / `render_h` | INT | 1280 / 720 | レンダリング解像度 |
| `size_start` | FLOAT | 0.45 | 最初stageの相対サイズ |
| `size_plateau_stage` | INT | 3 | このstage番号で最大サイズ1.0に到達 |
| `bg_gray` | FLOAT | 0.12 | 背景グレー値 0.0〜1.0 |

### RETURN_TYPES
```python
RETURN_TYPES  = ("STRING", "STRING")
RETURN_NAMES  = ("keyframes_dir", "status")
```

### 出力ディレクトリ構成

**grid_elev=1 (旧形式)**:
```
run_YYYYMMDD_HHMMSS/
  grid_meta.json
  stage_00/
    angle_000.png
    angle_001.png
    ...
  stage_01/
    ...
```

**grid_elev=7 (3D格子形式)**:
```
run_YYYYMMDD_HHMMSS/
  grid_meta.json
  stage_00/
    elev_00/
      angle_000.png  ← v=0, h=0
      angle_001.png  ← v=0, h=1
      ...
    elev_01/
      ...
    elev_06/
      ...
  stage_01/
    ...
```

### grid_meta.json フォーマット

```json
{
  "n_stages": 9,
  "grid_theta": 72,
  "grid_elev": 7,
  "elev_angles": [-30.0, -15.0, 0.0, 15.0, 30.0, 45.0, 60.0],
  "render_params": {
    "cam_dist": 2.5,
    "render_w": 1280,
    "render_h": 720,
    "bg_gray": 0.12,
    "size_start": 0.45,
    "size_plateau_stage": 3
  }
}
```

---

## 4. Grid4DTraverse — STEP2 仕様

### INPUT_TYPES

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|---------|------|
| `keyframes_dir` | STRING | — | Grid4DRenderKeyframes の出力 |
| `path_name` | COMBO | "diagonal" | 走査パス（全23種、下記参照） |
| `frames` | INT | 480 | 総フレーム数 |
| `fps` | INT | 24 | フレームレート |
| `output_dir` | STRING | — | 出力ディレクトリ |
| `video_codec` | COMBO | "libx264" | "libx264" / "h264_nvenc" |

### 走査パス全一覧（path_name）

**基本パス (8種)**:
```
diagonal        回りながら成長（斜め移動）★基本
spiral_2x       2周しながら成長
orbit_green     最初の状態を1周
orbit_ripe      最後の状態を1周
ripen_front     正面固定で変化
reverse_diag    過去→現在に戻りながら回る
time_wave       時間を往復しながらorbit
zoom_in_time    前半斜め・後半ゆっくり近づく
```

**ミツバチ視点 (3種)**:
```
bee_hover       不規則ホバリング + 成長観察
bee_spiral_in   遠くから螺旋降下
bee_inspect     接近→観察→離脱 (4フェーズ繰り返し)
```

**猫視点 (7種)**:
```
cat_passthrough   正面→背面 通り過ぎ
cat_reverse_pass  背面→正面 逆方向
cat_circle_age    1周(360°) しながら成長
cat_double_arc    正面→背面→正面 往復
cat_young_linger  序盤70%じっくり + 後半30%急速老化
cat_side_age      横顔(90°)固定で成長
cat_waltz         正面付近で左右にゆっくり揺れ
```

**共通スイープパス / _nb4d_paths.py から (5種)**:
```
sweep_pendulum         右45°→正面→左45°（振り子）
sweep_orbit_right_half 正面→右横→後ろ（右回り半周）
sweep_orbit_left_half  正面→左横→後ろ（左回り半周）
sweep_orbit_right_full 正面→後ろ→正面（右回り一周）
sweep_orbit_left_full  正面→後ろ→正面（左回り一周）
```

---

## 5. NB4D_LTXVStageInterpolatorV3 — STEP3 仕様

### 概要

grid_meta.json の全ステージを順に LTX-2 で補間し、
RGBA PNG 連番として保存する。

### INPUT_TYPES

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|---------|------|
| `keyframes_dir` | STRING | — | Grid4DRenderKeyframes の出力 |
| `positive_prompt` | STRING | "natural movement, smooth animation..." | |
| `negative_prompt` | STRING | "blurry, distorted..." | |
| `clip_frames` | INT | 49 | 1ステージのフレーム数 (8n+1: 25/49/97) |
| `sweep_path` | COMBO | "none" | 角度スイープパス (下記6種) |
| `fixed_angle` | INT | 0 | sweep_path="none" 時の固定角度インデックス |
| `cfg` | FLOAT | 3.5 | CFG スケール |
| `steps` | INT | 30 | サンプリングステップ数 |
| `seed` | INT | 42 | ランダムシード |
| `fps` | INT | 24 | 出力FPS |
| `zoom_start` | FLOAT | 1.0 | ズーム開始倍率（1.0=なし）|
| `zoom_frames` | INT | 72 | ズームにかけるフレーム数 |
| `alpha_thresh` | INT | 18 | アルファ抽出閾値 |
| `use_end_frame` | BOOLEAN | True | 終端フレームガイドを使うか |
| `end_frame_strength` | FLOAT | 0.9 | 終端フレームガイド強度 |
| `output_dir` | STRING | — | 出力ディレクトリ |
| `comfyui_url` | STRING | "http://127.0.0.1:8188" | 未使用（直接Python呼び出し）|
| `model_gguf` | STRING | "ltx2\ltx-2-19b-distilled_Q6_K.gguf" | |
| `model_vae` | STRING | "ltx2\LTX2_video_vae_bf16.safetensors" | |
| `gemma_fp4` | STRING | "ltx2\gemma_3_12B_it_fp4_mixed.safetensors" | |
| `embed_connector` | STRING | "ltx2\ltx-2-19b-embeddings_connector_distill_bf16.safetensors" | |

### sweep_path 選択肢（V3）

```
none              固定角度 (fixed_angle パラメータで指定)
pendulum          右45°→正面→左45° (振り子)
orbit_right_half  正面→右横→後ろ (右回り半周)
orbit_left_half   正面→左横→後ろ (左回り半周)
orbit_right_full  正面→後ろ→正面 (右回り一周)
orbit_left_full   正面→後ろ→正面 (左回り一周)
```

### RETURN_TYPES
```python
RETURN_TYPES  = ("STRING", "INT", "STRING")
RETURN_NAMES  = ("output_dir", "frame_count", "status")
```

### 内部処理フロー

```
モデル読み込み (1回):
  UnetLoaderGGUF → MODEL
  DualCLIPLoader → CLIP (type="ltxv")
  VAELoader → VAE

テキストエンコード (1回):
  CLIPTextEncode × 2 → pos/neg conditionings
  LTXVConditioning → ltxv_pos, ltxv_neg

サンプラー初期化 (1回):
  KSamplerSelect (euler)
  LTXVScheduler (steps, max_shift=2.05, base_shift=0.95)

ステージループ:
  for stage_idx in range(n_stages):
    角度決定: theta_for_stage(sweep_path, stage_idx, n_stages, n_angles, fixed_angle)
    画像読み込み: stage_XX/angle_YYY.png
    LTXVImgToVideo → latent
    LTXVPreprocess → preprocessed_start
    LTXVAddGuide (frame_idx=0, strength=1.0) → ガイド注入
    if use_end_frame:
      LTXVPreprocess → preprocessed_end
      LTXVAddGuide (frame_idx=-1, strength=end_frame_strength)
    CFGGuider → guider
    RandomNoise (seed + stage_idx)
    SamplerCustomAdvanced → output_latent
    VAEDecode → image_tensor
    フレーム結合 (先頭以外は overlap=1 除く)

RGBA PNG保存:
  アルファ抽出 (グレー背景除去)
  ズーム適用
  frame_00000.png 〜
```

---

## 6. NB4D_LTXVPathNavigator — 最新ノード仕様

### 概要

(t, h, v) のウェイポイントリストを定義し、
隣接ウェイポイント間を LTX-2 で1クリップずつ補間して
時空間を自由に飛び回る動画を生成する。

### INPUT_TYPES

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|---------|------|
| `keyframes_dir` | STRING | — | Grid4DRenderKeyframes の出力 (grid_elev>1 必須) |
| `path_preset` | COMBO | "orbit_flat" | ウェイポイントプリセット (8種) |
| `path_custom` | STRING multiline | "0, 0, 2\n1, 3, 2\n..." | custom_text 時のウェイポイント |
| `positive_prompt` | STRING | — | |
| `negative_prompt` | STRING | — | |
| `clip_frames` | INT | 49 | 1セグメントのフレーム数 |
| `cfg` | FLOAT | 3.5 | |
| `steps` | INT | 30 | |
| `seed` | INT | 42 | |
| `fps` | INT | 24 | |
| `zoom_start` | FLOAT | 1.0 | |
| `zoom_frames` | INT | 0 | |
| `alpha_thresh` | INT | 18 | |
| `use_end_frame` | BOOLEAN | True | 終端ガイドを使うか |
| `end_frame_strength` | FLOAT | 0.9 | 終端ガイド強度 |
| `output_dir` | STRING | — | |
| `comfyui_url` | STRING | — | 未使用 |
| `model_gguf` | STRING | — | |
| `model_vae` | STRING | — | |
| `gemma_fp4` | STRING | — | |
| `embed_connector` | STRING | — | |

### path_preset 選択肢（8種）

```
custom_text          テキスト入力でウェイポイント自由指定
orbit_flat           右回り一周・水平 (v=2, 0°) ・時間進行
spiral_ascend        右螺旋上昇 (水平→鳥瞰)
spiral_descend       右螺旋下降 (鳥瞰→水平)
top_survey           真上→水平へ降下 (正面固定)
pendulum_elevation   仰角振り子 (h固定、v: 水平↔見下ろし↔水平)
time_loop            時間往復 (行って帰る)
ground_to_sky        地面レベル→空へ上昇
```

### custom_text フォーマット

```
# コメント行は無視
# 整数インデックス形式: t_idx, h_idx, v_idx
0, 0, 2
1, 3, 2
2, 6, 3

# 正規化float形式 (0.0〜1.0): t_norm, h_norm, v_norm
0.0, 0.0, 0.333
0.25, 0.25, 0.5
0.5, 0.5, 0.833
```

### 内部処理フロー

```
1. grid_meta.json 読み込み (n_stages, grid_theta, grid_elev, elev_angles)
2. ウェイポイント解決:
   - preset → preset_to_waypoints(preset_name, n_stages, grid_theta, grid_elev)
   - custom_text → parse_path_text(text, n_stages, grid_theta, grid_elev)
3. 隣接チェック (警告のみ、実行継続)
4. モデル・テキスト・サンプラー初期化 (V3と同じ)
5. セグメントループ:
   for i in range(len(waypoints) - 1):
     (t0,h0,v0), (t1,h1,v1) = waypoints[i], waypoints[i+1]
     start_img = load_keyframe(kf_dir, t0, h0, v0, grid_elev)
     end_img   = load_keyframe(kf_dir, t1, h1, v1, grid_elev)
     [LTXVImgToVideo + LTXVAddGuide(先頭) + (use_end_frame → LTXVAddGuide(終端))]
     → サンプリング → VAEDecode
     フレーム結合 (overlap=1 除く)
6. RGBA PNG 保存
```

---

## 7. 共通パスライブラリ: `_nb4d_paths.py`

### 定数
```python
V_HORIZONTAL = 2 / 6  # 0°仰角 (v=2 in 7-level grid) の正規化値
```

### SWEEP_PATHS (dict)
step1/step2/step3 共通の水平スイープパス関数群。
各関数は `p: float (0→1)` を受け取り `(t_norm, h_norm, v_norm)` を返す。

```python
SWEEP_PATHS = {
    "pendulum":         _pendulum,           # h: 0.125→0→-0.125 (前後に揺れ)
    "orbit_right_half": _orbit_right_half,   # h: 0→0.5
    "orbit_left_half":  _orbit_left_half,    # h: 1→0.5
    "orbit_right_full": _orbit_right_full,   # h: 0→1
    "orbit_left_full":  _orbit_left_full,    # h: 1→0
}
```

### NAVIGATOR_PRESETS (dict)
PathNavigator 用ウェイポイントリスト。
各エントリは `(t_norm, h_norm, v_norm)` のリスト。

### 主要関数

```python
def theta_for_stage(path_name, stage_idx, n_stages, n_angles, fixed_angle) -> int:
    """V3 (step3) 用: stage_idx → 水平角インデックス"""

def sweep_lambda_for_grid(path_name, grid_t, grid_theta):
    """step1/step2 (Grid4DTraverse) 用: (i,N) → (t, theta) ラムダ生成"""

def preset_to_waypoints(preset_name, n_stages, grid_theta, grid_elev) -> list:
    """プリセット名 → [(t_idx, h_idx, v_idx), ...] に変換"""

def parse_path_text(text, n_stages, grid_theta, grid_elev) -> list:
    """テキスト入力 → [(t_idx, h_idx, v_idx), ...] に変換"""

def check_adjacency(waypoints) -> list:
    """隣接制約チェック。違反箇所の警告メッセージリストを返す"""
```

---

## 8. ワークフロー仕様

### workflow_4d_youtube_02.json (最新)

| ノードID | ノード種別 | 役割 |
|---------|----------|------|
| 10 | Grid4DRenderKeyframes | STEP1: GLB→グリッド (grid_elev=7) |
| 11 | Grid4DInfo | グリッド情報確認 |
| 12 | NB4D_LTXVPathNavigator | STEP3: 時空間ナビゲーター |
| 13 | ShowText | ステータス表示 |
| 14 | ShowText | ltxv_output_dir 表示 |

**接続**:
- ノード10 `keyframes_dir` → ノード11 `keyframes_dir`
- ノード10 `keyframes_dir` → ノード12 `keyframes_dir`
- ノード12 `output_dir` → ノード14

### workflow_4d_youtube.json (旧)

| ノードID | ノード種別 | 役割 |
|---------|----------|------|
| 10 | Grid4DRenderKeyframes | STEP1 (grid_elev=1) |
| 11 | Grid4DInfo | 情報確認 |
| 12 | NB4D_LTXVStageInterpolatorV2 | STEP3 V2 |

---

## 9. 依存パッケージ

```
pyrender>=0.1.45     3DレンダリングエンジンKEY
trimesh>=3.0.0       GLB読み込み
opencv-python>=4.5   光学フロー補間・動画生成
Pillow>=9.0          画像処理
numpy>=1.21          数値計算
google-genai>=1.0.0  Gemini API (GeminiImageGeneratorノード)
torch                ComfyUI経由で利用
```

---

## 10. 変更履歴

| バージョン | 日付 | 変更内容 |
|-----------|------|---------|
| 1.0.0 | 2026-03-05 | Gemini画像生成ノード実装 |
| 1.1.0 | 2026-03-07 | 4DGrid系ノード実装 (Grid4DRenderKeyframes/Traverse/Composite) |
| 1.2.0 | 2026-03-08 | LTXV補間 V1実装 (API呼び出し方式) |
| 1.3.0 | 2026-03-08 | LTXV補間 V2実装 (デッドロック修正・直接呼び出し方式) |
| 1.4.0 | 2026-03-09 | _nb4d_paths.py 共通パスライブラリ作成 |
| 1.5.0 | 2026-03-09 | LTXV補間 V3実装 (sweepパス統合) |
| 1.5.1 | 2026-03-09 | Grid4DRenderKeyframes に grid_elev 追加 |
| 2.0.0 | 2026-03-10 | PathNavigator実装・import修正・仰角7段階対応 |

---

*ComfyUI-RogoAI-NanoBanana v2.0.0 — Claude Sonnet 4.6 作成 — 2026-03-10*

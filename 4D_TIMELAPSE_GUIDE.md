# 4Dタイムラプス動画 制作ガイド
## ルリシジミ（Celastrina argiolus）9態ライフサイクル

**出力物**: `lifecycle_4D.mp4`
```
[Stage1 orbit 10s] → [遷移 2s] → [Stage2 orbit 10s] → ... → [Stage9 orbit 10s]
合計: 10s×9 + 2s×8 = 106秒
```

---

## 使用環境

| 用途 | Python環境 | パス |
|------|-----------|------|
| 3Dメッシュ生成 (ComfyUI) | Python 3.12 | `D:\Python_VENV\for_comfy_3d_wrapp_250507\ComfyUI\venv\Scripts\python.exe` |
| orbit動画・遷移動画生成 | Python 3.12 (同上) | 同上 |
| スクリプト置き場 | — | `D:\Python_VENV\for_comfy_ltx2_260115\Data\Packages\ComfyUI_for_LTX2\custom_nodes\ComfyUI-RogoAI-NanoBanana\` |
| 出力先 | — | `D:\NB4D_test\` |

---

## STEP 1: ハルシネーションのない素材収集

### 1-1. 収集方針
AIに画像生成を任せると各態間で形態的整合性が崩れる（ハルシネーション）。
**実写・標本写真を優先**して使用する。

### 1-2. 推奨素材源（実写）

| サイト | 特徴 | URL |
|--------|------|-----|
| **iNaturalist** | 世界中の観察記録、CC BY-NC、GPS付き | https://www.inaturalist.org |
| **BioLib** | ヨーロッパ中心、卵〜成虫の連続写真あり | https://www.biolib.cz |
| **日本蝶類学会** | 国内産の詳細写真 | — |
| **Wikimedia Commons** | 高解像度・著作権明確 | https://commons.wikimedia.org |
| **GBIF** | 標本・観察データ、学術利用可 | https://www.gbif.org |

検索キーワード例:
```
"Celastrina argiolus" egg
"Celastrina argiolus" larva instar
"Celastrina argiolus" pupa chrysalis
"Celastrina argiolus" adult
ルリシジミ 卵 幼虫 蛹 成虫
```

### 1-3. 9態の定義と撮影ポイント

| Stage | 態 | 撮影ポイント |
|-------|-----|------------|
| 01 | 卵 | 食草の葉上、単体、白色半球形 |
| 02 | 1齢幼虫 | 孵化直後、黄緑色、体長2mm程度 |
| 03 | 2〜3齢幼虫 | 緑色、背中の紋様が出始め |
| 04 | 4齢幼虫 | 終齢前、体が太くなる |
| 05 | 終齢幼虫 | 最大、蛹化直前の徘徊個体 |
| 06 | 前蛹 | 糸で固定、体が縮む |
| 07 | 蛹（初期） | 緑〜褐色、翅の模様が透けていない |
| 08 | 蛹（後期） | 翅の模様が透けて見える羽化直前 |
| 09 | 成虫 | 翅を広げた正面or翅を閉じた側面 |

### 1-4. 画像の選定基準（ハルシネーション防止）

- **背景が単純**（白・ベージュ・葉のみ）→ 背景除去が容易
- **正面または側面**から撮影（斜め視点は3D再構成が難しい）
- **解像度**: 最低 800×800 px 以上
- **ピントが全体に合っている**（被写界深度が深いもの）
- **各態で照明方向を統一**（できれば正面からの拡散光）

### 1-5. ディレクトリ構成

```
9stages/
  stage_01.png   ← 卵
  stage_02.png   ← 1齢幼虫
  stage_03.png   ← 2〜3齢幼虫
  stage_04.png   ← 4齢幼虫
  stage_05.png   ← 終齢幼虫
  stage_06.png   ← 前蛹
  stage_07.png   ← 蛹（初期）
  stage_08.png   ← 蛹（後期）
  stage_09.png   ← 成虫
```

---

## STEP 2: 画像の前処理

各 stage_XX.png を以下の状態にする:
- **サイズ**: 518×518 px（Hunyuan3D の入力サイズ）
- **背景**: 白または透明（PNGのアルファチャンネル推奨）
- **中央に被写体**

背景除去ツール:
- ComfyUI: `ImageRemoveBackground+` ノード（rembg）
- Web: https://remove.bg（無料・高精度）

---

## STEP 3: 3Dメッシュ生成（各stage × 9回）

### 3-1. 使用ComfyUI

```
D:\Python_VENV\for_comfy_3d_wrapp_250507\ComfyUI\
起動:
  cd D:\Python_VENV\for_comfy_3d_wrapp_250507\ComfyUI
  ./venv/scripts/activate.ps1
  python main.py
```

### 3-2. ワークフロー

ワークフローファイル:
```
G:\for_comfy_3d_wrapp_250507\user\default\workflows\hy3d_example_01.json
```
（フルテクスチャ版。custom_rasterizer 修正済み）

### 3-3. 各stageの実行手順

1. ComfyUI を起動、`hy3d_example_01.json` を読み込む
2. `LoadImage` ノードで `stage_XX.png` を選択
3. Queue Prompt で実行
4. 完了後 `output/3D/Hy3D_textured_XXXXX_.glb` が生成される
5. `stage_XX_orbit.glb` にリネームして保存

### 3-4. 生成パラメータ（推奨）

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| steps | 50 | 品質重視。30でも可 |
| guidance_scale | 5.5 | デフォルト |
| octree_resolution | 384 | メッシュ解像度 |
| max_facenum | 50000 | ポリゴン数上限 |

### 3-5. custom_rasterizer DLL修正（初回のみ）

Python 3.12 環境で custom_rasterizer の DLL が読み込めない場合は
以下のファイルが修正済みであることを確認する:

```
D:\Python_VENV\for_comfy_3d_wrapp_250507\ComfyUI\custom_nodes\
ComfyUI-Hunyuan3DWrapper\hy3dgen\texgen\differentiable_renderer\mesh_render.py
```

`raster_mode == 'cr':` のブロック先頭に以下が追加されていること:
```python
import os, torch as _torch
_torch_lib = os.path.join(os.path.dirname(_torch.__file__), 'lib')
if os.path.isdir(_torch_lib):
    os.add_dll_directory(_torch_lib)
```

---

## STEP 4: orbit動画生成（各stage × 9回）

### 4-1. スクリプト

```
D:\Python_VENV\for_comfy_ltx2_260115\Data\Packages\ComfyUI_for_LTX2\
custom_nodes\ComfyUI-RogoAI-NanoBanana\render_orbit_from_mesh.py
```

### 4-2. 実行コマンド（stage_XX ごとに繰り返す）

```powershell
$PY = "D:\Python_VENV\for_comfy_3d_wrapp_250507\ComfyUI\venv\Scripts\python.exe"
$SCRIPT = "D:\Python_VENV\for_comfy_ltx2_260115\Data\Packages\ComfyUI_for_LTX2\custom_nodes\ComfyUI-RogoAI-NanoBanana\render_orbit_from_mesh.py"

# stage_01
& $PY $SCRIPT --mesh "D:\NB4D_test\glb\stage_01.glb" --output "D:\NB4D_test\orbit_mesh\stage_01_orbit.mp4" --elev 20 --duration 10 --fps 24

# stage_02
& $PY $SCRIPT --mesh "D:\NB4D_test\glb\stage_02.glb" --output "D:\NB4D_test\orbit_mesh\stage_02_orbit.mp4" --elev 20 --duration 10 --fps 24

# ... stage_03 〜 stage_09 も同様
```

### 4-3. render_orbit_from_mesh.py の主要パラメータ

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `--elev` | 20 | 仰角（度）。大きいほど上から見下ろす |
| `--duration` | 10 | 動画の長さ（秒）= 360度 1周 |
| `--fps` | 24 | フレームレート |
| `--width` / `--height` | 1280 / 720 | 解像度 |
| `--output` | 自動生成 | 出力MP4パス |

### 4-4. 動作確認ポイント

実行ログに以下が表示されていれば正常:
```
  pyrender でレンダリングします（高品質モード）
  テクスチャ → 頂点カラー変換 完了
  f0000/240  0deg
  f0048/240  72deg
  ...
  f0240/240  360deg
完了: D:\NB4D_test\orbit_mesh\stage_01_orbit.mp4
```

---

## STEP 5: ステージ間遷移動画生成（自動）

### 5-1. スクリプト

```
D:\Python_VENV\for_comfy_ltx2_260115\Data\Packages\ComfyUI_for_LTX2\
custom_nodes\ComfyUI-RogoAI-NanoBanana\make_stage_transitions.py
```

### 5-2. スクリプト内の設定確認

ファイル冒頭の定数を環境に合わせて設定する:
```python
STAGES_DIR = Path(r"C:\Users\...\9stages")   # stage_XX.png のディレクトリ
OUTPUT_DIR  = Path(r"D:\NB4D_test\transitions")
ORBIT_DIR   = Path(r"D:\NB4D_test\orbit_mesh")  # orbit動画のディレクトリ
```

### 5-3. 実行コマンド

```powershell
$PY = "D:\Python_VENV\for_comfy_3d_wrapp_250507\ComfyUI\venv\Scripts\python.exe"
$SCRIPT = "D:\Python_VENV\for_comfy_ltx2_260115\Data\Packages\ComfyUI_for_LTX2\custom_nodes\ComfyUI-RogoAI-NanoBanana\make_stage_transitions.py"

& $PY $SCRIPT
```

### 5-4. アルゴリズム（ハルシネーションなし）

各ステージペアの SSIM（構造的類似度）を計算し、変化量で手法を自動選択:

| SSIM | 変化量 | 使用手法 |
|------|--------|---------|
| 0.55 以上 | 小（幼虫の成長など） | 双方向光学フローワープのみ |
| 0.55 未満 | 大（卵→幼虫、蛹→成虫） | 双方向フロー 60% + クロスディゾルブ 40% |

**双方向フローワープ**（FILM相当）の仕組み:
```
t=0.0  stage_A 原画
t=0.3  warp(A, 0.3×flow_AB)×0.7 + warp(B, 0.7×flow_BA)×0.3
t=0.5  warp(A, 0.5×flow_AB)×0.5 + warp(B, 0.5×flow_BA)×0.5  ← 中間点
t=0.7  warp(A, 0.7×flow_AB)×0.3 + warp(B, 0.3×flow_BA)×0.7
t=1.0  stage_B 原画
```
→ AIを一切使わないため視覚的破綻が起きない

### 5-5. 出力ファイル

```
D:\NB4D_test\transitions\transitions\
  transition_01_02.mp4   (48フレーム = 2秒)
  transition_02_03.mp4
  ...
  transition_08_09.mp4

D:\NB4D_test\transitions\
  concat_list.txt        (STEP 6 で使用)
```

---

## STEP 6: 最終動画の結合

### 6-1. concat_list.txt の確認・修正

`D:\NB4D_test\transitions\concat_list.txt` の内容が以下の順になっているか確認:

```
file 'D:\NB4D_test\orbit_mesh\stage_01_orbit.mp4'
file 'D:\NB4D_test\transitions\transitions\transition_01_02.mp4'
file 'D:\NB4D_test\orbit_mesh\stage_02_orbit.mp4'
file 'D:\NB4D_test\transitions\transitions\transition_02_03.mp4'
...
file 'D:\NB4D_test\orbit_mesh\stage_09_orbit.mp4'
```

### 6-2. ffmpeg で結合

```powershell
& "C:\Users\fareg\AppData\Local\FFmpeg\bin\ffmpeg.exe" -y `
  -f concat -safe 0 `
  -i "D:\NB4D_test\transitions\concat_list.txt" `
  -c:v libx264 -pix_fmt yuv420p -crf 18 `
  "D:\NB4D_test\transitions\lifecycle_4D.mp4"
```

### 6-3. 完成

```
D:\NB4D_test\transitions\lifecycle_4D.mp4
  合計: 10s×9 + 2s×8 = 106秒
  構成: orbit(卵) → 遷移 → orbit(1齢) → 遷移 → ... → orbit(成虫)
```

---

## トラブルシューティング

### custom_rasterizer が読み込めない
```
ImportError: DLL load failed while importing custom_rasterizer_kernel
```
→ mesh_render.py に `os.add_dll_directory(torch/lib)` が追加されているか確認（STEP 3-5）

### pyrender でテクスチャエラー
```
ctypes.ArgumentError: No array-type handler ...
```
→ render_orbit_from_mesh.py の `try_render_pyrender` 関数内に
`mesh.visual = mesh.visual.to_color()` が追加されているか確認

### カメラが動かず卵が動いて見える
→ `scene.set_pose(cam_node, cam_pose)` が `cam_node` を使っているか確認
（`list(scene.get_nodes())[0]` では最初のノード=メッシュが動く）

### 遷移でワープ歪みが激しい
- `--duration 3.0` で遷移を長くする（フレーム数を増やす）
- SSIM閾値を上げて常にディゾルブ混合にする（スクリプト内 `0.55` を `0.8` に変更）

---

## ファイル一覧

| スクリプト | 説明 |
|-----------|------|
| `render_orbit_from_mesh.py` | GLB → orbit MP4（pyrender使用） |
| `make_stage_transitions.py` | stage画像ペア → 遷移MP4（光学フロー） |
| `hy3d_example_01.json` | Hunyuan3D フルテクスチャ ワークフロー |
| `hy3d_example_02.json` | Hunyuan3D シェイプのみ ワークフロー（バックアップ） |

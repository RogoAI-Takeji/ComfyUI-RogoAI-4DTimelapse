# GitHub 初回Push手順

## 1. GitHubでリポジトリ作成

- URL: https://github.com/new
- Repository name: `ComfyUI-RogoAI-4DTimelapse`
- Description: `4D Timelapse video generation for ComfyUI — traverse Time × Angle × Elevation with LTX-Video 2`
- Public
- **Initialize: チェックを入れない（空リポジトリ）**
- License: なし（後でpushする）

## 2. ローカルリポジトリの初期化（PowerShell）

```powershell
# プロジェクトルートへ移動
cd "D:\Python_VENV\for_comfy_ltx2_260115\Data\Packages\ComfyUI_for_LTX2\custom_nodes\ComfyUI-RogoAI-NanoBanana"

# 今回追加したファイルをコピー
# README.md, pyproject.toml, .gitignore, LICENSE を上書き配置

# git初期化
git init
git add .
git commit -m "Initial release: 4D Timelapse + Gemini Image Generator v2.0.0"

# GitHubに接続してpush
git remote add origin https://github.com/RogoAI-Takeji/ComfyUI-RogoAI-4DTimelapse.git
git branch -M main
git push -u origin main
```

## 3. 確認事項

- [ ] `models/gemini_api_key.txt` が .gitignore で除外されているか
- [ ] `*.safetensors` `*.gguf` `*.glb` が除外されているか
- [ ] `__pycache__/` が除外されているか

## 4. GitHubリポジトリ設定（push後）

- About欄のDescription: `4D Timelapse video generation for ComfyUI`
- Topics（タグ）: `comfyui`, `ai`, `timelapse`, `ltx-video`, `3d`, `gemini`, `video-generation`
- Website: YouTubeチャンネルURL

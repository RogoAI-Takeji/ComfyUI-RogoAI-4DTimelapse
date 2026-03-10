# grid4d_nodes.py
#
# ComfyUI カスタムノード: 4D グリッドタイムラプス
#
# Node 1: Grid4DRenderKeyframes  - GLBディレクトリ → キーフレームグリッド生成
# Node 2: Grid4DTraverse         - キーフレームグリッド + パス選択 → MP4生成
#
# カテゴリ: NanoBanana/4DGrid

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from nodes._nb4d_paths import SWEEP_PATHS, sweep_lambda_for_grid

# ─── パス定義（grid_traverse.py と共有） ──────────────────────────────────────

GRID_T     = 120
GRID_THETA = 120
FPS        = 24

PATHS_INFO = {
    "diagonal":     "回りながら同時に熟れる（斜め移動）",
    "spiral_2x":    "2周しながら熟れる（螺旋）",
    "orbit_green":  "最初の状態の周りを1周orbit",
    "orbit_ripe":   "最後の状態の周りを1周orbit",
    "ripen_front":  "正面固定でどんどん変化",
    "reverse_diag": "過去から現在へ戻りながら回る",
    "time_wave":    "時間を往復しながらorbit（時間の波）",
    "zoom_in_time": "前半は斜め移動、後半はゆっくり近づく",
    # ─── ミツバチ視点パス ───────────────────────────────
    "bee_hover":       "ミツバチ: 不規則な高さでホバリングしながら成長観察",
    "bee_spiral_in":   "ミツバチ: 遠くから花に螺旋降下しながら成長観察",
    "bee_inspect":     "ミツバチ: 花を近くで調べる→離れる→次の成長ステージ",
    "cat_passthrough":  "猫視点①: 正面→背面(180°)に通り過ぎながら幼→老",
    "cat_reverse_pass": "猫視点②: 背面→正面(180°→0°)に通り過ぎながら幼→老（逆方向）",
    "cat_circle_age":   "猫視点③: 1周(360°)しながら幼→老（まるごと一周）",
    "cat_double_arc":   "猫視点④: 正面→背面→正面と往復しながら幼→老",
    "cat_young_linger": "猫視点⑤: 子猫を70%のフレームでじっくり、後半30%で急速に老化+1周",
    "cat_side_age":     "猫視点⑥: 横顔(90°)固定で幼→老の変遷（横顔プロファイル）",
    "cat_waltz":        "猫視点⑦: 正面付近で左右にゆっくり揺れながら幼→老（ワルツ）",
    # ─── 共通スイープパス (_nb4d_paths.py) ─────────────────────────────
    "sweep_pendulum":         "スイープ①: 右45°→正面→左45°（振り子）",
    "sweep_orbit_right_half": "スイープ②: 正面→右横→後ろ（右回り半周）",
    "sweep_orbit_left_half":  "スイープ③: 正面→左横→後ろ（左回り半周）",
    "sweep_orbit_right_full": "スイープ④: 正面→後ろ→正面（右回り一周）",
    "sweep_orbit_left_full":  "スイープ⑤: 正面→後ろ→正面（左回り一周）",
}

FFMPEG_CANDIDATES = [
    r"C:\Users\fareg\AppData\Local\FFmpeg\bin\ffmpeg.exe",
    "ffmpeg",
]


def _get_ffmpeg():
    for p in FFMPEG_CANDIDATES:
        try:
            subprocess.run([p, "-version"], capture_output=True, check=True)
            return p
        except Exception:
            pass
    raise RuntimeError("ffmpeg が見つかりません")


def _build_video_cmd(ffmpeg, fps, input_pattern, output_path, codec="libx264"):
    """ffmpeg コマンドを構築する（GPU/CPU コーデック対応）"""
    if codec == "h264_nvenc":
        quality = ["-preset", "p4", "-cq", "18"]
    else:  # libx264
        quality = ["-crf", "18"]
    return [
        ffmpeg, "-y",
        "-framerate", str(fps),
        "-i", str(input_pattern),
        "-c:v", codec,
        *quality,
        "-pix_fmt", "yuv420p",
        str(output_path),
    ]


# ─── Node 1: Grid4DRenderKeyframes ───────────────────────────────────────────

class Grid4DRenderKeyframes:
    """
    GLBファイルを指定ディレクトリから読み込み、
    全角度（grid_theta方向）でレンダリングしてキーフレームグリッドを生成する。

    run_name を指定すると keyframes_base_dir/run_name/ に保存される。
    空白の場合はタイムスタンプ（例: run_20260307_143022）が自動付与される。
    これにより異なるパラメータの試行が全て保存・比較できる。

    入力:
      glb_dir            - GLBファイルが入ったディレクトリ（stage順にソート）
      keyframes_base_dir - 出力の親ディレクトリ（run_nameのサブフォルダが作られる）
      run_name           - 試行名（空白=タイムスタンプ自動、例: "size045_plateau3"）
      grid_theta         - 空間軸（角度分割数）
      elev_start/end     - 仰角（最初→最後）
      cam_dist           - カメラ距離
      render_w/h         - 解像度
      size_start         - 最初のstageのサイズ比率（0.45など）
      size_plateau_stage - このステージ番号（0始まり）で最大サイズ1.0に到達
      bg_gray            - 背景グレー値 0.0〜1.0
    """

    CATEGORY = "NanoBanana/4DGrid"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "glb_dir":              ("STRING",  {"default": r"D:\NB4D_test\tomato\glb"}),
                "keyframes_base_dir":   ("STRING",  {"default": r"D:\NB4D_test\tomato\grid_keyframes"}),
                "run_name":             ("STRING",  {"default": ""}),
                "grid_theta":           ("INT",     {"default": 120, "min": 8, "max": 360, "step": 8}),
                "elev_start":           ("FLOAT",   {"default": 45.0, "min": 0.0, "max": 89.0, "step": 1.0}),
                "elev_end":             ("FLOAT",   {"default": 5.0,  "min": 0.0, "max": 89.0, "step": 1.0}),
                "cam_dist":             ("FLOAT",   {"default": 2.5,  "min": 1.0, "max": 10.0, "step": 0.1}),
                "render_w":             ("INT",     {"default": 1280, "min": 256, "max": 3840, "step": 64}),
                "render_h":             ("INT",     {"default": 720,  "min": 144, "max": 2160, "step": 64}),
                "size_start":           ("FLOAT",   {"default": 0.45, "min": 0.1, "max": 1.0,  "step": 0.05}),
                "size_plateau_stage":   ("INT",     {"default": 3,    "min": 0,   "max": 99,   "step": 1}),
                "bg_gray":              ("FLOAT",   {"default": 0.12, "min": 0.0, "max": 1.0,  "step": 0.01}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("keyframes_dir", "status")
    FUNCTION = "render_grid"
    OUTPUT_NODE = True

    def render_grid(self, glb_dir, keyframes_base_dir, run_name,
                    grid_theta, elev_start, elev_end, cam_dist,
                    render_w, render_h, size_start, size_plateau_stage, bg_gray):

        try:
            import pyrender
            import trimesh
        except ImportError:
            return ("", "[ERROR] pip install pyrender trimesh")

        from datetime import datetime

        glb_dir_p    = Path(glb_dir)
        base_dir     = Path(keyframes_base_dir)

        # run_name が空ならタイムスタンプを自動付与
        name = run_name.strip() or datetime.now().strftime("run_%Y%m%d_%H%M%S")
        keyframes_dir_p = base_dir / name
        keyframes_dir_p.mkdir(parents=True, exist_ok=True)

        glb_files = sorted(glb_dir_p.glob("*.glb"))
        if not glb_files:
            return ("", f"[ERROR] GLBが見つかりません: {glb_dir}")

        N          = len(glb_files)
        elevations = np.linspace(elev_start, elev_end, N)
        bg_color   = [bg_gray, bg_gray, bg_gray]

        plateau_idx = max(size_plateau_stage, 1)

        def size_curve(i):
            if i >= plateau_idx:
                return 1.0
            return size_start + (1.0 - size_start) * (i / plateau_idx)

        completed = 0
        skipped   = 0

        for i, glb_path in enumerate(glb_files):
            out_dir = keyframes_dir_p / f"stage_{i:02d}"
            elev    = float(elevations[i])
            scale   = size_curve(i)

            # 同一 run_name での再実行時: 完了済みstageはスキップ
            if out_dir.exists() and len(list(out_dir.glob("angle_*.png"))) == grid_theta:
                skipped += 1
                continue

            _render_glb_all_angles(
                glb_path, out_dir, grid_theta, elev, cam_dist,
                render_w, render_h, bg_color, scale
            )
            completed += 1

        # grid_meta.json 保存（パラメータ全込み）
        meta = {
            "run_name":    name,
            "n_stages":    N,
            "grid_theta":  grid_theta,
            "elev_start":  elev_start,
            "elev_end":    elev_end,
            "glb_files":   [g.name for g in glb_files],
            "stage_dirs":  [f"stage_{i:02d}" for i in range(N)],
            "render_params": {
                "grid_theta":         grid_theta,
                "elev_start":         elev_start,
                "elev_end":           elev_end,
                "cam_dist":           cam_dist,
                "render_w":           render_w,
                "render_h":           render_h,
                "size_start":         size_start,
                "size_plateau_stage": size_plateau_stage,
                "bg_gray":            bg_gray,
            },
        }
        meta_path = keyframes_dir_p / "grid_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        status = (
            f"完了: {name}\n"
            f"  {N}stages × {grid_theta}角度 = {N*grid_theta}枚\n"
            f"  レンダリング: {completed}  スキップ: {skipped}\n"
            f"  保存先: {keyframes_dir_p}"
        )
        return (str(keyframes_dir_p), status)


def _render_glb_all_angles(glb_path, out_dir, grid_theta, elev_deg, cam_dist,
                             W, H, bg, size_scale):
    """1つのGLBを grid_theta 角度でレンダリングして out_dir に保存"""
    import pyrender
    import trimesh
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)

    scene_mesh = trimesh.load(str(glb_path), force="scene")
    if isinstance(scene_mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(list(scene_mesh.geometry.values()))
    else:
        mesh = scene_mesh

    mesh.vertices -= mesh.centroid
    r = mesh.bounding_sphere.primitive.radius
    if r > 0:
        mesh.vertices /= r

    # 全stageの底面を Y=-1.0 に統一（形状が異なっても接地点を1点に固定）
    y_bottom = mesh.vertices[:, 1].min()
    mesh.vertices[:, 1] -= (y_bottom + 1.0)   # 底面を -1.0 に揃える

    # スケール後も底面 Y=-1.0 を維持
    mesh.vertices *= size_scale
    mesh.vertices[:, 1] -= (1.0 - size_scale)  # -1.0 * (1 - scale) を補正

    try:
        mesh.visual = mesh.visual.to_color()
    except Exception:
        pass

    pr_mesh  = pyrender.Mesh.from_trimesh(mesh, smooth=True)
    scene    = pyrender.Scene(
        ambient_light=[0.35, 0.35, 0.35],
        bg_color=bg + [1.0],
    )
    scene.add(pr_mesh)

    light      = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    light_node = scene.add(light, pose=np.eye(4))
    camera     = pyrender.PerspectiveCamera(yfov=np.radians(45.0))
    cam_node   = scene.add(camera, pose=np.eye(4))
    renderer   = pyrender.OffscreenRenderer(W, H)

    elev_rad = np.radians(elev_deg)

    for j in range(grid_theta):
        theta = (j / grid_theta) * 2.0 * np.pi
        cx = cam_dist * np.sin(theta) * np.cos(elev_rad)
        cy = cam_dist * np.sin(elev_rad)
        cz = cam_dist * np.cos(theta) * np.cos(elev_rad)

        look  = np.array([-cx, -cy, -cz], dtype=float)
        look /= np.linalg.norm(look)
        right = np.cross(look, [0, 1, 0])
        norm_r = np.linalg.norm(right)
        if norm_r < 1e-6:
            right = np.array([1.0, 0.0, 0.0])
        else:
            right /= norm_r
        up = np.cross(right, look)

        pose = np.eye(4)
        pose[:3, 0] = right
        pose[:3, 1] = up
        pose[:3, 2] = -look
        pose[:3, 3] = [cx, cy, cz]

        scene.set_pose(cam_node,   pose)
        scene.set_pose(light_node, pose)

        color, _ = renderer.render(scene)
        Image.fromarray(color).save(str(out_dir / f"angle_{j:03d}.png"))

    renderer.delete()


# ─── Node 2: Grid4DTraverse ───────────────────────────────────────────────────

class Grid4DTraverse:
    """
    キーフレームグリッドを指定パスで走査し、MP4動画を生成する。

    入力:
      keyframes_dir - Grid4DRenderKeyframesの出力（grid_meta.jsonがある場所）
      path_name     - 走査パス（diagonal / spiral_2x / orbit_green / etc.）
      frames        - 総フレーム数（120=5s@24fps, 480=20s@24fps, 720=30s@24fps）
      fps           - フレームレート
      output_dir    - MP4出力先ディレクトリ
      output_name   - ファイル名（空白で自動命名）
    """

    CATEGORY = "NanoBanana/4DGrid"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "keyframes_dir": ("STRING",  {"default": r"D:\NB4D_test\tomato\grid_keyframes"}),
                "path_name":     (list(PATHS_INFO.keys()), {}),
                "frames":        ("INT",     {"default": 480, "min": 24, "max": 3600, "step": 24}),
                "fps":           ("INT",     {"default": 24,  "min": 12, "max": 60,   "step": 1}),
                "output_dir":    ("STRING",  {"default": r"D:\NB4D_test\tomato\output"}),
                "video_codec":   (["libx264", "h264_nvenc"], {
                    "default": "libx264",
                    "tooltip": "libx264: CPU（安定）/ h264_nvenc: NVIDIA GPU（5〜10x高速、要対応ffmpeg）",
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("frames", "video_path", "status")
    FUNCTION = "traverse"
    OUTPUT_NODE = True

    def traverse(self, keyframes_dir, path_name, frames, fps, output_dir, video_codec="libx264"):
        try:
            import cv2
            from PIL import Image
            import torch
        except ImportError:
            dummy = torch.zeros(1, 64, 64, 3)
            return (dummy, "", "[ERROR] pip install opencv-python pillow")

        kf_dir  = Path(keyframes_dir)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # メタ読み込み
        meta_path = kf_dir / "grid_meta.json"
        if not meta_path.exists():
            return ("", f"[ERROR] grid_meta.json が見つかりません: {meta_path}")

        with open(meta_path) as f:
            meta = json.load(f)

        n_stages  = meta["n_stages"]
        grid_theta_val = meta["grid_theta"]

        # パス関数
        path_func = _get_path_func(path_name, grid_theta_val)

        # 出力ファイルパス
        from datetime import datetime
        ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"4d_{path_name}_{ts}.mp4"
        video_path = out_dir / fname

        # GridTraverserで走査 → 一時ディレクトリにフレームを保存 → ffmpeg
        traverser = _GridTraverser(kf_dir, n_stages, grid_theta_val)

        try:
            ffmpeg = _get_ffmpeg()
        except RuntimeError as e:
            return (torch.zeros(1, 64, 64, 3), "", f"[ERROR] {e}")

        try:
            from comfy.utils import ProgressBar
            pbar = ProgressBar(frames)
        except Exception:
            pbar = None

        print(f"[Grid4DTraverse] 開始: {path_name}  {frames}フレーム ({frames/fps:.1f}秒 @{fps}fps)")
        log_step = max(1, frames // 10)  # 10%刻みでコンソール出力

        frames_list = []
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp_dir = Path(tmp_str)

            for i in range(frames):
                t, theta = path_func(i, frames)
                frame    = traverser.get_frame(float(t), float(theta))
                Image.fromarray(frame).save(str(tmp_dir / f"frame_{i:05d}.png"))
                frames_list.append(frame)

                if pbar is not None:
                    pbar.update(1)
                if i % log_step == 0 or i == frames - 1:
                    pct = (i + 1) / frames * 100
                    print(f"  フレーム生成: {i+1}/{frames} ({pct:.0f}%)")

            print(f"[Grid4DTraverse] ffmpeg エンコード中... (codec={video_codec})")
            cmd = _build_video_cmd(
                ffmpeg, fps, tmp_dir / "frame_%05d.png", video_path, video_codec
            )
            r = subprocess.run(cmd, capture_output=True, text=True)

        frames_tensor = torch.from_numpy(
            np.stack(frames_list, axis=0).astype(np.float32) / 255.0
        )

        if r.returncode != 0:
            return (frames_tensor, "", f"[ffmpeg ERROR] {r.stderr[-400:]}")

        duration = frames / fps
        print(f"[Grid4DTraverse] 完成: {video_path}")
        status = (
            f"完成: {video_path}\n"
            f"  パス: {path_name}  |  {frames}フレーム = {duration:.1f}秒 @ {fps}fps\n"
            f"  説明: {PATHS_INFO.get(path_name, '')}"
        )
        return (frames_tensor, str(video_path), status)


def _get_path_func(path_name, grid_theta):
    """パス名 → (i, N) → (t, theta) 関数"""
    GT = grid_theta

    # ミツバチパス用シード固定ランダム（再現性のため）
    rng = np.random.default_rng(42)
    # bee_hover: 角速度にゆらぎ、時間軸は前進しながら小刻みに上下
    _bee_hover_jitter = rng.uniform(-0.15, 0.15, 2000)

    # ── 猫視点パス群 ────────────────────────────────────────────────────────────

    def cat_double_arc(i, N):
        p = i / N
        t = float(np.clip(p * (GRID_T - 1), 0, GRID_T - 1))
        if p < 0.5:
            theta = p / 0.5 * (GT * 0.5)
        else:
            theta = (1.0 - (p - 0.5) / 0.5) * (GT * 0.5)
        return (t, float(theta))

    def cat_young_linger(i, N):
        p = i / N
        if p < 0.7:
            t = float(np.clip(p / 0.7 * (GRID_T - 1) * 0.4, 0, GRID_T - 1))
        else:
            t = float(np.clip((GRID_T - 1) * 0.4 + (p - 0.7) / 0.3 * (GRID_T - 1) * 0.6, 0, GRID_T - 1))
        theta = float(p * GT % GT)
        return (t, theta)

    def cat_waltz(i, N):
        p = i / N
        t = float(np.clip(p * (GRID_T - 1), 0, GRID_T - 1))
        center = p * GT * 0.25
        swing  = GT * 0.10 * np.sin(2 * np.pi * p * 3)
        theta  = float((center + swing) % GT)
        return (t, theta)

    def bee_hover(i, N):
        p    = i / N
        t    = p * (GRID_T - 1)
        # 時間軸: 前進しながら高周波サイン波で揺れる（成長しながらホバリング感）
        t   += (GRID_T - 1) * 0.06 * np.sin(2 * np.pi * i * 7 / N)
        t    = float(np.clip(t, 0, GRID_T - 1))
        # 角度: 基本は1周しながら不規則なフラフラ
        base_theta = i * GT / N % GT
        jitter     = _bee_hover_jitter[i % len(_bee_hover_jitter)] * GT
        theta      = (base_theta + jitter) % GT
        return (t, theta)

    def bee_spiral_in(i, N):
        p = i / N
        # 前半(0→0.7): 遠くから3周しながら近づきつつ成長を観察
        # 後半(0.7→1): 花のそばで1周しながら満開/枯れを観察
        t = p * (GRID_T - 1)
        if p < 0.7:
            # 3周螺旋（外から内へ）
            turns = 3.0
            theta = (p / 0.7) * GT * turns % GT
        else:
            # 内側で1周
            theta = ((p - 0.7) / 0.3) * GT % GT
        return (float(np.clip(t, 0, GRID_T - 1)), float(theta))

    def bee_inspect(i, N):
        # 全体を4フェーズに分割: 接近→観察→離脱→次stageへ移動
        # これをステージ数回繰り返す
        p     = i / N
        cycle = 4  # 4フェーズ × stage数
        phase = (p * cycle) % 1.0  # 0〜1 の周期
        # 時間軸: ステージを順に進める（阶段的）
        t     = p * (GRID_T - 1)

        if phase < 0.3:
            # 接近: 正面から近づく（角度ゆっくり変化）
            theta = phase / 0.3 * (GT * 0.15)
        elif phase < 0.6:
            # 観察: 花の周りを素早く半周
            theta = GT * 0.15 + (phase - 0.3) / 0.3 * (GT * 0.5)
        elif phase < 0.8:
            # 離脱: 急旋回
            theta = GT * 0.65 + (phase - 0.6) / 0.2 * (GT * 0.2)
        else:
            # 次へ移動: 大きく角度変化
            theta = GT * 0.85 + (phase - 0.8) / 0.2 * (GT * 0.15)

        return (float(np.clip(t, 0, GRID_T - 1)), float(theta % GT))

    funcs = {
        "orbit_green":  lambda i, N: (0.0, i * GT / N % GT),
        "orbit_ripe":   lambda i, N: (GRID_T - 1, i * GT / N % GT),
        "ripen_front":  lambda i, N: (i * (GRID_T - 1) / N, 0.0),
        "diagonal":     lambda i, N: (i * (GRID_T - 1) / N, i * GT / N % GT),
        "spiral_2x":    lambda i, N: (i * (GRID_T - 1) / N, i * GT * 2 / N % GT),
        "reverse_diag": lambda i, N: ((GRID_T - 1) - i * (GRID_T - 1) / N, i * GT / N % GT),
        "time_wave":    lambda i, N: (
            (GRID_T - 1) / 2 * (1 + np.sin(4 * np.pi * i / N)),
            i * GT / N % GT,
        ),
        "zoom_in_time": lambda i, N: (
            i * (GRID_T - 1) / N,
            i * GT / N % GT if i < N * 0.8 else (GRID_T - 1) * 0.8 + (i - N * 0.8) * 0.2 * (GRID_T - 1) / (N * 0.2),
        ),
        # ミツバチ視点
        "bee_hover":       bee_hover,
        "bee_spiral_in":   bee_spiral_in,
        "bee_inspect":     bee_inspect,
        # ── 猫視点 7パス ─────────────────────────────────────────────────────────
        "cat_passthrough":  lambda i, N: (
            float(np.clip(i * (GRID_T - 1) / N, 0, GRID_T - 1)),
            float(i / N * (GT * 0.5)),
        ),
        "cat_reverse_pass": lambda i, N: (
            float(np.clip(i * (GRID_T - 1) / N, 0, GRID_T - 1)),
            float((1.0 - i / N) * (GT * 0.5)),
        ),
        "cat_circle_age":   lambda i, N: (
            float(np.clip(i * (GRID_T - 1) / N, 0, GRID_T - 1)),
            float(i * GT / N % GT),
        ),
        "cat_double_arc":   cat_double_arc,
        "cat_young_linger": cat_young_linger,
        "cat_side_age":     lambda i, N: (
            float(np.clip(i * (GRID_T - 1) / N, 0, GRID_T - 1)),
            float(GT * 0.25),
        ),
        "cat_waltz":        cat_waltz,
    }
    # ── 共通スイープパス (_nb4d_paths.py から生成) ─────────────────────────────
    for _name, _fn in SWEEP_PATHS.items():
        funcs[f"sweep_{_name}"] = sweep_lambda_for_grid(_name, GRID_T, GT)

    return funcs.get(path_name, funcs["diagonal"])


class _GridTraverser:
    def __init__(self, kf_dir, n_stages, grid_theta):
        self.kf_dir    = kf_dir
        self.n_stages  = n_stages
        self.grid_theta = grid_theta
        self.kf_times  = np.linspace(0, GRID_T - 1, n_stages)
        self._img_cache = {}

    def _get_img(self, stage, theta):
        import numpy as np
        from PIL import Image

        key = (stage, theta)
        if key not in self._img_cache:
            if len(self._img_cache) > 300:
                del self._img_cache[next(iter(self._img_cache))]
            path = self.kf_dir / f"stage_{stage:02d}" / f"angle_{theta:03d}.png"
            self._img_cache[key] = np.array(Image.open(str(path)).convert("RGB"), dtype=np.uint8)
        return self._img_cache[key]

    def get_frame(self, t, theta):
        import cv2

        t     = float(np.clip(t, 0, GRID_T - 1))
        theta = float(theta % self.grid_theta)
        theta_idx = int(round(theta)) % self.grid_theta

        kf_times = self.kf_times
        if t <= kf_times[0]:
            return self._get_img(0, theta_idx)
        if t >= kf_times[-1]:
            return self._get_img(self.n_stages - 1, theta_idx)

        ka    = int(np.searchsorted(kf_times, t) - 1)
        kb    = ka + 1
        t_a, t_b = kf_times[ka], kf_times[kb]
        alpha = (t - t_a) / (t_b - t_a)

        img_a = self._get_img(ka, theta_idx)
        img_b = self._get_img(kb, theta_idx)

        return _interpolate(img_a, img_b, float(alpha))


def _interpolate(img_a, img_b, alpha):
    import cv2

    if alpha <= 0.0:
        return img_a
    if alpha >= 1.0:
        return img_b

    ga = cv2.cvtColor(img_a, cv2.COLOR_RGB2GRAY)
    gb = cv2.cvtColor(img_b, cv2.COLOR_RGB2GRAY)

    flow_ab = cv2.calcOpticalFlowFarneback(
        ga, gb, None, 0.5, 5, 25, 10, 7, 1.5, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    flow_ba = cv2.calcOpticalFlowFarneback(
        gb, ga, None, 0.5, 5, 25, 10, 7, 1.5, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

    mse = float(np.mean((ga.astype(float) - gb.astype(float)) ** 2))
    is_large = mse > 2000

    H, W = img_a.shape[:2]
    map_x_ab = np.arange(W, dtype=np.float32)[None, :] + (alpha * flow_ab[..., 0])
    map_y_ab = np.arange(H, dtype=np.float32)[:, None] + (alpha * flow_ab[..., 1])
    map_x_ba = np.arange(W, dtype=np.float32)[None, :] + ((1 - alpha) * flow_ba[..., 0])
    map_y_ba = np.arange(H, dtype=np.float32)[:, None] + ((1 - alpha) * flow_ba[..., 1])

    warp_a = cv2.remap(img_a.astype(np.float32), map_x_ab, map_y_ab,
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    warp_b = cv2.remap(img_b.astype(np.float32), map_x_ba, map_y_ba,
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    flow_blend = (1 - alpha) * warp_a + alpha * warp_b

    if is_large:
        dissolve = img_a.astype(np.float32) * (1 - alpha) + img_b.astype(np.float32) * alpha
        result   = flow_blend * 0.60 + dissolve * 0.40
    else:
        result = flow_blend

    return np.clip(result, 0, 255).astype(np.uint8)


# ─── Node 3: Grid4DInfo ───────────────────────────────────────────────────────

class Grid4DInfo:
    """
    グリッドのメタ情報を表示するユーティリティノード。
    パスの一覧と説明も確認できる。
    """

    CATEGORY = "NanoBanana/4DGrid"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "keyframes_dir": ("STRING", {"default": r"D:\NB4D_test\tomato\grid_keyframes"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    FUNCTION = "show_info"
    OUTPUT_NODE = True

    def show_info(self, keyframes_dir):
        lines = []
        kf_dir = Path(keyframes_dir)
        meta_path = kf_dir / "grid_meta.json"

        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            n_stages  = meta["n_stages"]
            grid_theta = meta["grid_theta"]
            total     = n_stages * grid_theta
            lines.append("=== Grid メタ情報 ===")
            lines.append(f"  stageファイル数: {n_stages}")
            lines.append(f"  角度分割数: {grid_theta}")
            lines.append(f"  総フレーム数: {total}")
            lines.append(f"  仰角: {meta['elev_start']}° → {meta['elev_end']}°")
            lines.append("")

            # 既存stageの確認
            done = 0
            for sd in meta["stage_dirs"]:
                sd_path = kf_dir / sd
                if sd_path.exists():
                    cnt = len(list(sd_path.glob("angle_*.png")))
                    if cnt == grid_theta:
                        done += 1
            lines.append(f"  完了stage: {done}/{n_stages}")
            lines.append("")
        else:
            lines.append("[INFO] grid_meta.json なし（未生成）")
            lines.append("")

        lines.append("=== 利用可能なパス ===")
        for name, desc in PATHS_INFO.items():
            lines.append(f"  {name:<16} {desc}")

        info = "\n".join(lines)
        print(info)
        return (info,)


# ─── Node 4: Grid4DComposite ──────────────────────────────────────────────────

class Grid4DComposite:
    """
    4Dグリッドを走査しながら、背景画像に合成してMP4を生成する。

    アルファ抽出:
      グレー背景色（bg_gray）と各画素のL2距離でマスクを生成。
      黒白二重レンダリング不要のシンプル方式。

    入力:
      keyframes_dir - Grid4DRenderKeyframesの出力ディレクトリ
      bg_image_path - 背景画像（JPGまたはPNG）
      path_name     - 走査パス
      frames        - 総フレーム数
      fps           - フレームレート
      plate_x/y     - 背景画像上のオブジェクト中心（0〜1 相対座標）
      plate_scale   - 背景短辺×scale = オブジェクト幅
      alpha_thresh  - 背景除去閾値
      output_dir    - MP4出力先
      output_name   - ファイル名（空白=自動）
    """

    CATEGORY = "NanoBanana/4DGrid"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "keyframes_dir": ("STRING",  {"default": r"D:\NB4D_test\tomato\grid_keyframes\run_20260307_103932"}),
                "bg_image_path": ("STRING",  {"default": r"D:\NB4D_test\tomato\bg_farm.jpg"}),
                "path_name":     (list(PATHS_INFO.keys()), {}),
                "frames":        ("INT",     {"default": 480, "min": 24, "max": 3600, "step": 24}),
                "fps":           ("INT",     {"default": 24,  "min": 12, "max": 60,   "step": 1}),
                "plate_x":       ("FLOAT",   {"default": 0.5,  "min": 0.0, "max": 1.0, "step": 0.01}),
                "plate_y":       ("FLOAT",   {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.01}),
                "plate_scale":   ("FLOAT",   {"default": 0.45, "min": 0.1, "max": 1.5, "step": 0.05}),
                "alpha_thresh":  ("INT",     {"default": 18,  "min": 1,   "max": 80,  "step": 1}),
                "output_dir":    ("STRING",  {"default": r"D:\NB4D_test\tomato\output_composite"}),
                "video_codec":   (["libx264", "h264_nvenc"], {
                    "default": "libx264",
                    "tooltip": "libx264: CPU（安定）/ h264_nvenc: NVIDIA GPU（5〜10x高速、要対応ffmpeg）",
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("frames", "video_path", "status")
    FUNCTION = "composite"
    OUTPUT_NODE = True

    def composite(self, keyframes_dir, bg_image_path, path_name, frames, fps,
                  plate_x, plate_y, plate_scale, alpha_thresh,
                  output_dir, video_codec="libx264"):
        try:
            import cv2
            from PIL import Image as PILImage
            import torch
        except ImportError:
            dummy = torch.zeros(1, 64, 64, 3)
            return (dummy, "", "[ERROR] pip install opencv-python pillow")

        from datetime import datetime

        kf_dir    = Path(keyframes_dir)
        meta_path = kf_dir / "grid_meta.json"
        if not meta_path.exists():
            return ("", f"[ERROR] grid_meta.json が見つかりません: {meta_path}")

        with open(meta_path) as f:
            meta = json.load(f)

        n_stages   = meta["n_stages"]
        grid_theta_val = meta["grid_theta"]
        bg_gray    = meta.get("render_params", {}).get("bg_gray", 0.12)

        bg_path = Path(bg_image_path)
        if not bg_path.exists():
            return ("", f"[ERROR] 背景画像が見つかりません: {bg_path}")
        bg_rgb = np.array(PILImage.open(str(bg_path)).convert("RGB"), dtype=np.uint8)

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"composite_{path_name}_{ts}.mp4"
        video_path = out_dir / fname

        traverser = _GridTraverser(kf_dir, n_stages, grid_theta_val)
        path_func = _get_path_func(path_name, grid_theta_val)

        try:
            ffmpeg = _get_ffmpeg()
        except RuntimeError as e:
            return (torch.zeros(1, 64, 64, 3), "", f"[ERROR] {e}")

        try:
            from comfy.utils import ProgressBar
            pbar = ProgressBar(frames)
        except Exception:
            pbar = None

        print(f"[Grid4DComposite] 開始: {path_name}  {frames}フレーム ({frames/fps:.1f}秒 @{fps}fps)")
        log_step = max(1, frames // 10)

        frames_list = []
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)

            for i in range(frames):
                t, theta = path_func(i, frames)
                frame    = traverser.get_frame(float(t), float(theta))

                alpha    = _extract_alpha(frame, bg_gray, alpha_thresh)
                composed = _composite_on_bg(frame, alpha, bg_rgb, plate_x, plate_y, plate_scale)

                PILImage.fromarray(composed).save(str(tmp / f"frame_{i:05d}.png"))
                frames_list.append(composed)

                if pbar is not None:
                    pbar.update(1)
                if i % log_step == 0 or i == frames - 1:
                    pct = (i + 1) / frames * 100
                    print(f"  フレーム生成: {i+1}/{frames} ({pct:.0f}%)")

            print(f"[Grid4DComposite] ffmpeg エンコード中... (codec={video_codec})")
            cmd = _build_video_cmd(
                ffmpeg, fps, tmp / "frame_%05d.png", video_path, video_codec
            )
            r = subprocess.run(cmd, capture_output=True, text=True)

        frames_tensor = torch.from_numpy(
            np.stack(frames_list, axis=0).astype(np.float32) / 255.0
        )

        if r.returncode != 0:
            return (frames_tensor, "", f"[ffmpeg ERROR] {r.stderr[-400:]}")

        duration = frames / fps
        print(f"[Grid4DComposite] 完成: {video_path}")
        status = (
            f"完成: {video_path}\n"
            f"  パス: {path_name}  |  {frames}フレーム = {duration:.1f}秒 @ {fps}fps\n"
            f"  配置: center=({plate_x:.2f}, {plate_y:.2f}), scale={plate_scale:.2f}"
        )
        return (frames_tensor, str(video_path), status)


def _extract_alpha(frame_rgb: np.ndarray, bg_gray: float, threshold: int = 18) -> np.ndarray:
    """グレー背景からアルファマスクを生成（0=背景, 255=前景）"""
    import cv2
    bg_val = int(round(bg_gray * 255))
    bg_arr = np.array([bg_val, bg_val, bg_val], dtype=np.float32)
    dist   = np.linalg.norm(frame_rgb.astype(np.float32) - bg_arr, axis=2)
    mask   = (dist > threshold).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask   = cv2.GaussianBlur(mask, (7, 7), 0)
    return mask


def _composite_on_bg(
    frame_rgb: np.ndarray,
    alpha_mask: np.ndarray,
    bg_rgb: np.ndarray,
    plate_cx: float,
    plate_cy: float,
    plate_scale: float,
) -> np.ndarray:
    """背景画像にフレームをアルファ合成"""
    import cv2
    H_bg, W_bg = bg_rgb.shape[:2]
    H_f,  W_f  = frame_rgb.shape[:2]

    target_w = int(min(W_bg, H_bg) * plate_scale)
    target_h = int(target_w * H_f / W_f)

    frame_rs = cv2.resize(frame_rgb,  (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    alpha_rs = cv2.resize(alpha_mask, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    cx = int(W_bg * plate_cx)
    cy = int(H_bg * plate_cy)
    x0 = cx - target_w // 2
    y0 = cy - target_h // 2
    x1 = x0 + target_w
    y1 = y0 + target_h

    sx0 = max(0, -x0);   sy0 = max(0, -y0)
    ex0 = max(0, x0);    ey0 = max(0, y0)
    sx1 = target_w - max(0, x1 - W_bg)
    sy1 = target_h - max(0, y1 - H_bg)
    ex1 = min(W_bg, x1); ey1 = min(H_bg, y1)

    result  = bg_rgb.copy().astype(np.float32)
    fg_crop = frame_rs[sy0:sy1, sx0:sx1].astype(np.float32)
    al_crop = alpha_rs[sy0:sy1, sx0:sx1].astype(np.float32) / 255.0
    al_crop = al_crop[:, :, np.newaxis]

    roi = result[ey0:ey1, ex0:ex1]
    result[ey0:ey1, ex0:ex1] = roi * (1.0 - al_crop) + fg_crop * al_crop

    return np.clip(result, 0, 255).astype(np.uint8)


# ─── Node 5: NB4D_ProjectConfig ──────────────────────────────────────────────

class NB4D_ProjectConfig:
    """
    プロジェクトルートフォルダを1か所に設定し、
    全ノードのパスを自動生成するコンフィグノード。

    YouTube視聴者向け: project_root を自分のフォルダに変えるだけで
    全ステップのパスが自動で繋がります。

    入力:
      project_root - プロジェクトフォルダ（例: D:\\MyProjects\\tomato）
                     中に glb/ フォルダを作りGLBファイルを置いてください。

    出力:
      glb_dir            → Grid4DRenderKeyframes.glb_dir に接続
      keyframes_base_dir → Grid4DRenderKeyframes.keyframes_base_dir に接続
      run_name           → Grid4DRenderKeyframes.run_name に接続（自動命名）
      output_dir         → Grid4DTraverse.output_dir に接続
      ltxv_output_dir    → NB4D_LTXVStageInterpolatorV2.output_dir に接続
    """

    CATEGORY = "NanoBanana/4DGrid"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "project_root": ("STRING", {
                    "default": r"D:\NB4D_projects\my_project",
                    "tooltip": (
                        "プロジェクトフォルダのパス。\n"
                        "例: D:\\NB4D_projects\\tomato\n"
                        "この中に glb/ フォルダを作り stage_01.glb 〜 を置いてください。"
                    ),
                }),
            }
        }

    RETURN_TYPES  = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES  = ("glb_dir", "keyframes_base_dir", "run_name", "output_dir", "ltxv_output_dir")
    FUNCTION      = "configure"
    OUTPUT_NODE   = False

    def configure(self, project_root):
        root = Path(project_root.strip())

        glb_dir            = root / "glb"
        keyframes_base_dir = root / "grid_keyframes"
        run_name           = root.name          # フォルダ名 = 安定した試行名
        output_dir         = root / "output"
        ltxv_output_dir    = root / "output_ltxv"

        # ディレクトリを事前作成
        for d in [glb_dir, keyframes_base_dir, output_dir, ltxv_output_dir]:
            d.mkdir(parents=True, exist_ok=True)

        print(f"[NB4D_ProjectConfig] プロジェクト: {root.name}")
        print(f"  GLB入力:         {glb_dir}")
        print(f"  グリッド保存先:  {keyframes_base_dir / run_name}")
        print(f"  MP4出力:         {output_dir}")
        print(f"  LTXV PNG出力:    {ltxv_output_dir}")

        return (
            str(glb_dir),
            str(keyframes_base_dir),
            run_name,
            str(output_dir),
            str(ltxv_output_dir),
        )


# ─── Node 6: NB4D_PNGtoMP4 ───────────────────────────────────────────────────

class NB4D_PNGtoMP4:
    """
    RGBA PNG 連番をMP4動画に変換するノード。
    NB4D_LTXVStageInterpolatorV2 の出力（RGBA PNG連番）をMP4にします。

    入力:
      png_dir     - LTXVStageInterpolatorV2の output_dir 出力を接続
      fps         - フレームレート（24推奨）
      bg_color    - 背景色HEXコード（例: #000000=黒, #ffffff=白）
      output_dir  - MP4保存先（空=png_dirの親フォルダ）
      output_name - ファイル名（空=自動タイムスタンプ）
    """

    CATEGORY = "NanoBanana/4DGrid"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "png_dir": ("STRING", {
                    "default": "",
                    "tooltip": "NB4D_LTXVStageInterpolatorV2 の output_dir 出力を接続してください",
                }),
                "fps":      ("INT",    {"default": 24,  "min": 12, "max": 60}),
                "bg_color": ("STRING", {
                    "default": "#000000",
                    "tooltip": "背景色HEX: #000000=黒 / #ffffff=白 / #1a1a2e=ダーク青",
                }),
                "output_dir":  ("STRING", {"default": ""}),
                "video_codec": (["libx264", "h264_nvenc"], {
                    "default": "libx264",
                    "tooltip": "libx264: CPU（安定）/ h264_nvenc: NVIDIA GPU（5〜10x高速、要対応ffmpeg）",
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("frames", "video_path", "status")
    FUNCTION     = "assemble"
    OUTPUT_NODE  = True

    def assemble(self, png_dir, fps, bg_color, output_dir, video_codec="libx264"):
        try:
            from PIL import Image as PILImage
            import torch
        except ImportError:
            dummy = torch.zeros(1, 64, 64, 3)
            return (dummy, "", "[ERROR] pip install pillow")

        from datetime import datetime

        png_dir_p = Path(png_dir.strip())
        if not png_dir_p.exists():
            return ("", f"[ERROR] フォルダが見つかりません: {png_dir}")

        frames = sorted(png_dir_p.glob("frame_*.png"))
        if not frames:
            dummy = torch.zeros(1, 64, 64, 3)
            return (dummy, "", f"[ERROR] frame_*.png が見つかりません: {png_dir}")

        # 背景色パース
        hex_str = bg_color.strip().lstrip("#")
        try:
            bg_r, bg_g, bg_b = int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16)
        except Exception:
            bg_r, bg_g, bg_b = 0, 0, 0

        # 出力先
        if output_dir.strip():
            out_dir = Path(output_dir.strip())
        else:
            out_dir = png_dir_p.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = out_dir / f"ltxv_mp4_{ts}.mp4"

        try:
            ffmpeg = _get_ffmpeg()
        except RuntimeError as e:
            dummy = torch.zeros(1, 64, 64, 3)
            return (dummy, "", f"[ERROR] {e}")

        n_frames = len(frames)
        try:
            from comfy.utils import ProgressBar
            pbar = ProgressBar(n_frames)
        except Exception:
            pbar = None

        print(f"[NB4D_PNGtoMP4] 開始: {n_frames}フレーム ({n_frames/fps:.1f}秒 @{fps}fps)  背景: {bg_color}")
        log_step = max(1, n_frames // 10)

        frames_list = []
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            for i, frame_path in enumerate(frames):
                img = PILImage.open(str(frame_path))
                if img.mode == "RGBA":
                    bg = PILImage.new("RGB", img.size, (bg_r, bg_g, bg_b))
                    bg.paste(img, mask=img.split()[3])
                    rgb = bg
                else:
                    rgb = img.convert("RGB")
                rgb.save(str(tmp / f"frame_{i:05d}.png"))
                frames_list.append(np.array(rgb, dtype=np.uint8))

                if pbar is not None:
                    pbar.update(1)
                if i % log_step == 0 or i == n_frames - 1:
                    pct = (i + 1) / n_frames * 100
                    print(f"  フレーム変換: {i+1}/{n_frames} ({pct:.0f}%)")

            print(f"[NB4D_PNGtoMP4] ffmpeg エンコード中... (codec={video_codec})")
            cmd = _build_video_cmd(
                ffmpeg, fps, tmp / "frame_%05d.png", video_path, video_codec
            )
            r = subprocess.run(cmd, capture_output=True, text=True)

        frames_tensor = torch.from_numpy(
            np.stack(frames_list, axis=0).astype(np.float32) / 255.0
        )

        if r.returncode != 0:
            return (frames_tensor, "", f"[ffmpeg ERROR] {r.stderr[-400:]}")

        duration = n_frames / fps
        print(f"[NB4D_PNGtoMP4] 完成: {video_path}")
        status = (
            f"完成: {video_path}\n"
            f"  {n_frames}フレーム = {duration:.1f}秒 @ {fps}fps  背景: {bg_color}"
        )
        return (frames_tensor, str(video_path), status)


# ─── ノード登録 ──────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "Grid4DRenderKeyframes": Grid4DRenderKeyframes,
    "Grid4DTraverse":        Grid4DTraverse,
    "Grid4DInfo":            Grid4DInfo,
    "Grid4DComposite":       Grid4DComposite,
    "NB4D_ProjectConfig":    NB4D_ProjectConfig,
    "NB4D_PNGtoMP4":         NB4D_PNGtoMP4,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Grid4DRenderKeyframes": "4D Grid: Render Keyframes",
    "Grid4DTraverse":        "4D Grid: Traverse → MP4",
    "Grid4DInfo":            "4D Grid: Info",
    "Grid4DComposite":       "4D Grid: Composite → MP4",
    "NB4D_ProjectConfig":    "🚀 4D Grid: Project Config (START HERE)",
    "NB4D_PNGtoMP4":         "4D Grid: PNG Sequence → MP4",
}

# grid_render_keyframes.py
#
# 4D グリッド STEP 1: キーフレームGLBを 120角度でレンダリング
#
# 出力: KEYFRAMES_DIR/stage_XX/angle_YYY.png  (N_stages × 120 枚)
#
# 使い方:
#   python grid_render_keyframes.py
#   python grid_render_keyframes.py --glb_dir D:/path/to/glb --grid_theta 120

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# ─── 設定 ────────────────────────────────────────────────────────────────────

GLB_DIR      = Path(r"D:\NB4D_test\tomato\glb")
KEYFRAMES_DIR = Path(r"D:\NB4D_test\tomato\grid_keyframes")

GRID_THETA   = 120        # 空間軸: 120角度（3°刻み）
ELEV_START   = 45.0       # 最初のstageの仰角（緑のトマト：上から）
ELEV_END     = 5.0        # 最後のstageの仰角（完熟：目線）
CAM_DIST     = 2.5
RENDER_W     = 1280
RENDER_H     = 720
BG_COLOR     = [0.12, 0.12, 0.12]   # 暗いグレー背景

# サイズ成長カーブ
#   小さなトマト(SIZE_START) → 最大サイズ(1.0) → サイズ変化なし（色のみ変化）
#   SIZE_PLATEAU: 成長が止まるステージ比率（0.45 = 8枚中4枚目で最大サイズに達する）
SIZE_START         = 0.45   # 最初のstageのサイズ比率（1.0が最大）
SIZE_PLATEAU_STAGE = 3      # このステージ番号（0始まり）で最大サイズ1.0に到達
                            # 例: 8枚でstage_03を100%にしたい → 3

# ─── レンダリングコア ─────────────────────────────────────────────────────────

def render_glb_all_angles(glb_path: Path, out_dir: Path,
                           grid_theta: int, elev_deg: float,
                           W: int, H: int, bg: list,
                           size_scale: float = 1.0):
    """1つのGLBを grid_theta 角度でレンダリングして out_dir に保存"""
    try:
        import pyrender
        import trimesh
    except ImportError:
        print("  [ERROR] pip install pyrender trimesh")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    # --- メッシュ読み込み ---
    scene_mesh = trimesh.load(str(glb_path), force="scene")
    if isinstance(scene_mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(list(scene_mesh.geometry.values()))
    else:
        mesh = scene_mesh

    # 正規化（中心＝原点、単位球）→ サイズカーブを適用
    mesh.vertices -= mesh.centroid
    r = mesh.bounding_sphere.primitive.radius
    if r > 0:
        mesh.vertices /= r          # 全ステージを単位球に統一

    # 全stageの底面を Y=-1.0 に統一（形状が異なっても接地点を1点に固定）
    y_bottom = mesh.vertices[:, 1].min()
    mesh.vertices[:, 1] -= (y_bottom + 1.0)   # 底面を -1.0 に揃える

    # スケール後も底面 Y=-1.0 を維持
    mesh.vertices *= size_scale     # サイズ成長カーブを反映（0.45→1.0）
    mesh.vertices[:, 1] -= (1.0 - size_scale)  # -1.0 * (1 - scale) を補正

    # UV テクスチャ → 頂点カラー変換（pyrender OpenGL エラー回避）
    try:
        mesh.visual = mesh.visual.to_color()
    except Exception:
        pass

    # --- pyrender シーン ---
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
        theta = (j / grid_theta) * 2.0 * np.pi   # 0 〜 2π

        cx = CAM_DIST * np.sin(theta) * np.cos(elev_rad)
        cy = CAM_DIST * np.sin(elev_rad)
        cz = CAM_DIST * np.cos(theta) * np.cos(elev_rad)

        look  = np.array([-cx, -cy, -cz], dtype=float)
        look /= np.linalg.norm(look)
        right = np.cross(look, [0, 1, 0])
        right /= np.linalg.norm(right)
        up    = np.cross(right, look)

        pose = np.eye(4)
        pose[:3, 0] = right
        pose[:3, 1] = up
        pose[:3, 2] = -look
        pose[:3, 3] = [cx, cy, cz]

        scene.set_pose(cam_node, pose)
        scene.set_pose(light_node, pose)

        color, _ = renderer.render(scene)
        Image.fromarray(color).save(str(out_dir / f"angle_{j:03d}.png"))

    renderer.delete()


# ─── メイン ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="グリッド キーフレームレンダリング")
    parser.add_argument("--glb_dir",            default=str(GLB_DIR))
    parser.add_argument("--keyframes_base_dir",  default=str(KEYFRAMES_DIR),
                        help="出力の親ディレクトリ。run_nameのサブフォルダが作られる")
    parser.add_argument("--run_name",            default="",
                        help="試行名（空白=タイムスタンプ自動、例: size045_plateau3）")
    parser.add_argument("--grid_theta",          type=int,   default=GRID_THETA)
    parser.add_argument("--elev_start",          type=float, default=ELEV_START)
    parser.add_argument("--elev_end",            type=float, default=ELEV_END)
    parser.add_argument("--width",               type=int,   default=RENDER_W)
    parser.add_argument("--height",              type=int,   default=RENDER_H)
    parser.add_argument("--size_start",          type=float, default=SIZE_START,
                        help="最初のstageのサイズ比率（デフォルト0.45）")
    parser.add_argument("--size_plateau_stage",  type=int,   default=SIZE_PLATEAU_STAGE,
                        help="このステージ番号（0始まり）で最大サイズ1.0に到達（デフォルト3）")
    args = parser.parse_args()

    from datetime import datetime
    import json

    glb_dir  = Path(args.glb_dir)
    base_dir = Path(args.keyframes_base_dir)

    # run_name が空ならタイムスタンプを自動付与
    name = args.run_name.strip() or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    keyframes_dir = base_dir / name
    keyframes_dir.mkdir(parents=True, exist_ok=True)

    glb_files = sorted(glb_dir.glob("*.glb"))
    if not glb_files:
        print(f"[ERROR] GLBファイルが見つかりません: {glb_dir}")
        sys.exit(1)

    N = len(glb_files)
    elevations = np.linspace(args.elev_start, args.elev_end, N)

    plateau_idx = max(args.size_plateau_stage, 1)

    def size_curve(i):
        if i >= plateau_idx:
            return 1.0
        return args.size_start + (1.0 - args.size_start) * (i / plateau_idx)

    print("=" * 62)
    print("  grid_render_keyframes.py")
    print(f"  試行名: {name}")
    print(f"  GLB: {N}個  |  角度: {args.grid_theta}  |  合計: {N * args.grid_theta}枚")
    print(f"  仰角: {args.elev_start}° → {args.elev_end}°")
    print(f"  サイズ: {args.size_start} → 1.0  (stage_{plateau_idx:02d}で100%到達)")
    print(f"  保存先: {keyframes_dir}")
    print("=" * 62)

    for i, glb_path in enumerate(glb_files):
        out_dir = keyframes_dir / f"stage_{i:02d}"
        elev    = float(elevations[i])
        scale   = size_curve(i)

        # 同一 run_name 内での再実行時: 完了済みstageはスキップ
        if out_dir.exists() and len(list(out_dir.glob("angle_*.png"))) == args.grid_theta:
            print(f"  stage_{i:02d}: スキップ（完了済み）")
            continue

        print(f"\n  stage_{i:02d}: {glb_path.name}  仰角={elev:.1f}°  サイズ={scale:.2f}")
        render_glb_all_angles(
            glb_path, out_dir,
            args.grid_theta, elev,
            args.width, args.height, BG_COLOR,
            size_scale=scale,
        )
        print(f"  → {args.grid_theta}枚保存: {out_dir}")

    # grid_meta.json 保存（パラメータ全込み）
    meta = {
        "run_name":    name,
        "n_stages":    N,
        "grid_theta":  args.grid_theta,
        "elev_start":  args.elev_start,
        "elev_end":    args.elev_end,
        "glb_files":   [g.name for g in glb_files],
        "stage_dirs":  [f"stage_{i:02d}" for i in range(N)],
        "render_params": {
            "grid_theta":         args.grid_theta,
            "elev_start":         args.elev_start,
            "elev_end":           args.elev_end,
            "size_start":         args.size_start,
            "size_plateau_stage": args.size_plateau_stage,
        },
    }
    meta_path = keyframes_dir / "grid_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  メタ情報保存: {meta_path}")
    print("=" * 62)
    print(f"  完了。次のステップ:")
    print(f"    python grid_traverse.py --path diagonal")
    print("=" * 62)


if __name__ == "__main__":
    main()

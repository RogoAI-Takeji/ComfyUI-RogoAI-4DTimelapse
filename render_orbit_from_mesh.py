# render_orbit_from_mesh.py
#
# 3Dメッシュ(OBJ/GLB/PLY)を読み込んで360度orbit動画を生成する
#
# 前提: trimesh はインストール済み (pip show trimesh)
# 推奨: pip install pyrender  (高品質レンダリング。なければ numpy rasterizer を使用)
#
# 使い方:
#   & $PY render_orbit_from_mesh.py --mesh D:/path/to/model.glb
#   & $PY render_orbit_from_mesh.py --mesh D:/path/to/model.obj --elev 20
#
# 3Dメッシュの生成方法:
#   [方法A] ComfyUI + Hunyuan3D-2 ノード (最高品質・推奨)
#     1. comfyui-manager から "ComfyUI-Hunyuan3D" をインストール
#     2. HuggingFace から Tencent/Hunyuan3D-2 モデルをダウンロード
#     3. ComfyUI で stage_01.png -> GLBメッシュを生成
#
#   [方法B] ComfyUI + TripoSR ノード (高速・簡単)
#     1. comfyui-manager から "ComfyUI-TripoSR" をインストール
#     2. stabilityai/TripoSR モデルをダウンロード (~1.5GB)
#     3. ComfyUI で stage_01.png -> OBJメッシュを生成
#
#   [方法C] Hunyuan3D-2 スタンドアロン (ComfyUI不要)
#     pip install hunyuan3d
#     python -c "from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline; ..."

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

OUTPUT_DIR = Path(r"D:\NB4D_test\orbit_mesh")
FRAMES_DIR = OUTPUT_DIR / "frames"
FFMPEG_CANDIDATES = [
    r"C:\Users\fareg\AppData\Local\FFmpeg\bin\ffmpeg.exe",
    "ffmpeg",
]


def get_ffmpeg():
    for p in FFMPEG_CANDIDATES:
        try:
            subprocess.run([p, "-version"], capture_output=True, check=True)
            return p
        except Exception:
            pass
    raise RuntimeError("ffmpeg が見つかりません")


# ─── レンダラー: pyrender（高品質） ───────────────────────────────────────────
def try_render_pyrender(mesh_path, frames_dir, total_frames,
                         W, H, fps, elev_deg):
    """
    pyrender を使った高品質レンダリング。
    return True=成功, False=pyrender なし
    """
    try:
        import pyrender
        import trimesh
    except ImportError:
        return False

    print("  pyrender でレンダリングします（高品質モード）")

    scene_mesh = trimesh.load(str(mesh_path), force="scene")
    if isinstance(scene_mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            [g for g in scene_mesh.geometry.values()])
    else:
        mesh = scene_mesh

    # メッシュを正規化（中心を原点に、単位球に収める）
    mesh.vertices -= mesh.centroid
    scale = mesh.bounding_sphere.primitive.radius
    if scale > 0:
        mesh.vertices /= scale

    # UVテクスチャ → 頂点カラーに変換（pyrender の OpenGL テクスチャエラーを回避）
    try:
        mesh.visual = mesh.visual.to_color()
        print("  テクスチャ → 頂点カラー変換 完了")
    except Exception as e:
        print(f"  頂点カラー変換スキップ ({e})")

    pr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)
    scene   = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3])
    scene.add(pr_mesh)  # メッシュは原点固定

    # ライト（カメラノード参照を保持）
    light      = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    light_node = scene.add(light, pose=np.eye(4))

    # カメラ（ノード参照を保持 ← ここが修正の核心）
    camera   = pyrender.PerspectiveCamera(yfov=np.radians(45.0))
    cam_dist = 2.5
    cam_node = scene.add(camera, pose=np.eye(4))

    renderer = pyrender.OffscreenRenderer(W, H)
    elev_rad = np.radians(elev_deg)

    for fi in range(total_frames):
        theta = (fi / total_frames) * 2.0 * np.pi

        # カメラ位置: 球面上をorbit（メッシュは原点固定、カメラが周回）
        cx = cam_dist * np.sin(theta) * np.cos(elev_rad)
        cy = cam_dist * np.sin(elev_rad)
        cz = cam_dist * np.cos(theta) * np.cos(elev_rad)

        # look-at: カメラ(cx,cy,cz) → 原点(0,0,0) を向く
        look  = np.array([-cx, -cy, -cz], dtype=float)
        look /= np.linalg.norm(look)
        world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(look, world_up)
        right /= np.linalg.norm(right)
        cam_up = np.cross(right, look)

        # pyrender カメラpose: 列0=right, 列1=up, 列2=-look(pyrender は-Z前方)
        cam_pose = np.eye(4)
        cam_pose[:3, 0] = right
        cam_pose[:3, 1] = cam_up
        cam_pose[:3, 2] = -look
        cam_pose[:3, 3] = [cx, cy, cz]

        # カメラノードのみ更新（メッシュは動かさない）
        scene.set_pose(cam_node, cam_pose)
        scene.set_pose(light_node, cam_pose)

        color, _ = renderer.render(scene)
        Image.fromarray(color).save(str(frames_dir / f"frame_{fi:05d}.png"))

        if fi % (fps * 2) == 0:
            print(f"  f{fi:04d}/{total_frames}  {theta*180/np.pi:.0f}deg")

    renderer.delete()
    return True


# ─── レンダラー: trimesh + numpy rasterizer（フォールバック）───────────────────

def _project_vertices(verts, theta, elev, cam_dist, f, cx, cy):
    """
    verts: (N,3)
    return: screen_uv (N,2), depth (N,)
    """
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    cos_e, sin_e = np.cos(elev),  np.sin(elev)

    # Y軸回転
    X  = verts[:, 0]; Y = verts[:, 1]; Z = verts[:, 2]
    X1 =  X * cos_t + Z * sin_t
    Z1 = -X * sin_t + Z * cos_t

    # X軸回転（仰角）
    Y2 =  Y * cos_e - Z1 * sin_e
    Z2 =  Y * sin_e + Z1 * cos_e

    # カメラ空間: camera at Z = cam_dist, looking in -Z
    depth = cam_dist - Z2
    valid_depth = np.where(depth > 1e-4, depth, 1e-4)

    u = X1 / valid_depth * f + cx
    v = Y2 / valid_depth * f + cy

    return np.stack([u, v], axis=1), depth


def _rasterize_triangles(faces, verts_screen, depths, vert_colors, W, H):
    """
    Numpy ソフトウェアラスタライザ（Zバッファ + 三角形塗りつぶし）
    """
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    zbuf   = np.full((H, W), np.inf, dtype=np.float32)

    # 面をZ順（遠→近）でソート
    face_z = depths[faces].mean(axis=1)
    order  = np.argsort(face_z)[::-1]

    for idx in order:
        f  = faces[idx]
        p0 = verts_screen[f[0]]; p1 = verts_screen[f[1]]; p2 = verts_screen[f[2]]
        d0 = depths[f[0]];       d1 = depths[f[1]];       d2 = depths[f[2]]
        c0 = vert_colors[f[0]];  c1 = vert_colors[f[1]];  c2 = vert_colors[f[2]]

        # バウンディングボックス
        x_min = max(0, int(min(p0[0], p1[0], p2[0])))
        x_max = min(W-1, int(max(p0[0], p1[0], p2[0])) + 1)
        y_min = max(0, int(min(p0[1], p1[1], p2[1])))
        y_max = min(H-1, int(max(p0[1], p1[1], p2[1])) + 1)

        if x_max < x_min or y_max < y_min:
            continue

        # ピクセルグリッド
        px, py = np.meshgrid(
            np.arange(x_min, x_max+1, dtype=np.float32),
            np.arange(y_min, y_max+1, dtype=np.float32))

        # バリセントリック座標
        v0x, v0y = p1[0]-p0[0], p1[1]-p0[1]
        v1x, v1y = p2[0]-p0[0], p2[1]-p0[1]
        v2x = px - p0[0]; v2y = py - p0[1]

        denom = v0x*v1y - v1x*v0y
        if abs(denom) < 1e-8:
            continue

        t = (v2x*v1y - v1x*v2y) / denom
        s = (v0x*v2y - v2x*v0y) / denom

        inside = (t >= 0) & (s >= 0) & ((t + s) <= 1.0)
        if not inside.any():
            continue

        # 深度補間 + Zバッファテスト
        depth_interp = d0 + t*(d1-d0) + s*(d2-d0)
        xi = px.astype(np.int32)[inside]
        yi = py.astype(np.int32)[inside]
        di = depth_interp[inside]

        ztest = di < zbuf[yi, xi]
        xi, yi, di = xi[ztest], yi[ztest], di[ztest]
        t_ok = t[inside][ztest]
        s_ok = s[inside][ztest]

        zbuf[yi, xi] = di.astype(np.float32)

        # カラー補間
        col = (c0 + t_ok[:, None]*(c1 - c0) + s_ok[:, None]*(c2 - c0))
        col = np.clip(col, 0, 255).astype(np.uint8)
        canvas[yi, xi] = col

    return canvas


def render_trimesh_fallback(mesh_path, frames_dir, total_frames,
                             W, H, fps, elev_deg):
    """trimesh + numpy rasterizer によるフォールバックレンダリング"""
    import trimesh

    print("  trimesh + numpy rasterizer でレンダリングします")

    tm = trimesh.load(str(mesh_path), force="mesh")
    if isinstance(tm, trimesh.Scene):
        tm = trimesh.util.concatenate(list(tm.geometry.values()))

    # 正規化
    tm.vertices -= tm.centroid
    r = tm.bounding_sphere.primitive.radius
    if r > 0:
        tm.vertices /= r

    # 頂点カラー取得
    try:
        vc = tm.visual.to_color().vertex_colors[:, :3]
    except Exception:
        vc = np.full((len(tm.vertices), 3), 180, dtype=np.uint8)

    faces = np.array(tm.faces)
    verts = np.array(tm.vertices, dtype=np.float32)
    vc    = vc.astype(np.float32)

    fov_deg  = 45.0
    cam_dist = 2.5
    f        = H / (2.0 * np.tan(np.radians(fov_deg / 2.0)))
    cx, cy   = W / 2.0, H / 2.0
    elev_rad = np.radians(elev_deg)

    print(f"  頂点数: {len(verts):,}  面数: {len(faces):,}")
    print(f"  ※ numpy rasterizer は低速です。TripoSR/Hunyuan3D後に pyrender を推奨。")

    for fi in range(total_frames):
        theta = (fi / total_frames) * 2.0 * np.pi
        screen_uv, depths = _project_vertices(
            verts, theta, elev_rad, cam_dist, f, cx, cy)

        frame = _rasterize_triangles(faces, screen_uv, depths, vc, W, H)
        Image.fromarray(frame).save(str(frames_dir / f"frame_{fi:05d}.png"))

        if fi % (fps * 2) == 0:
            print(f"  f{fi:04d}/{total_frames}  {theta*180/np.pi:.0f}deg")


# ─── メイン ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="3Dメッシュ orbit レンダラー")
    parser.add_argument("--mesh", type=str, required=True,
                        help="入力メッシュ (OBJ/GLB/PLY)")
    parser.add_argument("--fps",      type=int,   default=24)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--width",    type=int,   default=1280)
    parser.add_argument("--height",   type=int,   default=720)
    parser.add_argument("--elev",     type=float, default=20.0,
                        help="仰角 deg (default 20)")
    parser.add_argument("--output",   type=str,   default=None,
                        help="出力MP4パス (省略時は自動)")
    parser.add_argument("--reuse",    action="store_true")
    args = parser.parse_args()

    mesh_path    = Path(args.mesh)
    total_frames = int(args.fps * args.duration)
    W, H         = args.width, args.height
    video_path   = Path(args.output) if args.output else \
                   OUTPUT_DIR / f"{mesh_path.stem}_orbit.mp4"

    print("=" * 62)
    print("  render_orbit_from_mesh.py")
    print(f"  入力メッシュ: {mesh_path.name}")
    print(f"  解像度: {W}x{H}  |  {args.fps}fps  |  {args.duration}s")
    print(f"  仰角: {args.elev}deg")
    print("=" * 62)

    if not mesh_path.exists():
        print(f"エラー: メッシュファイルが見つかりません: {mesh_path}")
        print()
        print("  3Dメッシュの生成方法:")
        print("  [A] ComfyUI Manager -> 'ComfyUI-Hunyuan3D' をインストール")
        print("      -> stage_01.png から GLBメッシュを生成")
        print("  [B] ComfyUI Manager -> 'ComfyUI-TripoSR' をインストール")
        print("      -> stage_01.png から OBJメッシュを生成")
        sys.exit(1)

    ffmpeg = get_ffmpeg()
    frames_dir = FRAMES_DIR
    frames_dir.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not (args.reuse and (frames_dir / f"frame_{total_frames-1:05d}.png").exists()):
        # pyrender を試みる → 失敗したら trimesh fallback
        success = try_render_pyrender(
            mesh_path, frames_dir, total_frames, W, H, args.fps, args.elev)
        if not success:
            print("  pyrender が見つかりません。")
            print("  高品質レンダリングを使いたい場合:")
            print("    pip install pyrender")
            print("  -> trimesh numpy rasterizer で続行します...\n")
            render_trimesh_fallback(
                mesh_path, frames_dir, total_frames, W, H, args.fps, args.elev)

    print(f"\n[MP4生成] {video_path}")
    cmd = [ffmpeg, "-y",
           "-framerate", str(args.fps),
           "-i", str(frames_dir / "frame_%05d.png"),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
           str(video_path)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 0:
        print(f"完了: {video_path}")
    else:
        print(f"エラー: {r.stderr[-200:]}")

    print(f"\n{'=' * 62}")
    print(f"  出力: {video_path}")
    print(f"\n  次のステップ（品質向上）:")
    print(f"    pip install pyrender   ← 高品質レンダリングを有効化")
    print(f"    ComfyUI-Hunyuan3D      ← 最高品質の3Dメッシュ生成")
    print(f"{'=' * 62}")


if __name__ == "__main__":
    main()

# NanoBanana 4D - 3D Orbit Camera (corrected)
#
# 設計:
#   - 卵(前景)の3D重心を回転軸に設定
#   - 背景は静止 (静止背景の上に前景点群を描画)
#   - カメラが卵の周囲を周回し、常に卵を画面中央に捉える
#   - 穴はぼかした背景で補完
#
# 実行:
#   & $PY run_3d_orbit.py
#   & $PY run_3d_orbit.py --midas --duration 12 --elev 20

import argparse
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

INPUT_IMG  = Path(r"C:\Users\fareg\Desktop\老後AI\work_folder\老後画像AI_シリーズ\015_comfyui_nanobanana\making_movie\Celastrina _argiolus\9stages\stage_01.png")
OUTPUT_DIR = Path(r"D:\NB4D_test\orbit_3d")
FRAMES_DIR = OUTPUT_DIR / "frames"
VIDEO_PATH = OUTPUT_DIR / "stage01_3d_orbit.mp4"

RENDER_W = 1280
RENDER_H = 720

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


# ─── 深度推定 ──────────────────────────────────────────────────────────────────
def estimate_depth_midas(img_pil):
    try:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  MiDaS_small ({device}) ...")
        midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
        midas = midas.to(device).eval()
        tfms  = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        img_np = np.array(img_pil.convert("RGB"))
        batch  = tfms.small_transform(img_np).to(device)
        with torch.no_grad():
            pred = midas(batch)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1), size=img_np.shape[:2],
                mode="bicubic", align_corners=False).squeeze()
        d = pred.cpu().numpy().astype(np.float32)
        d = (d - d.min()) / (d.max() - d.min() + 1e-8)
        print("  MiDaS 完了")
        return d
    except Exception as e:
        print(f"  MiDaS 失敗 ({e}) -> 合成深度")
        return estimate_depth_synthetic(img_pil)


def estimate_depth_synthetic(img_pil):
    H, W   = img_pil.height, img_pil.width
    img_np = np.array(img_pil.convert("RGB")).astype(np.float32) / 255.0
    gray   = 0.299*img_np[:,:,0] + 0.587*img_np[:,:,1] + 0.114*img_np[:,:,2]
    y, x   = np.mgrid[0:H, 0:W].astype(np.float32)
    dx     = (x - W/2) / (W/2 + 1e-8)
    dy     = (y - H/2) / (H/2 + 1e-8)
    cw     = np.clip(1.0 - np.sqrt(dx**2 + dy**2) * 0.85, 0.0, 1.0)
    depth  = 0.35*(1.0 - gray) + 0.65*cw
    depth  = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    blur_r = max(H, W) // 50
    dp = Image.fromarray((depth*255).astype(np.uint8))
    dp = dp.filter(ImageFilter.GaussianBlur(radius=blur_r))
    return np.array(dp).astype(np.float32) / 255.0


# ─── 3D 軌道カメラレンダラー ───────────────────────────────────────────────────
def build_scene(img_np, depth_np, fov_deg, fg_thresh):
    """
    前景点群(fg)と背景画像(bg)を分離して返す。
    前景の3D重心を (0,0,0) に平行移動済み。
    orbit_R: カメラと重心の距離 (=元の重心Z距離)
    """
    H, W = img_np.shape[:2]
    f    = W / (2.0 * np.tan(np.radians(fov_deg / 2.0)))
    cx, cy = W / 2.0, H / 2.0

    v_src, u_src = np.mgrid[0:H, 0:W].astype(np.float32)

    # Z: depth値が高い(=前景)ほど Z が小さい（カメラに近い）
    Z_NEAR, Z_FAR = 1.0, 5.0
    Z = Z_NEAR + (1.0 - depth_np) * (Z_FAR - Z_NEAR)

    X = (u_src - cx) / f * Z
    Y = (v_src - cy) / f * Z

    all_points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype(np.float32)
    all_colors = img_np.reshape(-1, 3)

    # 前景マスク（depth が fg_thresh 以上 = 手前にある点）
    fg_mask = depth_np.ravel() >= fg_thresh
    fg_pts  = all_points[fg_mask]
    fg_col  = all_colors[fg_mask]

    # 前景の重心 (orbit の中心)
    if len(fg_pts) == 0:
        print("  [警告] 前景点が見つかりません。fg_thresh を下げてください。")
        pivot = all_points.mean(axis=0)
    else:
        pivot = fg_pts.mean(axis=0)

    print(f"  前景点数: {len(fg_pts):,} / {len(all_points):,}")
    print(f"  pivot (orbit中心): X={pivot[0]:.3f}  Y={pivot[1]:.3f}  Z={pivot[2]:.3f}")

    # 前景点を pivot 中心に平行移動
    fg_pts_c = fg_pts - pivot  # pivotが原点に

    # orbit radius = pivot の元のZ距離（カメラ→重心）
    orbit_R = float(pivot[2])
    print(f"  orbit radius R = {orbit_R:.3f}")

    return fg_pts_c, fg_col, f, cx, cy, orbit_R, pivot


def render_frame(fg_pts_c, fg_col, theta, elev,
                 f, cx, cy, W, H, orbit_R, bg_np, splat_r):
    """
    fg_pts_c : (N,3) 前景点群。pivotが原点。
    theta    : 水平回転角 [rad]  0 -> 2pi
    elev     : 仰角 [rad]
    orbit_R  : カメラと原点(pivot)の距離
    bg_np    : (H,W,3) uint8 静止背景
    """
    X, Y, Z = fg_pts_c[:,0], fg_pts_c[:,1], fg_pts_c[:,2]

    # ── ワールドをY軸周りに回転（= カメラが水平軌道を移動するのと等価）─────
    cos_t, sin_t = float(np.cos(theta)), float(np.sin(theta))
    X1 =  X * cos_t + Z * sin_t
    Y1 =  Y
    Z1 = -X * sin_t + Z * cos_t

    # ── X軸周りに回転（仰角）──────────────────────────────────────────────
    cos_e, sin_e = float(np.cos(elev)), float(np.sin(elev))
    X2 =  X1
    Y2 =  Y1 * cos_e - Z1 * sin_e
    Z2 =  Y1 * sin_e + Z1 * cos_e

    # ── カメラはZ=orbit_R の位置から -Z 方向を向いている ─────────────────
    # カメラ空間の深度 = orbit_R - Z2  (正 = カメラ前方)
    depth_cam = orbit_R - Z2
    valid = depth_cam > 0.05
    X2v, Y2v, dv, colv = X2[valid], Y2[valid], depth_cam[valid], fg_col[valid]

    # 透視投影 → スクリーン座標
    u = (X2v / dv * f + cx)
    v = (Y2v / dv * f + cy)

    # 遠 -> 近 でソート（painter's algorithm）
    order = np.argsort(dv)[::-1]
    u, v, dv_s, colv = u[order], v[order], dv[order], colv[order]

    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)

    # ── 静止背景をベースキャンバスとして使用 ─────────────────────────────
    canvas = bg_np.copy()
    zbuf   = np.full((H, W), np.inf, dtype=np.float32)

    # スプラット描画（前景のみ）
    for dr in range(-splat_r, splat_r + 1):
        for dc in range(-splat_r, splat_r + 1):
            uc = np.clip(ui + dc, 0, W - 1)
            vc = np.clip(vi + dr, 0, H - 1)
            mask = dv_s < zbuf[vc, uc]
            zbuf[vc[mask], uc[mask]]   = dv_s[mask]
            canvas[vc[mask], uc[mask]] = colv[mask]

    # ── 前景書き込み領域の境界をなじませる ──────────────────────────────
    try:
        import cv2
        fg_painted = (zbuf < np.inf).astype(np.uint8) * 255
        # 前景エッジを少し広げた領域でブレンド
        kernel = np.ones((5, 5), np.uint8)
        edge   = cv2.dilate(fg_painted, kernel) ^ fg_painted
        if edge.any():
            blurred = cv2.GaussianBlur(canvas, (5, 5), 1)
            edge_m  = (edge[:,:,np.newaxis] / 255.0)
            canvas  = (canvas * (1 - edge_m) + blurred * edge_m).astype(np.uint8)
    except ImportError:
        pass

    return canvas


# ─── メイン ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="3D Orbit Camera - stage01")
    parser.add_argument("--midas",    action="store_true")
    parser.add_argument("--fps",      type=int,   default=24)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--fov",      type=float, default=55.0)
    parser.add_argument("--elev",     type=float, default=15.0,
                        help="仰角 deg: カメラが卵を見下ろす角度")
    parser.add_argument("--splat",    type=int,   default=2)
    parser.add_argument("--fg-thresh",type=float, default=0.45,
                        help="前景/背景分離の深度閾値 0-1 (default 0.45)")
    parser.add_argument("--reuse",    action="store_true")
    args = parser.parse_args()

    total_frames = int(args.fps * args.duration)
    elev_rad     = np.radians(args.elev)

    print("=" * 62)
    print("  NanoBanana 4D - 3D Orbit Camera (corrected)")
    print(f"  {args.fps}fps x {args.duration}s = {total_frames}f")
    print(f"  FOV={args.fov}deg  仰角={args.elev}deg  前景閾値={args.fg_thresh}")
    print("  設計: 卵の重心を回転軸 / 背景は静止 / 前景のみ3D orbit")
    print("=" * 62)

    ffmpeg = get_ffmpeg()
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: 入力画像
    print(f"\n[Step 1] 入力画像: {INPUT_IMG.name}")
    if not INPUT_IMG.exists():
        print(f"  エラー: {INPUT_IMG}")
        return
    img_pil = Image.open(str(INPUT_IMG)).convert("RGB").resize(
        (RENDER_W, RENDER_H), Image.LANCZOS)
    img_np  = np.array(img_pil, dtype=np.uint8)

    # 静止背景 = ソース画像をブラーしたもの
    bg_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=20))
    bg_np  = np.array(bg_pil, dtype=np.uint8)
    bg_pil.save(str(OUTPUT_DIR / "background.png"))
    print(f"  背景保存: background.png (静止用ぼかし背景)")

    # Step 2: 深度推定
    print(f"\n[Step 2] 深度推定")
    depth_cache = OUTPUT_DIR / "depth.npy"
    if args.reuse and depth_cache.exists():
        depth_np = np.load(str(depth_cache))
        print(f"  キャッシュ使用")
    else:
        depth_np = (estimate_depth_midas(img_pil) if args.midas
                    else estimate_depth_synthetic(img_pil))
        np.save(str(depth_cache), depth_np)
        Image.fromarray((depth_np * 255).astype(np.uint8)).save(
            str(OUTPUT_DIR / "depth_vis.png"))
        print(f"  保存: depth_vis.png")

    # Step 3: シーン構築（前景/背景分離 + pivot決定）
    print(f"\n[Step 3] シーン構築 (前景閾値={args.fg_thresh})")
    fg_pts_c, fg_col, f_focal, cx, cy, orbit_R, pivot = build_scene(
        img_np, depth_np, args.fov, args.fg_thresh)

    # Step 4: フレームレンダリング
    print(f"\n[Step 4] フレームレンダリング ({total_frames}f)")
    print(f"  カメラ軌道: 半径={orbit_R:.2f}  pivot中心の360度周回")

    for fi in range(total_frames):
        frame_path = FRAMES_DIR / f"frame_{fi:05d}.png"
        if args.reuse and frame_path.exists():
            continue

        theta = (fi / total_frames) * 2.0 * np.pi  # 0 → 360deg

        frame = render_frame(
            fg_pts_c, fg_col,
            theta=theta, elev=elev_rad,
            f=f_focal, cx=cx, cy=cy,
            W=RENDER_W, H=RENDER_H,
            orbit_R=orbit_R,
            bg_np=bg_np,
            splat_r=args.splat,
        )
        Image.fromarray(frame).save(str(frame_path))

        if fi % (args.fps * 2) == 0 or fi == total_frames - 1:
            deg = theta * 180 / np.pi
            print(f"  f{fi:04d}/{total_frames}  {deg:6.1f}deg  ({(fi+1)/total_frames*100:.0f}%)")

    # Step 5: MP4
    print(f"\n[Step 5] MP4生成")
    cmd = [ffmpeg, "-y",
           "-framerate", str(args.fps),
           "-i", str(FRAMES_DIR / "frame_%05d.png"),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
           str(VIDEO_PATH)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  エラー: {r.stderr[-200:]}")
    else:
        print(f"  完了: {VIDEO_PATH}")

    print(f"\n{'=' * 62}")
    print(f"  前景点数: {len(fg_pts_c):,}")
    print(f"  orbit R:  {orbit_R:.3f}")
    print(f"  出力: {VIDEO_PATH}")
    print()
    print(f"  うまくいかない場合のチューニング:")
    print(f"    --fg-thresh 0.3   前景をより広く取る（現在 {args.fg_thresh}）")
    print(f"    --midas           精度の高い深度マップ")
    print(f"    --elev 25         より上から見下ろす")
    print(f"    --splat 3         スプラット半径を大きく（穴が減る）")
    print(f"{'=' * 62}")


if __name__ == "__main__":
    main()

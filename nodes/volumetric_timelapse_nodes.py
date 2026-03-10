"""
NanoBanana 4D - 立体タイムラプス専用ノード
==========================================
時間軸(Time)と空間軸(Camera)を制御し、一貫性のある画像連番を生成するための補助ノード群。
"""

import json
import math
from typing import Any, Dict, List, Tuple
import torch

# ── ライフステージのデフォルト（ルリシジミ用） ───────────────────────────
DEFAULT_STAGES = [
    {"t": 0.00, "stage": "egg",         "desc": "A tiny, pale-green, lens-shaped egg of Celastrina argiolus (Holly Blue butterfly) laid on a flower bud of a host plant, extreme macro photography"},
    {"t": 0.15, "stage": "larva_early", "desc": "A very small, slug-like pale green caterpillar of Celastrina argiolus, camouflaged on a young flower bud"},
    {"t": 0.35, "stage": "larva_late",  "desc": "A mature slug-like green caterpillar of Celastrina argiolus (Pale Grass Blue), smooth textured, resting on a green leaf"},
    {"t": 0.50, "stage": "pupa",        "desc": "A small, rounded, brownish-grey pupa of Celastrina argiolus, attached to a stem with a silk girdle, macro photography"},
    {"t": 0.65, "stage": "eclosion",    "desc": "A small blue butterfly, Celastrina argiolus, emerging from its greyish pupa. The wings are wet and small, showing soft silvery-blue underside with black spots. No orange, no black veins like monarch."},
    {"t": 0.80, "stage": "adult_open",  "desc": "A beautiful Holly Blue butterfly (Celastrina argiolus) with vibrant sky-blue wings fully open, delicate black borders, resting on a flower"},
    {"t": 1.00, "stage": "flying",      "desc": "A small sky-blue butterfly, Celastrina argiolus, fluttering around green foliage in soft sunlight"}
]

# ── 共通ヘルパー ─────────────────────────────────────────────────────────────

def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t

# ── ノード実装 ──────────────────────────────────────────────────────────────

class NB4D_TimePlanner:
    """時間軸(t=0.0-1.0)に基づいて、ライフステージのプロンプトを補間・生成する"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "t": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "frame_index": ("INT", {"default": 0, "min": 0, "max": 10000, "forceInput": True}),
                "total_frames": ("INT", {"default": 100, "min": 1, "max": 10000, "forceInput": True}),
            },
            "optional": {
                "stages_json": ("STRING", {"multiline": True, "default": json.dumps(DEFAULT_STAGES, indent=2)}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "FLOAT")
    RETURN_NAMES = ("stage_prompt", "stage_name", "t_out")
    FUNCTION = "plan"
    CATEGORY = "RogoAI/4D"

    def plan(self, t, frame_index, total_frames, stages_json=""):
        # frame_index/total_frames が提供されている場合は、tを自動計算（優先）
        # frame_indexが0より大きい場合のみ自動計算を有効にする（スライダー優先モードとの切り替え）
        if total_frames > 1 and frame_index > 0:
            current_t = frame_index / (total_frames - 1)
        else:
            current_t = t
            
        try:
            stages = json.loads(stages_json)
        except Exception:
            stages = DEFAULT_STAGES
            
        stages = sorted(stages, key=lambda x: x["t"])
        
        # 前後のステージを特定
        prev_s = stages[0]
        next_s = stages[-1]
        
        for i in range(len(stages) - 1):
            if stages[i]["t"] <= current_t <= stages[i+1]["t"]:
                prev_s = stages[i]
                next_s = stages[i+1]
                break
        
        if prev_s == next_s:
            return (prev_s["desc"], prev_s["stage"], current_t)
            
        # ステージ間の重みを計算
        local_t = (current_t - prev_s["t"]) / (next_s["t"] - prev_s["t"])
        
        # 簡易プロンプト合成
        combined_prompt = f"Between {prev_s['desc']} and {next_s['desc']}, focusing on {next_s['stage']} stage"
        if local_t < 0.3:
            combined_prompt = prev_s["desc"]
        elif local_t > 0.7:
            combined_prompt = next_s["desc"]
            
        return (combined_prompt, f"{prev_s['stage']}_to_{next_s['stage']}", current_t)


class NB4D_CameraPath:
    """フレーム番号に基づいて球座標（水平角、垂直角、距離）を計算する"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frame_index": ("INT", {"default": 0, "min": 0, "max": 10000, "forceInput": True}),
                "total_frames": ("INT", {"default": 100, "min": 1, "max": 10000, "forceInput": True}),
                "path_type": (["spiral_ascend", "orbit_fixed", "static"], {"default": "spiral_ascend"}),
                "revolutions": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0}),
                "start_elevation": ("FLOAT", {"default": 0.0, "min": -90.0, "max": 90.0}),
                "end_elevation": ("FLOAT", {"default": 45.0, "min": -90.0, "max": 90.0}),
            }
        }

    RETURN_TYPES = ("FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("theta", "phi", "distance_desc")
    FUNCTION = "calculate_path"
    CATEGORY = "RogoAI/4D"

    def calculate_path(self, frame_index, total_frames, path_type, revolutions, start_elevation, end_elevation):
        t = frame_index / max(1, (total_frames - 1))
        
        theta = (t * 360 * revolutions) % 360
        phi = _lerp(start_elevation, end_elevation, t)
        
        dist_desc = "medium shot"
        if t < 0.2:
            dist_desc = "extreme close up"
        elif t < 0.5:
            dist_desc = "close up"
            
        return (theta, phi, dist_desc)


class NB4D_PromptComposer:
    """時間軸プロンプトとカメラ情報を統合して最終プロンプトを作成する"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stage_prompt": ("STRING", {"forceInput": True}),
                "theta": ("FLOAT", {"default": 0.0}),
                "phi": ("FLOAT", {"default": 0.0}),
                "distance_desc": ("STRING", {"default": "medium shot"}),
                "subject_name": ("STRING", {"default": "butterfly"}),
                "style_nodes": ("STRING", {"default": "photorealistic, 8k, highly detailed, nature documentary"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("final_prompt",)
    FUNCTION = "compose"
    CATEGORY = "RogoAI/4D"

    def compose(self, stage_prompt, theta, phi, distance_desc, subject_name, style_nodes):
        # 角度を言葉に変換
        lr = "center"
        if 15 < theta <= 165: lr = "right side"
        elif 195 < theta <= 345: lr = "left side"
        
        ud = "eye level"
        if phi > 20: ud = "high angle (looking down)"
        elif phi < -20: ud = "low angle (looking up)"
        
        final = (
            f"{stage_prompt}, "
            f"seen from the {lr} at a {ud}, "
            f"{distance_desc}, "
            f"{style_nodes}"
        )
        return (final,)


class NB4D_ConsistencyEngine:
    """フレーム間の一貫性を維持するための参照画像管理ノード"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "strategy": (["anchor_based", "sequential", "hybrid"], {"default": "hybrid"}),
                "frame_index": ("INT", {"default": 0, "min": 0, "max": 10000, "forceInput": True}),
                "total_frames": ("INT", {"default": 100, "min": 1, "max": 10000, "forceInput": True}),
            },
            "optional": {
                "anchor_images": ("IMAGE",), # ライフステージごとの基準画像バッチ
                "previous_frame": ("IMAGE",), # 1つ前の生成結果
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("ref_image_1", "ref_image_2", "ref_image_3")
    FUNCTION = "manage_refs"
    CATEGORY = "RogoAI/4D"

    def manage_refs(self, strategy, frame_index, total_frames, anchor_images=None, previous_frame=None):
        # ref_image_1: 直前フレーム（時間的一貫性）
        # ref_image_2: アンカー画像（形状・個体一貫性）
        # ref_image_3: 固定基準（スタイル・絶対一貫性）
        
        ref1 = previous_frame if previous_frame is not None else None
        
        ref2 = None
        if anchor_images is not None:
            # frame_indexに応じて適切なアンカーを選択
            num_anchors = anchor_images.shape[0]
            anchor_idx = min(int((frame_index / total_frames) * num_anchors), num_anchors - 1)
            ref2 = anchor_images[anchor_idx : anchor_idx + 1]
            
        ref3 = None
        if anchor_images is not None:
            # 最初のアンカーを固定基準として使用
            ref3 = anchor_images[0:1]
            
        return (ref1, ref2, ref3)


class NB4D_VideoAssembler:
    """連番画像をFFmpegでmp4動画に変換するノード"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "fps": ("INT", {"default": 30, "min": 1, "max": 60}),
                "filename_prefix": ("STRING", {"default": "nb4d_timelapse"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "assemble"
    OUTPUT_NODE = True
    CATEGORY = "RogoAI/4D"

    def assemble(self, images, fps, filename_prefix):
        import os
        import subprocess
        import folder_paths
        import numpy as np
        from PIL import Image
        import ctypes
        
        def get_short_path(path):
            """Windowsの日本語パス問題を回避するために、8.3形式の短縮パスを取得する"""
            if os.name != 'nt': return path
            try:
                # 入力パスがパターンの場合は親ディレクトリを短縮する
                if '%' in path:
                    dirname = os.path.dirname(path)
                    basename = os.path.basename(path)
                    buf = ctypes.create_unicode_buffer(1024)
                    ctypes.windll.kernel32.GetShortPathNameW(dirname, buf, 1024)
                    return os.path.join(buf.value, basename)
                
                buf = ctypes.create_unicode_buffer(1024)
                ctypes.windll.kernel32.GetShortPathNameW(path, buf, 1024)
                if not buf.value: # ファイルが存在しない場合などは、ディレクトリだけでも短縮を試みる
                    dirname = os.path.dirname(path)
                    basename = os.path.basename(path)
                    ctypes.windll.kernel32.GetShortPathNameW(dirname, buf, 1024)
                    return os.path.join(buf.value, basename)
                return buf.value
            except:
                return path
        
        def get_ffmpeg_path():
            """ffmpegの実行パスを特定する"""
            candidates = [
                r"C:\Users\fareg\AppData\Local\FFmpeg\bin\ffmpeg.exe",
                "ffmpeg", # PATHにある場合
            ]
            
            for path in candidates:
                try:
                    # check_output等で実行可能か確認
                    subprocess.run([path, "-version"], capture_output=True, check=True)
                    return path
                except:
                    continue
            return None

        output_dir = folder_paths.get_output_directory()
        temp_dir = os.path.join(output_dir, "nb4d_temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        # 清掃
        for f in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, f))
            
        # 画像保存
        for i, img_tensor in enumerate(images):
            # imagesがバッチ(Batch, H, W, C)であることを想定
            if len(img_tensor.shape) == 3: # (H, W, C)
                img_np = 255. * img_tensor.cpu().numpy()
            else: # (1, H, W, C) のようなケース
                img_np = 255. * img_tensor[0].cpu().numpy()
                
            img = Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))
            img.save(os.path.join(temp_dir, f"frame_{i:05d}.png"))
            
        video_path = os.path.join(output_dir, f"{filename_prefix}.mp4")
        
        ffmpeg_cmd = get_ffmpeg_path()
        if not ffmpeg_cmd:
            tried_path = r'C:\Users\fareg\AppData\Local\FFmpeg\bin\ffmpeg.exe'
            return (f"Error: ffmpeg not found. Please ensure it is installed and accessible. Tried: {tried_path} and system PATH.",)

        # パスの短縮化（日本語パス対策）
        short_input = get_short_path(os.path.join(temp_dir, 'frame_%05d.png'))
        short_output = get_short_path(video_path)
        
        # FFmpeg 実行
        cmd = [
            ffmpeg_cmd, '-y',
            '-framerate', str(fps),
            '-i', short_input,
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '18',
            short_output
        ]
        
        try:
            # 実行時のカレントディレクトリも念のため移動
            subprocess.run(cmd, check=True, capture_output=True, cwd=output_dir)
            return (video_path,)
        except Exception as e:
            return (f"Error execution ffmpeg: {str(e)}",)


# ── ノード登録用 ─────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "NB4D_TimePlanner": NB4D_TimePlanner,
    "NB4D_CameraPath": NB4D_CameraPath,
    "NB4D_PromptComposer": NB4D_PromptComposer,
    "NB4D_ConsistencyEngine": NB4D_ConsistencyEngine,
    "NB4D_VideoAssembler": NB4D_VideoAssembler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NB4D_TimePlanner": "NB4D Time Planner (時間軸制御)",
    "NB4D_CameraPath": "NB4D Camera Path (空間軸制御)",
    "NB4D_PromptComposer": "NB4D Prompt Composer (合成)",
    "NB4D_ConsistencyEngine": "NB4D Consistency Engine (一貫性維持)",
    "NB4D_VideoAssembler": "NB4D Video Assembler (動画生成)",
}

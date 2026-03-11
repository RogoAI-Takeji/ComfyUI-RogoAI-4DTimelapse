# ComfyUI-RogoAI-4DTimelapse

**4D Timelapse video generation system for ComfyUI**

Traverse **(Time × Horizontal Angle × Elevation)** space using LTX-Video 2 AI interpolation.  
Generate cinematic videos that move freely through both time and 3D viewpoint simultaneously.

---

## What is 4D Timelapse?

Traditional timelapse: camera fixed, subject changes over time.  
4D Timelapse: camera moves freely in 3D space **while** the subject evolves through time.

```
Axes:
  T (time)      — subject lifecycle  e.g. egg → larva → pupa → adult
  H (horizontal)— camera angle       0° (front) → 360° (full orbit)
  V (elevation) — camera elevation   -30° (low) → +60° (bird's eye)
```

Define a path through this **(T, H, V)** lattice using waypoints, and LTX-Video 2 interpolates each segment into a smooth video clip.

---

## Features

| Node | Description |
|------|-------------|
| `Grid4DRenderKeyframes` | GLB folder → keyframe grid (T × H × V) using pyrender |
| `Grid4DTraverse` | Keyframe grid → MP4 via 23 path presets + optical flow |
| `Grid4DComposite` | 4D traversal + background image compositing |
| `NB4D_LTXVStageInterpolatorV3` | LTX-2 AI interpolation with sweep path support |
| `NB4D_LTXVPathNavigator` | **Free traversal in (T, H, V) spacetime lattice** ← flagship |
| `GeminiImageGenerator` | Google Gemini / Imagen 3 image generation |

---

## Installation

### 1. Clone into ComfyUI custom_nodes

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/RogoAI-Takeji/ComfyUI-RogoAI-4DTimelapse.git
```

### 2. Install dependencies

```bash
pip install -r ComfyUI/custom_nodes/ComfyUI-RogoAI-4DTimelapse/requirements.txt
```

Or with ComfyUI portable:

```bash
python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI-RogoAI-4DTimelapse\requirements.txt
```

### 3. Restart ComfyUI

---

## Requirements

- Python 3.10+
- PyTorch 2.x + CUDA
- LTX-Video 2 models (for AI interpolation nodes)
- Google Gemini API key (for GeminiImageGenerator only)

### LTX-Video 2 model files

Place in `ComfyUI/models/`:

```
models/
  ltx2/ltx-2-19b-distilled_Q6_K.gguf
  ltx2/LTX2_video_vae_bf16.safetensors
  ltx2/gemma_3_12B_it_fp4_mixed.safetensors
  ltx2/ltx-2-19b-embeddings_connector_distill_bf16.safetensors
```

---

## Node Reference

### Grid4DRenderKeyframes

Renders a folder of GLB files into a keyframe grid.

| Parameter | Description |
|-----------|-------------|
| `glb_dir` | Folder containing stage_00.glb, stage_01.glb, ... |
| `output_dir` | Output directory for rendered images |
| `grid_theta` | Horizontal divisions (e.g. 72 = 5° per step) |
| `grid_elev` | Elevation levels (7 = -30° to +60°) |
| `width` / `height` | Render resolution |

Output structure: `stage_XX/elev_YY/angle_ZZZ.png` + `grid_meta.json`

---

### NB4D_LTXVPathNavigator

Flagship node. Define a path through **(T, H, V)** spacetime and generate AI-interpolated video.

| Parameter | Description |
|-----------|-------------|
| `grid_dir` | Output directory from Grid4DRenderKeyframes |
| `path_preset` | Built-in path preset |
| `path_custom` | Custom waypoints as text (overrides preset) |
| `frames_per_clip` | Frames per interpolated segment |
| `use_end_frame` | Enable end-frame guidance for better quality |

**path_custom format:**
```
# t, h, v  (integer indices)
0, 0, 2    # stage0, front, horizontal
2, 9, 3    # stage2, 45°, +15° elevation
4, 18, 4   # stage4, 90°, +30°
```

**Built-in presets:**

| Preset | Description |
|--------|-------------|
| `orbit_flat` | Horizontal orbit while time progresses |
| `spiral_ascend` | Spiral upward (right orbit + rising elevation) |
| `spiral_descend` | Spiral downward |
| `top_survey` | Bird's eye descent |
| `pendulum_elevation` | Elevation pendulum |
| `time_loop` | Time progression with orbit |
| `ground_to_sky` | Low angle rising to bird's eye |

---

### GeminiImageGenerator

Generate images using Google Gemini or Imagen 3 API.

| Parameter | Description |
|-----------|-------------|
| `api_key` | Gemini API key (auto-saved, masked display) |
| `model` | `gemini-2.0-flash-exp` / `imagen-3.0-generate-002` / etc. |
| `prompt` | Image generation prompt |
| `reference_image` | Optional style/subject reference |

---

## Workflow Examples

### workflow_4d_youtube_02.json

Step 1 (Grid4DRenderKeyframes, grid_elev=7) → Step 3 (PathNavigator)

Load from `workflows/workflow_4d_youtube_02.json`

---

## Coordinate System

```
V (elevation)
↑
6  +60°  bird's eye
5  +45°
4  +30°
3  +15°
2   0°   horizontal ← default
1  -15°
0  -30°  low angle
└──────────────────── T (time)
   0  1  2  3  4  5  6  7  8

H (horizontal, top view):
        h=0  (0°, front)
           ↑
  h=54 ←  ● → h=18   (● = subject)
 (270°)    ↓   (90°)
        h=36 (180°, rear)

grid_theta=72: 1 step = 5°
```

---

## Technical Notes

### Adjacency recommendation

For best interpolation quality, keep each waypoint step within:
- `|ΔT| ≤ 1`
- `|ΔH| ≤ 1`  
- `|ΔV| ≤ 1`

Larger jumps work but may reduce coherence between clips.

### ComfyUI deadlock (solved in V2+)

The `prompt_worker` thread in ComfyUI is single-threaded.  
Queueing internal jobs via API from within a node causes permanent deadlock.  
**V2+ uses direct Python class instantiation to avoid this.**

### Relative imports required

```python
# ✅ correct
from ._nb4d_paths import SWEEP_PATHS

# ❌ wrong — fails in ComfyUI
from nodes._nb4d_paths import SWEEP_PATHS
```

---

## License

MIT License — see [LICENSE](LICENSE)

---

## Tutorial

**▶ [【1枚の写真→猫の一生】AIで4Dタイムラプス作ってみた | NanoBanana2 ComfyUI](https://youtu.be/TYXX7ctKG4M)**

Step-by-step walkthrough: single photo → 4D timelapse using this node package.

---

## Credits

Developed by [RogoAI / たけ爺](https://www.youtube.com/@RogoAI)

Built on top of:
- [LTX-Video 2](https://github.com/Lightricks/LTX-Video) by Lightricks
- [Google Gemini API](https://ai.google.dev/)
- [pyrender](https://github.com/mmatl/pyrender)

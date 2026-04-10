"""
RunPod Serverless Handler for Hunyuan3D-Omni (shape) + Paint-v2-1 (texture).

Omni adds controllable generation: bbox, pose, point cloud, voxel.
Paint-v2-1 adds PBR textures (albedo, normal, metallic/roughness).

Input:
  image: base64-encoded image (with or without data: prefix)
  seed: int (default 42)
  steps: int (default 30)
  guidance_scale: float (default 4.5)
  octree_resolution: int (default 512)
  texture: bool (default true)
  face_count: int (default 90000)
  # Omni control signals (optional, pick one)
  control_type: str (none|bbox|pose|point|voxel, default "none")
  bbox: [w, h, d] — bounding box proportions
  pose: [[x1,y1,z1,x2,y2,z2], ...] — bone segments
  point: [[x,y,z], ...] — point cloud
  voxel: [[x,y,z], ...] — surface samples (81920 points)
  # Post-processing
  remesh_only: bool (default false)
  roblox_postprocess: bool (default false)
  clothing_type: str (default "shirt")
  target_tris: int (default 4000)

Output:
  files: [{ filename, type, data (base64) }]
  meta: { faces, vertices, inference_time, ... }
  # If roblox_postprocess:
  roblox_fbx: { filename, data }
  roblox_preview_glb: { filename, data }
  mannequin_preview_glb: { filename, data }
  roblox_meta: { ... }
"""

import runpod
import base64
import json
import os
import subprocess
import tempfile
import time
import traceback

# Global model pipeline (loaded once, reused across requests)
shape_pipeline = None


def get_weights_dir():
    """Get or create the model weights directory.
    Prefers /workspace/weights (RunPod network volume, persists across cold starts).
    Falls back to /app/weights (container-local, re-downloads each cold start).
    """
    for base in ['/workspace/weights', '/app/weights']:
        os.makedirs(base, exist_ok=True)
        return base
    return '/app/weights'


def download_models():
    """Download Omni model weights if not already cached."""
    weights_dir = get_weights_dir()
    omni_dir = os.path.join(weights_dir, 'Hunyuan3D-Omni')

    # Check if already downloaded (look for a model file)
    if os.path.exists(os.path.join(omni_dir, 'model', 'pytorch_model.bin')):
        print(f"[omni] Model weights found at {omni_dir}")
        return omni_dir

    print(f"[omni] Downloading Hunyuan3D-Omni to {omni_dir}...")
    t0 = time.time()

    from huggingface_hub import snapshot_download
    snapshot_download(
        'tencent/Hunyuan3D-Omni',
        local_dir=omni_dir,
        ignore_patterns=['*_ema.bin'],  # Skip EMA weights to save space/time
    )

    print(f"[omni] Download complete in {time.time()-t0:.1f}s")
    return omni_dir


def load_models():
    """Load Hunyuan3D-Omni shape model. Called once on cold start."""
    global shape_pipeline

    if shape_pipeline is not None:
        return

    import sys
    sys.path.insert(0, '/app/omni')

    import torch

    # Download if needed (cached on network volume for subsequent cold starts)
    omni_dir = download_models()

    print("[omni] Loading Hunyuan3D-Omni into VRAM...")
    t0 = time.time()

    from hy3dshape.pipelines import Hunyuan3DOmniSiTFlowMatchingPipeline

    shape_pipeline = Hunyuan3DOmniSiTFlowMatchingPipeline.from_pretrained(omni_dir)

    print(f"[omni] Shape model loaded in {time.time()-t0:.1f}s (10GB VRAM)")


def handler(job):
    """Process a single image-to-3D generation request."""
    try:
        load_models()
    except Exception as e:
        return {"error": f"Model loading failed: {e}\n{traceback.format_exc()}"}

    inp = job.get("input", {})

    # Parse image
    image_b64 = inp.get("image", "")
    if not image_b64:
        return {"error": "No image provided"}

    # Strip data URI prefix if present
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]

    # Settings
    seed = inp.get("seed", 42)
    steps = inp.get("steps", 30)
    guidance_scale = inp.get("guidance_scale", 4.5)
    octree_resolution = inp.get("octree_resolution", 512)
    face_count = inp.get("face_count", 90000)
    control_type = inp.get("control_type", "none")

    work_dir = tempfile.mkdtemp(prefix="hunyuan3d_")
    result = {}

    try:
        # Save input image and load as numpy array
        image_path = os.path.join(work_dir, "input.png")
        with open(image_path, "wb") as f:
            f.write(base64.b64decode(image_b64))

        import torch
        import numpy as np
        import cv2

        # Load image as numpy array (Omni preprocessor expects ndarray, not file path)
        image_np = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image_np is None:
            # cv2 failed — try PIL as fallback
            from PIL import Image as PILImage
            pil_img = PILImage.open(image_path)
            image_np = np.array(pil_img)
            print(f"[omni] Loaded image via PIL: {image_np.shape}")
        else:
            # cv2 loads as BGR/BGRA — convert to RGB/RGBA
            if len(image_np.shape) == 3:
                if image_np.shape[2] == 4:
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGRA2RGBA)
                else:
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            print(f"[omni] Loaded image via cv2: {image_np.shape}")

        # ── Shape generation with Omni ──
        print(f"[omni] Generating shape (steps={steps}, seed={seed}, control={control_type})...")
        t0 = time.time()

        generator = torch.Generator('cuda').manual_seed(seed)

        # Build control kwargs — Omni REQUIRES at least one control signal
        # Default to bbox [1,1,1] (cube proportions) when none specified
        control_kwargs = {}
        if control_type == "bbox" and inp.get("bbox"):
            bbox = torch.FloatTensor(inp["bbox"]).unsqueeze(0).unsqueeze(0).to(shape_pipeline.device).to(shape_pipeline.dtype)
            control_kwargs["bbox"] = bbox
            print(f"[omni] Bbox control: {inp['bbox']}")
        elif control_type == "pose" and inp.get("pose"):
            pose = torch.FloatTensor(inp["pose"]).unsqueeze(0).to(shape_pipeline.device).to(shape_pipeline.dtype)
            control_kwargs["pose"] = pose
            print(f"[omni] Pose control: {len(inp['pose'])} bones")
        elif control_type == "point" and inp.get("point"):
            point = torch.FloatTensor(inp["point"]).unsqueeze(0).to(shape_pipeline.device).to(shape_pipeline.dtype)
            control_kwargs["point"] = point
            print(f"[omni] Point cloud control: {len(inp['point'])} points")
        elif control_type == "voxel" and inp.get("voxel"):
            voxel = torch.FloatTensor(inp["voxel"]).unsqueeze(0).to(shape_pipeline.device).to(shape_pipeline.dtype)
            control_kwargs["voxel"] = voxel
            print(f"[omni] Voxel control: {len(inp['voxel'])} samples")
        else:
            # Default: bbox [1,1,1] — uniform cube proportions (lets model decide shape)
            bbox = torch.FloatTensor([1.0, 1.0, 1.0]).unsqueeze(0).unsqueeze(0).to(shape_pipeline.device).to(shape_pipeline.dtype)
            control_kwargs["bbox"] = bbox
            print(f"[omni] Default bbox control: [1, 1, 1]")

        # Pass numpy array to Omni pipeline (preprocessor expects ndarray)
        shape_result = shape_pipeline(
            image=image_np,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            octree_resolution=octree_resolution,
            mc_level=0,
            generator=generator,
            **control_kwargs,
        )
        mesh = shape_result['shapes'][0][0]  # trimesh.Trimesh

        shape_time = time.time() - t0
        print(f"[omni] Shape done in {shape_time:.1f}s, faces={len(mesh.faces)}")

        # ── Smooth normals (fixes polygonal/faceted look) ──
        import trimesh
        try:
            trimesh.smoothing.filter_taubin(mesh, iterations=3)
            print(f"[omni] Applied Taubin smoothing (3 iterations)")
        except Exception as e:
            print(f"[omni] Taubin smoothing skipped: {e}")

        # Force smooth vertex normals
        if hasattr(mesh, 'vertex_normals'):
            _ = mesh.vertex_normals
            print(f"[omni] Smooth vertex normals computed")

        # Export GLB
        glb_path = os.path.join(work_dir, "output.glb")
        mesh.export(glb_path, include_normals=True)
        # TODO: Add Paint-v2-1 PBR texture pipeline once CUDA extensions are resolved

        # Read GLB
        with open(glb_path, "rb") as f:
            glb_data = f.read()

        result["files"] = [{
            "filename": "hunyuan3d_output.glb",
            "type": "base64",
            "data": base64.b64encode(glb_data).decode(),
        }]

        # Count faces/vertices
        loaded = trimesh.load(glb_path)
        if hasattr(loaded, 'faces'):
            n_faces = len(loaded.faces)
            n_verts = len(loaded.vertices)
        elif hasattr(loaded, 'geometry'):
            n_faces = sum(len(g.faces) for g in loaded.geometry.values())
            n_verts = sum(len(g.vertices) for g in loaded.geometry.values())
        else:
            n_faces = n_verts = 0

        result["meta"] = {
            "faces": n_faces,
            "vertices": n_verts,
            "shape_time_sec": round(shape_time, 1),
            "total_inference_sec": round(shape_time, 1),
            "glb_size_bytes": len(glb_data),
            "seed": seed,
            "steps": steps,
            "octree_resolution": octree_resolution,
            "textured": False,  # TODO: Paint-v2-1 PBR textures coming soon
            "control_type": control_type,
            "model": "hunyuan3d-omni",
        }

        print(f"[omni] GLB: {len(glb_data)} bytes, {n_faces} faces, {n_verts} verts")

        # ── Post-processing ──
        if inp.get("remesh_only", False):
            _run_remesh(glb_path, work_dir, inp, result)
        elif inp.get("roblox_postprocess", False):
            _run_roblox_pipeline(glb_path, work_dir, inp, result)

    except Exception as e:
        result["error"] = f"{e}\n{traceback.format_exc()}"
        print(f"[omni] ERROR: {e}")

    return result


def _run_remesh(glb_path, work_dir, inp, result):
    """Run Blender-based retopology (preserves UVs/textures)."""
    target_tris = inp.get("target_tris", 4000)
    print(f"[omni] Remesh: target={target_tris} (Blender decimation)")

    try:
        from retopo import retopologize
        remesh_glb = os.path.join(work_dir, "remeshed.glb")
        stats = retopologize(glb_path, remesh_glb, target_tris)

        with open(remesh_glb, "rb") as f:
            data = f.read()

        result["remeshed_glb"] = {
            "filename": "remeshed.glb", "type": "base64",
            "data": base64.b64encode(data).decode(),
        }
        result["remesh_meta"] = {
            "target_tris": target_tris,
            "original_faces": stats.get("original_faces", 0),
            "final_faces": stats.get("final_faces", 0),
            "reduction_pct": stats.get("reduction_pct", 0),
        }
        print(f"[omni] Remesh done: {stats.get('final_faces', '?')} faces, {len(data)} bytes")

    except Exception as e:
        result["remesh_error"] = str(e)
        print(f"[omni] Remesh error: {e}")


def _run_roblox_pipeline(glb_path, work_dir, inp, result):
    """Run full Roblox layered clothing pipeline with Blender decimation."""
    clothing_type = inp.get("clothing_type", "shirt")
    target_tris = inp.get("target_tris", 4000)
    out_dir = os.path.join(work_dir, "roblox-output")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[omni] Roblox pipeline: type={clothing_type}, tris={target_tris}")

    try:
        # Step 1: Blender-based decimation (preserves UVs/textures)
        print("[omni] Step 1: Blender decimation...")
        from retopo import retopologize
        decimated_glb = os.path.join(work_dir, "decimated.glb")
        retopo_stats = retopologize(glb_path, decimated_glb, target_tris)
        print(f"[omni] Decimated: {retopo_stats.get('original_faces','?')} -> {retopo_stats.get('final_faces','?')} faces")

        # Step 2: Roblox post-processing (cage, rig, FBX)
        print("[omni] Step 2: Roblox LC (cage + rig + FBX)...")
        pp_result = subprocess.run(
            ["python3", "/app/postprocess_clothing.py",
             "--input", decimated_glb,
             "--output-dir", out_dir,
             "--clothing-type", clothing_type,
             "--target-tris", str(target_tris),
             "--skip-retopo"],
            capture_output=True, text=True, timeout=180,
        )

        if pp_result.stdout:
            for line in pp_result.stdout.strip().split("\n")[-10:]:
                print(f"  {line}")
        if pp_result.returncode != 0:
            print(f"[omni] Post-process stderr: {pp_result.stderr[-1000:]}")

        # Collect outputs
        fbx_path = os.path.join(out_dir, "clothing_roblox.fbx")
        preview_path = os.path.join(out_dir, "clothing_preview.glb")
        mannequin_path = os.path.join(out_dir, "clothing_on_mannequin.glb")
        meta_path = os.path.join(out_dir, "metadata.json")

        if os.path.exists(fbx_path):
            with open(fbx_path, "rb") as f:
                result["roblox_fbx"] = {
                    "filename": "clothing_roblox.fbx", "type": "base64",
                    "data": base64.b64encode(f.read()).decode(),
                }
            print(f"[omni] FBX: {os.path.getsize(fbx_path)} bytes")

        if os.path.exists(preview_path):
            with open(preview_path, "rb") as f:
                result["roblox_preview_glb"] = {
                    "filename": "clothing_preview.glb", "type": "base64",
                    "data": base64.b64encode(f.read()).decode(),
                }

        if os.path.exists(mannequin_path):
            with open(mannequin_path, "rb") as f:
                result["mannequin_preview_glb"] = {
                    "filename": "clothing_on_mannequin.glb", "type": "base64",
                    "data": base64.b64encode(f.read()).decode(),
                }

        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            meta["retopo"] = retopo_stats
            result["roblox_meta"] = meta
        else:
            result["roblox_meta"] = {"retopo": retopo_stats}

    except Exception as e:
        result["roblox_error"] = str(e)
        print(f"[omni] Roblox pipeline error: {e}")


# Start RunPod serverless worker
runpod.serverless.start({"handler": handler})

"""
RunPod Serverless Handler for Hunyuan3D-2.0 Image-to-3D.

Input:
  image: base64-encoded image (with or without data: prefix)
  seed: int (default 42)
  steps: int (default 5)
  guidance_scale: float (default 5.0)
  octree_resolution: int (default 256)
  texture: bool (default true)
  face_count: int (default 40000)
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

# Global model pipelines (loaded once, reused across requests)
shape_pipeline = None
paint_pipeline = None


def load_models():
    """Load Hunyuan3D-2.0 models into VRAM. Called once on cold start."""
    global shape_pipeline, paint_pipeline

    if shape_pipeline is not None:
        return

    print("[hunyuan3d] Loading shape generation model...")
    t0 = time.time()

    import sys
    sys.path.insert(0, '/app')

    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    from hy3dgen.texgen import Hunyuan3DPaintPipeline

    shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        '/app/weights/Hunyuan3D-2',
        subfolder='hunyuan3d-dit-v2-0',
        use_safetensors=True,
    )
    shape_pipeline.to('cuda')

    print(f"[hunyuan3d] Shape model loaded in {time.time()-t0:.1f}s")

    print("[hunyuan3d] Loading texture paint model...")
    t1 = time.time()

    paint_pipeline = Hunyuan3DPaintPipeline.from_pretrained(
        '/app/weights/Hunyuan3D-2',
        subfolder='hunyuan3d-paint-v2-0',
        use_safetensors=True,
    )

    print(f"[hunyuan3d] Paint model loaded in {time.time()-t1:.1f}s")
    print(f"[hunyuan3d] All models ready ({time.time()-t0:.1f}s total)")


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
    steps = inp.get("steps", 5)
    guidance_scale = inp.get("guidance_scale", 5.0)
    octree_resolution = inp.get("octree_resolution", 256)
    do_texture = inp.get("texture", True)
    face_count = inp.get("face_count", 40000)

    work_dir = tempfile.mkdtemp(prefix="hunyuan3d_")
    result = {}

    try:
        # Save input image
        image_path = os.path.join(work_dir, "input.png")
        with open(image_path, "wb") as f:
            f.write(base64.b64decode(image_b64))

        from PIL import Image
        image = Image.open(image_path).convert("RGBA")

        # ── Shape generation ──
        print(f"[hunyuan3d] Generating shape (steps={steps}, seed={seed})...")
        t0 = time.time()

        import torch
        torch.manual_seed(seed)

        mesh = shape_pipeline(
            image=image,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            octree_resolution=octree_resolution,
        )[0]

        shape_time = time.time() - t0
        print(f"[hunyuan3d] Shape done in {shape_time:.1f}s")

        # Export untextured GLB
        untextured_path = os.path.join(work_dir, "untextured.glb")
        mesh.export(untextured_path)

        # ── Texture generation ──
        tex_time = 0
        if do_texture and paint_pipeline:
            print("[hunyuan3d] Generating textures...")
            t1 = time.time()

            textured_mesh = paint_pipeline(mesh, image=image)

            tex_time = time.time() - t1
            print(f"[hunyuan3d] Texture done in {tex_time:.1f}s")

            # Export textured GLB
            glb_path = os.path.join(work_dir, "output.glb")
            textured_mesh.export(glb_path)
        else:
            glb_path = untextured_path

        # Read GLB
        with open(glb_path, "rb") as f:
            glb_data = f.read()

        result["files"] = [{
            "filename": "hunyuan3d_output.glb",
            "type": "base64",
            "data": base64.b64encode(glb_data).decode(),
        }]

        # Count faces/vertices
        import trimesh
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
            "texture_time_sec": round(tex_time, 1),
            "total_inference_sec": round(shape_time + tex_time, 1),
            "glb_size_bytes": len(glb_data),
            "seed": seed,
            "steps": steps,
            "octree_resolution": octree_resolution,
            "textured": do_texture,
        }

        print(f"[hunyuan3d] GLB: {len(glb_data)} bytes, {n_faces} faces, {n_verts} verts")

        # ── Post-processing ──
        if inp.get("remesh_only", False):
            _run_remesh(glb_path, work_dir, inp, result)
        elif inp.get("roblox_postprocess", False):
            _run_roblox_pipeline(glb_path, work_dir, inp, result)

    except Exception as e:
        result["error"] = f"{e}\n{traceback.format_exc()}"
        print(f"[hunyuan3d] ERROR: {e}")

    return result


def _run_remesh(glb_path, work_dir, inp, result):
    """Run Blender-based retopology (preserves UVs/textures)."""
    target_tris = inp.get("target_tris", 4000)
    print(f"[hunyuan3d] Remesh: target={target_tris} (Blender decimation)")

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
        print(f"[hunyuan3d] Remesh done: {stats.get('final_faces', '?')} faces, {len(data)} bytes")

    except Exception as e:
        result["remesh_error"] = str(e)
        print(f"[hunyuan3d] Remesh error: {e}")


def _run_roblox_pipeline(glb_path, work_dir, inp, result):
    """Run full Roblox layered clothing pipeline with Blender decimation."""
    clothing_type = inp.get("clothing_type", "shirt")
    target_tris = inp.get("target_tris", 4000)
    out_dir = os.path.join(work_dir, "roblox-output")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[hunyuan3d] Roblox pipeline: type={clothing_type}, tris={target_tris}")

    try:
        # Step 1: Blender-based decimation (preserves UVs/textures)
        print("[hunyuan3d] Step 1: Blender decimation...")
        from retopo import retopologize
        decimated_glb = os.path.join(work_dir, "decimated.glb")
        retopo_stats = retopologize(glb_path, decimated_glb, target_tris)
        print(f"[hunyuan3d] Decimated: {retopo_stats.get('original_faces','?')} → {retopo_stats.get('final_faces','?')} faces")

        # Step 2: Roblox post-processing (cage, rig, FBX)
        print("[hunyuan3d] Step 2: Roblox LC (cage + rig + FBX)...")
        pp_result = subprocess.run(
            ["python3", "/app/postprocess_clothing.py",
             "--input", decimated_glb,
             "--output-dir", out_dir,
             "--clothing-type", clothing_type,
             "--target-tris", str(target_tris),
             "--skip-retopo"],  # Already decimated, skip retopo in postprocess
            capture_output=True, text=True, timeout=180,
        )

        if pp_result.stdout:
            for line in pp_result.stdout.strip().split("\n")[-10:]:
                print(f"  {line}")
        if pp_result.returncode != 0:
            print(f"[hunyuan3d] Post-process stderr: {pp_result.stderr[-1000:]}")

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
            print(f"[hunyuan3d] FBX: {os.path.getsize(fbx_path)} bytes")

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
            meta["retopo"] = retopo_stats  # Add decimation stats
            result["roblox_meta"] = meta
        else:
            result["roblox_meta"] = {"retopo": retopo_stats}

    except Exception as e:
        result["roblox_error"] = str(e)
        print(f"[hunyuan3d] Roblox pipeline error: {e}")


# Start RunPod serverless worker
runpod.serverless.start({"handler": handler})

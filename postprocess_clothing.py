#!/usr/bin/env python3
"""
postprocess_clothing.py — Full pipeline: TRELLIS2 GLB → Roblox-ready FBX

Called by the RunPod handler after TRELLIS2 generates a raw GLB.
Chains: retopology (PyMeshLab) → fitting + caging + rigging (Blender) → FBX

Usage:
    python postprocess_clothing.py \
        --input /comfyui/output/trellis2_00001.glb \
        --output-dir /tmp/roblox-output \
        --clothing-type shirt \
        --target-tris 4000

Outputs:
    /tmp/roblox-output/clothing_preview.glb   (decimated GLB for web preview)
    /tmp/roblox-output/clothing_roblox.fbx    (Roblox-ready layered clothing FBX)
    /tmp/roblox-output/metadata.json          (pipeline stats + Roblox info)
"""

import argparse
import json
import os
import subprocess
import sys
import time
import shutil
import base64


TEMPLATES_DIR = '/opt/roblox-templates'
BLENDER_BIN = '/usr/bin/blender'  # System blender in Docker


def run_retopo(input_glb, output_glb, target_tris=4000):
    """Run PyMeshLab retopology."""
    print(f"\n{'='*60}")
    print(f"STEP 1: Retopology ({target_tris} target tris)")
    print(f"{'='*60}")

    start = time.time()

    # Import retopo module directly (same Python env)
    sys.path.insert(0, os.path.dirname(__file__))
    from retopo import retopologize

    # PyMeshLab can't read GLB directly — convert via trimesh first
    import trimesh

    # Load GLB
    scene = trimesh.load(input_glb, force='scene')
    if isinstance(scene, trimesh.Scene):
        # Merge all meshes into one
        mesh = scene.dump(concatenate=True) if hasattr(scene, 'dump') else trimesh.util.concatenate(scene.dump())
    else:
        mesh = scene

    # Save as OBJ for PyMeshLab
    temp_obj = output_glb.replace('.glb', '_temp.obj')
    mesh.export(temp_obj)
    print(f"[postprocess] Converted GLB → OBJ: {temp_obj}")
    print(f"[postprocess] Input mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

    # Run retopology
    temp_retopo_obj = output_glb.replace('.glb', '_retopo.obj')
    stats = retopologize(temp_obj, temp_retopo_obj, target_tris)

    # Convert back to GLB
    retopo_mesh = trimesh.load(temp_retopo_obj)
    retopo_mesh.export(output_glb)

    # Cleanup temp files
    for f in [temp_obj, temp_retopo_obj,
              temp_obj.replace('.obj', '.mtl'),
              temp_retopo_obj.replace('.obj', '.mtl')]:
        if os.path.exists(f):
            os.remove(f)

    elapsed = time.time() - start
    print(f"[postprocess] Retopology complete in {elapsed:.1f}s")
    stats['retopo_time_sec'] = round(elapsed, 1)
    return stats


def run_blender(input_glb, output_fbx, output_glb, mannequin_glb, clothing_type, meta_path):
    """Run Blender headless post-processing."""
    print(f"\n{'='*60}")
    print(f"STEP 2: Blender Post-Processing (fit, cage, rig, export)")
    print(f"{'='*60}")

    start = time.time()

    blender_script = os.path.join(os.path.dirname(__file__), 'blender_postprocess.py')

    cmd = [
        BLENDER_BIN,
        '--background',
        '--python', blender_script,
        '--',
        '--input', input_glb,
        '--output', output_fbx,
        '--output-glb', output_glb,
        '--output-mannequin-glb', mannequin_glb,
        '--clothing-type', clothing_type,
        '--templates-dir', TEMPLATES_DIR,
        '--meta-output', meta_path,
    ]

    print(f"[postprocess] Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    if result.stdout:
        # Print only lines with [blender] prefix
        for line in result.stdout.split('\n'):
            if '[blender]' in line:
                print(line)

    if result.returncode != 0:
        print(f"[postprocess] Blender stderr: {result.stderr[-1000:]}")
        raise RuntimeError(f"Blender failed with exit code {result.returncode}")

    elapsed = time.time() - start
    print(f"[postprocess] Blender complete in {elapsed:.1f}s")
    return elapsed


def encode_file_base64(filepath):
    """Read file and return base64 string."""
    with open(filepath, 'rb') as f:
        return base64.b64encode(f.read()).decode('ascii')


def main():
    parser = argparse.ArgumentParser(description='Post-process TRELLIS2 output for Roblox')
    parser.add_argument('--input', required=True, help='Input GLB from TRELLIS2')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--clothing-type', default='shirt',
                        choices=['shirt', 'tshirt', 'jacket', 'sweater', 'pants',
                                 'shorts', 'dress', 'skirt', 'full'])
    parser.add_argument('--target-tris', type=int, default=4000,
                        help='Target triangle count for Roblox (default: 4000)')
    args = parser.parse_args()

    total_start = time.time()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Output paths
    retopo_glb = os.path.join(args.output_dir, 'clothing_retopo.glb')
    preview_glb = os.path.join(args.output_dir, 'clothing_preview.glb')
    mannequin_glb = os.path.join(args.output_dir, 'clothing_on_mannequin.glb')
    roblox_fbx = os.path.join(args.output_dir, 'clothing_roblox.fbx')
    meta_path = os.path.join(args.output_dir, 'metadata.json')

    print(f"\n{'#'*60}")
    print(f"# ROBLOX LAYERED CLOTHING PIPELINE")
    print(f"# Input:         {args.input}")
    print(f"# Clothing type: {args.clothing_type}")
    print(f"# Target tris:   {args.target_tris}")
    print(f"{'#'*60}")

    # ── Step 1: Retopology ───────────────────────────────────────────────

    retopo_stats = run_retopo(args.input, retopo_glb, args.target_tris)

    # ── Step 2: Blender (fit, cage, rig, export FBX) ────────────────────

    blender_time = run_blender(retopo_glb, roblox_fbx, preview_glb,
                               mannequin_glb, args.clothing_type, meta_path)

    # ── Step 3: Compile final metadata ───────────────────────────────────

    total_time = time.time() - total_start

    # Read Blender metadata if it was written
    blender_meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            blender_meta = json.load(f)

    # File sizes
    fbx_size = os.path.getsize(roblox_fbx) if os.path.exists(roblox_fbx) else 0
    glb_size = os.path.getsize(preview_glb) if os.path.exists(preview_glb) else 0
    mannequin_glb_size = os.path.getsize(mannequin_glb) if os.path.exists(mannequin_glb) else 0

    final_meta = {
        'pipeline': 'trellis2-roblox-layered-clothing',
        'clothing_type': args.clothing_type,
        'target_tris': args.target_tris,
        'retopo': retopo_stats,
        'blender': blender_meta,
        'timing': {
            'retopo_sec': retopo_stats.get('retopo_time_sec', 0),
            'blender_sec': round(blender_time, 1),
            'total_sec': round(total_time, 1),
        },
        'output': {
            'fbx_path': roblox_fbx,
            'fbx_size_bytes': fbx_size,
            'fbx_size_mb': round(fbx_size / 1024 / 1024, 2),
            'glb_path': preview_glb,
            'glb_size_bytes': glb_size,
            'glb_size_mb': round(glb_size / 1024 / 1024, 2),
            'mannequin_glb_path': mannequin_glb,
            'mannequin_glb_size_bytes': mannequin_glb_size,
            'mannequin_glb_size_mb': round(mannequin_glb_size / 1024 / 1024, 2),
        },
        'roblox': {
            'ready': blender_meta.get('roblox_ready', False),
            'has_cages': blender_meta.get('has_cages', False),
            'has_armature': blender_meta.get('has_armature', False),
            'attachment': blender_meta.get('attachment', 'BodyFrontAttachment'),
            'auto_skin_recommended': True,
            'max_tris_limit': 4000,
        },
    }

    with open(meta_path, 'w') as f:
        json.dump(final_meta, f, indent=2)

    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"  Total time:    {total_time:.1f}s")
    print(f"  FBX:           {roblox_fbx} ({final_meta['output']['fbx_size_mb']} MB)")
    print(f"  Preview GLB:   {preview_glb} ({final_meta['output']['glb_size_mb']} MB)")
    print(f"  Mannequin GLB: {mannequin_glb} ({final_meta['output']['mannequin_glb_size_mb']} MB)")
    print(f"  Metadata:      {meta_path}")
    print(f"  Roblox ready:  {final_meta['roblox']['ready']}")
    print(f"  Has cages:     {final_meta['roblox']['has_cages']}")
    print(f"  Tris:          {retopo_stats.get('final_faces', '?')}")

    return final_meta


if __name__ == '__main__':
    main()

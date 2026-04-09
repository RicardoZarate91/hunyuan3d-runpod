#!/usr/bin/env python3
"""
retopo.py — Retopologize a mesh to a target triangle count.

Uses Blender's Decimate modifier (headless) which preserves:
  - UV maps and textures
  - Materials and vertex colors
  - Clean topology at high reduction ratios

Falls back to trimesh + fast-simplification if Blender is not available
(but trimesh strips UVs/textures).

Usage:
    python retopo.py --input input.glb --output output.glb --target-tris 4000
"""

import argparse
import subprocess
import sys
import os
import json

BLENDER_BIN = '/usr/bin/blender'


def retopologize(input_path, output_path, target_tris=4000):
    """
    Decimate mesh to target triangle count, preserving textures.

    Uses Blender headless for UV/texture-preserving decimation.
    Falls back to trimesh if Blender is unavailable.
    """
    blender_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'blender_decimate.py')

    if os.path.exists(BLENDER_BIN) and os.path.exists(blender_script):
        return _retopo_blender(input_path, output_path, target_tris, blender_script)
    else:
        print("[retopo] Blender not available, falling back to trimesh (no texture preservation)")
        return _retopo_trimesh(input_path, output_path, target_tris)


def _retopo_blender(input_path, output_path, target_tris, blender_script):
    """Run Blender headless decimation."""
    cmd = [
        BLENDER_BIN,
        '--background',
        '--python', blender_script,
        '--',
        '--input', input_path,
        '--output', output_path,
        '--target-tris', str(target_tris),
    ]

    print(f"[retopo] Running Blender decimate: {target_tris} target tris")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    # Parse stats from stdout
    stats = None
    for line in (result.stdout or '').split('\n'):
        if line.startswith('RETOPO_STATS:'):
            stats = json.loads(line[len('RETOPO_STATS:'):])
        if '[blender_decimate]' in line:
            print(line.strip())

    if result.returncode != 0:
        stderr_tail = (result.stderr or '')[-500:]
        print(f"[retopo] Blender stderr: {stderr_tail}")
        raise RuntimeError(f"Blender decimate failed (exit {result.returncode})")

    if not os.path.exists(output_path):
        raise RuntimeError("Blender decimate produced no output file")

    if stats:
        print(f"[retopo] Result: {stats['final_verts']} verts, {stats['final_faces']} tris ({stats['reduction_pct']}% reduction)")
        return stats

    # If we couldn't parse stats, return minimal info
    return {
        'original_faces': 0,
        'original_verts': 0,
        'final_faces': target_tris,
        'final_verts': 0,
        'reduction_pct': 0.0,
    }


def _retopo_trimesh(input_path, output_path, target_tris):
    """Fallback: trimesh decimation (strips UVs/textures)."""
    import trimesh

    loaded = trimesh.load(input_path, force='mesh')
    if isinstance(loaded, trimesh.Scene):
        mesh = trimesh.util.concatenate(loaded.dump())
    else:
        mesh = loaded

    original_faces = len(mesh.faces)
    original_verts = len(mesh.vertices)
    print(f"[retopo] Input: {original_verts} vertices, {original_faces} faces")

    if original_faces <= target_tris:
        print(f"[retopo] Already under target ({original_faces} <= {target_tris}), skipping")
        mesh.export(output_path)
        return {
            'original_faces': original_faces,
            'original_verts': original_verts,
            'final_faces': original_faces,
            'final_verts': original_verts,
            'reduction_pct': 0.0,
        }

    print(f"[retopo] Decimating {original_faces} -> {target_tris} faces...")
    decimated = mesh.simplify_quadric_decimation(face_count=target_tris)

    final_faces = len(decimated.faces)
    final_verts = len(decimated.vertices)
    reduction = round((1 - final_faces / max(original_faces, 1)) * 100, 1)
    print(f"[retopo] Output: {final_verts} verts, {final_faces} faces ({reduction}% reduction)")

    decimated.export(output_path)
    return {
        'original_faces': original_faces,
        'original_verts': original_verts,
        'final_faces': final_faces,
        'final_verts': final_verts,
        'reduction_pct': reduction,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Retopologize mesh for Roblox')
    parser.add_argument('--input', required=True, help='Input mesh file (GLB/OBJ/PLY/STL)')
    parser.add_argument('--output', required=True, help='Output mesh file')
    parser.add_argument('--target-tris', type=int, default=4000,
                        help='Target triangle count (default: 4000)')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found")
        sys.exit(1)

    retopologize(args.input, args.output, args.target_tris)

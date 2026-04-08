#!/usr/bin/env python3
"""
retopo.py — Retopologize a GLB/OBJ mesh to a target triangle count.

Uses PyMeshLab for quadric edge collapse decimation (much better quality
than naive decimation). Also cleans up non-manifold geometry and makes
the mesh watertight for Roblox compatibility.

Usage:
    python retopo.py input.glb output.glb --target-tris 4000
    python retopo.py input.glb output.obj --target-tris 4000
"""

import argparse
import sys
import os

def retopologize(input_path, output_path, target_tris=4000):
    """
    Decimate mesh to target triangle count with quality preservation.

    Steps:
    1. Load mesh
    2. Remove duplicate vertices/faces
    3. Remove non-manifold edges
    4. Quadric edge collapse decimation to target
    5. Smooth (light Laplacian)
    6. Save
    """
    import pymeshlab

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(input_path)

    mesh = ms.current_mesh()
    original_faces = mesh.face_number()
    original_verts = mesh.vertex_number()
    print(f"[retopo] Input: {original_verts} vertices, {original_faces} faces")

    # Step 1: Clean up geometry
    print("[retopo] Cleaning geometry...")
    ms.apply_filter('meshing_remove_duplicate_faces')
    ms.apply_filter('meshing_remove_duplicate_vertices')

    # Remove unreferenced vertices
    ms.apply_filter('meshing_remove_unreferenced_vertices')

    # Remove zero-area faces
    try:
        ms.apply_filter('meshing_remove_null_faces')
    except Exception:
        pass  # Filter might not exist in all versions

    # Step 2: Repair non-manifold (important for Roblox watertight requirement)
    print("[retopo] Repairing non-manifold geometry...")
    try:
        ms.apply_filter('meshing_repair_non_manifold_edges')
        ms.apply_filter('meshing_repair_non_manifold_vertices')
    except Exception:
        pass  # Best effort

    # Step 3: Close holes (watertight)
    print("[retopo] Closing holes...")
    try:
        ms.apply_filter('meshing_close_holes', maxholesize=100)
    except Exception:
        pass

    mesh = ms.current_mesh()
    cleaned_faces = mesh.face_number()
    print(f"[retopo] After cleanup: {mesh.vertex_number()} vertices, {cleaned_faces} faces")

    # Step 4: Decimate if needed
    if cleaned_faces > target_tris:
        print(f"[retopo] Decimating {cleaned_faces} -> {target_tris} faces...")

        # Use quadric edge collapse — best quality decimation
        ms.apply_filter('meshing_decimation_quadric_edge_collapse',
                        targetfacenum=target_tris,
                        qualitythr=0.5,
                        preserveboundary=True,
                        preservenormal=True,
                        preservetopology=True,
                        optimalplacement=True,
                        planarquadric=True)

        mesh = ms.current_mesh()
        print(f"[retopo] After decimation: {mesh.vertex_number()} vertices, {mesh.face_number()} faces")
    else:
        print(f"[retopo] Already under target ({cleaned_faces} <= {target_tris}), skipping decimation")

    # Step 5: Light smoothing to clean up decimation artifacts
    print("[retopo] Light smoothing...")
    try:
        ms.apply_filter('apply_coord_laplacian_smoothing',
                        stepsmoothnum=1,
                        cotangentweight=True)
    except Exception:
        pass

    # Step 6: Re-clean after decimation
    ms.apply_filter('meshing_remove_duplicate_faces')
    ms.apply_filter('meshing_remove_duplicate_vertices')

    # Final stats
    mesh = ms.current_mesh()
    final_faces = mesh.face_number()
    final_verts = mesh.vertex_number()
    print(f"[retopo] Output: {final_verts} vertices, {final_faces} faces")
    print(f"[retopo] Reduction: {original_faces} -> {final_faces} ({(1 - final_faces/max(original_faces,1))*100:.1f}%)")

    # Save
    ms.save_current_mesh(output_path)
    print(f"[retopo] Saved to {output_path}")

    return {
        'original_faces': original_faces,
        'original_verts': original_verts,
        'final_faces': final_faces,
        'final_verts': final_verts,
        'reduction_pct': round((1 - final_faces / max(original_faces, 1)) * 100, 1),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Retopologize mesh for Roblox')
    parser.add_argument('input', help='Input mesh file (GLB/OBJ/PLY)')
    parser.add_argument('output', help='Output mesh file')
    parser.add_argument('--target-tris', type=int, default=4000,
                        help='Target triangle count (default: 4000)')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found")
        sys.exit(1)

    retopologize(args.input, args.output, args.target_tris)

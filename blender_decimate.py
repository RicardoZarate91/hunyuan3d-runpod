#!/usr/bin/env python3
"""
blender_decimate.py — Blender headless decimation that preserves UVs and textures.

Unlike trimesh's simplify_quadric_decimation (which strips UVs/materials),
Blender's Decimate modifier preserves:
  - UV maps → textures stay mapped correctly
  - Materials and vertex colors
  - Clean topology at high reduction ratios

Usage (run via Blender headless):
    blender --background --python blender_decimate.py -- \
        --input input.glb \
        --output output.glb \
        --target-tris 4000

Output: JSON stats printed to stdout (last line), parseable by caller.
"""

import bpy
import sys
import os
import json


def parse_args():
    argv = sys.argv
    if '--' in argv:
        argv = argv[argv.index('--') + 1:]
    else:
        argv = []

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input GLB file')
    parser.add_argument('--output', required=True, help='Output GLB file')
    parser.add_argument('--target-tris', type=int, default=4000,
                        help='Target triangle count')
    return parser.parse_args(argv)


def clear_scene():
    """Remove all objects from the scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    # Clear orphan data
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)
    for block in bpy.data.images:
        if block.users == 0:
            bpy.data.images.remove(block)


def count_tris():
    """Count total triangles across all mesh objects."""
    total = 0
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            # Evaluate to get triangulated count
            depsgraph = bpy.context.evaluated_depsgraph_get()
            eval_obj = obj.evaluated_get(depsgraph)
            mesh = eval_obj.to_mesh()
            mesh.calc_loop_triangles()
            total += len(mesh.loop_triangles)
            eval_obj.to_mesh_clear()
    return total


def count_verts():
    """Count total vertices across all mesh objects."""
    total = 0
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            total += len(obj.data.vertices)
    return total


def main():
    args = parse_args()

    print(f"[blender_decimate] Input: {args.input}")
    print(f"[blender_decimate] Target: {args.target_tris} tris")

    # Clear default scene
    clear_scene()

    # Import GLB
    bpy.ops.import_scene.gltf(filepath=args.input)

    # Count original geometry
    original_tris = count_tris()
    original_verts = count_verts()
    print(f"[blender_decimate] Original: {original_verts} verts, {original_tris} tris")

    if original_tris <= args.target_tris:
        print(f"[blender_decimate] Already under target, exporting as-is")
        bpy.ops.export_scene.gltf(
            filepath=args.output,
            export_format='GLB',
            export_image_format='AUTO',
            export_materials='EXPORT',
            export_texcoords=True,
            export_normals=True,
            export_colors=True,
        )
        stats = {
            'original_faces': original_tris,
            'original_verts': original_verts,
            'final_faces': original_tris,
            'final_verts': original_verts,
            'reduction_pct': 0.0,
        }
        print(f"RETOPO_STATS:{json.dumps(stats)}")
        return

    # Calculate decimation ratio
    ratio = args.target_tris / max(original_tris, 1)
    ratio = max(0.001, min(ratio, 1.0))
    print(f"[blender_decimate] Decimation ratio: {ratio:.4f}")

    # Apply Decimate modifier to each mesh object
    mesh_objects = [obj for obj in bpy.data.objects if obj.type == 'MESH']

    for obj in mesh_objects:
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)

        # Add Decimate modifier
        mod = obj.modifiers.new(name='Decimate', type='DECIMATE')
        mod.decimate_type = 'COLLAPSE'  # Quadric edge collapse — best quality
        mod.ratio = ratio
        mod.use_collapse_triangulate = True  # Ensure output is triangulated

        # Apply the modifier
        bpy.ops.object.modifier_apply(modifier=mod.name)
        obj.select_set(False)

    # Count result
    final_tris = count_tris()
    final_verts = count_verts()
    reduction = round((1 - final_tris / max(original_tris, 1)) * 100, 1)
    print(f"[blender_decimate] Result: {final_verts} verts, {final_tris} tris")
    print(f"[blender_decimate] Reduction: {reduction}%")

    # If we overshot the target significantly, do a second pass with adjusted ratio
    if final_tris > args.target_tris * 1.15:
        second_ratio = args.target_tris / max(final_tris, 1)
        second_ratio = max(0.01, min(second_ratio, 1.0))
        print(f"[blender_decimate] Second pass: ratio {second_ratio:.4f}")

        for obj in mesh_objects:
            if obj.type != 'MESH':
                continue
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
            mod = obj.modifiers.new(name='Decimate2', type='DECIMATE')
            mod.decimate_type = 'COLLAPSE'
            mod.ratio = second_ratio
            mod.use_collapse_triangulate = True
            bpy.ops.object.modifier_apply(modifier=mod.name)
            obj.select_set(False)

        final_tris = count_tris()
        final_verts = count_verts()
        reduction = round((1 - final_tris / max(original_tris, 1)) * 100, 1)
        print(f"[blender_decimate] After 2nd pass: {final_verts} verts, {final_tris} tris")

    # Export GLB — preserving textures and materials
    bpy.ops.export_scene.gltf(
        filepath=args.output,
        export_format='GLB',
        export_image_format='AUTO',
        export_materials='EXPORT',
        export_texcoords=True,
        export_normals=True,
        export_colors=True,
    )

    print(f"[blender_decimate] Saved to {args.output}")

    stats = {
        'original_faces': original_tris,
        'original_verts': original_verts,
        'final_faces': final_tris,
        'final_verts': final_verts,
        'reduction_pct': reduction,
    }
    # Print stats as parseable JSON on last line
    print(f"RETOPO_STATS:{json.dumps(stats)}")


if __name__ == '__main__':
    main()

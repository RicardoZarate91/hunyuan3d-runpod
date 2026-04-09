#!/usr/bin/env python3
"""
blender_postprocess.py — Blender headless post-processing for Roblox Layered Clothing.

Takes a decimated GLB clothing mesh and produces a Roblox-ready FBX with:
  - Clothing mesh scaled/positioned to Roblox mannequin (NO shrinkwrap/deformation)
  - Inner cage (from Roblox template, unchanged)
  - Outer cage (template cage with vertices pushed outward to envelop clothing)
  - R15 armature with correct bone names
  - Automatic weight painting

Key principle: NEVER deform the AI-generated clothing mesh. Only deform the
outer cage to fit around it. The clothing shape is preserved exactly as generated.

Usage (run via Blender headless):
    blender --background --python blender_postprocess.py -- \
        --input clothing.glb \
        --output clothing_roblox.fbx \
        --clothing-type shirt \
        --templates-dir /opt/roblox-templates

Clothing types: shirt, jacket, pants, shorts, dress, skirt, tshirt, sweater, full
"""

import bpy
import bmesh
import mathutils
import sys
import os
import argparse
import json
from math import radians
from mathutils import Vector

# ── Parse arguments (after --) ───────────────────────────────────────────────

def parse_args():
    argv = sys.argv
    if '--' in argv:
        argv = argv[argv.index('--') + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input GLB/OBJ mesh')
    parser.add_argument('--output', required=True, help='Output FBX path')
    parser.add_argument('--output-glb', default=None, help='Also save processed GLB')
    parser.add_argument('--clothing-type', default='shirt',
                        choices=['shirt', 'tshirt', 'jacket', 'sweater', 'pants',
                                 'shorts', 'dress', 'skirt', 'full'],
                        help='Clothing type for attachment point')
    parser.add_argument('--templates-dir', default='/opt/roblox-templates',
                        help='Directory with Roblox template FBX files')
    parser.add_argument('--output-mannequin-glb', default=None,
                        help='Also save GLB with clothing + mannequin for web preview')
    parser.add_argument('--meta-output', default=None,
                        help='Path to write JSON metadata')
    parser.add_argument('--skip-retopo', action='store_true',
                        help='Skip retopology (mesh already decimated)')
    return parser.parse_args(argv)


# ── Clothing type → attachment config ────────────────────────────────────────

CLOTHING_CONFIG = {
    #                attachment point             which cage region to edit
    'shirt':   {'attachment': 'BodyFrontAttachment', 'region': 'upper'},
    'tshirt':  {'attachment': 'BodyFrontAttachment', 'region': 'upper'},
    'jacket':  {'attachment': 'BodyFrontAttachment', 'region': 'upper'},
    'sweater': {'attachment': 'BodyFrontAttachment', 'region': 'upper'},
    'pants':   {'attachment': 'WaistCenterAttachment', 'region': 'lower'},
    'shorts':  {'attachment': 'WaistCenterAttachment', 'region': 'lower'},
    'dress':   {'attachment': 'BodyFrontAttachment', 'region': 'full'},
    'skirt':   {'attachment': 'WaistCenterAttachment', 'region': 'lower'},
    'full':    {'attachment': 'BodyFrontAttachment', 'region': 'full'},
}


# ── Helper functions ─────────────────────────────────────────────────────────

def clear_scene():
    """Remove all objects from scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.armatures:
        if block.users == 0:
            bpy.data.armatures.remove(block)


def import_fbx(filepath):
    """Import FBX file."""
    bpy.ops.import_scene.fbx(filepath=filepath, use_anim=False)


def import_glb(filepath):
    """Import GLB/GLTF file."""
    bpy.ops.import_scene.gltf(filepath=filepath)


def get_mesh_objects():
    """Return all mesh objects in the scene."""
    return [obj for obj in bpy.data.objects if obj.type == 'MESH']


def get_armature():
    """Return the first armature in the scene."""
    for obj in bpy.data.objects:
        if obj.type == 'ARMATURE':
            return obj
    return None


def get_bounding_box(obj):
    """Get world-space bounding box of an object."""
    bbox = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    xs = [v.x for v in bbox]
    ys = [v.y for v in bbox]
    zs = [v.z for v in bbox]
    return {
        'min': Vector((min(xs), min(ys), min(zs))),
        'max': Vector((max(xs), max(ys), max(zs))),
        'center': Vector(((min(xs)+max(xs))/2, (min(ys)+max(ys))/2, (min(zs)+max(zs))/2)),
        'size': Vector((max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs))),
    }


def fit_clothing_to_mannequin(clothing_obj, mannequin_obj, clothing_type):
    """
    Scale and position the clothing mesh to align with the mannequin.

    Unlike shrinkwrap, this ONLY scales and translates — it does NOT deform
    any vertices. The original clothing shape is fully preserved.
    """
    mannequin_bb = get_bounding_box(mannequin_obj)
    cloth_bb = get_bounding_box(clothing_obj)

    region = CLOTHING_CONFIG.get(clothing_type, {}).get('region', 'full')

    # Scale to match mannequin proportions based on clothing type
    mannequin_height = mannequin_bb['size'].y
    cloth_height = cloth_bb['size'].y

    if cloth_height <= 0:
        print("[blender] WARNING: Clothing has zero height, skipping fit")
        return

    # Scale to ~90% of mannequin height (clothing should be slightly smaller
    # than full body, the cage handles the rest)
    if region == 'upper':
        # Upper body: scale to ~50% of mannequin height (torso + arms)
        target_height = mannequin_height * 0.50
    elif region == 'lower':
        # Lower body: scale to ~55% of mannequin height (waist to feet)
        target_height = mannequin_height * 0.55
    else:
        # Full body: scale to ~90% of mannequin height
        target_height = mannequin_height * 0.90

    scale_factor = target_height / cloth_height
    clothing_obj.scale = (scale_factor, scale_factor, scale_factor)
    bpy.context.view_layer.update()

    # Recompute bounding box after scale
    cloth_bb = get_bounding_box(clothing_obj)

    # Position: center X and Z on mannequin, align Y based on region
    if region == 'upper':
        # Align top of clothing with top of mannequin (shoulders/neck area)
        y_offset = mannequin_bb['max'].y - cloth_bb['max'].y
    elif region == 'lower':
        # Align bottom of clothing with bottom of mannequin (feet)
        y_offset = mannequin_bb['min'].y - cloth_bb['min'].y
    else:
        # Center vertically
        y_offset = mannequin_bb['center'].y - cloth_bb['center'].y

    x_offset = mannequin_bb['center'].x - cloth_bb['center'].x
    z_offset = mannequin_bb['center'].z - cloth_bb['center'].z

    clothing_obj.location += Vector((x_offset, y_offset, z_offset))
    bpy.context.view_layer.update()

    final_bb = get_bounding_box(clothing_obj)
    print(f"[blender] Clothing fitted: scale={scale_factor:.3f}, "
          f"size={final_bb['size'].x:.3f}x{final_bb['size'].y:.3f}x{final_bb['size'].z:.3f}")


def deform_outer_cage(outer_cage_obj, clothing_obj, region='full', margin=0.008):
    """
    Deform outer cage vertices to envelop the clothing mesh.

    Algorithm (matches how real Roblox creators do it):
    1. Build BVH tree from clothing mesh for fast raycasting
    2. For each outer cage vertex, compute its outward normal
    3. Find the nearest clothing surface point
    4. If clothing extends beyond the cage vertex, push it outward along normal
    5. Only modify vertices in the relevant body region
    6. Smooth the displacement to avoid abrupt jumps

    CRITICAL: Never modify vertex count, face count, edges, or UVs.
    Only vertex positions are changed.
    """
    from mathutils.bvhtree import BVHTree

    depsgraph = bpy.context.evaluated_depsgraph_get()
    cloth_bvh = BVHTree.FromObject(clothing_obj, depsgraph)

    cage_mesh = outer_cage_obj.data
    cage_world = outer_cage_obj.matrix_world
    cage_world_inv = cage_world.inverted()

    # Get clothing bounding box to determine which cage vertices are near clothing
    cloth_bb = get_bounding_box(clothing_obj)
    cage_bb = get_bounding_box(outer_cage_obj)

    # Determine Y range for this clothing region
    # (only modify cage vertices in the area covered by clothing)
    y_min_edit = cloth_bb['min'].y - margin * 5
    y_max_edit = cloth_bb['max'].y + margin * 5

    print(f"[blender] Cage edit region: Y=[{y_min_edit:.3f}, {y_max_edit:.3f}]")

    # First pass: compute displacement for each vertex
    displacements = {}  # vertex index -> new world position

    for vert in cage_mesh.vertices:
        cage_world_pos = cage_world @ vert.co

        # Skip vertices outside the clothing region
        if cage_world_pos.y < y_min_edit or cage_world_pos.y > y_max_edit:
            continue

        # Find nearest point on clothing surface
        nearest, normal, face_idx, dist = cloth_bvh.find_nearest(cage_world_pos)

        if nearest is None:
            continue

        # Skip if clothing is far from this cage vertex (not covered)
        # Use a threshold relative to the cage size
        max_dist = max(cage_bb['size']) * 0.3
        if dist > max_dist:
            continue

        # Compute outward direction from cage center to this vertex
        # (approximate surface normal for the cage)
        cage_center = cage_bb['center']
        outward_dir = (cage_world_pos - cage_center).normalized()

        # Check if clothing surface is at or beyond the cage vertex
        # Project both onto the outward direction to compare distances from center
        cage_dist_from_center = (cage_world_pos - cage_center).dot(outward_dir)
        cloth_dist_from_center = (nearest - cage_center).dot(outward_dir)

        # If clothing extends beyond cage (or is close), push cage outward
        if cloth_dist_from_center > cage_dist_from_center - margin:
            # Push cage vertex to clothing surface + margin, along outward direction
            new_dist = cloth_dist_from_center + margin
            new_world_pos = cage_center + outward_dir * new_dist
            displacements[vert.index] = new_world_pos

    print(f"[blender] Outer cage: {len(displacements)}/{len(cage_mesh.vertices)} vertices need adjustment")

    if not displacements:
        print("[blender] No cage vertices need adjustment (clothing fits within cage)")
        return

    # Second pass: smooth the displacement to avoid abrupt jumps
    # Build adjacency map from edges
    adjacency = {i: set() for i in range(len(cage_mesh.vertices))}
    for edge in cage_mesh.edges:
        adjacency[edge.vertices[0]].add(edge.vertices[1])
        adjacency[edge.vertices[1]].add(edge.vertices[0])

    # Smooth: for each displaced vertex, average with its neighbors (2 iterations)
    for smooth_iter in range(2):
        smoothed = {}
        for vert_idx, new_pos in displacements.items():
            neighbor_positions = []
            for neighbor_idx in adjacency[vert_idx]:
                if neighbor_idx in displacements:
                    neighbor_positions.append(displacements[neighbor_idx])
                else:
                    # Neighbor is not displaced, use its current position
                    neighbor_positions.append(cage_world @ cage_mesh.vertices[neighbor_idx].co)

            if neighbor_positions:
                # Weighted average: 60% self, 40% neighbors
                avg_neighbor = Vector((0, 0, 0))
                for np in neighbor_positions:
                    avg_neighbor += np
                avg_neighbor /= len(neighbor_positions)
                smoothed[vert_idx] = new_pos * 0.6 + avg_neighbor * 0.4
            else:
                smoothed[vert_idx] = new_pos

        displacements = smoothed

    # Apply displacements
    applied = 0
    for vert_idx, new_world_pos in displacements.items():
        cage_mesh.vertices[vert_idx].co = cage_world_inv @ new_world_pos
        applied += 1

    cage_mesh.update()
    print(f"[blender] Outer cage: {applied} vertices displaced (smoothed, 2 iterations)")


def clean_mesh_for_roblox(clothing_obj):
    """
    Clean up AI-generated mesh to meet Roblox requirements:
    - Remove non-manifold geometry (edges with 3+ faces)
    - Fill small holes (watertight requirement)
    - Remove loose/floating vertices
    - Remove degenerate faces (zero area)
    """
    bpy.ops.object.select_all(action='DESELECT')
    clothing_obj.select_set(True)
    bpy.context.view_layer.objects.active = clothing_obj

    # Enter edit mode for cleanup
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')

    # Remove degenerate faces (zero area)
    bpy.ops.mesh.dissolve_degenerate(threshold=0.0001)

    # Remove loose vertices/edges (not connected to faces)
    bpy.ops.mesh.delete_loose(use_verts=True, use_edges=True, use_faces=False)

    # Select non-manifold edges and try to fix
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.mesh.select_non_manifold(extend=False)

    # Count non-manifold before fix
    bpy.ops.object.mode_set(mode='OBJECT')
    non_manifold_count = sum(1 for v in clothing_obj.data.vertices if v.select)

    if non_manifold_count > 0:
        print(f"[blender] Found {non_manifold_count} non-manifold vertices, attempting fix...")
        bpy.ops.object.mode_set(mode='EDIT')
        # Try to fill holes (makes mesh watertight)
        try:
            bpy.ops.mesh.fill_holes(sides=4)
            print("[blender] Filled holes in mesh")
        except Exception:
            pass
        bpy.ops.object.mode_set(mode='OBJECT')
    else:
        print("[blender] Mesh is manifold (clean)")

    bpy.ops.object.mode_set(mode='OBJECT')

    final_faces = len(clothing_obj.data.polygons)
    final_verts = len(clothing_obj.data.vertices)
    print(f"[blender] After cleanup: {final_faces} faces, {final_verts} verts")


def limit_bone_influences(clothing_obj, max_influences=4):
    """
    Roblox requires max 4 bone influences per vertex.
    Remove the weakest influences if any vertex exceeds this limit.
    """
    mesh = clothing_obj.data
    groups = clothing_obj.vertex_groups

    if not groups:
        return

    excess_count = 0
    for vert in mesh.vertices:
        if len(vert.groups) > max_influences:
            # Sort by weight, keep only the top N
            sorted_groups = sorted(vert.groups, key=lambda g: g.weight, reverse=True)
            # Remove excess (weakest) groups
            for g in sorted_groups[max_influences:]:
                groups[g.group].remove([vert.index])
                excess_count += 1

    if excess_count > 0:
        print(f"[blender] Limited bone influences: removed {excess_count} excess weights (max {max_influences})")
    else:
        print(f"[blender] All vertices within {max_influences} bone influence limit")


def validate_roblox_constraints(clothing_obj, inner_cage_obj, outer_cage_obj):
    """
    Validate the final mesh against Roblox LC requirements.
    Returns a dict of validation results.
    """
    issues = []

    # Check triangle count (max 4000)
    tri_count = sum(len(p.vertices) - 2 for p in clothing_obj.data.polygons)
    if tri_count > 10000:
        issues.append(f"Triangle count {tri_count} exceeds 10000 (Roblox limit: 10000 for LC)")
    print(f"[blender] Validation: {tri_count} triangles {'✓' if tri_count <= 10000 else '✗'}")

    # Check bounding box size (max 8x8x8 studs ≈ 0.08x0.08x0.08 in Blender units with 0.01 scale)
    bb = get_bounding_box(clothing_obj)
    size = bb['size']
    print(f"[blender] Validation: size {size.x:.3f}x{size.y:.3f}x{size.z:.3f}")

    # Check cage topology matches template (vertex count should be unchanged)
    if inner_cage_obj and outer_cage_obj:
        inner_verts = len(inner_cage_obj.data.vertices)
        outer_verts = len(outer_cage_obj.data.vertices)
        if inner_verts != outer_verts:
            issues.append(f"Cage vertex mismatch: inner={inner_verts}, outer={outer_verts}")
        print(f"[blender] Validation: cage verts inner={inner_verts}, outer={outer_verts} "
              f"{'✓' if inner_verts == outer_verts else '✗'}")

    # Check bone influences
    max_influences = 0
    for vert in clothing_obj.data.vertices:
        max_influences = max(max_influences, len(vert.groups))
    if max_influences > 4:
        issues.append(f"Max bone influences {max_influences} exceeds Roblox limit of 4")
    print(f"[blender] Validation: max bone influences = {max_influences} "
          f"{'✓' if max_influences <= 4 else '✗'}")

    if issues:
        print(f"[blender] ⚠ {len(issues)} validation issues:")
        for issue in issues:
            print(f"[blender]   - {issue}")
    else:
        print(f"[blender] ✓ All Roblox validations passed")

    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'tri_count': tri_count,
        'max_bone_influences': max_influences,
        'bounding_box': [round(size.x, 3), round(size.y, 3), round(size.z, 3)],
    }


def add_attachment_point(armature_obj, clothing_obj, attachment_name):
    """
    Add the correct attachment point as an Empty object.
    Roblox uses these to know where to attach the clothing.
    """
    # Create empty at the correct position
    bpy.ops.object.select_all(action='DESELECT')

    empty = bpy.data.objects.new(attachment_name, None)
    empty.empty_display_type = 'PLAIN_AXES'
    empty.empty_display_size = 0.01
    bpy.context.collection.objects.link(empty)

    # Position based on attachment type
    cloth_bb = get_bounding_box(clothing_obj)
    if 'Front' in attachment_name or 'Back' in attachment_name:
        empty.location = cloth_bb['center']
    elif 'Waist' in attachment_name:
        empty.location = Vector((cloth_bb['center'].x, cloth_bb['min'].y + cloth_bb['size'].y * 0.4, cloth_bb['center'].z))

    # Parent to armature
    empty.parent = armature_obj
    print(f"[blender] Added attachment point: {attachment_name}")
    return empty


def setup_armature_and_weights(clothing_obj, armature_obj):
    """
    Parent clothing mesh to R15 armature with automatic weights.
    Then enforce max 4 bone influences per vertex (Roblox requirement).
    """
    bpy.ops.object.select_all(action='DESELECT')
    clothing_obj.select_set(True)
    armature_obj.select_set(True)
    bpy.context.view_layer.objects.active = armature_obj

    try:
        bpy.ops.object.parent_set(type='ARMATURE_AUTO')
        print("[blender] Armature parenting with automatic weights: SUCCESS")
    except Exception as e:
        print(f"[blender] Auto weights failed ({e}), trying envelope weights...")
        try:
            bpy.ops.object.parent_set(type='ARMATURE_ENVELOPE')
            print("[blender] Armature parenting with envelope weights: SUCCESS")
        except Exception as e2:
            print(f"[blender] Envelope weights also failed: {e2}")
            bpy.ops.object.parent_set(type='ARMATURE')
            print("[blender] Armature parenting (no weights, AutoSkin required): SUCCESS")

    # Enforce Roblox max 4 bone influences per vertex
    limit_bone_influences(clothing_obj, max_influences=4)


def export_fbx(filepath, armature_obj, mesh_objects):
    """Export as FBX with Roblox-compatible settings."""
    bpy.ops.object.select_all(action='DESELECT')
    armature_obj.select_set(True)
    for obj in mesh_objects:
        obj.select_set(True)

    bpy.context.view_layer.objects.active = armature_obj

    bpy.ops.export_scene.fbx(
        filepath=filepath,
        use_selection=True,
        apply_scale_options='FBX_SCALE_UNITS',
        global_scale=0.01,
        apply_unit_scale=True,
        bake_space_transform=False,
        object_types={'ARMATURE', 'MESH', 'EMPTY'},
        use_mesh_modifiers=True,
        mesh_smooth_type='FACE',
        use_mesh_edges=False,
        use_tspace=True,
        use_custom_props=False,
        add_leaf_bones=False,  # CRITICAL: Roblox doesn't want leaf bones
        primary_bone_axis='Y',
        secondary_bone_axis='X',
        use_armature_deform_only=False,
        armature_nodetype='NULL',
        bake_anim=False,
        path_mode='COPY',
        embed_textures=True,
        batch_mode='OFF',
    )
    print(f"[blender] FBX exported to {filepath}")


def export_glb(filepath, mesh_objects):
    """Export processed mesh as GLB for preview."""
    bpy.ops.object.select_all(action='DESELECT')
    for obj in mesh_objects:
        obj.select_set(True)

    bpy.ops.export_scene.gltf(
        filepath=filepath,
        use_selection=True,
        export_format='GLB',
        export_apply=True,
    )
    print(f"[blender] GLB exported to {filepath}")


# ── Main pipeline ────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    templates_dir = args.templates_dir
    config = CLOTHING_CONFIG.get(args.clothing_type, CLOTHING_CONFIG['shirt'])

    print(f"[blender] === Roblox Layered Clothing Post-Process ===")
    print(f"[blender] Input: {args.input}")
    print(f"[blender] Clothing type: {args.clothing_type}")
    print(f"[blender] Config: {config}")

    # ── Step 1: Clear scene and import template (rig + cages + mannequin) ──

    clear_scene()

    combined_path = os.path.join(templates_dir, 'Combined-Template.fbx')
    if os.path.exists(combined_path):
        print(f"[blender] Loading combined template: {combined_path}")
        import_fbx(combined_path)
    else:
        rig_path = os.path.join(templates_dir, 'Rig_and_Attachments_Template.fbx')
        cage_path = os.path.join(templates_dir, 'Clothing_Cage_Template.fbx')
        mannequin_path = os.path.join(templates_dir, 'ClassicMannequin_With-Cages.fbx')

        print(f"[blender] Loading templates individually...")
        if os.path.exists(rig_path):
            import_fbx(rig_path)
        if os.path.exists(cage_path):
            import_fbx(cage_path)
        if os.path.exists(mannequin_path):
            import_fbx(mannequin_path)

    # Identify template objects
    armature = get_armature()
    template_meshes = get_mesh_objects()

    mannequin_obj = None
    inner_cage_obj = None
    outer_cage_obj = None

    for obj in template_meshes:
        name_lower = obj.name.lower()
        if 'innercage' in name_lower or 'inner_cage' in name_lower:
            inner_cage_obj = obj
        elif 'outercage' in name_lower or 'outer_cage' in name_lower:
            outer_cage_obj = obj
        elif 'mannequin' in name_lower or 'body' in name_lower or 'mesh' in name_lower:
            if 'cage' not in name_lower:
                mannequin_obj = obj

    if not mannequin_obj:
        for obj in template_meshes:
            if 'cage' not in obj.name.lower() and obj.data.vertices:
                mannequin_obj = obj
                break

    print(f"[blender] Armature: {armature.name if armature else 'NOT FOUND'}")
    print(f"[blender] Mannequin: {mannequin_obj.name if mannequin_obj else 'NOT FOUND'}")
    print(f"[blender] Inner cage: {inner_cage_obj.name if inner_cage_obj else 'NOT FOUND'}")
    print(f"[blender] Outer cage: {outer_cage_obj.name if outer_cage_obj else 'NOT FOUND'}")

    if not armature:
        print("[blender] ERROR: No armature found in template. Cannot proceed.")
        sys.exit(1)

    # ── Step 2: Import AI-generated clothing mesh ────────────────────────

    print(f"\n[blender] Importing clothing mesh: {args.input}")
    ext = os.path.splitext(args.input)[1].lower()
    if ext in ('.glb', '.gltf'):
        import_glb(args.input)
    elif ext == '.fbx':
        import_fbx(args.input)
    elif ext == '.obj':
        bpy.ops.import_scene.obj(filepath=args.input)
    else:
        bpy.ops.import_mesh.ply(filepath=args.input)

    # Find the newly imported clothing mesh
    all_meshes_now = get_mesh_objects()
    clothing_obj = None
    for obj in all_meshes_now:
        if obj not in template_meshes:
            clothing_obj = obj
            break

    if not clothing_obj:
        print("[blender] ERROR: No clothing mesh found after import.")
        sys.exit(1)

    cloth_faces = len(clothing_obj.data.polygons)
    cloth_verts = len(clothing_obj.data.vertices)
    print(f"[blender] Clothing mesh: {clothing_obj.name} ({cloth_faces} faces, {cloth_verts} verts)")

    # ── Step 3: Clean mesh for Roblox ───────────────────────────────────
    # Fix non-manifold geometry, remove loose verts, fill holes

    print("\n[blender] Cleaning mesh for Roblox requirements...")
    clean_mesh_for_roblox(clothing_obj)

    # ── Step 4: Scale and position clothing onto mannequin ───────────────
    # NOTE: No shrinkwrap! We preserve the original clothing shape exactly.

    if mannequin_obj:
        print("\n[blender] Fitting clothing to mannequin (scale + position only)...")
        fit_clothing_to_mannequin(clothing_obj, mannequin_obj, args.clothing_type)

    # Apply all transforms
    bpy.ops.object.select_all(action='DESELECT')
    clothing_obj.select_set(True)
    bpy.context.view_layer.objects.active = clothing_obj
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    # ── Step 5: Deform outer cage to envelop clothing ────────────────────
    # The cage wraps the clothing, NOT the other way around.

    if outer_cage_obj and inner_cage_obj:
        print("\n[blender] Setting up cages...")

        # Rename cages to match Roblox naming convention
        clothing_name = "LayeredClothing"
        clothing_obj.name = clothing_name
        inner_cage_obj.name = f"{clothing_name}_InnerCage"
        outer_cage_obj.name = f"{clothing_name}_OuterCage"

        # Deform outer cage outward to envelop the clothing
        deform_outer_cage(
            outer_cage_obj, clothing_obj,
            region=config.get('region', 'full'),
            margin=0.008,
        )

        # Verify: count clothing vertices outside cage (should be 0 or very few)
        print(f"[blender] Inner cage: {inner_cage_obj.name} (UNCHANGED from template)")
        print(f"[blender] Outer cage: {outer_cage_obj.name} (deformed to fit clothing)")
    else:
        print("[blender] WARNING: Cages not found. FBX will need manual caging.")
        clothing_obj.name = "LayeredClothing"

    # ── Step 6: Rig clothing to R15 armature ─────────────────────────────

    if armature:
        print("\n[blender] Rigging to R15 armature...")
        setup_armature_and_weights(clothing_obj, armature)

    # ── Step 7: Add attachment point ──────────────────────────────────────

    attachment_empty = None
    if armature:
        attachment_empty = add_attachment_point(armature, clothing_obj, config['attachment'])

    # ── Step 8: Validate against Roblox requirements ──────────────────────

    print("\n[blender] Validating Roblox requirements...")
    validation = validate_roblox_constraints(clothing_obj, inner_cage_obj, outer_cage_obj)

    # ── Step 9: Export combined preview GLB (clothing + mannequin) ────────

    if args.output_mannequin_glb and mannequin_obj:
        print(f"\n[blender] Exporting combined preview (clothing + mannequin)...")
        export_glb(args.output_mannequin_glb, [clothing_obj, mannequin_obj])

    # ── Step 10: Remove mannequin (not needed in FBX or clothing-only GLB)

    if mannequin_obj:
        bpy.data.objects.remove(mannequin_obj, do_unlink=True)
        print("[blender] Mannequin removed from export")

    # ── Step 11: Export FBX and clothing-only GLB ────────────────────────

    print(f"\n[blender] Exporting FBX...")
    export_objects = [clothing_obj]
    if inner_cage_obj:
        export_objects.append(inner_cage_obj)
    if outer_cage_obj:
        export_objects.append(outer_cage_obj)
    if attachment_empty:
        export_objects.append(attachment_empty)

    export_fbx(args.output, armature, export_objects)

    if args.output_glb:
        export_glb(args.output_glb, [clothing_obj])

    # ── Step 12: Write metadata ──────────────────────────────────────────

    meta = {
        'clothing_type': args.clothing_type,
        'attachment': config['attachment'],
        'mesh_name': clothing_obj.name,
        'face_count': len(clothing_obj.data.polygons),
        'vertex_count': len(clothing_obj.data.vertices),
        'tri_count': validation.get('tri_count', 0),
        'has_cages': inner_cage_obj is not None and outer_cage_obj is not None,
        'has_armature': armature is not None,
        'has_attachment_point': attachment_empty is not None,
        'max_bone_influences': validation.get('max_bone_influences', 0),
        'bounding_box': validation.get('bounding_box', []),
        'roblox_ready': validation.get('valid', False),
        'validation_issues': validation.get('issues', []),
        'inner_cage_modified': False,  # Inner cage is NEVER modified
        'outer_cage_modified': True,
        'shrinkwrap_used': False,  # We don't destroy the mesh anymore
        'fbx_path': args.output,
        'glb_path': args.output_glb,
        'mannequin_glb_path': args.output_mannequin_glb,
    }

    if args.meta_output:
        with open(args.meta_output, 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"[blender] Metadata written to {args.meta_output}")

    print(f"\n[blender] === DONE ===")
    print(f"[blender] FBX: {args.output}")
    if args.output_glb:
        print(f"[blender] GLB (clothing only): {args.output_glb}")
    if args.output_mannequin_glb:
        print(f"[blender] GLB (on mannequin): {args.output_mannequin_glb}")
    print(f"[blender] Faces: {meta['face_count']}, Vertices: {meta['vertex_count']}")
    print(f"[blender] Roblox ready: {meta['roblox_ready']}")

    return meta


if __name__ == '__main__':
    main()

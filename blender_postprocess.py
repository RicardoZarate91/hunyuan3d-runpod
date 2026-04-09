#!/usr/bin/env python3
"""
blender_postprocess.py — Blender headless post-processing for Roblox Layered Clothing.

Built from official Roblox documentation:
  - https://create.roblox.com/docs/art/accessories/clothing-specifications
  - https://create.roblox.com/docs/art/accessories/caging-best-practices
  - https://create.roblox.com/docs/art/accessories/creating/exporting

Key principles:
  1. NEVER deform the clothing mesh — preserve original AI-generated shape
  2. Use Roblox's official cage templates — NEVER create cages from scratch
  3. Only move outer cage vertex POSITIONS — never add/remove verts or change UVs
  4. Inner cage stays UNTOUCHED from template
  5. Max 4 bone influences per vertex
  6. Max 4,000 triangles per accessory
  7. Attachment point with _Att suffix for auto-recognition
  8. FBX: scale 0.01, no leaf bones, no animation, embed textures
  9. Remove ALL extra objects before export (lights, cameras, mannequin)
  10. Freeze all transforms (loc 0, rot 0, scale 1)

Usage:
    blender --background --python blender_postprocess.py -- \\
        --input clothing.glb --output clothing_roblox.fbx --clothing-type shirt
"""

import bpy
import bmesh
import mathutils
import sys
import os
import argparse
import json
from mathutils import Vector

# ── Parse arguments ─────────────────────────────────────────────────────────

def parse_args():
    argv = sys.argv
    if '--' in argv:
        argv = argv[argv.index('--') + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--output-glb', default=None)
    parser.add_argument('--clothing-type', default='shirt',
                        choices=['shirt', 'tshirt', 'jacket', 'sweater', 'pants',
                                 'shorts', 'dress', 'skirt', 'full'])
    parser.add_argument('--templates-dir', default='/opt/roblox-templates')
    parser.add_argument('--output-mannequin-glb', default=None)
    parser.add_argument('--meta-output', default=None)
    parser.add_argument('--skip-retopo', action='store_true')
    return parser.parse_args(argv)


# ── Clothing type config (from Roblox docs) ─────────────────────────────────

CLOTHING_CONFIG = {
    # attachment_name must match Roblox's exact names
    # _Att suffix is added at export for Studio auto-recognition
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

# R15 bone hierarchy (from Roblox docs — exact names required)
R15_BONES = [
    'Root', 'HumanoidRootNode', 'LowerTorso', 'UpperTorso', 'Head',
    'LeftUpperArm', 'LeftLowerArm', 'LeftHand',
    'RightUpperArm', 'RightLowerArm', 'RightHand',
    'LeftUpperLeg', 'LeftLowerLeg', 'LeftFoot',
    'RightUpperLeg', 'RightLowerLeg', 'RightFoot',
]

# ── Scene helpers ────────────────────────────────────────────────────────────

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.armatures:
        if block.users == 0:
            bpy.data.armatures.remove(block)


def import_fbx(filepath):
    bpy.ops.import_scene.fbx(filepath=filepath, use_anim=False)


def import_glb(filepath):
    bpy.ops.import_scene.gltf(filepath=filepath)


def get_mesh_objects():
    return [obj for obj in bpy.data.objects if obj.type == 'MESH']


def get_armature():
    for obj in bpy.data.objects:
        if obj.type == 'ARMATURE':
            return obj
    return None


def get_bounding_box(obj):
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


# ── Mesh cleanup (Roblox requires watertight, manifold geometry) ─────────────

def clean_mesh_for_roblox(clothing_obj):
    """
    Roblox requires: watertight, no holes, no non-manifold edges,
    no degenerate faces, single mesh, prefer quads.
    """
    bpy.ops.object.select_all(action='DESELECT')
    clothing_obj.select_set(True)
    bpy.context.view_layer.objects.active = clothing_obj

    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')

    # Remove zero-area faces
    bpy.ops.mesh.dissolve_degenerate(threshold=0.0001)

    # Remove loose vertices/edges not connected to faces
    bpy.ops.mesh.delete_loose(use_verts=True, use_edges=True, use_faces=False)

    # Fix non-manifold geometry
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.mesh.select_non_manifold(extend=False)

    bpy.ops.object.mode_set(mode='OBJECT')
    non_manifold = sum(1 for v in clothing_obj.data.vertices if v.select)

    if non_manifold > 0:
        print(f"[blender] {non_manifold} non-manifold vertices found, filling holes...")
        bpy.ops.object.mode_set(mode='EDIT')
        try:
            bpy.ops.mesh.fill_holes(sides=4)
        except Exception:
            pass
        # Recalculate normals outward (required for watertight check)
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.object.mode_set(mode='OBJECT')
    else:
        print("[blender] Mesh is manifold ✓")

    # Ensure normals face outward even if mesh was clean
    bpy.ops.object.select_all(action='DESELECT')
    clothing_obj.select_set(True)
    bpy.context.view_layer.objects.active = clothing_obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode='OBJECT')

    print(f"[blender] After cleanup: {len(clothing_obj.data.polygons)} faces, "
          f"{len(clothing_obj.data.vertices)} verts")


# ── Fit clothing to mannequin (scale + translate ONLY, no vertex deformation) ─

def fit_clothing_to_mannequin(clothing_obj, mannequin_obj, clothing_type):
    """
    Scale and position clothing onto mannequin. NEVER deforms vertices.
    """
    mannequin_bb = get_bounding_box(mannequin_obj)
    cloth_bb = get_bounding_box(clothing_obj)
    region = CLOTHING_CONFIG.get(clothing_type, {}).get('region', 'full')

    mannequin_height = mannequin_bb['size'].y
    cloth_height = cloth_bb['size'].y

    if cloth_height <= 0:
        print("[blender] WARNING: Zero-height clothing, skipping fit")
        return

    # Scale targets per region
    target_ratios = {'upper': 0.50, 'lower': 0.55, 'full': 0.90}
    target_height = mannequin_height * target_ratios.get(region, 0.90)

    scale_factor = target_height / cloth_height
    clothing_obj.scale = (scale_factor, scale_factor, scale_factor)
    bpy.context.view_layer.update()

    # Recompute after scale
    cloth_bb = get_bounding_box(clothing_obj)

    # Position alignment
    x_off = mannequin_bb['center'].x - cloth_bb['center'].x
    z_off = mannequin_bb['center'].z - cloth_bb['center'].z

    if region == 'upper':
        y_off = mannequin_bb['max'].y - cloth_bb['max'].y
    elif region == 'lower':
        y_off = mannequin_bb['min'].y - cloth_bb['min'].y
    else:
        y_off = mannequin_bb['center'].y - cloth_bb['center'].y

    clothing_obj.location += Vector((x_off, y_off, z_off))
    bpy.context.view_layer.update()

    final_bb = get_bounding_box(clothing_obj)
    print(f"[blender] Fitted: scale={scale_factor:.3f}, "
          f"size={final_bb['size'].x:.3f}x{final_bb['size'].y:.3f}x{final_bb['size'].z:.3f}")


# ── Outer cage deformation (official Roblox workflow) ────────────────────────

def deform_outer_cage(outer_cage_obj, clothing_obj, region='full', margin=0.008):
    """
    Push outer cage vertices outward to envelop the clothing mesh.

    CRITICAL Roblox rules enforced:
    - NEVER add/remove vertices (topology must match template exactly)
    - NEVER modify UVs
    - Only modify vertices in regions covered by clothing
    - Keep cage as tight as possible (oversized = deformation issues)
    - Smooth displacement to avoid abrupt jumps
    - Never push outer cage inside inner cage
    """
    from mathutils.bvhtree import BVHTree

    depsgraph = bpy.context.evaluated_depsgraph_get()
    cloth_bvh = BVHTree.FromObject(clothing_obj, depsgraph)

    cage_mesh = outer_cage_obj.data
    cage_world = outer_cage_obj.matrix_world
    cage_world_inv = cage_world.inverted()

    # Record original vertex count for safety check
    original_vert_count = len(cage_mesh.vertices)
    original_face_count = len(cage_mesh.polygons)

    cloth_bb = get_bounding_box(clothing_obj)
    cage_bb = get_bounding_box(outer_cage_obj)

    # Only edit cage vertices in the Y-range covered by clothing
    y_min_edit = cloth_bb['min'].y - margin * 5
    y_max_edit = cloth_bb['max'].y + margin * 5

    print(f"[blender] Cage edit Y-range: [{y_min_edit:.3f}, {y_max_edit:.3f}]")

    # First pass: compute displacements
    displacements = {}

    for vert in cage_mesh.vertices:
        cage_world_pos = cage_world @ vert.co

        # Skip vertices outside clothing region
        if cage_world_pos.y < y_min_edit or cage_world_pos.y > y_max_edit:
            continue

        nearest, normal, face_idx, dist = cloth_bvh.find_nearest(cage_world_pos)
        if nearest is None:
            continue

        # Skip if clothing is far from this vertex (body part not covered)
        max_dist = max(cage_bb['size']) * 0.3
        if dist > max_dist:
            continue

        # Outward direction from cage center
        cage_center = cage_bb['center']
        outward_dir = (cage_world_pos - cage_center).normalized()

        # Project distances along outward direction
        cage_dist = (cage_world_pos - cage_center).dot(outward_dir)
        cloth_dist = (nearest - cage_center).dot(outward_dir)

        # Push cage outward if clothing is at or beyond cage surface
        if cloth_dist > cage_dist - margin:
            new_dist = cloth_dist + margin
            displacements[vert.index] = cage_center + outward_dir * new_dist

    print(f"[blender] Outer cage: {len(displacements)}/{len(cage_mesh.vertices)} vertices need adjustment")

    if not displacements:
        print("[blender] Clothing fits within cage, no adjustment needed")
        return

    # Second pass: smooth displacement (2 iterations, avoid abrupt jumps)
    adjacency = {i: set() for i in range(len(cage_mesh.vertices))}
    for edge in cage_mesh.edges:
        adjacency[edge.vertices[0]].add(edge.vertices[1])
        adjacency[edge.vertices[1]].add(edge.vertices[0])

    for _ in range(2):
        smoothed = {}
        for vi, new_pos in displacements.items():
            neighbors = []
            for ni in adjacency[vi]:
                if ni in displacements:
                    neighbors.append(displacements[ni])
                else:
                    neighbors.append(cage_world @ cage_mesh.vertices[ni].co)

            if neighbors:
                avg = Vector((0, 0, 0))
                for n in neighbors:
                    avg += n
                avg /= len(neighbors)
                smoothed[vi] = new_pos * 0.6 + avg * 0.4
            else:
                smoothed[vi] = new_pos
        displacements = smoothed

    # Apply
    for vi, new_world_pos in displacements.items():
        cage_mesh.vertices[vi].co = cage_world_inv @ new_world_pos

    cage_mesh.update()

    # SAFETY: verify topology unchanged (Roblox will reject modified topology)
    assert len(cage_mesh.vertices) == original_vert_count, \
        f"CAGE VERTEX COUNT CHANGED: {original_vert_count} -> {len(cage_mesh.vertices)}"
    assert len(cage_mesh.polygons) == original_face_count, \
        f"CAGE FACE COUNT CHANGED: {original_face_count} -> {len(cage_mesh.polygons)}"

    print(f"[blender] Outer cage: {len(displacements)} vertices displaced (topology preserved ✓)")


# ── Bone influence limiting (Roblox max 4 per vertex) ────────────────────────

def limit_bone_influences(clothing_obj, max_influences=4):
    """Remove weakest bone weights if any vertex exceeds the limit."""
    mesh = clothing_obj.data
    groups = clothing_obj.vertex_groups
    if not groups:
        return

    excess_count = 0
    for vert in mesh.vertices:
        if len(vert.groups) > max_influences:
            sorted_groups = sorted(vert.groups, key=lambda g: g.weight, reverse=True)
            for g in sorted_groups[max_influences:]:
                groups[g.group].remove([vert.index])
                excess_count += 1

    if excess_count > 0:
        # Normalize remaining weights so they sum to 1.0
        for vert in mesh.vertices:
            total = sum(g.weight for g in vert.groups)
            if total > 0 and abs(total - 1.0) > 0.001:
                for g in vert.groups:
                    groups[g.group].add([vert.index], g.weight / total, 'REPLACE')

        print(f"[blender] Bone influences: removed {excess_count} excess, normalized weights ✓")
    else:
        print(f"[blender] Bone influences: all within {max_influences} limit ✓")


# ── Armature setup ───────────────────────────────────────────────────────────

def setup_armature_and_weights(clothing_obj, armature_obj):
    """Parent to R15 armature with auto weights, then limit to 4 influences."""
    bpy.ops.object.select_all(action='DESELECT')
    clothing_obj.select_set(True)
    armature_obj.select_set(True)
    bpy.context.view_layer.objects.active = armature_obj

    try:
        bpy.ops.object.parent_set(type='ARMATURE_AUTO')
        print("[blender] Auto weights: SUCCESS ✓")
    except Exception as e:
        print(f"[blender] Auto weights failed ({e}), trying envelope...")
        try:
            bpy.ops.object.parent_set(type='ARMATURE_ENVELOPE')
            print("[blender] Envelope weights: SUCCESS ✓")
        except Exception as e2:
            print(f"[blender] Envelope failed ({e2}), parenting without weights...")
            bpy.ops.object.parent_set(type='ARMATURE')
            print("[blender] No weights (Roblox AutoSkin will handle it)")

    # Roblox: max 4 bone influences per vertex
    limit_bone_influences(clothing_obj, max_influences=4)

    # Verify: do NOT apply weights to Root bone (Roblox docs say don't)
    if 'Root' in [g.name for g in clothing_obj.vertex_groups]:
        root_group = clothing_obj.vertex_groups['Root']
        # Zero out all Root weights
        for vert in clothing_obj.data.vertices:
            try:
                root_group.remove([vert.index])
            except RuntimeError:
                pass
        print("[blender] Removed Root bone weights (Roblox requirement) ✓")


# ── Attachment point (Roblox _Att suffix for auto-recognition) ────────────────

def add_attachment_point(armature_obj, clothing_obj, attachment_name):
    """
    Add attachment as an Empty with _Att suffix.
    Roblox Studio auto-recognizes objects with _Att suffix as attachment points.
    """
    # Use _Att suffix for auto-recognition by Studio
    att_obj_name = f"{attachment_name}_Att"

    empty = bpy.data.objects.new(att_obj_name, None)
    empty.empty_display_type = 'PLAIN_AXES'
    empty.empty_display_size = 0.01
    bpy.context.collection.objects.link(empty)

    # Position at clothing center (attachment origin)
    cloth_bb = get_bounding_box(clothing_obj)
    if 'Waist' in attachment_name:
        # Waist attachments: position at upper part of lower clothing
        empty.location = Vector((
            cloth_bb['center'].x,
            cloth_bb['min'].y + cloth_bb['size'].y * 0.4,
            cloth_bb['center'].z
        ))
    else:
        # Body front/back: center of clothing
        empty.location = Vector((
            cloth_bb['center'].x,
            cloth_bb['center'].y,
            cloth_bb['center'].z
        ))

    # Parent to armature
    empty.parent = armature_obj
    print(f"[blender] Attachment: {att_obj_name} ✓")
    return empty


# ── Freeze transforms (Roblox requires loc=0, rot=0, scale=1) ────────────────

def freeze_transforms(obj):
    """Apply all transforms so object has identity transform."""
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)


# ── Validation (matches Roblox Studio upload checks) ─────────────────────────

def validate_roblox(clothing_obj, inner_cage_obj, outer_cage_obj, armature_obj):
    """
    Validate against official Roblox LC requirements.
    Returns dict with 'valid' bool and 'issues' list.
    """
    issues = []

    # 1. Triangle count (max 4,000 for accessories)
    tri_count = sum(len(p.vertices) - 2 for p in clothing_obj.data.polygons)
    if tri_count > 4000:
        issues.append(f"Triangles: {tri_count} > 4,000 limit")
    print(f"[blender] Validation: {tri_count} tris {'✓' if tri_count <= 4000 else '✗ OVER LIMIT'}")

    # 2. Single mesh check
    mesh_count = len([o for o in bpy.data.objects if o.type == 'MESH'
                      and 'cage' not in o.name.lower()
                      and 'mannequin' not in o.name.lower()])
    if mesh_count > 1:
        issues.append(f"Multiple clothing meshes found ({mesh_count}), Roblox requires 1")

    # 3. Bone influences (max 4)
    max_influences = 0
    for vert in clothing_obj.data.vertices:
        max_influences = max(max_influences, len(vert.groups))
    if max_influences > 4:
        issues.append(f"Bone influences: {max_influences} > 4 limit")
    print(f"[blender] Validation: max bone influences = {max_influences} "
          f"{'✓' if max_influences <= 4 else '✗'}")

    # 4. Cage topology integrity
    if inner_cage_obj and outer_cage_obj:
        inner_v = len(inner_cage_obj.data.vertices)
        outer_v = len(outer_cage_obj.data.vertices)
        if inner_v != outer_v:
            issues.append(f"Cage vertex mismatch: inner={inner_v} != outer={outer_v}")
        print(f"[blender] Validation: cages inner={inner_v}, outer={outer_v} "
              f"{'✓' if inner_v == outer_v else '✗'}")
    else:
        issues.append("Missing cage meshes")

    # 5. Armature bone names
    if armature_obj:
        bone_names = [b.name for b in armature_obj.data.bones]
        missing_bones = [b for b in ['LowerTorso', 'UpperTorso', 'Head'] if b not in bone_names]
        if missing_bones:
            issues.append(f"Missing R15 bones: {missing_bones}")
        print(f"[blender] Validation: R15 bones {'✓' if not missing_bones else '✗ missing: ' + str(missing_bones)}")

    # 6. Bounding box (max 8x8x8 studs, but at 0.01 scale that's larger in Blender units)
    bb = get_bounding_box(clothing_obj)
    size = bb['size']
    print(f"[blender] Validation: bbox {size.x:.3f}x{size.y:.3f}x{size.z:.3f}")

    # 7. Transforms frozen
    loc = clothing_obj.location
    rot = clothing_obj.rotation_euler
    scl = clothing_obj.scale
    transforms_clean = (
        abs(loc.x) < 0.001 and abs(loc.y) < 0.001 and abs(loc.z) < 0.001 and
        abs(rot.x) < 0.001 and abs(rot.y) < 0.001 and abs(rot.z) < 0.001 and
        abs(scl.x - 1) < 0.001 and abs(scl.y - 1) < 0.001 and abs(scl.z - 1) < 0.001
    )
    if not transforms_clean:
        issues.append("Transforms not frozen (loc/rot/scale should be 0/0/1)")
    print(f"[blender] Validation: transforms {'✓ frozen' if transforms_clean else '✗ not frozen'}")

    if issues:
        print(f"[blender] ⚠ {len(issues)} validation issues:")
        for i in issues:
            print(f"[blender]   - {i}")
    else:
        print(f"[blender] ✓ ALL Roblox validations passed")

    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'tri_count': tri_count,
        'max_bone_influences': max_influences,
        'bounding_box': [round(size.x, 3), round(size.y, 3), round(size.z, 3)],
    }


# ── FBX export (exact Roblox settings from official docs) ────────────────────

def export_fbx(filepath, armature_obj, mesh_objects):
    """
    Export FBX with exact Roblox-required settings.
    From: https://create.roblox.com/docs/art/accessories/creating/exporting
    """
    bpy.ops.object.select_all(action='DESELECT')
    armature_obj.select_set(True)
    for obj in mesh_objects:
        obj.select_set(True)

    bpy.context.view_layer.objects.active = armature_obj

    # Set scene units to centimeters with 0.01 scale (Roblox requirement)
    bpy.context.scene.unit_settings.system = 'METRIC'
    bpy.context.scene.unit_settings.scale_length = 0.01
    bpy.context.scene.unit_settings.length_unit = 'CENTIMETERS'

    bpy.ops.export_scene.fbx(
        filepath=filepath,
        use_selection=True,
        apply_scale_options='FBX_SCALE_UNITS',  # Use scene Unit Scale
        global_scale=0.01,                       # Roblox: 0.01
        apply_unit_scale=True,
        bake_space_transform=False,
        object_types={'ARMATURE', 'MESH', 'EMPTY'},
        use_mesh_modifiers=True,
        mesh_smooth_type='FACE',
        use_mesh_edges=False,
        use_tspace=True,
        use_custom_props=False,
        add_leaf_bones=False,    # CRITICAL: Roblox rejects leaf bones
        primary_bone_axis='Y',
        secondary_bone_axis='X',
        use_armature_deform_only=False,
        armature_nodetype='NULL',
        bake_anim=False,         # CRITICAL: no animation data
        path_mode='COPY',        # Roblox: Copy mode
        embed_textures=True,     # Roblox: embed textures
        batch_mode='OFF',
    )
    print(f"[blender] FBX exported: {filepath}")


def export_glb(filepath, mesh_objects):
    bpy.ops.object.select_all(action='DESELECT')
    for obj in mesh_objects:
        obj.select_set(True)
    bpy.ops.export_scene.gltf(
        filepath=filepath, use_selection=True,
        export_format='GLB', export_apply=True,
    )
    print(f"[blender] GLB exported: {filepath}")


# ── Main pipeline ────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    config = CLOTHING_CONFIG.get(args.clothing_type, CLOTHING_CONFIG['shirt'])

    print(f"[blender] ═══════════════════════════════════════════════")
    print(f"[blender] ROBLOX LAYERED CLOTHING POST-PROCESS")
    print(f"[blender] Input: {args.input}")
    print(f"[blender] Type: {args.clothing_type} → {config['attachment']}")
    print(f"[blender] ═══════════════════════════════════════════════")

    # ── 1. Load Roblox template ──────────────────────────────────────────

    clear_scene()

    templates_dir = args.templates_dir
    combined_path = os.path.join(templates_dir, 'Combined-Template.fbx')

    if os.path.exists(combined_path):
        print(f"\n[blender] Loading: Combined-Template.fbx")
        import_fbx(combined_path)
    else:
        # Fallback to individual files
        for fname in ['Rig_and_Attachments_Template.fbx',
                      'Clothing_Cage_Template.fbx',
                      'ClassicMannequin_With-Cages.fbx']:
            fpath = os.path.join(templates_dir, fname)
            if os.path.exists(fpath):
                print(f"[blender] Loading: {fname}")
                import_fbx(fpath)

    # Find template objects
    armature = get_armature()
    template_meshes = get_mesh_objects()

    mannequin_obj = None
    inner_cage_obj = None
    outer_cage_obj = None

    for obj in template_meshes:
        name = obj.name.lower()
        if 'innercage' in name or 'inner_cage' in name:
            inner_cage_obj = obj
        elif 'outercage' in name or 'outer_cage' in name:
            outer_cage_obj = obj
        elif 'cage' not in name and len(obj.data.vertices) > 100:
            if not mannequin_obj:
                mannequin_obj = obj

    print(f"[blender] Armature:    {armature.name if armature else '✗ NOT FOUND'}")
    print(f"[blender] Mannequin:   {mannequin_obj.name if mannequin_obj else '✗ NOT FOUND'}")
    print(f"[blender] Inner cage:  {inner_cage_obj.name if inner_cage_obj else '✗ NOT FOUND'}")
    print(f"[blender] Outer cage:  {outer_cage_obj.name if outer_cage_obj else '✗ NOT FOUND'}")

    if not armature:
        print("[blender] FATAL: No armature in template")
        sys.exit(1)

    # Record cage vertex/face counts (for post-validation)
    cage_vert_count = len(outer_cage_obj.data.vertices) if outer_cage_obj else 0
    cage_face_count = len(outer_cage_obj.data.polygons) if outer_cage_obj else 0
    print(f"[blender] Cage template: {cage_vert_count} verts, {cage_face_count} faces")

    # ── 2. Import clothing mesh ──────────────────────────────────────────

    print(f"\n[blender] Importing: {args.input}")
    ext = os.path.splitext(args.input)[1].lower()
    if ext in ('.glb', '.gltf'):
        import_glb(args.input)
    elif ext == '.fbx':
        import_fbx(args.input)
    elif ext == '.obj':
        bpy.ops.import_scene.obj(filepath=args.input)
    else:
        bpy.ops.import_mesh.ply(filepath=args.input)

    # Find the new mesh (not in template_meshes)
    clothing_obj = None
    for obj in get_mesh_objects():
        if obj not in template_meshes:
            clothing_obj = obj
            break

    if not clothing_obj:
        print("[blender] FATAL: No clothing mesh found after import")
        sys.exit(1)

    print(f"[blender] Clothing: {clothing_obj.name} "
          f"({len(clothing_obj.data.polygons)} faces, {len(clothing_obj.data.vertices)} verts)")

    # ── 3. Clean mesh ────────────────────────────────────────────────────

    print(f"\n[blender] Step 3: Mesh cleanup")
    clean_mesh_for_roblox(clothing_obj)

    # ── 4. Fit to mannequin (NO vertex deformation) ──────────────────────

    if mannequin_obj:
        print(f"\n[blender] Step 4: Fit to mannequin (scale + position only)")
        fit_clothing_to_mannequin(clothing_obj, mannequin_obj, args.clothing_type)

    # Freeze transforms (Roblox requires 0/0/1)
    freeze_transforms(clothing_obj)

    # ── 5. Rename + deform outer cage ────────────────────────────────────

    clothing_name = "LayeredClothing"
    clothing_obj.name = clothing_name

    if outer_cage_obj and inner_cage_obj:
        print(f"\n[blender] Step 5: Cage setup")

        # Roblox naming: MeshName_InnerCage / MeshName_OuterCage
        inner_cage_obj.name = f"{clothing_name}_InnerCage"
        outer_cage_obj.name = f"{clothing_name}_OuterCage"

        deform_outer_cage(
            outer_cage_obj, clothing_obj,
            region=config['region'],
            margin=0.008,
        )

        # Post-check: cage topology MUST be unchanged
        assert len(outer_cage_obj.data.vertices) == cage_vert_count, "Cage topology corrupted!"
        print(f"[blender] Inner cage: UNTOUCHED from template ✓")
        print(f"[blender] Outer cage: vertices repositioned, topology preserved ✓")
    else:
        print("[blender] WARNING: Cages not found, FBX needs manual caging")

    # ── 6. Rig to R15 armature ───────────────────────────────────────────

    print(f"\n[blender] Step 6: R15 rigging")
    setup_armature_and_weights(clothing_obj, armature)

    # ── 7. Attachment point ──────────────────────────────────────────────

    print(f"\n[blender] Step 7: Attachment point")
    attachment_empty = add_attachment_point(armature, clothing_obj, config['attachment'])

    # ── 8. Validate ──────────────────────────────────────────────────────

    print(f"\n[blender] Step 8: Roblox validation")
    validation = validate_roblox(clothing_obj, inner_cage_obj, outer_cage_obj, armature)

    # ── 9. Preview GLB (clothing + mannequin, for web viewer) ────────────

    if args.output_mannequin_glb and mannequin_obj:
        print(f"\n[blender] Exporting preview with mannequin...")
        export_glb(args.output_mannequin_glb, [clothing_obj, mannequin_obj])

    # ── 10. Clean scene for export ───────────────────────────────────────
    # Roblox docs: remove ALL extra objects (lights, cameras, mannequin)

    # Remove mannequin
    if mannequin_obj:
        bpy.data.objects.remove(mannequin_obj, do_unlink=True)

    # Remove any lights or cameras
    for obj in list(bpy.data.objects):
        if obj.type in ('LIGHT', 'CAMERA'):
            bpy.data.objects.remove(obj, do_unlink=True)

    print("[blender] Scene cleaned: only clothing + cages + armature + attachment remain")

    # ── 11. Export FBX ───────────────────────────────────────────────────

    print(f"\n[blender] Step 11: FBX export")
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

    # ── 12. Metadata ─────────────────────────────────────────────────────

    meta = {
        'clothing_type': args.clothing_type,
        'attachment': config['attachment'],
        'mesh_name': clothing_name,
        'face_count': len(clothing_obj.data.polygons),
        'vertex_count': len(clothing_obj.data.vertices),
        'tri_count': validation['tri_count'],
        'has_cages': inner_cage_obj is not None and outer_cage_obj is not None,
        'has_armature': True,
        'has_attachment_point': True,
        'attachment_name': f"{config['attachment']}_Att",
        'max_bone_influences': validation['max_bone_influences'],
        'bounding_box': validation['bounding_box'],
        'roblox_ready': validation['valid'],
        'validation_issues': validation['issues'],
        'cage_verts_preserved': cage_vert_count,
        'inner_cage_modified': False,
        'outer_cage_modified': True,
        'shrinkwrap_used': False,
        'auto_skin_recommended': 'EnabledPreserve',
        'fbx_path': args.output,
    }

    if args.meta_output:
        with open(args.meta_output, 'w') as f:
            json.dump(meta, f, indent=2)

    print(f"\n[blender] ═══════════════════════════════════════════════")
    print(f"[blender] DONE — Roblox LC FBX ready")
    print(f"[blender] FBX:          {args.output}")
    print(f"[blender] Tris:         {meta['tri_count']} / 4,000 limit")
    print(f"[blender] Cages:        {'✓' if meta['has_cages'] else '✗'}")
    print(f"[blender] R15 rig:      ✓")
    print(f"[blender] Attachment:   {meta['attachment_name']}")
    print(f"[blender] Bone infl.:   {meta['max_bone_influences']} / 4 max")
    print(f"[blender] Valid:        {'✓ PASS' if meta['roblox_ready'] else '✗ ISSUES: ' + str(meta['validation_issues'])}")
    print(f"[blender] AutoSkin:     {meta['auto_skin_recommended']}")
    print(f"[blender] ═══════════════════════════════════════════════")

    return meta


if __name__ == '__main__':
    main()

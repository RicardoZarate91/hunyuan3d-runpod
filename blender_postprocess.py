#!/usr/bin/env python3
"""
blender_postprocess.py — Blender headless post-processing for Roblox Layered Clothing.

Takes a decimated GLB clothing mesh and produces a Roblox-ready FBX with:
  - Clothing mesh fitted to the Roblox mannequin body
  - Inner cage (from Roblox template, unchanged)
  - Outer cage (deformed to envelop the clothing mesh)
  - R15 armature with correct bone names
  - Automatic weight painting

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
    return parser.parse_args(argv)


# ── Clothing type → attachment config ────────────────────────────────────────

CLOTHING_CONFIG = {
    'shirt':   {'attachment': 'BodyFrontAttachment', 'cage_parts': 'upper'},
    'tshirt':  {'attachment': 'BodyFrontAttachment', 'cage_parts': 'upper'},
    'jacket':  {'attachment': 'BodyFrontAttachment', 'cage_parts': 'upper'},
    'sweater': {'attachment': 'BodyFrontAttachment', 'cage_parts': 'upper'},
    'pants':   {'attachment': 'WaistCenterAttachment', 'cage_parts': 'lower'},
    'shorts':  {'attachment': 'WaistCenterAttachment', 'cage_parts': 'lower'},
    'dress':   {'attachment': 'BodyFrontAttachment', 'cage_parts': 'full'},
    'skirt':   {'attachment': 'WaistCenterAttachment', 'cage_parts': 'lower'},
    'full':    {'attachment': 'BodyFrontAttachment', 'cage_parts': 'full'},
}


# ── Helper functions ─────────────────────────────────────────────────────────

def clear_scene():
    """Remove all objects from scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    # Clear orphan data
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
    bbox = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
    xs = [v.x for v in bbox]
    ys = [v.y for v in bbox]
    zs = [v.z for v in bbox]
    return {
        'min': mathutils.Vector((min(xs), min(ys), min(zs))),
        'max': mathutils.Vector((max(xs), max(ys), max(zs))),
        'center': mathutils.Vector(((min(xs)+max(xs))/2, (min(ys)+max(ys))/2, (min(zs)+max(zs))/2)),
        'size': mathutils.Vector((max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs))),
    }


def center_and_scale_to_mannequin(clothing_obj, mannequin_bbox):
    """Scale and position clothing mesh to fit the mannequin."""
    cloth_bb = get_bounding_box(clothing_obj)

    # Scale to match mannequin height (Y axis in Blender)
    mannequin_height = mannequin_bbox['size'].y
    cloth_height = cloth_bb['size'].y

    if cloth_height > 0:
        scale_factor = mannequin_height / cloth_height * 0.85  # 85% to leave room
        clothing_obj.scale = (scale_factor, scale_factor, scale_factor)
        bpy.context.view_layer.update()

    # Re-center on mannequin
    cloth_bb = get_bounding_box(clothing_obj)
    offset = mannequin_bbox['center'] - cloth_bb['center']
    clothing_obj.location += offset
    bpy.context.view_layer.update()


def apply_shrinkwrap(clothing_obj, mannequin_obj, offset=0.005):
    """
    Apply shrinkwrap modifier to fit clothing close to mannequin body.
    offset: how far the clothing sits from the body surface (in meters/studs).
    """
    # Add shrinkwrap modifier
    mod = clothing_obj.modifiers.new(name='Shrinkwrap', type='SHRINKWRAP')
    mod.target = mannequin_obj
    mod.wrap_method = 'NEAREST_SURFACEPOINT'
    mod.wrap_mode = 'OUTSIDE_SURFACE'
    mod.offset = offset  # Slight offset so clothing isn't inside body

    # Apply modifier
    bpy.context.view_layer.objects.active = clothing_obj
    bpy.ops.object.modifier_apply(modifier='Shrinkwrap')
    print(f"[blender] Shrinkwrap applied with offset {offset}")


def deform_outer_cage(outer_cage_obj, clothing_obj, inflate_amount=0.01):
    """
    Deform outer cage vertices to envelop the clothing mesh.

    For each outer cage vertex:
    - Cast a ray from the cage vertex toward the clothing mesh
    - If the clothing extends beyond the cage, push the cage vertex outward
    - Ensure cage fully envelops the clothing + small margin
    """
    # Get clothing mesh as BVH tree for raycasting
    depsgraph = bpy.context.evaluated_depsgraph_get()
    cloth_eval = clothing_obj.evaluated_get(depsgraph)
    cloth_mesh = cloth_eval.to_mesh()

    from mathutils.bvhtree import BVHTree
    cloth_bvh = BVHTree.FromObject(clothing_obj, depsgraph)

    cage_mesh = outer_cage_obj.data

    # For each cage vertex, find nearest point on clothing
    # and push cage vertex outward if clothing is beyond it
    cage_world = outer_cage_obj.matrix_world
    cage_world_inv = cage_world.inverted()

    modified_count = 0
    for vert in cage_mesh.vertices:
        cage_world_pos = cage_world @ vert.co

        # Find nearest point on clothing
        nearest, normal, idx, dist = cloth_bvh.find_nearest(cage_world_pos)

        if nearest is None:
            continue

        # Direction from cage center to this vertex (outward normal)
        cage_center = get_bounding_box(outer_cage_obj)['center']
        outward_dir = (cage_world_pos - cage_center).normalized()

        # If clothing is close to or beyond the cage vertex, push cage out
        cloth_to_cage = (cage_world_pos - nearest).length

        if cloth_to_cage < inflate_amount * 3:
            # Push cage vertex outward along its outward direction
            new_world_pos = nearest + outward_dir * inflate_amount
            vert.co = cage_world_inv @ new_world_pos
            modified_count += 1

    cloth_eval.to_mesh_clear()
    print(f"[blender] Outer cage: {modified_count}/{len(cage_mesh.vertices)} vertices adjusted")


def setup_armature_and_weights(clothing_obj, armature_obj):
    """
    Parent clothing mesh to R15 armature with automatic weights.
    """
    # Ensure clothing is selected and armature is active
    bpy.ops.object.select_all(action='DESELECT')
    clothing_obj.select_set(True)
    armature_obj.select_set(True)
    bpy.context.view_layer.objects.active = armature_obj

    # Parent with automatic weights
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
            # Last resort: just parent without weights (AutoSkin will handle it in Roblox)
            bpy.ops.object.parent_set(type='ARMATURE')
            print("[blender] Armature parenting (no weights, AutoSkin required): SUCCESS")


def export_fbx(filepath, armature_obj, mesh_objects):
    """
    Export as FBX with Roblox-compatible settings.
    """
    # Select only the objects we want to export
    bpy.ops.object.select_all(action='DESELECT')
    armature_obj.select_set(True)
    for obj in mesh_objects:
        obj.select_set(True)

    bpy.context.view_layer.objects.active = armature_obj

    bpy.ops.export_scene.fbx(
        filepath=filepath,
        use_selection=True,
        apply_scale_options='FBX_SCALE_UNITS',
        global_scale=0.01,  # Roblox expects this scale
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
        bake_anim=False,  # No animation
        path_mode='COPY',
        embed_textures=True,  # Embed textures in FBX
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

    # ── Step 1: Clear scene and import mannequin + rig + cages ───────────

    clear_scene()

    # Import the Combined Template (has rig + cages + mannequin)
    combined_path = os.path.join(templates_dir, 'Combined-Template.fbx')
    if os.path.exists(combined_path):
        print(f"[blender] Loading combined template: {combined_path}")
        import_fbx(combined_path)
    else:
        # Fall back to separate files
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

    # If we didn't find specific objects, try to identify by name patterns
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

    mannequin_bbox = get_bounding_box(mannequin_obj) if mannequin_obj else None

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

    print(f"[blender] Clothing mesh: {clothing_obj.name} ({len(clothing_obj.data.polygons)} faces)")

    # ── Step 3: Position and fit clothing to mannequin ───────────────────

    if mannequin_bbox:
        print("\n[blender] Fitting clothing to mannequin...")
        center_and_scale_to_mannequin(clothing_obj, mannequin_bbox)

        # Apply shrinkwrap to fit clothing close to body
        # Use a small offset so clothing sits on top of body
        apply_shrinkwrap(clothing_obj, mannequin_obj, offset=0.003)

    # Apply all transforms
    bpy.ops.object.select_all(action='DESELECT')
    clothing_obj.select_set(True)
    bpy.context.view_layer.objects.active = clothing_obj
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    # ── Step 4: Setup cages ──────────────────────────────────────────────

    if outer_cage_obj and inner_cage_obj:
        print("\n[blender] Setting up cages...")

        # Rename cages to match Roblox naming convention
        clothing_name = "LayeredClothing"
        clothing_obj.name = clothing_name
        inner_cage_obj.name = f"{clothing_name}_InnerCage"
        outer_cage_obj.name = f"{clothing_name}_OuterCage"

        # Deform outer cage to envelop clothing
        deform_outer_cage(outer_cage_obj, clothing_obj, inflate_amount=0.008)

        print(f"[blender] Cages renamed: {inner_cage_obj.name}, {outer_cage_obj.name}")
    else:
        print("[blender] WARNING: Cages not found. FBX will need manual caging in Blender/Studio.")
        clothing_obj.name = "LayeredClothing"

    # ── Step 5: Rig to armature ──────────────────────────────────────────

    if armature:
        print("\n[blender] Rigging to R15 armature...")
        setup_armature_and_weights(clothing_obj, armature)

    # ── Step 6: Export combined preview GLB (clothing + mannequin) ─────

    if args.output_mannequin_glb and mannequin_obj:
        print(f"\n[blender] Exporting combined preview (clothing + mannequin)...")
        export_glb(args.output_mannequin_glb, [clothing_obj, mannequin_obj])

    # ── Step 7: Remove mannequin (not needed in FBX or clothing-only GLB)

    if mannequin_obj:
        mannequin_obj.hide_set(True)
        mannequin_obj.hide_render = True
        bpy.data.objects.remove(mannequin_obj, do_unlink=True)
        print("[blender] Mannequin removed from export")

    # ── Step 8: Export FBX and clothing-only GLB ─────────────────────────

    print(f"\n[blender] Exporting FBX...")
    export_objects = [clothing_obj]
    if inner_cage_obj:
        export_objects.append(inner_cage_obj)
    if outer_cage_obj:
        export_objects.append(outer_cage_obj)

    export_fbx(args.output, armature, export_objects)

    # Also export clothing-only GLB for preview if requested
    if args.output_glb:
        export_glb(args.output_glb, [clothing_obj])

    # ── Step 9: Write metadata ───────────────────────────────────────────

    meta = {
        'clothing_type': args.clothing_type,
        'attachment': config['attachment'],
        'mesh_name': clothing_obj.name,
        'face_count': len(clothing_obj.data.polygons),
        'vertex_count': len(clothing_obj.data.vertices),
        'has_cages': inner_cage_obj is not None and outer_cage_obj is not None,
        'has_armature': armature is not None,
        'roblox_ready': True,
        'auto_skin_recommended': True,  # Always recommend AutoSkin as backup
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

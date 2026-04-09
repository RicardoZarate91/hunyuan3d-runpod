#!/usr/bin/env python3
"""
blender_accessory.py — Blender headless post-processing for Roblox Rigid Accessories.

Takes a decimated GLB mesh and produces a Roblox-ready FBX with:
  - Mesh positioned at the correct attachment point
  - R15 armature with correct bone hierarchy
  - Mesh parented to the appropriate bone
  - Proper scale for Roblox Studio import

Unlike Layered Clothing, accessories do NOT need:
  - Inner/Outer cage meshes
  - Shrinkwrap fitting
  - Weight painting (single bone parent)

Usage (run via Blender headless):
    blender --background --python blender_accessory.py -- \
        --input hat.glb \
        --output hat_roblox.fbx \
        --accessory-type hat \
        --templates-dir /opt/roblox-templates

Accessory types: hat, hair, face, neck, shoulder, back, waist, front
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
    parser.add_argument('--input', required=True, help='Input GLB mesh')
    parser.add_argument('--output', required=True, help='Output FBX path')
    parser.add_argument('--output-glb', default=None, help='Also save processed GLB')
    parser.add_argument('--accessory-type', default='hat',
                        choices=['hat', 'hair', 'face', 'neck', 'shoulder',
                                 'back', 'waist', 'front'],
                        help='Accessory type for attachment point')
    parser.add_argument('--templates-dir', default='/opt/roblox-templates',
                        help='Directory with Roblox template FBX files')
    parser.add_argument('--output-preview-glb', default=None,
                        help='Save GLB with accessory on mannequin for web preview')
    parser.add_argument('--meta-output', default=None,
                        help='Path to write JSON metadata')
    return parser.parse_args(argv)


# ── Accessory type → attachment config ──────────────────────────────────────

# Each accessory type maps to:
#   - attachment: The Roblox attachment name (becomes an Attachment instance in Studio)
#   - bone: The R15 bone the mesh parents to
#   - offset: (x, y, z) offset from bone origin in Roblox coords (studs)
#   - scale_ref: Reference dimension for auto-scaling ('head' or 'torso')
ACCESSORY_CONFIG = {
    'hat': {
        'attachment': 'HatAttachment',
        'bone': 'Head',
        'offset': (0, 0.75, 0),     # Top of head
        'scale_ref': 'head',
        'max_size': 3.0,             # Max studs in any dimension
    },
    'hair': {
        'attachment': 'HairAttachment',
        'bone': 'Head',
        'offset': (0, 0.6, 0),      # Slightly lower than hat
        'scale_ref': 'head',
        'max_size': 3.0,
    },
    'face': {
        'attachment': 'FaceFrontAttachment',
        'bone': 'Head',
        'offset': (0, 0, 0.6),      # Front of face
        'scale_ref': 'head',
        'max_size': 2.0,
    },
    'neck': {
        'attachment': 'NeckAttachment',
        'bone': 'Head',
        'offset': (0, -0.5, 0),     # Base of head/top of neck
        'scale_ref': 'head',
        'max_size': 2.5,
    },
    'shoulder': {
        'attachment': 'RightCollarAttachment',
        'bone': 'UpperTorso',
        'offset': (1.0, 0.5, 0),    # Right shoulder area
        'scale_ref': 'torso',
        'max_size': 2.5,
    },
    'back': {
        'attachment': 'BodyBackAttachment',
        'bone': 'UpperTorso',
        'offset': (0, 0, -0.75),    # Behind torso
        'scale_ref': 'torso',
        'max_size': 4.0,
    },
    'waist': {
        'attachment': 'WaistBackAttachment',
        'bone': 'LowerTorso',
        'offset': (0, -0.2, -0.6),  # Lower back / waist
        'scale_ref': 'torso',
        'max_size': 3.5,
    },
    'front': {
        'attachment': 'BodyFrontAttachment',
        'bone': 'UpperTorso',
        'offset': (0, 0, 0.6),      # Front of torso
        'scale_ref': 'torso',
        'max_size': 3.0,
    },
}

# Reference dimensions in Blender units (matching Roblox mannequin scale)
# These are approximate for the standard R15 mannequin
REF_DIMENSIONS = {
    'head':  {'height': 1.2, 'width': 1.2},
    'torso': {'height': 2.0, 'width': 2.0},
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


def import_glb(filepath):
    """Import GLB/GLTF file."""
    bpy.ops.import_scene.gltf(filepath=filepath)


def import_fbx(filepath):
    """Import FBX file."""
    bpy.ops.import_scene.fbx(filepath=filepath, use_anim=False)


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


def count_tris(obj):
    """Count triangles in a mesh object."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.triangulate(bm, faces=bm.faces)
    count = len(bm.faces)
    bm.free()
    return count


def join_meshes(mesh_objects):
    """Join multiple mesh objects into one."""
    if len(mesh_objects) <= 1:
        return mesh_objects[0] if mesh_objects else None

    bpy.ops.object.select_all(action='DESELECT')
    for obj in mesh_objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_objects[0]
    bpy.ops.object.join()
    return bpy.context.active_object


# ── Main pipeline ────────────────────────────────────────────────────────────

def process_accessory(args):
    """
    Main accessory processing pipeline:
    1. Import accessory mesh
    2. Import R15 armature from template
    3. Scale/position accessory at attachment point
    4. Parent to correct bone
    5. Export FBX
    """
    config = ACCESSORY_CONFIG[args.accessory_type]
    meta = {
        'accessory_type': args.accessory_type,
        'attachment': config['attachment'],
        'bone': config['bone'],
    }

    # ── Step 1: Import accessory mesh ──────────────────────────────────────
    print(f"\n[accessory] Importing mesh: {args.input}")
    clear_scene()
    import_glb(args.input)

    meshes = get_mesh_objects()
    if not meshes:
        raise RuntimeError("No mesh objects found in input GLB")

    # Join all meshes into one
    accessory = join_meshes(meshes)
    accessory.name = "AccessoryMesh"

    # Apply transforms
    bpy.context.view_layer.objects.active = accessory
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    input_tris = count_tris(accessory)
    meta['input_tris'] = input_tris
    print(f"[accessory] Input mesh: {input_tris} tris")

    # ── Step 2: Import R15 armature ────────────────────────────────────────
    # Look for the R15 template
    template_candidates = [
        os.path.join(args.templates_dir, 'R15-Armature.fbx'),
        os.path.join(args.templates_dir, 'Combined-Template.fbx'),
        os.path.join(args.templates_dir, 'Mannequin.fbx'),
    ]

    template_path = None
    for p in template_candidates:
        if os.path.exists(p):
            template_path = p
            break

    if template_path:
        print(f"[accessory] Loading armature from: {template_path}")
        import_fbx(template_path)

        armature = get_armature()
        if armature:
            # Remove template meshes (mannequin body, cages) — we only want the armature
            template_meshes = [obj for obj in get_mesh_objects() if obj != accessory]
            mannequin_mesh = None

            for obj in template_meshes:
                # Keep one mannequin mesh for preview if requested
                if args.output_preview_glb and mannequin_mesh is None and 'cage' not in obj.name.lower():
                    mannequin_mesh = obj
                    continue
                bpy.data.objects.remove(obj, do_unlink=True)

            meta['has_armature'] = True
            meta['armature_bones'] = len(armature.data.bones)
        else:
            print("[accessory] WARNING: No armature found in template")
            armature = create_simple_armature(config)
            meta['has_armature'] = True
            meta['armature_type'] = 'generated'
    else:
        print("[accessory] WARNING: No R15 template found, creating simple armature")
        armature = create_simple_armature(config)
        meta['has_armature'] = True
        meta['armature_type'] = 'generated'

    # ── Step 3: Scale and position accessory ───────────────────────────────
    print(f"[accessory] Positioning at {config['attachment']} (bone: {config['bone']})")

    # Center the mesh at origin first
    acc_bb = get_bounding_box(accessory)
    accessory.location = -acc_bb['center']
    bpy.context.view_layer.objects.active = accessory
    bpy.ops.object.transform_apply(location=True)
    acc_bb = get_bounding_box(accessory)

    # Scale to fit within max_size
    max_dim = max(acc_bb['size'].x, acc_bb['size'].y, acc_bb['size'].z)
    if max_dim > 0:
        target_size = config['max_size']
        ref = REF_DIMENSIONS[config['scale_ref']]
        # Scale relative to reference body part
        scale_factor = min(target_size / max_dim, ref['height'] / max_dim)
        accessory.scale = (scale_factor, scale_factor, scale_factor)
        bpy.ops.object.transform_apply(scale=True)
        meta['scale_factor'] = round(scale_factor, 4)

    # Position at attachment point
    # Find the bone in the armature
    target_bone_name = config['bone']
    bone_pos = mathutils.Vector((0, 0, 0))

    if armature and target_bone_name in armature.data.bones:
        bone = armature.data.bones[target_bone_name]
        bone_pos = armature.matrix_world @ bone.head_local
        print(f"[accessory] Bone '{target_bone_name}' position: {bone_pos}")

    # Apply offset (in Roblox studs, roughly 1:1 with Blender units for our mannequin)
    offset = mathutils.Vector(config['offset'])
    accessory.location = bone_pos + offset
    meta['position'] = [round(v, 3) for v in accessory.location]

    # ── Step 4: Parent to armature bone ────────────────────────────────────
    if armature:
        bpy.ops.object.select_all(action='DESELECT')
        accessory.select_set(True)
        armature.select_set(True)
        bpy.context.view_layer.objects.active = armature

        # Set parent to bone
        bpy.ops.object.mode_set(mode='POSE')

        # Find and select the target bone
        for pbone in armature.pose.bones:
            pbone.bone.select = (pbone.name == target_bone_name)
            if pbone.name == target_bone_name:
                armature.data.bones.active = pbone.bone

        bpy.ops.object.mode_set(mode='OBJECT')

        # Parent with bone
        accessory.select_set(True)
        bpy.context.view_layer.objects.active = armature
        bpy.ops.object.parent_set(type='BONE', keep_transform=True)

        meta['parented_to'] = target_bone_name
        print(f"[accessory] Parented to bone: {target_bone_name}")

    # Final tri count
    final_tris = count_tris(accessory)
    meta['final_tris'] = final_tris
    print(f"[accessory] Final mesh: {final_tris} tris")

    # ── Step 5: Save preview GLB (with mannequin if available) ─────────────
    if args.output_preview_glb:
        print(f"[accessory] Saving preview GLB: {args.output_preview_glb}")
        bpy.ops.export_scene.gltf(
            filepath=args.output_preview_glb,
            export_format='GLB',
            use_selection=False,
        )
        meta['preview_glb'] = True

    # ── Step 6: Export FBX ─────────────────────────────────────────────────
    # Remove mannequin mesh before FBX export (only accessory + armature)
    remaining_meshes = [obj for obj in get_mesh_objects() if obj != accessory]
    for obj in remaining_meshes:
        bpy.data.objects.remove(obj, do_unlink=True)

    print(f"[accessory] Exporting FBX: {args.output}")

    # Select accessory and armature for export
    bpy.ops.object.select_all(action='DESELECT')
    accessory.select_set(True)
    if armature:
        armature.select_set(True)

    bpy.ops.export_scene.fbx(
        filepath=args.output,
        use_selection=True,
        apply_scale_options='FBX_SCALE_ALL',
        axis_forward='-Z',
        axis_up='Y',
        use_mesh_modifiers=True,
        add_leaf_bones=False,
        bake_anim=False,
    )

    meta['roblox'] = {
        'ready': True,
        'type': 'accessory',
        'attachment': config['attachment'],
        'has_armature': armature is not None,
    }

    # Also export processed GLB if requested
    if args.output_glb:
        print(f"[accessory] Saving processed GLB: {args.output_glb}")
        bpy.ops.export_scene.gltf(
            filepath=args.output_glb,
            export_format='GLB',
            use_selection=True,
        )

    # ── Step 7: Write metadata ─────────────────────────────────────────────
    if args.meta_output:
        with open(args.meta_output, 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"[accessory] Metadata written to: {args.meta_output}")

    print(f"\n[accessory] Done! Accessory type: {args.accessory_type}")
    print(f"[accessory]   Attachment: {config['attachment']}")
    print(f"[accessory]   Bone: {config['bone']}")
    print(f"[accessory]   Tris: {final_tris}")

    return meta


def create_simple_armature(config):
    """Create a minimal armature with the required bone if no template available."""
    bpy.ops.object.armature_add(enter_editmode=True, location=(0, 0, 0))
    armature = bpy.context.active_object
    armature.name = "Armature"

    # Remove any default bones
    for bone in list(armature.data.edit_bones):
        armature.data.edit_bones.remove(bone)

    # Create HumanoidRootPart as root
    root = armature.data.edit_bones.new('HumanoidRootPart')
    root.head = (0, 0, 0)
    root.tail = (0, 0.5, 0)

    # Create the target bone (may be the same as root)
    if config['bone'] != 'HumanoidRootPart':
        bone = armature.data.edit_bones.new(config['bone'])
        bone.head = (0, 0, 0)
        bone.tail = (0, 1, 0)
        bone.parent = root

    bpy.ops.object.mode_set(mode='OBJECT')
    return armature


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    args = parse_args()
    try:
        meta = process_accessory(args)
        print(f"\nACCESSORY_META:{json.dumps(meta)}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nACCESSORY_ERROR:{str(e)}")
        sys.exit(1)

import os.path
import pdb
import sys
import bpy
import bmesh
import numpy as np
import cv2
import argparse
from skimage.exposure import match_histograms

# Argument parsing
if "--" not in sys.argv:
    pass
else:
    sys.argv = [""] + sys.argv[sys.argv.index("--") + 1:]
parser = argparse.ArgumentParser("Merge texture", add_help=True)
parser.add_argument("--data_dir", type=str, required=True)
args = parser.parse_args()

# Black color threshold and proportion limit
black_threshold = 0.1
black_pixel_limit = 0.2

working_path = args.data_dir
# Define file paths
blend_file_path = f'{working_path}/scene_stage1C.blend'
append_file_path = f'{working_path}/scene_stage1B.blend'
out_file_path = f'{working_path}/scene_stage1_merged.blend'

# Open the target Blender file
bpy.ops.wm.open_mainfile(filepath=blend_file_path)

# Create a new collection named 'new'
collection_name = "new"
if collection_name not in bpy.data.collections:
    new_collection = bpy.data.collections.new(collection_name)
    bpy.context.scene.collection.children.link(new_collection)
else:
    new_collection = bpy.data.collections[collection_name]

# Find a collection whose name ends with 'osm_buildings'
osm_buildings_collection = None
for col in bpy.data.collections:
    if col.name.endswith("osm_buildings"):
        osm_buildings_collection = col
        break

# If the target collection is found, modify the names of the buildings in it
building_pairs = []
if osm_buildings_collection:
    for obj in osm_buildings_collection.objects:
        if obj.type == 'MESH':  # Only process mesh objects
            original_name = obj.name
            new_name = f"{obj.name}_osm"  # Modify the name
            obj.name = new_name
            building_pairs.append((original_name, new_name))

# Load all meshes from another Blender file, except for 'Terrain' and 'Roof', and append to 'original'
with bpy.data.libraries.load(append_file_path, link=False) as (data_from, data_to):
    mesh_names_to_append = [name for name in data_from.objects if name not in ('Terrain', 'Roof')]
    data_to.objects = mesh_names_to_append

# List of imported building names
imported_building_names = [obj.name for obj in data_to.objects if obj is not None and obj.type == 'MESH']
# Check for unmatched buildings in osm_buildings_collection
unmatched_buildings_in_C = [a_mesh for a_mesh in mesh_names_to_append if a_mesh.name not in [pair[0] for pair in building_pairs]]
for building in unmatched_buildings_in_C:
    if building.type == 'MESH':
        print(f"Copying unmatched building: {building.name} from scene_stage1B.")
        new_obj = building.copy()
        new_obj.data = building.data.copy()
        bpy.context.scene.collection.objects.link(new_obj)
        new_collection.objects.link(new_obj)

# Continue with original code to match and process textures for each building pair
for original_name, project_name in building_pairs:
    original_obj = bpy.data.objects.get(original_name)
    project_obj = bpy.data.objects.get(project_name)

    if original_obj and project_obj:
        print(f"Processing: Original - {original_name}, Project - {project_name}")

        # Create a copy of the original object and link to new collection
        new_obj = original_obj.copy()
        new_obj.data = original_obj.data.copy()
        bpy.context.scene.collection.objects.link(new_obj)
        new_collection.objects.link(new_obj)

        # Set up texture extraction for original and project objects
        bm_ori = bmesh.new()
        bm_ori.from_mesh(original_obj.data)  # Use original object's mesh
        uv_layer = bm_ori.loops.layers.uv.active  # Get the UV map layer

        # Extract the original and project textures
        original_texture = original_obj.active_material.node_tree.nodes['Image Texture'].image
        project_texture = project_obj.active_material.node_tree.nodes['Image Texture'].image

        texture_width = original_texture.size[0]
        texture_height = original_texture.size[1]
        original_pixels = np.array(original_texture.pixels).reshape((texture_height, texture_width, 4))

        # Define the blue filter conditions for original texture
        filter_original_blue1 = original_pixels[:, :, 2] * 1.5 < original_pixels[:, :, 0]
        filter_original_blue2 = original_pixels[:, :, 2] * 1.5 < original_pixels[:, :, 1]
        rows, cols = np.where(filter_original_blue1 & filter_original_blue2)
        original_pixels[rows, cols, :3] = 0.0

        if original_texture.size[0] != project_texture.size[0] or original_texture.size[1] != project_texture.size[1]:
            project_pixels = np.zeros_like(original_pixels)
        else:
            project_pixels = np.array(project_texture.pixels).reshape((texture_height, texture_width, 4))
        original_pixels_filtered = original_pixels[original_pixels[:, :, :3].sum(axis=-1) > black_threshold][:, :3]
        if original_pixels_filtered.sum() > 0:
            target_obj_r, target_obj_g, target_obj_b = cv2.split(np.expand_dims(original_pixels_filtered, 0))
        else:
            target_obj_r, target_obj_g, target_obj_b = None, None, None

# Remove all objects that are not part of the new collection
for obj in bpy.data.objects:
    if obj.name not in new_collection.objects:
        bpy.data.objects.remove(obj, do_unlink=True)

# Remove orphan data blocks to ensure a clean state
bpy.ops.outliner.orphans_purge(do_recursive=True)

# Pack all textures into the Blender file
bpy.ops.file.pack_all()

# Save the final result to out_file_path
bpy.ops.wm.save_mainfile(filepath=out_file_path)

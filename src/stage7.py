import pdb

import bpy
import argparse
import os
import sys
from tqdm import tqdm


def process_mesh_materials(type):
    """
    Iterate over all meshes and modify material node trees:
    - Check if 'TEX_IMAGE' output 'Color' is connected to 'BSDF_PRINCIPLED' 'Emission Color'.
    - Set 'Emission Strength' to 0.
    - Remove the connection from 'Emission Color'.
    - Connect 'TEX_IMAGE' output 'Color' to 'Base Color' of 'BSDF_PRINCIPLED'.
    """
    for obj in bpy.data.objects:
        if obj.type != 'MESH':
            continue

        for mat_slot in obj.material_slots:
            material = mat_slot.material
            if not material or not material.use_nodes:
                continue

            node_tree = material.node_tree
            nodes = node_tree.nodes
            links = node_tree.links

            for node in nodes:
                if node.type == 'TEX_IMAGE':
                    if type == "buildings" or type == "terrain":
                        tex_image_node = node
                        for link in links:
                            if link.from_node == tex_image_node and link.from_socket.name == 'Color':
                                to_node = link.to_node
                                to_socket = link.to_socket
                                if to_node.type == 'BSDF_PRINCIPLED' and to_socket.name == 'Emission Color':
                                    print(f"Processing material '{material.name}' in object '{obj.name}'")

                                    to_node.inputs['Emission Strength'].default_value = 0
                                    links.remove(link)
                                    links.new(tex_image_node.outputs['Color'], to_node.inputs['Base Color'])
                                    print(f"Reconnected 'Color' from TEX_IMAGE to 'Base Color' of BSDF_PRINCIPLED.")
                    else:
                        pass
            if type == "roof":
                nodes.remove(node_tree.nodes["Emission"])
                nodes.remove(node_tree.nodes["Transparent BSDF"])
                nodes.remove(node_tree.nodes["Light Path"])
                nodes.remove(node_tree.nodes["Mix Shader"])
                links.new(node_tree.nodes["Image Texture"].outputs['Color'],
                          node_tree.nodes["Material Output"].inputs['Surface'])

def process_directory(input_dir):
    """
    Traverse the input directory and process all buildings.glb files.
    - Load buildings.glb into Blender.
    - Apply material modifications.
    - Export as buildings_basic.glb in the same folder.
    """
    input_path_list = []
    for file in os.listdir(input_dir):
        if file in ['roof.glb', 'terrain.glb']:
            input_path_list.append((os.path.join(input_dir, file), file.split(".")[0],
                                    os.path.join(input_dir, f'{file.split(".")[0]}_basic.glb')))
        elif file == 'buildings':
            for building in os.listdir(os.path.join(input_dir, file)):
                input_path_list.append((os.path.join(input_dir, file, building), 'buildings',
                                        os.path.join(input_dir, "buildings_basic", building)))

    os.makedirs(os.path.join(input_dir, "buildings_basic"), exist_ok=True)
    for input_path, type, output_path in tqdm(input_path_list):
        print(f"Processing: {input_path}")

        # Reset Blender to clear previous data
        bpy.ops.wm.read_factory_settings(use_empty=True)

        # Import the GLB file
        bpy.ops.import_scene.gltf(filepath=input_path)

        # Apply material modifications
        process_mesh_materials(type)

        # Select all objects in the scene
        bpy.ops.object.select_all(action='DESELECT')  # Deselect all objects first
        for obj in bpy.context.scene.objects:
            obj.select_set(True)  # Select all objects

        # Export only selected objects
        bpy.ops.export_scene.gltf(filepath=output_path, export_format='GLB', use_selection=True)
        print(f"Exported: {output_path}")

        # Clear the scene for the next file
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()


if __name__ == "__main__":
    if "--" not in sys.argv:
        pass
    else:
        sys.argv = [""] + sys.argv[sys.argv.index("--") + 1:]
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Process GLB files and apply material modifications.")
    parser.add_argument('--input_dir', type=str, required=True)
    args = parser.parse_args()

    # Process the directory
    process_directory(args.input_dir)

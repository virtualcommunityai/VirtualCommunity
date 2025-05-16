import bpy
import os
import sys
import argparse
from tqdm import tqdm


# Function to ensure directory exists
def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory created: {path}")
    else:
        print(f"Directory already exists: {path}")


def load_first_mesh_from_blend(file_path):
    object_dir = file_path + "/Object/"
    bpy.ops.wm.append(filepath=file_path, directory=object_dir, filename="Mesh_0")


if __name__ == "__main__":
    if "--" not in sys.argv:
        pass
    else:
        sys.argv = [""] + sys.argv[sys.argv.index("--") + 1:]
    parser = argparse.ArgumentParser("Align Google Stree View to 3D tiles data", add_help=True)
    parser.add_argument("--input_path", type=str, required=False, default="meshes/input.blend")
    parser.add_argument("--output_path", type=str, required=False, default="meshes/merged.blend")
    parser.add_argument("--overwrite", action='store_true')
    args = parser.parse_args()
    if os.path.isfile(args.output_path) and not args.overwrite:
        print(f"{args.output_path} already exists, skipping...")
        exit(0)
    # Clear existing objects in the scene
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    load_first_mesh_from_blend(file_path=args.input_path)
    # help(bpy.ops.export_scene.gltf)
    # Select all mesh objects
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            print("Export")
            bpy.ops.export_scene.gltf(
                filepath=args.output_path,
                export_format='GLB',
                export_materials='NONE',
                use_selection=True
            )

import numpy._core
import bpy
import os
from pathlib import Path
from tqdm import tqdm
import sys
import shutil

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from blenderlib import MeshObject

def emit_texture_maps(blender_file_path: str) -> str:
    """Given a blender file with bundled texture maps internally, emit the texture maps to a directory and return the path
    of that directory.
    """
    bpy.ops.file.unpack_all(method="WRITE_LOCAL")
    return str(Path(Path(blender_file_path).parent, "textures"))

def superres_texture_maps(textures_path: str, image_dir: str):
    """Use upscayl executable to run superresolution  on the texture maps. Will overwrite texturemaps in place.

    $upscayl_executable -i {textures_path} -o {textures_path} -z 4 -s 4 -n ultrasharp
    opt/Upscayl/resources/bin/upscayl-bin -i ~/Desktop/CityGenData/Data/textures-orig/ -o ~/Desktop/CityGenData/Data/textures/ -z 4 -s 4 -m /opt/Upscayl/resources/models/ -n ultrasharp -f jpg -v
    """
    image_dir_path = Path(image_dir)
    for file in tqdm(Path(textures_path).iterdir()):
        print(file)
        file_str = str(file)
        try:
            file_str = file_str.split('Material_')[1].split('_Image')[0]
            file_str = file_str.replace('_', ' ')
            new_image_path = image_dir_path / file_str / f'{file_str}_texture.png'
            if new_image_path.exists():
                shutil.copy(new_image_path, file)
        except:
            continue

def rebundle_texture_maps(save_to: str):
    """Bundle texture maps back to the blender file and save the new blender file.
    """
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=save_to)
    bpy.ops.wm.quit_blender()

def import_and_replace_textures(blender_file_path: str, image_dir: str, save_path: str):
    """
    Import textures from image_dir and replace existing textures in the Blender file.
    """
    # Ensure the image directory exists
    image_dir_path = Path(image_dir)
    if not image_dir_path.exists() or not image_dir_path.is_dir():
        print(f"Error: The provided image directory '{image_dir}' does not exist or is not a directory.")
        return
    
    bpy.ops.wm.open_mainfile(filepath=blender_file_path)
    EXCLUDE_LIST = ["Roof"]
    for exclude in EXCLUDE_LIST: MeshObject.withName(exclude).delete()
    bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)

    texture_path = emit_texture_maps(blender_file_path)
    superres_texture_maps(texture_path, image_dir)
    rebundle_texture_maps(save_path)

    bpy.ops.wm.open_mainfile(filepath=save_path)
    for exclude in EXCLUDE_LIST: MeshObject.remoteAppend(blender_file_path, exclude)
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=save_path)
    
    return
    # Iterate through all objects and materials in the Blender scene
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            for mat in obj.data.materials:
                if mat and mat.use_nodes:
                    for node in mat.node_tree.nodes:
                        if node.type == 'TEX_IMAGE' and node.image:
                            old_image_name = node.image.name
                            new_image_path = image_dir_path / obj.name / f'{obj.name}_texture.png'

                            if new_image_path.exists():
                                # Load the new image
                                new_image = bpy.data.images.load(str(new_image_path), check_existing=True)
                                node.image = new_image
                                node.image.pack()
                                print(f"Updated texture in material '{mat.name}' with new image '{new_image.name}'")
                            else:
                                print(f"Warning: Texture '{new_image_path}' not found, skipping update.")

    # Save the updated Blender file
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=save_path)
    print(f"Blender file saved with updated textures at '{save_path}'")


if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--blender_file", type=str, required=True, help="Path to the Blender file")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing new texture images")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the updated Blender file")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])

    import_and_replace_textures(args.blender_file, args.image_dir, args.save_path)

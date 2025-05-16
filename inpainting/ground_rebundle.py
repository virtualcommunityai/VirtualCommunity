import bpy
import bmesh
from pathlib import Path
import numpy as np
from PIL import Image
import json
import os
import shutil

blend_dir=""

def load_img_to_array(img_p):
    img = Image.open(img_p)
    if img.mode == "RGBA":
        img = img.convert("RGB")
    return np.array(img)

def emit_texture_maps(blender_file_path: str) -> str:
    """Given a blender file with bundled texture maps internally, emit the texture maps to a directory and return the path
    of that directory.
    """
    # shutil.rmtree(str(Path(Path(blender_file_path).parent, "textures")))
    bpy.ops.file.unpack_all(method="WRITE_LOCAL")
    shutil.move(str(Path(Path(blender_file_path).parent, "textures")),str(Path(Path(blender_file_path).parent, f"textures_{blend_dir.stem}")))
    # os.system(f"rm {str(Path(Path(blender_file_path).parent, 'textures'))}/Image*")
    return str(Path(Path(blender_file_path).parent, f"textures_{blend_dir.stem}"))

def save_mesh_texture_maps(mesh_name, output_directory):
    # Get the mesh object by name
    obj = bpy.data.objects.get(mesh_name)
    
    if obj is None or obj.type != 'MESH':
        print(f"Object '{mesh_name}' not found or is not a mesh.")
        return
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Iterate through the materials of the mesh
    for mat in obj.data.materials:
        if mat is None:
            continue
        
        # Check if the material uses nodes
        if mat.use_nodes:
            for node in mat.node_tree.nodes:
                if node.type == 'TEX_IMAGE' and node.image:
                    image_name = f"{mat.name}_{node.name}.png"
                    image_path = os.path.join(output_directory, image_name)
                    node.image.save_render(image_path)
                    print(f"Saved texture: {image_path}")
        else:
            # Check texture slots if not using nodes
            for texture_slot in mat.texture_slots:
                if texture_slot and texture_slot.texture and texture_slot.texture.type == 'IMAGE':
                    image_name = f"{mat.name}_{texture_slot.texture.name}.png"
                    image_path = os.path.join(output_directory, image_name)
                    texture_slot.texture.image.save_render(image_path)
                    print(f"Saved texture: {image_path}")

def modify_and_save_textures(mesh_name, new_image_dir_path):
    # Get the mesh object by name
    obj = bpy.data.objects.get(mesh_name)
    
    if obj is None or obj.type != 'MESH':
        print(f"Object '{mesh_name}' not found or is not a mesh.")
        return
    
    # Check if the new image file exists
    # new_image_path = str(Path(new_image_dir_path) / f"Material_{mesh_name.replace(' ','_')}_Image Texture.png")
    # if not os.path.isfile(new_image_path):
    #     print(f"Image file '{new_image_path}' not found.")
    #     return
    
    # Iterate through the materials of the mesh
    for mat in obj.data.materials:
        if mat is None:
            continue
        
        # Check if the material uses nodes
        if mat.use_nodes:
            for node in mat.node_tree.nodes:
                if node.type == 'TEX_IMAGE' and node.image:
                    # Load the new image
                    new_image_path = str(Path(new_image_dir_path) / f"{mat.name}.png")
                    if not os.path.isfile(new_image_path):
                        print(f"Image file '{new_image_path}' not found.")
                        return
                    new_image = bpy.data.images.load(new_image_path)
                    node.image = new_image
                    print(f"Updated texture in material '{mat.name}' with new image '{new_image.name}'")
        else:
            # Check texture slots if not using nodes
            for texture_slot in mat.texture_slots:
                if texture_slot and texture_slot.texture and texture_slot.texture.type == 'IMAGE':
                    # Load the new image
                    new_image_path = str(Path(new_image_dir_path) / f"{mat.name}.png")
                    if not os.path.isfile(new_image_path):
                        print(f"Image file '{new_image_path}' not found.")
                        return
                    new_image = bpy.data.images.load(new_image_path)
                    texture_slot.texture.image = new_image
                    print(f"Updated texture in material '{mat.name}' with new image '{new_image.name}'")
    
    # Optionally, save the Blender file to preserve changes
    # bpy.ops.wm.save_mainfile()
    # print("Changes saved to the Blender file.")

def rebundle_texture_maps(save_to: str):
    """Bundle texture maps back to the blender file and save the new blender file.
    """
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=save_to)
    bpy.ops.wm.quit_blender()

def clear_transformations(bpy_obj):
    bpy_obj.location = (0.0, 0.0, 0.0)
    bpy_obj.rotation_quaternion = (1., 0., 0., 0.)
    bpy_obj.scale = (1.0, 1.0, 1.0)
    bpy.context.view_layer.update()    

dx,dy=[-1,0,1,0],[0,-1,0,1]
lx,ly,ux,uy=0,0,0,0
mp={}

def main(blender_file_path: str, image_dir:str, mesh_name: str, save_to: str):
    global blend_dir
    blend_dir=Path(Path(blender_file_path).parent)
    bpy.ops.wm.open_mainfile(filepath=blender_file_path)

    modify_and_save_textures(mesh_name, image_dir)
    # texture_path = emit_texture_maps(blender_file_path)
    # superres_texture_maps(texture_path, upscayl_executable)
    rebundle_texture_maps(save_to)
    


if __name__ == "__main__":
    import argparse, sys
    parser = argparse.ArgumentParser()
    parser.add_argument("--blender_file", type=str, required=True, help="Path to the blender file that contains the mesh and building vertex groups")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to the blender file that contains the mesh and building vertex groups")
    parser.add_argument("--mesh", type=str, required=True, help="Name of the mesh object in the blender file")
    # parser.add_argument("--upscayl_exec", type=str, required=True, help="Executable for upscayl - https://github.com/upscayl/upscayl")
    parser.add_argument("--save_to", type=str, required=True, help="Save modified blender file to")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])

    main(args.blender_file, args.image_dir, args.mesh, args.save_to)

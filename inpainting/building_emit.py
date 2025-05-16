import bpy
import bmesh
from pathlib import Path
import numpy as np
from PIL import Image
import json
import os
import shutil
import cv2

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
    shutil.move(str(Path(Path(blender_file_path).parent, "textures")),str(Path(Path(blender_file_path).parent, f"textures_{blend_dir.stem}_building")))
    # os.system(f"rm {str(Path(Path(blender_file_path).parent, 'textures'))}/Image*")
    return str(Path(Path(blender_file_path).parent, f"textures_{blend_dir.stem}_building"))

def save_mesh_texture_maps(mesh_name, output_directory, scale_by):
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
                    # if "element" in mat.name: continue
                    image_name = f"{mat.name.replace('/', '-')}_{node.name}.png"
                    image_path = os.path.join(output_directory, image_name)
                    node.image.save_render(image_path)
                    image = cv2.imread(image_path)
                    downscaled_image = cv2.resize(image, None, fx=scale_by, fy=scale_by, interpolation=cv2.INTER_AREA)
                    cv2.imwrite(image_path, downscaled_image)
                    print(f"Saved texture: {image_path}")
        else:
            # Check texture slots if not using nodes
            for texture_slot in mat.texture_slots:
                if texture_slot and texture_slot.texture and texture_slot.texture.type == 'IMAGE':
                    # if "element" in mat.name: continue
                    image_name = f"{mat.name.replace('/', '-')}_{texture_slot.texture.name}.png"
                    image_path = os.path.join(output_directory, image_name)
                    texture_slot.texture.image.save_render(image_path)
                    image = cv2.imread(image_path)
                    downscaled_image = cv2.resize(image, None, fx=scale_by, fy=scale_by, interpolation=cv2.INTER_AREA)
                    cv2.imwrite(image_path, downscaled_image)
                    print(f"Saved texture: {image_path}")

def superres_texture_maps(textures_path: str, upscayl_executable: str):
    """Use upscayl executable to run superresolution  on the texture maps. Will overwrite texturemaps in place.

    $upscayl_executable -i {textures_path} -o {textures_path} -z 4 -s 4 -n ultrasharp
    opt/Upscayl/resources/bin/upscayl-bin -i ~/Desktop/CityGenData/Data/textures-orig/ -o ~/Desktop/CityGenData/Data/textures/ -z 4 -s 4 -m /opt/Upscayl/resources/models/ -n ultrasharp -f jpg -v
    """
    for file in Path(textures_path).iterdir():
        print(file)
        args = [upscayl_executable, "-i", str(file), "-o", str(file), "-z", "4", "-s", "4", "-n", "ultrasharp",
                        "-m", str(Path(Path(upscayl_executable).parent.parent, "models"))]
        ret_code = subprocess.call(args)
        assert ret_code == 0

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

def main(blender_file_path: str, mesh_name: str, output_dir: str, scale_by: float):
    global blend_dir
    blend_dir=Path(Path(blender_file_path).parent)
    print(blend_dir)
    #os.makedirs(blend_dir/output_dir,exist_ok=True)
    bpy.ops.wm.open_mainfile(filepath=blender_file_path)

    building_meshes = [obj.name for obj in bpy.data.objects if obj.name not in ["Cube", "Roof"]]
    for building_mesh in building_meshes:
        save_mesh_texture_maps(building_mesh, output_dir, scale_by)
    # texture_path = emit_texture_maps(blender_file_path)
    # superres_texture_maps(texture_path, upscayl_executable)
    # rebundle_texture_maps(save_to)
    


if __name__ == "__main__":
    import argparse, sys
    parser = argparse.ArgumentParser()
    parser.add_argument("--blender_file", type=str, required=True, help="Path to the blender file that contains the mesh and building vertex groups")
    parser.add_argument("--mesh", type=str, required=True, help="Name of the mesh object in the blender file")
    parser.add_argument("--output_dir", type=str, required=True, help="Name of the output dir")
    parser.add_argument("--scale_by", type=float, required=False, default=1.0, help="Save modified blender file to")
    # parser.add_argument("--upscayl_exec", type=str, required=True, help="Executable for upscayl - https://github.com/upscayl/upscayl")
    # parser.add_argument("--save_to", type=str, required=True, help="Save modified blender file to")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])

    main(args.blender_file, args.mesh, args.output_dir, args.scale_by)

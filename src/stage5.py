import bpy
import subprocess
import shutil
from pathlib import Path
import os, sys
import copy

from tqdm import tqdm
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
dir = os.path.dirname(bpy.data.filepath)
if not dir in sys.path: sys.path.append(dir)

from blenderlib import MeshObject

def emit_texture_maps(blender_file_path: str) -> str:
    """Given a blender file with bundled texture maps internally, emit the texture maps to a directory and return the path
    of that directory.
    """
    bpy.ops.file.unpack_all(method="WRITE_LOCAL")
    return str(Path(Path(blender_file_path).parent, "textures"))

def superres_texture_maps(textures_path: str, upscayl_executable: str):
    """Use upscayl executable to run superresolution  on the texture maps. Will overwrite texturemaps in place.

    $upscayl_executable -i {textures_path} -o {textures_path} -z 4 -s 4 -n ultrasharp
    opt/Upscayl/resources/bin/upscayl-bin -i ~/Desktop/CityGenData/Data/textures-orig/ -o ~/Desktop/CityGenData/Data/textures/ -z 4 -s 4 -m /opt/Upscayl/resources/models/ -n ultrasharp -f jpg -v
    """
    for file in tqdm(Path(textures_path).iterdir()):
        print(file)
        args = [upscayl_executable, "-i", str(file), "-o", str(file), "-z", "2", "-s", "2", "-n", "ultrasharp-4x",
                        "-m", str(Path(Path(upscayl_executable).parent.parent, "models"))]
        # args = upscayl_executable.split(" ") + ["-i", str(file), "-o", str(file), "-z", "4", "-s", "2", "-n", "ultrasharp",
        #                 "-m", str(Path(Path(upscayl_executable).parent.parent, "models"))]
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

def main(blender_file_path: str, upscayl_executable: str, save_to: str, glb_to: str):
    if os.path.exists(save_to):
        bpy.ops.wm.open_mainfile(filepath=save_to)
    else:
        bpy.ops.wm.open_mainfile(filepath=blender_file_path)
        # EXCLUDE_LIST = ["Roof"]
        # for exclude in EXCLUDE_LIST: MeshObject.withName(exclude).delete()
        # bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)

        # texture_path = emit_texture_maps(blender_file_path)
        # superres_texture_maps(texture_path, upscayl_executable)
        rebundle_texture_maps(save_to)

        bpy.ops.wm.open_mainfile(filepath=save_to)
        for exclude in EXCLUDE_LIST: MeshObject.remoteAppend(blender_file_path, exclude)
        bpy.ops.file.pack_all()
        bpy.ops.wm.save_as_mainfile(filepath=save_to)

    mesh_objects = [MeshObject(obj) for obj in bpy.context.scene.objects if obj.type == 'MESH']

    # Save all buildings
    bpy.ops.object.select_all(action='DESELECT')
    os.makedirs(os.path.join(Path(glb_to).parent, "buildings"), exist_ok=True)
    for obj in tqdm(mesh_objects):
        if obj.name in ("Terrain", "Roof"): continue
        obj.is_select = True
        glb_file_path = f'buildings_{copy.copy(obj.name).replace("/", "_")}.glb'
        bpy.ops.export_scene.gltf(
            filepath=str(Path(Path(glb_to).parent, "buildings", glb_file_path)),
            check_existing=True,
            export_format='GLB',
            use_selection=True,
            export_apply=True,
            export_texcoords=True,
            export_normals=True,
            export_materials='EXPORT',
            export_cameras=True,
            export_lights=True,
            export_extras=True,
            export_yup=True,
            export_animations=True,
            export_image_format='AUTO'
        )
        obj.is_select = False

    # Save roof
    bpy.ops.object.select_all(action='DESELECT')
    for obj in mesh_objects:
        if obj.name == "Roof":
            obj.is_select = True
        else:
            obj.is_select = False
    bpy.ops.export_scene.gltf(
        filepath=str(Path(Path(glb_to).parent, "roof.glb")),
        check_existing=True,
        export_format='GLB',
        use_selection=True,
        export_apply=True,
        export_texcoords=True,
        export_normals=True,
        export_materials='EXPORT',
        export_cameras=True,
        export_lights=True,
        export_extras=True,
        export_yup=True,
        export_animations=True,
        export_image_format='AUTO'
    )

    # Save Terrain
    bpy.ops.object.select_all(action='DESELECT')
    for obj in mesh_objects:
        if obj.name == "Terrain":
            obj.is_select = True
        else:
            obj.is_select = False
    bpy.ops.export_scene.gltf(
        filepath=str(Path(Path(glb_to).parent, "terrain.glb")),
        check_existing=True,
        export_format='GLB',
        use_selection=True,
        export_apply=True,
        export_texcoords=True,
        export_normals=True,
        export_materials='EXPORT',
        export_cameras=True,
        export_lights=True,
        export_extras=True,
        export_yup=True,
        export_animations=True,
        export_image_format='AUTO'
    )

    bpy.ops.wm.quit_blender()
    print("Done")

    bpy.ops.wm.quit_blender()
    shutil.rmtree(Path(Path(blender_file_path).parent, "textures"), ignore_errors=True)


if __name__ == "__main__":
    import argparse, sys
    parser = argparse.ArgumentParser()
    parser.add_argument("--blender_file", type=str, required=True, help="Path to the blender file that contains the mesh and building vertex groups")
    parser.add_argument("--upscayl_exec", type=str, required=True, help="Executable for upscayl - https://github.com/upscayl/upscayl")
    parser.add_argument("--save_to", type=str, required=True, help="Save modified blender file to")
    parser.add_argument("--glb_to", type=str, required=True, help="Save glb file to")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])

    main(args.blender_file, args.upscayl_exec, args.save_to, args.glb_to)

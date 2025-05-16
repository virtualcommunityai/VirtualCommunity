import bpy
import os, sys
import math
from mathutils import Euler
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from blenderlib import MeshObject

def main(ground_blender_file: str, ground_mesh_name: str, save_to: str, glb_to: str, roof_blender: str, roof_name: str):
    # Append ground to building
    MeshObject.remoteAppend(ground_blender_file, ground_mesh_name)
    MeshObject.remoteAppend(roof_blender, roof_name)
    
    # Join meshes
    mesh_objects = [MeshObject(obj) for obj in bpy.context.scene.objects if obj.type == 'MESH']
    for obj in mesh_objects:
        obj.mesh_object.location.y += 100.
        obj.apply_transform()
        

    # Add Triangulate modifier
    for idx, mesh in enumerate(mesh_objects):
        print(f"\rConverting {idx + 1} / {len(mesh_objects)}".ljust(40), end="")
        with mesh.scoped_select(True), mesh.scoped_active():
            mesh.mesh_object.rotation_mode = 'XYZ'
            mesh.mesh_object.rotation_euler = Euler((math.radians(90), 0., 0.), 'XYZ')
            mesh.apply_transform()
            mesh.mesh_object.modifiers.new(name="Triangulate", type='TRIANGULATE')
            bpy.ops.object.modifier_apply(modifier="Triangulate")
            
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            uv_layers = obj.data.uv_layers
            # 检查是否存在 "size" 和 "UVMap" 两个 UV 图层
            if "size" in uv_layers and "UVMap" in uv_layers:
                # 复制 "size" 图层的 UV 数据到 "UVMap" 图层
                # 假设两个图层的 UV 数据数量相同
                size_layer = uv_layers["size"]
                uvmap_layer = uv_layers["UVMap"]
                for i, uv in enumerate(size_layer.data):
                    uvmap_layer.data[i].uv = uv.uv[:]
                
                # 删除 "size" 图层
                uv_layers.remove(size_layer)
                print(f"在对象 {obj.name} 中删除了激活的 UV 映射")
                
                # 将 active_index 设为 0
                uv_layers.active_index = 0
    
    bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
    bpy.ops.wm.save_as_mainfile(filepath=save_to)
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.wm.quit_blender()
    print("Done")


def init(building_blender_file: str, ground_blender_file: str, ground_mesh_name: str, save_to: str, glb_to: str, roof_blender: str, roof_name: str):
    bpy.ops.wm.open_mainfile(filepath=building_blender_file)
    main(ground_blender_file, ground_mesh_name, save_to, glb_to, roof_blender, roof_name)
    # bpy.app.timers.register(
    #     lambda: main(ground_blender_file, ground_mesh_name, save_to, glb_to, roof_blender, roof_name),
    #     first_interval=0.5
    # )


if __name__ == "__main__":
    import argparse, sys
    parser = argparse.ArgumentParser()
    parser.add_argument("--building_blender", type=str, required=True,
                        help="Blender file with vertex groups of buildings with corresponding textures of building")
    parser.add_argument("--terrain_blender", type=str, required=True,
                        help="Blender file with ground plane of scene with corresponding textures")
    parser.add_argument("--roof_blender", type=str, required=True,
                        help="Blender file with roof mesh of scene with corresponding textures")
    parser.add_argument("--save_to", type=str, required=True,
                        help="Save resulted blender file to ...")
    parser.add_argument("--glb_to", type=str, required=True,
                        help="Save resulted glb file to ...")
    
    parser.add_argument("--terrain_name", type=str, required=True, help="Name of Terrain Mesh")
    parser.add_argument("--roof_name", type=str, required=True, help="Name of Roof Mesh")
    
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])

    init(args.building_blender, args.terrain_blender, args.terrain_name, args.save_to, args.glb_to, args.roof_blender, args.roof_name)

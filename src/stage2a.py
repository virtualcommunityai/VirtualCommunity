import os, sys
import bpy
import numpy as np
from bpy.app.handlers import persistent

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from blenderlib import MeshObject, BakeService, CoordSystem, AssertLiteralType

# Register save handlers

@persistent
def save_mod_images(_):
    """ Save all modified images """
    if any(i.is_dirty for i in bpy.data.images):
        bpy.ops.image.save_all_modified()

@persistent
def pack_dirty_images(_):
    """ Pack all modified images """ 
    for i in bpy.data.images:
        if i.is_dirty:
            i.pack()
            print("Packed:", i.name)

bpy.app.handlers.save_pre.append(save_mod_images)
bpy.app.handlers.save_pre.append(pack_dirty_images)

# End

def apply_bsdf_material_per_face(mesh_name: str):
    mesh_obj = bpy.data.objects[mesh_name]
    bpy.context.view_layer.objects.active = mesh_obj
    mesh_obj.select_set(True)
    
    bpy.ops.object.mode_set(mode = 'OBJECT')
    material_names = []

    for idx in range(len(mesh_obj.data.polygons)):
        print(f"\rCreating materials: {idx + 1} / {len(mesh_obj.data.polygons)}", end="")
        material_name = f"Material_Ground{str(idx).zfill(3)}"
    
        material = bpy.data.materials.new(name=material_name)
        material.use_nodes = True
        nodes = material.node_tree.nodes
        
        # Clean up default nodes
        for node in nodes: nodes.remove(node)
        
        # Define new material nodes
        image_texture = nodes.new(type="ShaderNodeTexImage")
        # bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
        bsdf = nodes.new(type='ShaderNodeEmission')
        material_output = nodes.new(type="ShaderNodeOutputMaterial")
        
        image = bpy.data.images.new(name=f"Texture_{material_name}", height=1024, width=1024)
        image.file_format = 'JPEG'
        image.pack()
        image_texture.image = image
        
        # bsdf.inputs['Specular IOR Level'].default_value = 0.000
        
        image_texture.location = (-600, 0)
        bsdf.location=(0, 0)
        material_output.location = (300, 0)
        
        material.node_tree.links.new(image_texture.outputs["Color"], bsdf.inputs["Color"])
        material.node_tree.links.new(bsdf.outputs["Emission"], material_output.inputs["Surface"])
        mesh_obj.data.materials.append(material)
        material_names.append(material_name)
    print ("")

    bpy.ops.object.mode_set(mode='OBJECT')
    for idx in range(len(mesh_obj.data.polygons)):
        print(f"\rApplying materials: {idx + 1} / {len(mesh_obj.data.polygons)}", end="")
        bpy.ops.object.mode_set(mode='OBJECT')
        
        mesh_obj.data.polygons[idx].select = True

        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_mode(type="FACE")
        
        material_index = mesh_obj.data.materials.find(material_names[idx])
        mesh_obj.active_material_index = material_index
        bpy.ops.object.material_slot_assign()
        bpy.ops.uv.smart_project(correct_aspect=False)
        
        bpy.ops.mesh.select_all(action='DESELECT')

    print(f"\nCreated {len(material_names)} material(s)")
    return material_names


def align_mesh_alt(terrain_mesh: MeshObject, tile_mesh: MeshObject, coord_sys: CoordSystem) -> float:
    AssertLiteralType(coord_sys, CoordSystem)
    terrain_coords = terrain_mesh.verts_Tworld
    raycast_source = terrain_coords.copy()
    
    if coord_sys == "Z+":
        alt_axis = 2
        raycast_source[..., 2] = -100
        direction   = (0., 0., 1.)
    else:
        alt_axis = 1
        raycast_source[..., 1] = -100
        direction   = (0., 1., 0.)
    
    target_alts, mask, bvhtree = tile_mesh.cast_ray_on(
        raycast_source, direction, 500., None, use_modifiers=False
    )
    
    offset = np.median((terrain_coords[mask][..., alt_axis] - target_alts[mask][..., alt_axis]))
    tile_mesh.mesh_object.matrix_world[alt_axis][3] = offset + 10


def main(terrain_file, terrain_name, tile_name, save_as):
    bakery = BakeService()
    coord_sys: CoordSystem = "Y+"
    
    terrain_mesh = MeshObject.remoteAppend(terrain_file, terrain_name)
    tile_mesh = MeshObject.withName(tile_name)
    tile_mesh.apply_transform()
    
    terrain_mesh.apply_transform()
    terrain_mesh.set_active()
    terrain_mesh.mode = "OBJECT"
    
    def cage_fn(cage: MeshObject):
        cage.mesh_object.matrix_world[1 if coord_sys == "Y+" else 2][3] += 100
    
    bakery.bake_with_custom_cage([tile_mesh], terrain_mesh, cage_fn=cage_fn)
    
    tile_mesh.delete()
    bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
    bpy.ops.wm.save_as_mainfile(filepath=save_as)
    bpy.ops.wm.quit_blender()
    print("\aDone.")


def init(terrain_file: str, tile_file: str, terrain_name: str, tile_name: str, save_as: str):
    # Since tile is much larger than texture-less terrain, this can be faster
    bpy.ops.wm.open_mainfile(filepath=tile_file)
    main(terrain_file, terrain_name, tile_name, save_as)
    
    # bpy.app.timers.register(
    #     lambda: main(terrain_file, terrain_name, tile_name, save_as),
    #     first_interval=0.5
    # )


if __name__ == "__main__":
    import argparse, sys
    parser = argparse.ArgumentParser()
    parser.add_argument("--terrain_file", type=str, required=True, help="Path to the blender file that contains the terrain mesh")
    parser.add_argument("--tile_file"   , type=str, required=True, help="Path to the blender file that contains the 3D tile mesh")
    parser.add_argument("--terrain_name", type=str, required=True, help="Name of terrain mesh in terrain_file")
    parser.add_argument("--tile_name"   , type=str, required=True, help="Name of tile mesh in tile_file")

    parser.add_argument("--save_as"     , type=str, required=True, help="Name of blend file to save as")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])

    init(args.terrain_file, args.tile_file, args.terrain_name, args.tile_name, args.save_as)
    
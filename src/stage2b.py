import os, sys
import bpy
import math
import numpy as np
import typing as T
import bmesh
from contextlib import ExitStack
from tqdm import tqdm
from bpy.app.handlers import persistent

dir = os.path.dirname(bpy.data.filepath)
if not dir in sys.path: sys.path.append(dir)

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# End
from BlenderLib.Mesh import MeshObject as MeshObject2
from BlenderLib.Material import BSDF_Textured_Material, MaterialBase
from blenderlib import (
    MeshObject, BakeService, VertexGroup,
    CoordSystem, AssertLiteralType
)

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


## Start

# 30,000 m^2 per 1024x1024 texture
TEXTURE_RESOLUTION_LUT = [
    # (Maximum Area in m^2, Tex_height, Tex_width)
    (128    , 64   , 64   ),
    (500    , 128  , 128  ),
    (1875   , 256  , 256  ),
    (7500   , 512  , 512  ),
    (30000  , 1024 , 1024 ),
    (-1     , 1024 , 1024 ),   # -1 means no limit, 
    # can't process tex larger than this since there will be a 
    # 4x superresolution pipeline after this step
]
def lookup_resolution(area: float) -> tuple[int, int]:
    for max_area, tex_height, tex_width in TEXTURE_RESOLUTION_LUT:
        if max_area == -1 or area <= max_area:
            return tex_height, tex_width
    return TEXTURE_RESOLUTION_LUT[-1][1], TEXTURE_RESOLUTION_LUT[-1][2]


def apply_bsdf_material(mesh: MeshObject):
    # mesh = MeshObject2(mesh.mesh_object)
    # AREA_THRESHOLD = 100.
    # def get_material_resolution(area: float) -> tuple[int, int]:
    #     if   area < 256   : return (64, 64)
    #     elif area < 1024  : return (128, 128)
    #     else: return (256, 256)
    #     # elif area < 3750  : return (256, 256)
    #     # elif area < 15000 : return (512, 512)
    #     # else: return (1024, 1024)
    
    # def generate_material(p):
    #     if (poly_area := p.area) < AREA_THRESHOLD: return None
    #     texture_size = get_material_resolution(poly_area)
    #     return BSDF_Textured_Material(
    #         f"{mesh.name.replace(' ', '_')}_{str(p.index).zfill(3)}",
    #         tex_height=texture_size[0], tex_width=texture_size[1]
    #     )
    
    # # for idx, building in enumerate(buildings):
    # # print(f"Applying UV Grid Material to building #{idx}")
    # mesh.clear_material()
    # base_mat = BSDF_Textured_Material(f"{mesh.name.replace(' ', '_')}_base", tex_height=512, tex_width=512)
    # base_mat.apply_on_certain_faces(mesh, lambda p: p.area < AREA_THRESHOLD)
        
    # MaterialBase.apply_per_face(mesh, generate_material)
    with mesh.scoped_select(), mesh.scoped_mode("OBJECT", scoped_active=True):
        material_name = f"Material {mesh.name}".replace(" ", "_")
    
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
        
        height, width = lookup_resolution(mesh.surface_area)
        print(f"Create new material - {material_name},\tResolution={height}x{width}")
        image = bpy.data.images.new(name=f"TextureMap {mesh.name}", height=height, width=width)
        image.file_format = 'JPEG'
        image.pack()
        image_texture.image = image
        
        # bsdf.inputs[2].default_value = 1.0
        # bsdf.inputs['Specular IOR Level'].default_value = 0.000
        
        image_texture.location = (-600, 0)
        bsdf.location=(0, 0)
        material_output.location = (300, 0)
        
        # material.node_tree.links.new(image_texture.outputs["Color"], bsdf.inputs["Base Color"])
        # material.node_tree.links.new(bsdf.outputs["BSDF"], material_output.inputs["Surface"])
        material.node_tree.links.new(image_texture.outputs["Color"], bsdf.inputs["Color"])
        material.node_tree.links.new(bsdf.outputs["Emission"], material_output.inputs["Surface"])
        mesh.data.materials.append(material)
        
        material_index = mesh.data.materials.find(material.name)
        #
        
        with mesh.scoped_mode("EDIT", True):
            bpy.ops.mesh.select_all(action='SELECT')
            mesh.mesh_object.active_material_index = material_index
            bpy.ops.object.material_slot_assign()
            
            try:
                bpy.ops.uv.smart_project()
            except KeyboardInterrupt:
                raise KeyboardInterrupt() from None
            except:
                print(f"Failed to reproject UV on {mesh.name}, expected to see deteriorated result on this building")

        bpy.ops.object.material_slot_remove_unused()


def get_aabb(obj: MeshObject):
    verts = obj.verts_Tworld
    return verts.min(axis=0), verts.max(axis=0)


def align_mesh_alt(target_mesh: MeshObject, move_meshes: list[MeshObject], coord_sys: CoordSystem,
                   reduction: T.Literal["Median", "Mean", "Min", "Max"], only_bottom_verts: bool,
                   direction: T.Literal["TopDown", "BottomUp"],
                   sample_strategy: T.Literal["Vertex", "Uniform"]="Vertex") -> list[float]:
    AssertLiteralType(coord_sys, CoordSystem)
    target_mesh.apply_transform()
    
    if coord_sys == "Z+":
        alt_axis = 2
        plane_axis_1 = 0
        plane_axis_2 = 1
        ray_direction   = (0., 0., 1.) if direction == "BottomUp" else (0., 0., -1)
    else:
        alt_axis = 1
        plane_axis_1 = 0
        plane_axis_2 = 2
        ray_direction   = (0., 1., 0.) if direction == "BottomUp" else (0., -1., 0.)
    
    alt_offsets: list[float] = []
    target_bvhtree = None
    
    for move_mesh in move_meshes:
        move_mesh.apply_transform()
        move_mesh_verts = move_mesh.verts_Tworld
        
        if move_mesh_verts.size == 0:
            alt_offsets.append(0.)
            continue
        
        if only_bottom_verts:
            move_mesh_min_alt = move_mesh_verts[..., alt_axis].min()
            is_bottom_verts = np.isclose(move_mesh_verts[..., alt_axis], move_mesh_min_alt, atol=.5)
            move_mesh_verts = move_mesh_verts[is_bottom_verts]
        
        if sample_strategy == "Vertex":
            ray_sources = move_mesh_verts.copy()
        else:
            min_coord, max_coord = get_aabb(move_mesh)
            sample_x, sample_y = np.meshgrid(
                np.linspace(min_coord[plane_axis_1], max_coord[plane_axis_1], num=20),
                np.linspace(min_coord[plane_axis_2], max_coord[plane_axis_2], num=20)
            )
            ray_sources = np.empty((sample_x.size, 3))
            ray_sources[..., plane_axis_1] = sample_x.flatten()
            ray_sources[..., plane_axis_2] = sample_y.flatten()
        ray_sources[..., alt_axis] = -100. if direction == "BottomUp" else 100.
        
        target_alts, mask, target_bvhtree = target_mesh.cast_ray_on(
            ray_sources, ray_direction, 500., target_bvhtree, use_modifiers=False
        )
        
        if sample_strategy == "Vertex":
            offsets = (move_mesh_verts[mask][..., alt_axis] - target_alts[mask][..., alt_axis])
        else:
            orig_alts, orig_mask, _ = move_mesh.cast_ray_on(
                ray_sources, ray_direction, 500., None, use_modifiers=False
            )
            mask = mask & orig_mask
            offsets = (orig_alts[mask][..., alt_axis] - target_alts[mask][..., alt_axis])
        
        if offsets.size == 0: offset = 0.
        elif reduction == "Min": offset = offsets.min()
        elif reduction == "Max": offset = offsets.max()
        elif reduction == "Mean": offset = offsets.mean()
        elif reduction == "Median": offset = np.median(offsets)
        
        alt_offsets.append(offset)
    
    return alt_offsets


def convert_z2y(mesh: MeshObject):
    mesh.apply_transform()
    mesh.mesh_object.rotation_mode = 'XYZ'
    mesh.mesh_object.rotation_euler[0] = math.radians(270)
    mesh.apply_transform()


def add_delta_z_to_vertices(obj: MeshObject, delta_z, pred, coord: CoordSystem):
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()
    world_matrix = obj.mesh_object.matrix_world
    
    alt_axis = 1 if coord == "Y+" else 2
    
    mod_count = 0
    for v in bm.verts:
        world_coord = world_matrix @ v.co
        if pred(world_coord):
            v.co[alt_axis] += delta_z
            mod_count += 1
    
    print(f"\t{obj.name}, Modified : {mod_count}, delta_z : {delta_z}")
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()


def extract_roof_vertices(osm: MeshObject, tile: MeshObject, coord: CoordSystem) -> VertexGroup:
    roof = VertexGroup.create(tile, "Roof")
    
    if coord == "Y+": direction = (0., -1., 0.)
    else: direction = (0., 0., -1.)
    
    def is_roof_vertex(tile_verts):
        _, mask, _ = osm.cast_ray_on(
            sources=tile_verts, direction=direction, distance=200., 
            self_bvh=None, use_modifiers=False
        )
        return mask
    roof.add(is_roof_vertex)
    return roof


def separate_roofs(buildings: list[MeshObject], roof: VertexGroup, coord: CoordSystem) -> list[VertexGroup]:
    if coord == "Y+": direction = (0., -1., 0.)
    else: direction = (0., 0., -1.)
    
    roof_vert_mask = np.array([False] * len(roof.mesh.data.vertices), dtype=bool)
    roof_vert_mask[roof.verts_id] = True
    result_vgs: list[VertexGroup] = []
    
    for building in buildings:
        roof_vg = VertexGroup.create(roof.mesh, building.name)
        
        def is_specific_roof(tile_verts):
            roof_verts = tile_verts[roof_vert_mask]
            
            _, mask, _ = building.cast_ray_on(
                sources=roof_verts, direction=direction, distance=200., 
                self_bvh=None, use_modifiers=False
            )
            
            result_mask = roof_vert_mask.copy()
            result_mask[roof_vert_mask] = mask
            return result_mask

        roof_vg.add(is_specific_roof)
        result_vgs.append(roof_vg)
        
    return result_vgs


def main(args):
    coord: CoordSystem = "Y+"
    AssertLiteralType(coord, CoordSystem)
    alt_axis = 1 if coord == "Y+" else 2
    bakery = BakeService()
    
    osm_buildings: list[MeshObject] = [
        MeshObject(obj) for obj in bpy.data.objects
        if obj.type == "MESH"
    ]
    
    if args.remove_small:
        objects_filter = [ (obj, obj.surface_area) for obj in osm_buildings]
        osm_buildings = []
        for obj, surface in objects_filter:
            if surface < 500.: obj.delete()
            else: osm_buildings.append(obj)
    
    for osm_building in tqdm(osm_buildings):
        convert_z2y(osm_building)
    
    terrain_mesh = MeshObject.remoteAppend(args.terrain_blender, args.terrain_name)
    
    # Align meshes
    alt_offsets = align_mesh_alt(terrain_mesh, osm_buildings, coord, reduction="Median", only_bottom_verts=True, direction="TopDown")
    for building, offset in zip(osm_buildings, alt_offsets):
        building.mesh_object.matrix_world[alt_axis][3] -= offset
        building.apply_transform()
    
    tile_mesh    = MeshObject.remoteAppend(args.tile_blender, args.tile_name)
    tile_mesh.apply_transform()
    
    roof_offsets = align_mesh_alt(
        tile_mesh, osm_buildings, coord, reduction="Mean", only_bottom_verts=False, direction="TopDown",
        sample_strategy="Uniform"
    )
    for building, roof_offset in zip(osm_buildings, roof_offsets):
        try:
            min_alt = building.verts_Tworld[..., alt_axis].min()
        except ValueError:
            continue
        add_delta_z_to_vertices(building, -1 * roof_offset, lambda x: x[alt_axis] > min_alt + 1., coord)
    
    bottom_offsets = align_mesh_alt(terrain_mesh, osm_buildings, coord, reduction="Max", only_bottom_verts=True, direction="TopDown")
    for building, bottom_offset in zip(osm_buildings, bottom_offsets):
        try:
            bottom = building.verts_Tworld[..., alt_axis].min()
        except ValueError:
            continue
        add_delta_z_to_vertices(building, -1 * bottom_offset, lambda x: abs(x[alt_axis] - bottom) < 0.5, coord)
    
    # Generate roof vertex group for decorative purpose
    with ExitStack() as batch_selection:
        # Deselect everything
        for obj in bpy.data.objects:
            if obj.type != "MESH": continue
            batch_selection.enter_context(MeshObject(obj).scoped_select(False))
        
        # Create a OSM geometry mesh for efficient ray casting
        # Select everything we are interested in
        dup_buildings = [building.copy(keep_material=False) for building in osm_buildings]
        for dup_building in dup_buildings:
            dup_building.is_select = True
    
        with dup_buildings[0].scoped_active():
            bpy.ops.object.join()
    osm_geometry = dup_buildings[0]
    roof_verts = extract_roof_vertices(osm_geometry, tile_mesh, coord)
    # per_roof_verts = separate_roofs(osm_buildings, roof_verts, coord)
    # roof_verts.expand(1)
    
    osm_geometry.delete()
    
    with ExitStack() as stack:
        for obj in bpy.data.objects:
            if obj.type != "MESH" or obj.name == terrain_mesh.name: continue
            stack.enter_context(MeshObject(obj).scoped_select(False))
        
        for idx, building in enumerate(osm_buildings):
            print(f"\rApplying material {idx} / {len(osm_buildings)}", flush=True, end="")
            apply_bsdf_material(building)
        print("")
    
    if args.no_bake:
        tile_mesh.delete()
        return

    terrain_mesh.delete()

    print("Start baking")
    for building in tqdm(osm_buildings):
        try:
            bakery.bake_limited_dist([tile_mesh], building, in_offset=None, ray_distance=15.0)
        except:
            pass
    
    print("Done")
    
    
    tile_mesh.mask(roof_verts)
    tile_mesh.delete_loose()
    tile_mesh.name = "Roof"
    # tile_mesh.delete()
    bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
    bpy.ops.wm.save_as_mainfile(filepath=args.save_as)
    bpy.ops.wm.quit_blender()


def init(args):
    bpy.ops.wm.open_mainfile(filepath=args.osm_blender)
    main(args)
    # bpy.app.timers.register(lambda: main(args),first_interval=0.5)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--osm_blender", type=str, required=True, help="Path to the osm file")
    parser.add_argument("--terrain_blender", type=str, required=True, help="Blender file with terrain information for elevation alignment")
    parser.add_argument("--tile_blender", type=str, required=True, help="Blender file with ground plane of scene with corresponding textures")
    
    parser.add_argument("--terrain_name", type=str, required=True, help="Name of Terrain Mesh")
    parser.add_argument("--tile_name", type=str, required=True, help="Name of Tile Mesh")
    
    parser.add_argument("--save_as", type=str, required=True, help="Save resulted blender file to ...")
    parser.add_argument("--no-bake", action="store_true")
    parser.add_argument("--remove_small", action="store_true")
    
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    
    init(args)

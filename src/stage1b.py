import os, sys
import math
import pickle
import bpy
import bmesh
import numpy as np
import mathutils
from mathutils import Euler
from bpy.app.handlers import persistent

import scipy
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from blenderlib import MeshObject, CoordSystem, AssertLiteralType, BakeService
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


class LinearNDInterpolatorExt(object):
  def __init__(self, points,values):
    self.funcinterp = scipy.interpolate.LinearNDInterpolator(points,values)
    self.funcnearest= scipy.interpolate.NearestNDInterpolator(points,values)
  def __call__(self,*args):
    t=self.funcinterp(*args)
    if not np.isnan(t):
      return t.item(0)
    else:
      return self.funcnearest(*args)


def apply_emissive_material_per_face(mesh_name: str):
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
        
        image = bpy.data.images.new(name=f"Texture_{material_name}", height=512, width=512)
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


def create_base_material():
    material = bpy.data.materials.new(name="Ground_base_mat")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    node_emissive = nodes.new(type='ShaderNodeEmission')
    
    node_emissive.inputs['Color'].default_value = (0.448, 0.292, 0.137, 1)  # Brown color
    
    node_output = nodes.new(type='ShaderNodeOutputMaterial')
    links = material.node_tree.links
    links.new(node_emissive.outputs['Emission'], node_output.inputs['Surface'])
    return material


def main(radius: float, ref: str, save: str):
    coord: CoordSystem = "Y+"
    radius += 50. # FIXME: this is not good, should fundamentally fix this problem.
    
    # Cleanup
    try: MeshObject.withName("Cube").delete()
    except: pass
    
    # Terrain plane
    bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, location=(0, 0, 0))
    terrain = MeshObject(bpy.context.active_object)
    terrain.name = "Terrain"
    terrain.mesh_object.scale = (radius, radius, 1.)
    
    if coord == "Y+":
        terrain.mesh_object.rotation_mode = 'XYZ'
        terrain.mesh_object.rotation_euler = Euler((math.radians(90), 0., 0.), 'XYZ')
    else: pass
    
    terrain.apply_transform()
    
    # Subdivide terrain plane (initial subdivision for material partition)
    material_subdivide_lv = 3
    terrain.subdivide(material_subdivide_lv)
    
    # Apply material for eachface of terrain mesh
    with terrain.scoped_mode("OBJECT", True):
        apply_emissive_material_per_face(terrain.name)
    
    # Subdivide terrain plane (for higher terrain height resolution)
    terrain.subdivide(6 - material_subdivide_lv)
    
    # Create a base material for the ground
    base_material = create_base_material()    
    base_idx      = len(terrain.data.materials)
    terrain.data.materials.append(base_material)
    
    # Extrude the plane for creating volumn
    terrain_thickness = 20.0
    ground_surface_verts = []
    
    with terrain.scoped_mode("EDIT", True):
        bm = bmesh.from_edit_mesh(terrain.data)
        ground_surface_verts = [v.co.copy() for v in bm.verts]
        for face in bm.faces: face.select = True
        
        extrude_offset = (0, 0, -1 * terrain_thickness) if coord == "Z+" else (0, -1 * terrain_thickness, 0)
        
        # Store the original face
        original_face = bm.faces[:]
        
        # Extrude
        ret = bmesh.ops.extrude_face_region(bm, geom=original_face)
        extruded_geom = ret["geom"]
        
        # Move the extruded faces down
        extruded_verts = [v for v in extruded_geom if isinstance(v, bmesh.types.BMVert)]
        bmesh.ops.translate(bm, vec=extrude_offset, verts=extruded_verts)

        for face in bm.faces:
            if face in original_face: pass
            else: face.material_index = base_idx
        
        bmesh.update_edit_mesh(terrain.data)

    with open(ref, "rb") as fb: Reference_Points = pickle.load(fb)
    alt_axis = 1 if coord == "Y+" else 2
    plane_2nd_axis = 2 if coord == "Y+" else 1
    
    alts     = Reference_Points[..., alt_axis]
    terrain.mesh_object.location = (0., alts.min(), 0.) if coord == "Y+" else (0., 0., alts.min())
    terrain.apply_transform()
    
    xs = Reference_Points[..., 0]
    ys = Reference_Points[..., plane_2nd_axis]
    interpolator = LinearNDInterpolatorExt(np.stack([xs, ys], axis=-1), alts)

    # Shape the terrain
    with terrain.scoped_mode("EDIT", True):
        bm = bmesh.from_edit_mesh(terrain.data)
        bm.verts.ensure_lookup_table()
        
        for idx, vert_co in enumerate(ground_surface_verts):
            new_alt   = interpolator(vert_co[0], vert_co[plane_2nd_axis])
            new_coord = (vert_co[0], new_alt, vert_co[2]) if coord == "Y+" else (vert_co[0], vert_co[1], new_alt)
            bm.verts[idx].co = mathutils.Vector(new_coord)
        
        bmesh.update_edit_mesh(terrain.data)

    bpy.ops.wm.save_as_mainfile(filepath=save)
    bpy.ops.wm.quit_blender()
    print("\aDone.")

# def init(args):
#     bpy.app.timers.register(
#         lambda: main(args.radius, args.ref, args.save),
#         first_interval=0.5
#     )


if __name__ == "__main__":
    import argparse, sys
    parser = argparse.ArgumentParser()
    parser.add_argument('--radius'  , type=float, required=True, help="Radius of the scene")
    parser.add_argument('--ref'     , type=str, required=True, help="Reference Points")
    parser.add_argument('--save'    , type=str, required=True, help="Save terrain to")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    main(args.radius, args.ref, args.save)

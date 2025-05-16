from __future__ import annotations
from typing import Any
from typing_extensions import Self
import logging

import numpy as np
import bpy
from mathutils.bvhtree import BVHTree
from mathutils import Matrix
bpy: Any        # Just ignore type checking on bpy since it does not provide any information

from ..BaseObject import BlenderObject
from ..Context import (
    SetObjectMode, SetObjectActive, SetObjectSelect, GetBMesh,
    T_OBJECT_MODE
)


_logger = logging.getLogger("BlenderLib")


bpy.types.Object.mesh_proxy_type = bpy.props.StringProperty(
    name="Proxy Type",
    description="Proxy type for current mesh object in SceneGen codebase.",
    default="MeshObject"
)

class PROXYMESH_PT_CustomPanel(bpy.types.Panel):
    bl_label = "SceneGen Properties"
    bl_idname = "PROXYMESH_PT_CustomPanel"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "object"
    
    def draw(self, context):
        layout = self.layout
        obj = context.object

        # Display custom properties in the panel
        layout.prop(obj, "mesh_proxy_type")


class MeshObject(BlenderObject):
    """A proxy class for blender Object with type 'MESH'.
    """
    bpy.utils.register_class(PROXYMESH_PT_CustomPanel)
    _HIERARCHY: dict[str, type] = dict()
    
    def __init__(self, mesh):
        """Mesh: a blender object that has type 'MESH'.
        """
        assert mesh is not None and mesh.type == 'MESH', f"MeshObject can only be instantiated with blender object with type 'MESH'. Get {mesh.type} instead."
        mesh.mesh_proxy_type = self.__class__.__name__
        super().__init__(mesh)
    
    
    # Proxy auto-speicialization and register.
    # This is similar to DynReflect class in Library.Reflect, but we want to separate DynReflect
    # with the Blender library, so re-implement it here.
    #
    # This allows us to create type-specific proxy class for a blender project automatically.
    @classmethod
    def proxy(cls, mesh) -> Self:
        return cls(mesh)
    
    def __init_subclass__(cls, **kwargs) -> None:
        cls._HIERARCHY = {"": cls}
        checkbase = list(filter(lambda x: issubclass(x, MeshObject), cls.__bases__))
        assert len(checkbase) == 1, "Does not support diamond inheritance in SubclassRegistry"
        
        for pcls in cls.mro()[1:]:
            if not issubclass(pcls, MeshObject): continue
            if cls.__name__ in pcls._HIERARCHY:
                _logger.fatal(f"SubclassRegistry Error: There more than one descendent of class 'DynReflect' with name of {cls.__name__}. "
                             "This introduces ambiguity to dynamic reflection and is therefore disallowed.")
                raise NameError(f"Repetitive class name {cls.__name__}")
            else:
                pcls._HIERARCHY[cls.__name__] = cls

    @classmethod
    def get_class(cls, type: str) -> type[Self]:
        clsname = type  # bad to have variable name as keyword, but the most fitting name
        if clsname in cls._HIERARCHY: return cls._HIERARCHY[clsname]
        _logger.fatal(f"Get '{clsname}' from class {cls.__name__}, expect to be one of {list(cls._HIERARCHY.keys())}")
        raise KeyError(f"Get '{clsname}' from class {cls.__name__}, expect to be one of {list(cls._HIERARCHY.keys())}")
    
    @classmethod
    def specialize(cls, mesh) -> MeshObject:
        proxy_class_name = mesh.mesh_proxy_type
        return cls.get_class(proxy_class_name).proxy(mesh)
    
    @classmethod
    def specialize_all(cls) -> list[MeshObject]:
        results = []
        for obj in bpy.data.objects:
            if obj.type != "MESH": continue
            results.append(MeshObject.specialize(obj))
        return results
    # End
    
    # Methods for generate / retrieve a mesh from blender #####################
    @classmethod
    def get_withName(cls, name: str) -> MeshObject:
        return cls(mesh=bpy.data.objects[name])
    
    @classmethod
    def get_fromFile(cls, blender_file: str, mesh_name: str) -> MeshObject:
        with SetObjectMode("OBJECT"):
            before_object = set(bpy.data.objects)
            bpy.ops.wm.append(
                filepath  = f"{blender_file}\\Object\\{mesh_name}", 
                directory = f"{blender_file}\\Object",
                filename  = mesh_name
            )
            after_object = set(bpy.data.objects)
            appended_object = list(after_object - before_object)[0]
        return MeshObject(appended_object)

    # Properties ##############################################################
    @property
    def is_active(self) -> bool: return self.object == bpy.context.active_object
    
    def set_active(self) -> None: bpy.context.view_layer.objects.active = self.object
    
    @property
    def is_select(self) -> bool: return self.object.select_get()

    @is_select.setter
    def is_select(self, mode: bool) -> None: return self.object.select_set(mode)

    @property
    def mode(self) -> T_OBJECT_MODE: return self.object.mode

    @mode.setter
    def mode(self, mode: T_OBJECT_MODE) -> None:
        assert self.is_active, "Can only change mode of active object"
        bpy.ops.object.mode_set(mode=mode)

    @property
    def coord_Tobj(self) -> np.ndarray:
        """
        Get vertex coordinates under object coordinate frame
        """
        vertices = np.empty(len(self.data.vertices) * 3, dtype=np.float32)
        self.data.vertices.foreach_get("co", vertices)
        return vertices.reshape(-1, 3)

    @property
    def coord_Tworld(self) -> np.ndarray:
        """
        Get vertex coordinates under world coordinate frame
        """
        world_matrix = self.T_obj2world
        local_coords = self.coord_Tobj

        local_coords_homogeneous = np.column_stack((local_coords, np.ones(len(local_coords))))
        world_coords_homogeneous = local_coords_homogeneous @ world_matrix.T

        w = world_coords_homogeneous[:, 3]
        world_coords = world_coords_homogeneous[:, :3] / w[:, np.newaxis]
        return world_coords

    @property
    def aabb_corner(self) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
        """Returns (min xyz), (max xyz) if the mesh has at least one vertex, None otherwise"""
        coords = self.coord_Tworld
        if coords.size == 0: return None
        min_coord = coords.min(axis=0)
        max_coord = coords.max(axis=0)
        return (min_coord[0], min_coord[1], min_coord[2]), (max_coord[0], max_coord[1], max_coord[2])
    
    @property
    def num_faces(self) -> int: return len(self.data.polygons)
    
    @property
    def num_edges(self) -> int: return len(self.data.edges)
    
    @property
    def num_vertices(self) -> int: return len(self.data.vertices)
    
    # Slow operations   #######################################################
    def surface_area(self) -> float:
        total_area = 0.0
        for p in self.object.data.polygons: total_area += p.area
        return total_area
    
    def as_BVHTree(self, use_modifier: bool) -> BVHTree:
        with GetBMesh(self.object, use_modifier, write_back=False) as bm:
            tree = BVHTree.FromBMesh(bm)
        return tree
    
    def clone(self, keep_material: bool=True) -> MeshObject:
        """Clone the current object and underlying mesh.

        Args:
            keep_material (bool, optional): Retain material of original mesh, if set to false, the copied mesh will have no material. Defaults to True.

        Returns:
            MeshObject: Cloned mesh object.
        """
        new_object = self.object.copy()
        new_object.data = self.object.data.copy()
        bpy.context.collection.objects.link(new_object)
        
        if not keep_material:
            new_object.data.materials.clear()
        
        return MeshObject(new_object)
    
    def delete(self):
        _logger.info("Remove mesh", self.object.name)
        bpy.data.objects.remove(self.object, do_unlink=True)

    def clear_material(self):
        with SetObjectActive(self.object), SetObjectMode("OBJECT"):
            for _ in range(1,len(self.object.material_slots)):
                self.object.active_material_index = 1
                bpy.ops.object.material_slot_remove()
            
            with SetObjectMode("EDIT"):
                bpy.ops.mesh.select_all(action = 'SELECT')
                bpy.ops.object.material_slot_assign()
            bpy.ops.object.material_slot_remove_unused()

    def apply_transform(self, location: bool=True, rotation: bool=True, scale: bool=True):
        with SetObjectActive(self.object), SetObjectMode("OBJECT"):
            bpy.ops.object.transform_apply(location=location, rotation=rotation, scale=scale)
    
    def delete_loose(self):
        with SetObjectSelect(self.object, True), SetObjectActive(self.object), SetObjectMode("EDIT"):
            bpy.ops.mesh.select_all(action = 'SELECT')
            bpy.ops.mesh.delete_loose()
            bpy.ops.mesh.select_all(action = 'DESELECT')

MeshObject._HIERARCHY['MeshObject'] = MeshObject

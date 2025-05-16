from typing import Any, Callable
import bpy
bpy: Any

from ..Mesh import MeshObject
from ..Context import SetObjectActive, SetObjectMode, SetObjectSelect, GetBMesh

class MaterialBase:
    def __init__(self, material) -> None:
        """Base class of all sorts of materials

        Args:
            material (bpy.Material): Blender python Material
        """
        self.material = material


    def register_to_object(self, mesh: MeshObject) -> int:
        """Register current material to the Mesh object, return handle.

        Args:
            mesh (MeshObject): The object to add current material to.
        """
        is_material_self = [slot.material.name == self.material.name for slot in mesh.object.material_slots if slot.material is not None]
        try:
            return is_material_self.index(True)
        except ValueError: ...  # Not found, therefore register it to mesh.
        
        mat_idx      = len(mesh.data.materials)
        mesh.data.materials.append(self.material)
        return mat_idx
    

    def apply_entire_mesh(self, mesh: MeshObject):
        """Apply material to every face of the mesh.
        """
        mesh.data.materials.clear()
        mat_idx = self.register_to_object(mesh)
        with SetObjectSelect(mesh.object), SetObjectActive(mesh.object), SetObjectMode("EDIT"):
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.uv.smart_project(correct_aspect=False)
    
    def apply_on_certain_faces(self, mesh: MeshObject, predicate: Callable[[Any,], bool]):
        with SetObjectActive(mesh.object), SetObjectSelect(mesh.object), SetObjectMode("OBJECT"):
            with SetObjectMode("EDIT"):
                bpy.ops.mesh.select_mode(type="FACE")
                bpy.ops.mesh.select_all(action='DESELECT')
                
            with SetObjectMode("OBJECT"):
                for idx in range(len(mesh.data.polygons)):
                    mesh.data.polygons[idx].select = predicate(mesh.data.polygons[idx])
                    
            with SetObjectMode("EDIT"):
                material_index = self.register_to_object(mesh)
                mesh.object.active_material_index = material_index
                
                bpy.ops.object.material_slot_assign()
                bpy.ops.uv.smart_project(correct_aspect=False)

    @classmethod
    def apply_per_face(cls, mesh: MeshObject, face_to_mat: Callable[[Any,], "MaterialBase | None"]) -> list["MaterialBase"]:
        """Apply material to each face of the mesh separately. Specifically a callback function that
        takes in a blender polygon (from mesh.data) will be called on each face. The callback is expected to 
        generate an material for each call and the resulted material will be assigned to that face specifically.
        """
        materials = []
        
        with SetObjectActive(mesh.object), SetObjectSelect(mesh.object), SetObjectMode("OBJECT"):
            for idx in range(len(mesh.data.polygons)):
                print(f"\rApplying materials: {idx + 1} / {len(mesh.data.polygons)}", end="", flush=True)
                
                with SetObjectMode("EDIT"):
                    bpy.ops.mesh.select_mode(type="FACE")
                    bpy.ops.mesh.select_all(action='DESELECT')
                
                with SetObjectMode("OBJECT"):
                    mesh.data.polygons[idx].select = True
                    
                with SetObjectMode("EDIT"):
                    material = face_to_mat(mesh.data.polygons[idx])
                    if material is None: continue
                    
                    material_index = material.register_to_object(mesh)
                    mesh.object.active_material_index = material_index
                    
                    bpy.ops.object.material_slot_assign()
                    bpy.ops.uv.smart_project(correct_aspect=False)
        
        return materials

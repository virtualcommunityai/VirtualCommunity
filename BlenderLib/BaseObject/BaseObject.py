from __future__ import annotations
from mathutils import Matrix
import numpy as np

class BlenderObject:
    def __init__(self, obj):
        self.object = obj

    # Hierarchy Relationship
    def add_child(self, mesh: BlenderObject) -> None:
        mesh.object.parent = self.object
    
    def del_child(self, mesh: BlenderObject) -> None:
        if mesh.object.parent != self.object: return
        mesh.object.parent = None
    
    @property
    def children(self) -> list[BlenderObject]:
        return [BlenderObject(child) for child in self.object.children]
    
    @property
    def name(self) -> str: return self.object.name
    
    @name.setter
    def name(self, new_name: str) -> None:
        self.object.name = new_name
        self.object.data.name = new_name

    @property
    def data(self): return self.object.data

    @property
    def T_obj2world(self) -> np.ndarray:
        return np.array(self.object.matrix_world)
    
    @T_obj2world.setter
    def T_obj2world(self, new_transform: np.ndarray) -> None:
        blender_matrix = Matrix(new_transform.tolist())
        self.object.matrix_world = blender_matrix
    
    @property
    def T_world2obj(self) -> np.ndarray:
        return np.linalg.inv(np.array(self.object.matrix_world))

import numpy as np
import mathutils

from .MeshObject import MeshObject
from ..Context.ModeSetter import SetObjectMode, SetObjectActive, SetObjectSelect



class RayCastMesh(MeshObject):
    """A specialized mesh that support efficient ray casting logic.
    """
    def __init__(self, mesh, use_modifier: bool):
        super().__init__(mesh)
        with SetObjectSelect(mesh, True), SetObjectActive(mesh), SetObjectMode('EDIT'):
            self.bvh_tree = self.as_BVHTree(use_modifier=use_modifier)
    
    @classmethod
    def get_fromMeshObject(cls, mesh_object: MeshObject, use_modifier: bool):
        return cls(mesh_object.object, use_modifier)

    @property
    def as_MeshObject(self) -> MeshObject:
        return MeshObject(self.object)
    
    # Ray Casting Logic #######################################################
    def cast_ray_on(self, sources: np.ndarray, direction: tuple[float, float, float], max_dist: float) -> tuple[np.ndarray, np.ndarray]:
        """Cast ray ono the mesh from a sequence of sources along the specified direction

        Args:
            sources (np.ndarray): Nx3 np array as the source of ray
            direction (tuple[float, float, float]): direction of ray casting
            max_dist (float): distance threshold for ray casting, will be invalid after this distance threshold

        Returns:
            tuple[np.ndarray, np.ndarray, BVHTree]: 
                * Ray casting positions - Nx3 array
                * Validity Mask - N array with boolean value, True means valid
        """
        queries = [mathutils.Vector((sources[idx, 0], sources[idx, 1], sources[idx, 2]))
                   for idx in range(sources.shape[0])]
        
        result_position = np.zeros_like(sources)
        result_validity = np.zeros((sources.shape[0],), dtype=bool)
        
        for idx, query in enumerate(queries):
            on_mesh_pt, _, _, _ = self.bvh_tree.ray_cast(
                query, mathutils.Vector(direction), max_dist
            )
            if on_mesh_pt is None:
                result_validity[idx] = False
            else:
                result_validity[idx] = True
                result_position[idx, 0] = on_mesh_pt.x
                result_position[idx, 1] = on_mesh_pt.y
                result_position[idx, 2] = on_mesh_pt.z
        
        return result_position, result_validity

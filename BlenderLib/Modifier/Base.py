import bpy
from typing import Any
bpy: Any

from ..Mesh import MeshObject
from ..Context import SetObjectActive, SetObjectMode


class ModifierBase:
    def __init__(self, modifier, mesh: MeshObject):
        """Modifier: blender object modifier that can be applied to
        """
        self.modifier = modifier
        self.mesh     = mesh
    
    def apply(self):
        with SetObjectActive(self.mesh.object), SetObjectMode("OBJECT"):
            bpy.ops.object.modifier_apply(modifier=self.modifier.name)

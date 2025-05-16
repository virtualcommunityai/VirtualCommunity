import bpy
from typing import Any
bpy: Any

from ..Mesh import MeshObject
from ..Context import SetObjectActive, SetObjectMode

from .Base import ModifierBase

class SubdivideModifier(ModifierBase):
    def __init__(self, mesh: MeshObject, iteration: int, name: str) -> None:
        with SetObjectActive(mesh.object), SetObjectMode("OBJECT"):
            subs_mod = mesh.object.modifiers.new(name=name, type="SUBSURF")
            subs_mod.levels = iteration
            subs_mod.subdivision_type = "SIMPLE"
        
        super().__init__(subs_mod, mesh)


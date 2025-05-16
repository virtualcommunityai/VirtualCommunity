import bpy
from typing import Any
bpy: Any

from ..Mesh import MeshObject

def clear_objects():
    to_remove: list[MeshObject] = []
    for obj in bpy.data.objects:
        if obj.type == 'MESH': to_remove.append(MeshObject(obj))
    for obj in to_remove: obj.delete()

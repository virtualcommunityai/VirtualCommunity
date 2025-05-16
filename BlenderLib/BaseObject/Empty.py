import bpy
from typing import Any
bpy: Any

from .BaseObject import BlenderObject


class EmptyObject(BlenderObject):
    def __init__(self, name, loc: tuple[float, float, float]) -> None:
        empty = bpy.data.objects.new(name, None)
        bpy.context.collection.objects.link(empty)
        empty.empty_display_type = 'PLAIN_AXES'
        empty.location = loc
        
        empty.name = name
        super().__init__(empty)

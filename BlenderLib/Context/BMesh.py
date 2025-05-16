from contextlib import contextmanager
from typing import Generator, Any, Literal

import bpy
import bmesh    #type: ignore
bpy  : Any
bmesh: Any


@contextmanager
def GetBMesh(mesh, use_modifier: bool, write_back: bool) -> Generator[Any, None, None]:
    """Get a BMesh object that can operate on and modify freely. All changes will be write back
    to the underlying mesh when exiting the context.

    Args:
        mesh (bpy.Object): Blender Object with type 'MESH'
        use_modifier     : Apply modifier on the object to bmesh before returning one

    Yields:
        Generator[Any]: The BMesh that can be edited / apply operations on.
    """
    if use_modifier:
        depsgraph = bpy.context.evaluated_depsgraph_get()
        object_eval = mesh.evaluated_get(depsgraph)
        me = object_eval.to_mesh()
        bm = bmesh.new()
        bm.from_mesh(me)
    else:
        me = mesh.data
        bm = bmesh.new()
        bm.from_mesh(me)
        object_eval = None
    
    try: yield bm
    finally:
        if write_back:
            bm.to_mesh(me)
            me.update()
            bm.free()
            bmesh.update_edit_mesh(me)
            if object_eval is not None: object_eval.to_mesh_clear()

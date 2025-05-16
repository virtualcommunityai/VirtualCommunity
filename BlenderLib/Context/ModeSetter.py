from contextlib import contextmanager
from typing import Generator, Any, Literal

import bpy
bpy: Any


T_OBJECT_MODE = Literal["OBJECT", "EDIT", "POSE", "SCULPT", "VERTEX_PAINT", "WEIGHT_PAINT", "TEXTURE_PAINT", "PARTICLE_EDIT"]


@contextmanager
def SetObjectMode(set_to: T_OBJECT_MODE) -> Generator[None, None, None]:
    """Enter an object mode of currently active object in Blender and restore original mode upon exit.
    """
    try: 
        enter_mode = bpy.context.object.mode
        bpy.ops.object.mode_set(mode=set_to)
    except:
        enter_mode = None
    
    try: yield
    except Exception as e: raise e
    finally:
        if enter_mode is not None: 
            bpy.ops.object.mode_set(mode=enter_mode)


@contextmanager
def SetObjectActive(obj) -> Generator[None, None, None]:
    """Set an object to active, re-activate the original active object upon exit.

    Args:
        obj (bpy.Object): Object with type 'MESH'.
    """
    original_active_obj = bpy.context.active_object
    bpy.context.view_layer.objects.active = obj
    try: yield
    finally: bpy.context.view_layer.objects.active = original_active_obj


@contextmanager
def SetObjectSelect(obj, select: bool=True) -> Generator[None, None, None]:
    """Set an object to be selected in Blender, restore its original selection status upon exit.

    Args:
        obj (bpy.Object): Blender Object.
        select (bool, optional): Whether to select/deselect the object. Defaults to True.
    """
    original_select = obj.select_get()
    obj.select_set(select)
    try: yield
    finally: obj.select_set(original_select)

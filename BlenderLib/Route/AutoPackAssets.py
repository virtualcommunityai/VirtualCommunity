from typing import Any
import bpy
bpy: Any

from bpy.app.handlers import persistent #type: ignore

@persistent
def save_mod_images(_):
    """ Save all modified images """
    if any(i.is_dirty for i in bpy.data.images):
        bpy.ops.image.save_all_modified()

@persistent
def pack_dirty_images(_):
    """ Pack all modified images """ 
    for i in bpy.data.images:
        if i.is_dirty:
            i.pack()
            print("Packed:", i.name)

def set_save_all_assets():
    bpy.app.handlers.save_pre.append(save_mod_images)
    bpy.app.handlers.save_pre.append(pack_dirty_images)

def save_and_exit_blender(save_to: str):
    bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
    bpy.ops.wm.save_as_mainfile(filepath=save_to)
    bpy.ops.wm.quit_blender()

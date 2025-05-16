import bpy
from typing import Any, Literal
bpy: Any

from .UseDevice import configure_cycles_devices


T_BakeType = Literal["COMBINED", "AO", "SHADOW", "NORMAL", "UV", "ROUGHNESS", "EMIT", "ENVIRONMENT", "DIFFUSE", "GLOSSY", "TRANSMISSION", "SUBSURFACE"]
_T_Device   = Literal["CPU", "GPU"]

def setup_cycles_engine(bake_type: T_BakeType, device: _T_Device="GPU", sample_cnt: int=256):
    assert sample_cnt >= 64, "https://docs.blender.org/api/current/bpy.types.RenderSettings.html Claims the minimum sample size to be 64." 
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.render.bake_settings.bake_samples = sample_cnt
    bpy.context.scene.cycles.device = device
    bpy.context.scene.cycles.bake_type = bake_type
    configure_cycles_devices(user_pref_rank=["CUDA", "OPTIX", "CPU"])

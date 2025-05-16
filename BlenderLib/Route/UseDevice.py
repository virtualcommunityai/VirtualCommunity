import logging
from typing import Literal, Any

import bpy
bpy: Any

_logger = logging.getLogger("BlenderLib")


T_Preference = Literal["OPTIX", "CUDA", "CPU"]

def configure_cycles_devices(user_pref_rank: list[T_Preference]):
    assert bpy.context.scene.render.engine == "CYCLES"
    bpy.context.scene.cycles.device = "GPU"
    prefs = bpy.context.preferences.addons["cycles"].preferences

    # Necessary to "remind" cycles that the devices exist? Not sure. Without this no devices are found.
    for dt in prefs.get_device_types(bpy.context):
        prefs.get_devices_for_type(dt[0])

    assert len(prefs.devices) != 0, prefs.devices

    types = list(d.type for d in prefs.devices)

    types = sorted(types, key=user_pref_rank.index)
    use_device_type = types[0]

    if use_device_type == "CPU":
        _logger.warning(f"Render will use CPU-only, only found {types=}")
        bpy.context.scene.cycles.device = "CPU"
        return

    bpy.context.preferences.addons["cycles"].preferences.compute_device_type = use_device_type
    use_devices = [d for d in prefs.devices if d.type == use_device_type]

    _logger.info(f"Cycles will use {use_device_type=}, {len(use_devices)=}")

    for d in prefs.devices:
        d.use = False
    for d in use_devices:
        d.use = True

    return use_devices

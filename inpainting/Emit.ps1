<#

Sub-pipeline for emitting all textures of a scene

#>

param(
    # Scene Name (no _suffix, for instance, NY_ok will just be NY)
    [Parameter(Mandatory=$true)]
    [string]
    $scene_tag,

    # Blender path
    [Parameter(Mandatory=$true)]
    [string]
    $blender_exec,

    # Data Root (that stores all scenes)
    [Parameter(Mandatory=$true)]
    [string]
    $data_root
)

&$blender_exec -b -P ./inpainting/ground_emit.py -- `
    --blender_file ${data_root}/${scene_tag}_ok/${scene_tag}_stage1A.blend `
    --mesh Terrain `
    --save_to ${data_root}/${scene_tag}_ok/textures_${scene_tag}

&$blender_exec -b -P ./inpainting/building_emit.py -- `
    --blender_file ${data_root}/${scene_tag}_ok/${scene_tag}_stage1_merged.blend `
    --mesh Terrain `
    --output_dir ${data_root}/${scene_tag}_ok/textures_${scene_tag}_building `
    --scale_by 0.75

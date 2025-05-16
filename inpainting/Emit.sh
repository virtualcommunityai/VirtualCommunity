#!/bin/bash

# Sub-pipeline for emitting all textures of a scene

# Check arguments
if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <scene_tag> <blender_exec> <data_root>"
    exit 1
fi

# Parse arguments
scene_tag=$1
blender_exec=$2
data_root=$3

# Emit ground textures
"$blender_exec" -b -P ./inpainting/ground_emit.py -- \
    --blender_file "${data_root}/${scene_tag}/${scene_tag}_stage1A.blend" \
    --mesh Terrain \
    --save_to "${data_root}/${scene_tag}/textures_${scene_tag}"

# Emit building textures
"$blender_exec" -b -P ./inpainting/building_emit.py -- \
    --blender_file "${data_root}/${scene_tag}/${scene_tag}_stage1B_merged.blend" \
    --mesh Terrain \
    --output_dir "${data_root}/${scene_tag}/textures_${scene_tag}_building" \
    --scale_by 0.75

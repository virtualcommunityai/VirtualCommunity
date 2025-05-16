#!/bin/bash

# Pipeline for converting rough Google 3D tile to simulation ready mesh.
# Last update: 2025-1-24
# Example usage: bash pipeline_unity.sh --radius 400 --dataroot data/ --cacheroot cache/ --datapoint datapoint/MIT.txt

# Parameters
dataroot=""
cacheroot=""
datapoint_file=""
radius=""

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --dataroot)
      dataroot="$2"
      shift 2
      ;;
    --cacheroot)
      cacheroot="$2"
      shift 2
      ;;
    --datapoint)
      datapoint_file="$2"
      shift 2
      ;;
    --radius)
      radius="$2"
      shift 2
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
done

lat_lon_name=$(cat "$datapoint_file")
center_lat=$(echo $lat_lon_name | cut -d' ' -f1)
center_lon=$(echo $lat_lon_name | cut -d' ' -f2)
scene_name=$(echo $lat_lon_name | cut -d' ' -f3)

# Utility functions
add_suffix_to_filename() {
  local path="$1"
  local suffix="$2"
  local directory
  local filename_without_ext
  local extension
  directory=$(dirname "$path")
  filename_without_ext=$(basename "$path" | cut -f 1 -d '.')
  extension=$(basename "$path" | grep -o "\.[^.]*$")
  echo "${directory}/${filename_without_ext}${suffix}${extension}"
}

write_color_output() {
  local color="$1"  # The color name passed as the first argument
  shift  # Remove the first argument to handle the rest as the output message
  local color_code  # Variable to store the ANSI escape sequence for the color

  # Map color names to ANSI escape sequences
  case $color in
    green) color_code="\e[32m" ;;   # Green
    yellow) color_code="\e[33m" ;;  # Yellow
    red) color_code="\e[31m" ;;     # Red
    magenta) color_code="\e[35m" ;; # Magenta
    *) color_code="\e[0m" ;;        # Default (reset to no color)
  esac

  # Print the message with the selected color
  echo -e "${color_code}$*${RESET}"
}

# Executables
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
  blender="your_blender_path"
  upscayl="upscayl-ncnn/build/upscayl-bin"
  dst_path="VirtualCommunity/assets/scene/final/"
elif [[ "$OSTYPE" == "darwin"* ]]; then
  blender="/Applications/Blender.app/Contents/MacOS/Blender"
  upscayl="upscayl"
else
  blender="blender"
  upscayl="upscayl"
fi

pano_cache="streetview/${scene_name}"
inpainted_cache="streetview_inpaint/${scene_name}"
# make dirs
for dir in ${dataroot} ${cacheroot} ${cacheroot}/${scene_name} ${dataroot}/${scene_name} ${pano_cache} ${inpainted_cache}
do
  mkdir -p ${dir}
done

# IN Parameters
in_blender_file="${dataroot}/${scene_name}/${scene_name}.blend"
tile_blender_file="${dataroot}/${scene_name}/blender_output_${scene_name}_${radius}.blend"
mask_blender_file="${dataroot}/${scene_name}/mask_${scene_name}_${radius}.blend"
glb_file="${dataroot}/${scene_name}/${scene_name}.glb"

# OUT Parameters
terrain_name="Terrain"
mask_tile_name="Mesh_0"
tile_name="Mesh_0"
osm_blender_file=$(add_suffix_to_filename "$in_blender_file" "_osm")
terrain_blender_file=$(add_suffix_to_filename "$in_blender_file" "_terrain")

height_field_file="${dataroot}/${scene_name}/height_field.npz"
aabb_file="${dataroot}/${scene_name}/building_to_osm_tags.json"
pano_file="${dataroot}/${scene_name}/streetview_locs.pkl"
ref_ground_file="${dataroot}/${scene_name}/street_view_loc_clean_all.pkl"
solve_result_file="${dataroot}/${scene_name}/camera_solve_result.pkl"

out1C_blender_file=$(add_suffix_to_filename "$in_blender_file" "_stage1C")

# Prologue
write_color_output green "DataRoot=${dataroot}"
write_color_output green "SceneName=${scene_name}"
write_color_output green "Center=(${center_lat}, ${center_lon}), Radius=${radius}"

# Additional Outputs
out1A_blender_file=$(add_suffix_to_filename "$in_blender_file" "_stage1A")
out1A_inpaint_file=$(add_suffix_to_filename "$in_blender_file" "_stage1A_inpaint")
out1B_blender_file=$(add_suffix_to_filename "$in_blender_file" "_stage1B")
out1B_merged_file=$(add_suffix_to_filename "$in_blender_file" "_stage1B_merged")
out1B_inpaint_file=$(add_suffix_to_filename "$in_blender_file" "_stage1B_inpaint")
out1C_blender_file=$(add_suffix_to_filename "$in_blender_file" "_stage1C")
out4_blender_file=$(add_suffix_to_filename "$in_blender_file" "_stage2")
out5_blender_file=$(add_suffix_to_filename "$in_blender_file" "_stage2_superres")
out3A_blender_file=$(add_suffix_to_filename "$in_blender_file" "_stage3A")

# Stage 0a: Lat lon to Mask
if [[ ! -f "$mask_blender_file" ]]; then
  "$blender" -b --python ./src/stage0a.py -- \
    --lat ${center_lat} \
    --lng ${center_lon} \
    --rad 400 \
    --api_key_file secret.txt \
    --tag ${scene_name} \
    --mapillary_token_file mapillary_key.txt \
    --cache_root ${cacheroot}/${scene_name} \
    --tile_output_path ${tile_blender_file} \
    --mask_output_path ${mask_blender_file} \
    --ref_ground_output_path ${ref_ground_file}
  write_color_output green "    [OK ] Stage 0a Done."
else
  write_color_output yellow "    [Ign] Stage 0a Skip."
fi

# Stage 0b: Fetch Street view meta data
if [[ ! -f "$pano_file" ]]; then
  "$blender" -b --python ./src/stage0b.py -- \
    --lat ${center_lat} \
    --lng ${center_lon} \
    --rad 400 \
    --api_key_file secret.txt \
    --mapillary_token_file mapillary_key.txt \
    --streetview_locs_output_path $pano_file \
    --cache_root ${cacheroot}/${scene_name}
  write_color_output green "    [OK ] Stage 0b Done."
else
  write_color_output yellow "    [Ign] Stage 0b Skip."
fi

# Stage 0c: Fetch OSM Buildings
if [[ ! -f "$osm_blender_file" ]]; then
  "$blender" -b --python ./src/stage1a.py -- \
    --export "$osm_blender_file" \
    osm \
    --source server \
    --server overpass-api.de \
    --mode 3Dsimple \
    --circle "$center_lat" "$center_lon" "$radius" \
    --no-single-object --no-relative-to-initial-import \
    --no-water --no-forests --no-vegetation --no-highways --no-railways \
    --roof-shape flat \
    --level-height 3.0 \
    --no-subdivide
  write_color_output green "    [OK ] Fetch OSM Done."
else
  write_color_output yellow "    [Ign] Fetch OSM Skip."
fi

# Build Terrain
if [[ ! -f "$terrain_blender_file" ]]; then
  "$blender" -b --python ./src/stage1b.py -- \
    --radius "$radius" \
    --ref "$ref_ground_file" \
    --save "$terrain_blender_file"
  write_color_output green "    [OK ] Build Terrain Done."
else
  write_color_output yellow "    [Ign] Build Terrain Skip."
fi

# Create height field file
if [[ ! -f "$height_field_file" ]]; then
  "$blender" -b --python ./src/export_heightfield.py -- \
    --ref "$ref_ground_file" \
    --save_as "$height_field_file"
  write_color_output green "    [OK ] Build Heightfield Done."
else
  write_color_output yellow "    [Ign] Build Heightfield Skip."
fi

if [[ ! -f "$out1A_blender_file" ]]; then
  "$blender" -b --python ./src/stage2a.py -- \
    --terrain_file "$terrain_blender_file" \
    --tile_file "$mask_blender_file" \
    --terrain_name "$terrain_name" \
    --tile_name "$mask_tile_name" \
    --save_as "$out1A_blender_file"
  write_color_output green "    [OK ] Bake Terrain Done."
else
  write_color_output yellow "    [Ign] Bake Terrain Skip."
fi

# Stage 1B: Bake OSM Buildings
if [[ ! -f "$out1B_blender_file" ]]; then
  "$blender" -b --python ./src/stage2b.py -- \
    --osm_blender "$osm_blender_file" \
    --terrain_blender "$terrain_blender_file" \
    --tile_blender "$tile_blender_file" \
    --terrain_name "$terrain_name" \
    --tile_name "$tile_name" \
    --save_as "$out1B_blender_file"
  write_color_output green "    [OK ] Bake OSM Done."
else
  write_color_output yellow "    [Ign] Bake OSM Skip."
fi

# Stage 3A: Solve street view cameras
if [[ ! -f "$out3A_blender_file" ]]; then
  "$blender" -b --python ./src/stage3a.py -- \
    --input_blend_path "$out1B_blender_file" \
    --streetview_locs_path "$pano_file" \
    --output_solve_result_path "$solve_result_file" \
    --output_blend_path "$out3A_blender_file"
  write_color_output green "    [OK ] Solve Done."
else
  write_color_output yellow "    [Ign] Skip street view solving."
fi

# Stage 3B: Fetch street views
if [[ ! -f "$pano_cache/done.txt" ]]; then
  "$blender" -b --python ./src/stage3b.py -- \
    --solve_result_path "$solve_result_file" \
    --api_key_path secret.txt \
    --output_gsv_dir "$pano_cache"
  write_color_output green "    [OK ] Download GSV Done."
else
  write_color_output yellow "    [Ign] Skip gsv downloading."
fi

# Inpaint street views
if [[ ! -f "${cacheroot}/${scene_name}/inpaint_done.txt" ]]; then
  PYTHONPATH=${PWD}/inpainting/Inpaint_Anything/:${PYTHONPATH} python inpainting/remove_street_lsa.py \
  --input_img $pano_cache \
  --coords_type key_in \
  --point_coords 343 382 \
  --point_labels 1 \
  --dilate_kernel_size 15 \
  --output_dir $inpainted_cache \
  --sam_model_type "vit_h" \
  --sam_ckpt ./inpainting/Inpaint_Anything/pretrained_models/sam_vit_h_4b8939.pth \
  --lama_config ./inpainting/Inpaint_Anything/lama/configs/prediction/default.yaml \
  --lama_ckpt ./inpainting/Inpaint_Anything/pretrained_models/big-lama \
  --job_num 1 --job_id 0
  echo "Done" > ${cacheroot}/${scene_name}/inpaint_done.txt
  write_color_output green "    [OK ] Inpaint Street Views Done."
else
  write_color_output yellow "    [Ign] Skip inpainting street views."
fi

# Stage 3C: Project street views
if [[ ! -f "$out1B_merged_file" ]]; then
  "$blender" -b --python ./src/stage3c.py -- \
    --input_blend_path "$out3A_blender_file" \
    --streetview_locs_path "$pano_file" \
    --input_gsv_dir "$inpainted_cache" \
    --solve_result_path "$solve_result_file" \
    --cache_root "$cacheroot"/${scene_name} \
    --blender_save_path "$out1B_merged_file"
  write_color_output green "    [OK ] Projection Done."
else
  write_color_output yellow "    [Ign] Skip gsv projection."
fi

# Inpaint the building and terrains
if [[ ! -f "$out1A_inpaint_file" || ! -f "$out1B_inpaint_file" ]]; then
  write_color_output green "    Emitting Texture Maps..."
  ./inpainting/Emit.sh "$scene_name" "${blender[0]}" "$dataroot"

  if [[ ! -f "${dataroot}/${scene_name}/ground_inpaint_done.txt" ]]; then
    # Inpaint Terrain
    ./inpainting/Inpaint.sh \
      "${dataroot}/${scene_name}/textures_${scene_name}" \
      remove_black_batch.py \
      "${dataroot}/${scene_name}/textures_${scene_name}_ground_inpaint"
    echo "done" > "${dataroot}/${scene_name}/ground_inpaint_done.txt"
    write_color_output green "    [OK ] Ground inpaint Done."
  else
    write_color_output yellow "    [Ign] Skip ground inpaint."
  fi

  if [[ ! -f "${dataroot}/${scene_name}/building_inpaint_done.txt" ]]; then
    # Inpaint Building
    ./inpainting/Inpaint.sh \
      "${dataroot}/${scene_name}/textures_${scene_name}_building" \
      remove_black_batch_building.py \
      "${dataroot}/${scene_name}/textures_${scene_name}_building_inpaint"
    echo "done" > "${dataroot}/${scene_name}/building_inpaint_done.txt"
    write_color_output green "    [OK ] Building inpaint Done."
  else
    write_color_output yellow "    [Ign] Skip building inpaint."
  fi
  echo DEBUG $out1A_inpaint_file $out1B_inpaint_file
  # Rebundle Texture Maps
  "$blender" -b --python ./inpainting/ground_rebundle.py -- \
    --blender_file "$out1A_blender_file" \
    --mesh Terrain \
    --image_dir "${dataroot}/${scene_name}/textures_${scene_name}_ground_inpaint" \
    --save_to "$out1A_inpaint_file"

  "$blender" -b --python ./inpainting/building_rebundle.py -- \
    --blender_file "$out1B_merged_file" \
    --mesh Terrain \
    --image_dir "${dataroot}/${scene_name}/textures_${scene_name}_building_inpaint" \
    --save_to "$out1B_inpaint_file"

#   Cleanup temporary texture files
#  rm -rf "${dataroot}/${scene_name}/textures_${scene_name}" \
#         "${dataroot}/${scene_name}/textures_${scene_name}_ground_inpaint" \
#         "${dataroot}/${scene_name}/textures_${scene_name}_building" \
#         "${dataroot}/${scene_name}/textures_${scene_name}_building_inpaint"
else
  write_color_output yellow "    [Ign] Skip texture inpainting."
fi

## Stage 4: Combine Building and Terrain
if [[ ! -f "$out4_blender_file" && -f "$out1A_inpaint_file" && -f "$out1B_inpaint_file" ]]; then
  "$blender" -b --python ./src/stage4.py -- \
    --terrain_blender "$out1A_inpaint_file" \
    --building_blender "$out1B_inpaint_file" \
    --terrain_name "$terrain_name" \
    --save_to "$out4_blender_file" \
    --glb_to "$glb_file" \
    --roof_blender "$out1B_blender_file" \
    --roof_name "Roof"
  write_color_output green "    [OK ] Building placement done."
elif [[ ! -f "$out1A_inpaint_file" || ! -f "$out1B_inpaint_file" ]]; then
  write_color_output magenta "    [WRN] Early stop pipeline since required file is not received yet."
  exit 1
else
  write_color_output yellow "    [Ign] Skip building placement."
fi

# Stage 5: Upscale the scene
if [[ ! -f "$out5_blender_file" ]]; then
"$blender" -b --python ./src/stage5.py -- \
  --blender_file "$out4_blender_file" \
  --glb_to "$glb_file" \
  --upscayl_exec ${upscayl} \
  --save_to "$out5_blender_file"
  write_color_output green "    [OK ] Up-scaling done."
else
  write_color_output yellow "    [Ign] Skip up-scaling."
fi

# Stage 6: Create building name <=> 3D AABB JSON
if [[ ! -f "$aabb_file" ]]; then
  "$blender" -b --python ./src/stage6.py -- \
    --building_file "$out4_blender_file" \
    --exclude_names "Roof" "Terrain" \
    --save_as "$aabb_file" \
    --circle "$center_lat" "$center_lon" "$radius"
  write_color_output green "    [OK ] Building meta & AABB generation done."
else
  write_color_output yellow "    [Ign] Skip generating metadata & AABB."
fi

# Stage 7: Converting emissive to basics
if [[ ! -f "$aabb_file" ]]; then
  "$blender" -b --python ./stage7.py -- \
    --input_dir ${dataroot}/${scene_name}
  write_color_output green "    [OK ] Converting emissive to basic done."
else
  write_color_output yellow "    [Ign] Skip converting emissive to basic."
fi

for file in roof_basic.glb terrain_basic.glb building_to_osm_tags.json height_field.npz
do
  echo "Updating file ${file}"
  cp data/${scene_name}/${file} ${dst_path}/${scene_name}_ok/
done

echo "Updating file buildings_basic"
cp -r data/${scene_name}/buildings_basic ${dst_path}/${scene_name}_ok/

write_color_output green "Pipeline completed successfully!"


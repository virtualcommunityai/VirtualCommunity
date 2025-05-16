#!/bin/bash
submit="srun --cpus-per-task=16 -N 1 --mem=24G --time 3:59:00 --pty "
BLENDER=your_blender_path
# Set base directory on rclone and local data directory
base_dir="onedrive:/Raw_Scenes"
local_data_dir="data/"

# Create local data directory
mkdir -p $local_data_dir

# Iterate over directories in the rclone base directory
for raw_dir in $(rclone lsd $base_dir | awk '{print $5}'); do
  # Define file paths for stage1B, stage1C, and merged files
  dir="${raw_dir%_ok}"
  stage1b_file="${base_dir}/${raw_dir}/${dir}_stage1B.blend"
  stage1c_file="${base_dir}/${raw_dir}/${dir}_stage1C.blend"
  merged_file="${base_dir}/${raw_dir}/${dir}_stage1_merged.blend"

  # Check if both stage1B and stage1C exist, and merged file does not exist
  if rclone ls $stage1b_file > /dev/null 2>&1 && rclone ls $stage1c_file > /dev/null 2>&1 && ! rclone ls $merged_file > /dev/null 2>&1; then
    echo "Processing $dir: Found both stage1B and stage1C, but merged file not present."

    # Download both stage1B and stage1C files to local data directory
    rclone copy $stage1b_file $local_data_dir
    rclone copy $stage1c_file $local_data_dir

    # Rename downloaded files to scene_stage1B.blend and scene_stage1C.blend
    mv "${local_data_dir}/${dir}_stage1B.blend" "${local_data_dir}/scene_stage1B.blend"
    mv "${local_data_dir}/${dir}_stage1C.blend" "${local_data_dir}/scene_stage1C.blend"

    # Run the merge script on the local data directory
    $submit $BLENDER -b --python merge_texture.py -- --data_dir $local_data_dir

    # Check if the merged file was created successfully
    if [[ -f "${local_data_dir}/scene_stage1_merged.blend" ]]; then
      # Rename the merged file to include the directory name (e.g., DIR_scene_stage1_merged.blend)
      mv "${local_data_dir}/scene_stage1_merged.blend" "${local_data_dir}/${dir}_stage1_merged.blend"

      # Upload the renamed merged file back to the corresponding directory on rclone
      rclone copy "${local_data_dir}/${dir}_stage1_merged.blend" "${base_dir}/${raw_dir}/"
      echo "Uploaded ${dir}_stage1_merged.blend to ${base_dir}/${raw_dir}/"

      # Clean up local data directory
      rm -rf ${local_data_dir}/*
    else
      echo "Merge failed for $dir, scene_stage1_merged.blend not found."
    fi
  else
    echo "Skipping $dir: Conditions not met (either stage1B or stage1C not found, or merged already exists)."
  fi
done

echo "Processing complete."

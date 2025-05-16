#!/bin/bash

# Sub-pipeline for inpainting all images in a folder.

# Check arguments
if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <input_dir> <process> <output_dir>"
    exit 1
fi

# Parse arguments
input_dir=$1
process=$2
output_dir=$3

# Utility function for colored output
write_color_output() {
    local color="$1"
    shift
    case $color in
        green) echo -e "\e[32m$*\e[0m" ;;
        yellow) echo -e "\e[33m$*\e[0m" ;;
        red) echo -e "\e[31m$*\e[0m" ;;
        *) echo "$*" ;;
    esac
}

write_color_output green "Inpainting ${input_dir} with pipeline variant ${process}"

# Run the Python script
PYTHONPATH=${PWD}/inpainting/Inpaint_Anything/:${PYTHONPATH} python ./inpainting/${process} \
    --input_img "${input_dir}" \
    --coords_type key_in \
    --point_coords 343 382 \
    --point_labels 1 \
    --dilate_kernel_size 5 \
    --output_dir "${output_dir}" \
    --sam_model_type vit_h \
    --sam_ckpt ./inpainting/Inpaint_Anything/pretrained_models/sam_vit_h_4b8939.pth \
    --lama_config ./inpainting/Inpaint_Anything/lama/configs/prediction/default.yaml \
    --lama_ckpt ./inpainting/Inpaint_Anything/pretrained_models/big-lama

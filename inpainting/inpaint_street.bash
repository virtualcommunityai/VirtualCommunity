# Check arguments
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <input_dir> <output_dir>"
    exit 1
fi

# Parse arguments
INPUT_DIR=$1
OUTPUT_DIR=$2

python inpainting/remove_street_lsa.py \
  --input_img ${INPUT_DIR} \
  --coords_type key_in \
  --point_coords 343 382 \
  --point_labels 1 \
  --dilate_kernel_size 15 \
  --output_dir ${OUTPUT_DIR} \
  --sam_model_type "vit_h" \
  --sam_ckpt ./Inpaint_Anything/pretrained_models/sam_vit_h_4b8939.pth \
  --lama_config ./Inpaint_Anything/lama/configs/prediction/default.yaml \
  --lama_ckpt ./Inpaint_Anything/pretrained_models/big-lama \
  --job_num 1 --job_id 0

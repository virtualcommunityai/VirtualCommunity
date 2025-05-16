SCENE_NAME=$1
BLENDER=/gpfs/u/home/AICD/AICDzhqn/scratch/lxy/blender-3.6.9-linux-x64/blender
ROOT=/gpfs/u/home/AICD/AICDzhqn/scratch/lxy/Inpaint
PYTHON="/gpfs/u/home/AICD/AICDzhqn/scratch/qh_npl_conda/envs/linpaint/bin/python"

echo "${SCENE_NAME}: Emit"
# $BLENDER -b -P ./ground_emit.py -- --blender_file ./data/Raw_Scenes/${SCENE_NAME}_ok/${SCENE_NAME}_stage1A.blend --mesh Terrain --save_to ./data/Raw_Scenes/${SCENE_NAME}_ok/textures_${SCENE_NAME}

echo "${SCENE_NAME}: Inpainting Step 1"
if [[ "${SCENE_NAME}" == *"_"* ]]; then
    ${PYTHON} Inpaint_Anything/remove_black_batch_2.py     --input_img ${ROOT}/data/Raw_Scenes/${SCENE_NAME}_ok/textures_${SCENE_NAME}     --coords_type key_in     --point_coords 343 382     --point_labels 1     --dilate_kernel_size 5     --output_dir ${ROOT}/results/inpaint_step1/textures_${SCENE_NAME}_mask_ip     --sam_model_type "vit_h"     --sam_ckpt ${ROOT}/Inpaint_Anything/pretrained_models/sam_vit_h_4b8939.pth     --lama_config ${ROOT}/Inpaint_Anything/lama/configs/prediction/default.yaml     --lama_ckpt ${ROOT}/Inpaint_Anything/pretrained_models/big-lama
else
    ${PYTHON} Inpaint_Anything/remove_black_batch.py     --input_img ${ROOT}/data/Raw_Scenes/${SCENE_NAME}_ok/textures_${SCENE_NAME}     --coords_type key_in     --point_coords 343 382     --point_labels 1     --dilate_kernel_size 5     --output_dir ${ROOT}/results/inpaint_step1/textures_${SCENE_NAME}_mask_ip     --sam_model_type "vit_h"     --sam_ckpt ${ROOT}/Inpaint_Anything/pretrained_models/sam_vit_h_4b8939.pth     --lama_config ${ROOT}/Inpaint_Anything/lama/configs/prediction/default.yaml     --lama_ckpt ${ROOT}/Inpaint_Anything/pretrained_models/big-lama
fi

echo "${SCENE_NAME}: Inpainting Step 2"
${PYTHON} Inpaint_Anything/remove_black_batch_lsa.py     --input_img ${ROOT}/results/inpaint_step1/textures_${SCENE_NAME}_mask_ip     --coords_type key_in     --point_coords 343 382     --point_labels 1     --dilate_kernel_size 5     --output_dir ${ROOT}/results/lsa/textures_${SCENE_NAME}_mask_ip_lsa     --sam_model_type "vit_h"     --sam_ckpt ${ROOT}/Inpaint_Anything/pretrained_models/sam_vit_h_4b8939.pth     --lama_config ${ROOT}/Inpaint_Anything/lama/configs/prediction/default.yaml     --lama_ckpt ${ROOT}/Inpaint_Anything/pretrained_models/big-lama

echo "${SCENE_NAME}: Rebundle"
$BLENDER -b -P ./ground_rebundle.py -- --blender_file ./data/Raw_Scenes/${SCENE_NAME}_ok/scene_stage1A.blend --mesh Terrain  --image_dir ./results/lsa/textures_${SCENE_NAME}_mask_ip_lsa --save_to ./data/Raw_Scenes/${SCENE_NAME}_ok/scene_stage1A_inpaint.blend

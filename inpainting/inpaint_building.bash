SCENE_NAME=$1
BLENDER=/gpfs/u/home/AICD/AICDzhqn/scratch/lxy/blender-3.3.16-linux-x64/blender
ROOT=/gpfs/u/home/AICD/AICDzhqn/scratch/lxy/Inpaint
PYTHON="/gpfs/u/home/AICD/AICDzhqn/scratch/qh_npl_conda/envs/linpaint/bin/python"
echo "${SCENE_NAME}_building: Emit"
#$BLENDER -b -P ./building_emit.py -- --blender_file ./data/Raw_Scenes/${SCENE_NAME}_ok/${SCENE_NAME}_stage1B.blend --mesh Terrain --save_to ./data/Raw_Scenes/${SCENE_NAME}_ok/textures_${SCENE_NAME}_building
echo "${SCENE_NAME}_building: Inpainting"
${PYTHON} Inpaint_Anything/remove_black_batch_building.py     --input_img ${ROOT}/data/Raw_Scenes/${SCENE_NAME}_ok/textures_${SCENE_NAME}_building     --coords_type key_in     --point_coords 343 382     --point_labels 1     --dilate_kernel_size 5     --output_dir ${ROOT}/results/render_result/textures_${SCENE_NAME}_building_inpaint     --sam_model_type "vit_h"     --sam_ckpt ${ROOT}/Inpaint_Anything/pretrained_models/sam_vit_h_4b8939.pth     --lama_config ${ROOT}/Inpaint_Anything/lama/configs/prediction/default.yaml     --lama_ckpt ${ROOT}/Inpaint_Anything/pretrained_models/big-lama
echo "${SCENE_NAME}_building: Rebundle"
$BLENDER -b -P ./building_rebundle.py -- --blender_file ./data/Raw_Scenes/${SCENE_NAME}_ok/${SCENE_NAME}_stage1C.blend --mesh Terrain  --image_dir ./results/render_result/textures_${SCENE_NAME}_building_inpaint --save_to ./data/Raw_Scenes/${SCENE_NAME}_ok/${SCENE_NAME}_stage1C_inpaint.blend

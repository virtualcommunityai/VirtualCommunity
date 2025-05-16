import torch
import numpy as np
from PIL import Image, ImageOps, ImageChops
from diffusers import AutoencoderKL, TCDScheduler
from diffusers.models.model_loading_utils import load_state_dict
from huggingface_hub import hf_hub_download

from detect_corrupt import detect_damage_mask
from detect_corrupt_lap import detect_damage_mask_lap
from controlnet_union import ControlNetModel_Union
from pipeline_fill_sd_xl import StableDiffusionXLFillPipeline

# 加载模型
MODELS = {"RealVisXL V5.0 Lightning": "SG161222/RealVisXL_V5.0_Lightning"}

config_file = hf_hub_download("xinsir/controlnet-union-sdxl-1.0", filename="config_promax.json")
config = ControlNetModel_Union.load_config(config_file)
controlnet_model = ControlNetModel_Union.from_config(config)
model_file = hf_hub_download("xinsir/controlnet-union-sdxl-1.0", filename="diffusion_pytorch_model_promax.safetensors")
state_dict = load_state_dict(model_file)
model, _, _, _, _ = ControlNetModel_Union._load_pretrained_model(
    controlnet_model, state_dict, model_file, "xinsir/controlnet-union-sdxl-1.0"
)
model.to(device="cuda", dtype=torch.float16)

vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).to("cuda")

pipe = StableDiffusionXLFillPipeline.from_pretrained(
    "SG161222/RealVisXL_V5.0_Lightning", torch_dtype=torch.float16, vae=vae, controlnet=model, variant="fp16"
).to("cuda")

pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)

prompt = "high quality, ultra-detailed, modern building, no sky, full-frame architecture, no background"
(
    prompt_embeds,
    negative_prompt_embeds,
    pooled_prompt_embeds,
    negative_pooled_prompt_embeds,
) = pipe.encode_prompt(prompt, "cuda", True)

def fill_image(image_path, output_path):
    
    img = Image.open(image_path)
    
    # 获取图像尺寸
    width, height = img.size
    if width * height <= 256 * 256:
        # 直接保存图像
        img.save(output_path)
        return
    source = Image.open(image_path).convert("RGBA")
    
    mask = detect_damage_mask(source)
    mask_lap = detect_damage_mask_lap(source)
    
    if isinstance(mask, np.ndarray):
        mask = Image.fromarray(mask)
    if isinstance(mask_lap, np.ndarray):
       mask_lap = Image.fromarray(mask_lap)
    
    mask.putalpha(ImageOps.invert(mask.getchannel("A")))
    mask_lap.putalpha(ImageOps.invert(mask_lap.getchannel("A")))
    
    alpha_channel = ImageChops.add(mask.split()[3], mask_lap.split()[3])
    # alpha_channel = mask.split()[3]
    
    binary_mask = alpha_channel.point(lambda p: p > 0 and 255)
    cnet_image = source.copy()
    cnet_image.paste(0, (0, 0), binary_mask)
    
    
    for image in pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        image=cnet_image,
    ):
        pass
    
    
    final_image = image.convert("RGBA")
    cnet_image.paste(final_image, (0, 0), binary_mask)
    cnet_image.save(output_path)
    print(f"Processed image saved at: {output_path}")
    
    
    
import os
import sys
from PIL import Image
from tqdm import tqdm

src_dir = sys.argv[1]
dst_dir = sys.argv[2]

# 先遍历目录，收集所有需要处理的图片信息
image_files = []
for root, dirs, files in os.walk(src_dir):
    for file in files:
        if file.endswith("Texture.png"):
            src_path = os.path.join(root, file)
            # 计算相对于 src_dir 的子目录路径
            relative_path = os.path.relpath(root, src_dir)
            dst_subdir = os.path.join(dst_dir, relative_path)
            image_files.append((src_path, dst_subdir, file))

# 使用 tqdm 显示进度条
for src_path, dst_subdir, file in tqdm(image_files, desc="Processing images"):
    os.makedirs(dst_subdir, exist_ok=True)
    dst_path = os.path.join(dst_subdir, file)
    if os.path.exists(dst_path):
        continue
    fill_image(src_path, dst_path)
    print(f"Processed and saved: {dst_path}")
                

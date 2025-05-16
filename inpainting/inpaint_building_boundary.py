import torch
import math
import numpy as np
from PIL import Image, ImageOps, ImageChops
from diffusers import AutoencoderKL, TCDScheduler
from diffusers.models.model_loading_utils import load_state_dict
from huggingface_hub import hf_hub_download

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

prompt = "high quality, ultra-detailed, modern building"
(
    prompt_embeds,
    negative_prompt_embeds,
    pooled_prompt_embeds,
    negative_pooled_prompt_embeds,
) = pipe.encode_prompt(prompt, "cuda", True)


import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import binary_dilation

def imageTo_binary_array(image_path):
    # 读取图像，转换为灰度图
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # 设定黑色的阈值（可以根据需要调整）
    _, binary_image = cv2.threshold(image, 1, 1, cv2.THRESH_BINARY_INV)
    return binary_image

# 读取两个图像并转换为二值数组

def fix(c_array, source):
    # 上下翻转 c_array
    c_array_flipped = cv2.flip(c_array, 0)

    # 计算新的尺寸
    new_size = source.size

    # 调整尺寸
    c_array_resized = cv2.resize(c_array_flipped, new_size, interpolation=cv2.INTER_NEAREST)
    return c_array_resized



def get_mask_bound(image_path, boundary_dir, source):
    img_name = image_path.split('Material_')[1].split('_Image')[0].replace('_', ' ')+'.png'
    print(img_name)

    mask = imageTo_binary_array(f"{boundary_dir}/{img_name}")
    mask = fix(mask, source)
    
    # import pdb; pdb.setTrace()

    rgba_mask = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    rgba_mask[:, :, 0:3] = 0  # 白色区域（R, G, B 全部为 255）
    rgba_mask[:, :, 3] = mask * 255   # 反转后的 Alpha 通道

    return rgba_mask


def fill_image(image_path, output_path, boundary_dir):
    
    img = Image.open(image_path)
    
    # 获取图像尺寸
    width, height = img.size
    
    image_path2 = os.path.join(boundary_dir, image_path.split('Material_')[1].split('_Image')[0].replace('_', ' ')+'.png')
    # import pdb; pdb.set_trace()
    
    if not os.path.exists(image_path2):
        print(f"skip {output_path}")
        # 直接保存图像
        img.save(output_path)
        return
    
    
    
    source = Image.open(image_path).convert("RGBA")
    original_size = source.size  # 记录原始尺寸
    w, h = original_size

    # 如果 source 的面积小于 1024×1024，则按最小整数倍放大
    if w * h < 1024 * 1024:
        scale_factor = math.ceil(math.sqrt((1024 * 1024) / (w * h)))
        new_size = (w * scale_factor, h * scale_factor)
        source = source.resize(new_size, Image.LANCZOS)
        # print(f"图像放大了 {scale_factor} 倍，新尺寸为: {new_size}")
    
    
    mask = get_mask_bound(image_path, boundary_dir, source)
    
    if isinstance(mask, np.ndarray):
        mask = Image.fromarray(mask)
    
    mask.putalpha(ImageOps.invert(mask.getchannel("A")))
    alpha_channel = mask.split()[3]
    
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
    
    # 如果之前放大过图像，则将 final_image 缩小回原始尺寸
    if source.size != original_size:
        cnet_image = cnet_image.resize(original_size, Image.LANCZOS)
        # print(f"将 final_image 缩小回原始尺寸: {original_size}")
    
    cnet_image.save(output_path)
    # print(f"Processed image saved at: {output_path}")
    
    
    
import os
import sys
from PIL import Image
from tqdm import tqdm

boundary_dir = sys.argv[1]

src_dir = sys.argv[2]
dst_dir = sys.argv[3]

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
    fill_image(src_path, dst_path, boundary_dir)
    print(f"Processed and saved: {dst_path}")
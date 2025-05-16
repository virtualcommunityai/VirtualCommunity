import os
import random
import sys
import os
import random
import sys
import time
from typing import Sequence, Mapping, Any, Union
import torch
import numpy as np


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    try:
        from main import load_extra_path_config
    except ImportError:
        print(
            "Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead."
        )
        from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_extra_nodes()


from nodes import NODE_CLASS_MAPPINGS



import_custom_nodes()
with torch.inference_mode():
    upscalemodelloader = NODE_CLASS_MAPPINGS["UpscaleModelLoader"]()
    upscalemodelloader_2 = upscalemodelloader.load_model(
        model_name="RealESRGAN_x2plus.pth"
    )


def fix_image(image_path, output_path, relative_path, file):
    with torch.inference_mode():

        loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
        loadimage_3 = loadimage.load_image(image=os.path.join(relative_path, file))

        imageupscalewithmodel = NODE_CLASS_MAPPINGS["ImageUpscaleWithModel"]()
        saveimage = NODE_CLASS_MAPPINGS["SaveImage"]()

        for q in range(1):
            imageupscalewithmodel_1 = imageupscalewithmodel.upscale(
                upscale_model=get_value_at_index(upscalemodelloader_2, 0),
                image=get_value_at_index(loadimage_3, 0),
            )
            
            
            images = get_value_at_index(imageupscalewithmodel_1, 0)
            
            i = 255. * images[0].cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            img.save(output_path)




import os
from PIL import Image
from tqdm import tqdm

from comfy.cli_args import args
src_dir = args.input_directory
dst_dir = args.output_directory

# 先遍历目录，收集所有需要处理的图片信息
image_files = []
for root, dirs, files in os.walk(src_dir):
    for file in files:
        if file.endswith(".png"):
            src_path = os.path.join(root, file)
            # 计算相对于 src_dir 的子目录路径
            relative_path = os.path.relpath(root, src_dir)
            dst_subdir = os.path.join(dst_dir, relative_path)
            
            image_files.append((src_path, dst_subdir, relative_path, file))

random.seed(time.perf_counter_ns())
random.shuffle(image_files)

# 使用 tqdm 显示进度条
for src_path, dst_subdir, relative_path, file in tqdm(image_files, desc="Processing images"):
    dst_path = os.path.join(dst_subdir, file)
    if os.path.exists(dst_path):
        continue
    os.makedirs(dst_subdir, exist_ok=True)
    fix_image(src_path, dst_path, relative_path, file)
    print(f"Processed and saved: {dst_path}")
                
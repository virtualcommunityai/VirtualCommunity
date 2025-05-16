import numpy._core
import bpy
import os
from pathlib import Path
from tqdm import tqdm
import sys
import shutil

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from blenderlib import MeshObject


def fix_uvmap(blender_file_path: str, save_path: str):

    bpy.ops.wm.open_mainfile(filepath=blender_file_path)

    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            uv_layers = obj.data.uv_layers
            # 检查是否存在 "size" 和 "UVMap" 两个 UV 图层
            if "size" in uv_layers and "UVMap" in uv_layers:
                # 复制 "size" 图层的 UV 数据到 "UVMap" 图层
                # 假设两个图层的 UV 数据数量相同
                size_layer = uv_layers["size"]
                uvmap_layer = uv_layers["UVMap"]
                for i, uv in enumerate(size_layer.data):
                    uvmap_layer.data[i].uv = uv.uv[:]
                
                # 删除 "size" 图层
                uv_layers.remove(size_layer)
                print(f"在对象 {obj.name} 中删除了激活的 UV 映射")
                
                # 将 active_index 设为 0
                uv_layers.active_index = 0
                        
    # 保存修改后的 .blend 文件（覆盖原文件）
    bpy.ops.wm.save_mainfile(filepath=save_path)
    print("已保存修改后的 blend 文件")


if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--blender_file", type=str, required=True, help="Path to the Blender file")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the updated Blender file")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])

    fix_uvmap(args.blender_file, args.save_path)

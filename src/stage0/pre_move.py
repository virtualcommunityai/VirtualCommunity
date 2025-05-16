import pickle
import numpy as np
from pygltflib import GLTF2
import glob
import shutil
import os


def read_glb(file_path):
    gltf = GLTF2().load(file_path)
    return gltf


def modify_vertices(gltf, offset):
    gltf.nodes[0].translation[0] -= offset[0]
    gltf.nodes[0].translation[1] -= offset[1]
    gltf.nodes[0].translation[2] -= offset[2]
    return gltf


def read_vertices(gltf):
    return gltf.nodes[0].translation


def save_glb(gltf, output_path):
    gltf.save(output_path)


def read_coord(input_file_path):
    gltf = read_glb(input_file_path)
    coord = read_vertices(gltf)
    return coord


def process(input_file_path, output_file_path, offset):
    gltf = read_glb(input_file_path)
    gltf = modify_vertices(gltf, offset)
    save_glb(gltf, output_file_path)


def pre_move_tiles(input_dir, output_dir):
    offset = None
    for file in os.listdir(input_dir):
        if file.endswith(".glb"):
            offset = read_coord(os.path.join(input_dir, file))
            break
    if offset is None:
        assert "Offset matching failed!"
    offset = np.array(offset, dtype=np.float64)
    for file in os.listdir(input_dir):
        if file.endswith(".glb"):
            process(os.path.join(input_dir, file), os.path.join(output_dir, file), offset)
    for json_file in glob.glob(os.path.join(input_dir, "*.json")):
        shutil.copy(json_file, output_dir)
    return offset

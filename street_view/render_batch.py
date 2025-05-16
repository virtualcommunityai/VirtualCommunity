import pdb

import bpy
import mathutils
import utm

import os 
import json
import sys 

from dataclasses import dataclass

import argparse
import numpy as np

from tqdm import tqdm

@dataclass
class StreetViewMetaData:
    lat: float
    lon: float
    alt: float
    fov: float
    heading: float
    tilt: float
    roll: float
    width: int
    height: int 

    def get_xyz(self, base_lat: float, base_lon: float):
        self.base_e, self.base_n, _, _ = utm.from_latlon(base_lat, base_lon)
        self.e, self.n, _, _ = utm.from_latlon(self.lat, self.lon)
        x, y = self.e - self.base_e, self.n - self.base_n
        z = self.alt
        return x, y, z
    
    def get_camera_intrinsics(self):
        cx = self.width / 2.0
        fx = cx / np.tan(np.radians(self.fov / 2.0))
        cy = self.height / 2.0
        fy = cy / np.tan(np.radians(self.fov / 2.0))
        return fx, fy, cx, cy
    
    def get_camera_extrinsics(self, base_lat: float, base_lon: float):
        # Convert degrees to radians
        heading = np.radians(-self.heading)
        tilt = np.radians(self.tilt)
        roll = np.radians(-self.roll)
        
        # Compute rotation matrices around each axis
        R_heading = mathutils.Matrix.Rotation(heading, 3, 'Z')
        R_tilt = mathutils.Matrix.Rotation(tilt, 3, 'X')
        R_roll = mathutils.Matrix.Rotation(roll, 3, 'Z')
        
        # Combine the rotation matrices
        R = R_heading @ (R_tilt @ R_roll)
        rotation_euler = R.to_euler('XYZ')

        # Compute the camera location
        location = mathutils.Vector(self.get_xyz(base_lat, base_lon))
        
        return location, rotation_euler


class DepthRenderer:
    @dataclass
    class Config:
        base_lat: float 
        base_lon: float 
        
        input_path: str
        output_dir: str

    def __init__(self, cfg: Config):
        self.cfg = cfg

        bpy.ops.wm.open_mainfile(filepath=cfg.input_path)
        if 'Camera' not in bpy.data.objects:
            self.camera = bpy.data.objects.new('Camera', bpy.data.cameras.new('Camera'))
        else:
            self.camera = bpy.data.objects['Camera']
        self.scene = bpy.context.scene
        self.depth_scene, self.depth_output_node = self.setup_depth_scene()

    def setup_depth_scene(self):
        # Create a new scene for depth rendering
        depth_scene = bpy.data.scenes.new('DepthScene')
        # depth_scene.render.resolution_x = self.scene.render.resolution_x
        # depth_scene.render.resolution_y = self.scene.render.resolution_y
        # depth_scene.render.resolution_percentage = 100

        depth_scene.camera = self.camera
        bpy.context.window.scene = depth_scene

        # Create a new view layer for depth rendering
        depth_scene.view_layers.new('DepthLayer')
        depth_scene.view_layers['DepthLayer'].use_pass_z = True
        
        # Link objects from the original scene
        for obj in self.scene.objects:
            depth_scene.collection.objects.link(obj)
        
        # Create a new depth node tree
        depth_scene.use_nodes = True
        depth_tree = depth_scene.node_tree
        for node in depth_tree.nodes:
            depth_tree.nodes.remove(node)
        
        # Create render layer node
        rl_node = depth_tree.nodes.new('CompositorNodeRLayers')
        rl_node.layer = 'DepthLayer'
        
        # Create depth output node
        depth_output_node = depth_tree.nodes.new('CompositorNodeOutputFile')
        # depth_output_node.base_path = ''
        # depth_output_node.file_slots['Image'].path = output_path
        depth_output_node.format.file_format = 'OPEN_EXR'
        depth_output_node.format.color_depth = '32'
        
        # Link nodes
        depth_tree.links.new(rl_node.outputs['Depth'], depth_output_node.inputs['Image'])
        return depth_scene, depth_output_node

    def render(self, id: str, metadata: StreetViewMetaData):
        # camera extrinsics
        location, rotation = metadata.get_camera_extrinsics(self.cfg.base_lat, self.cfg.base_lon)
        self.camera.location = location
        self.camera.rotation_euler = rotation

        # camera intrinsics
        fx, fy, cx, cy = metadata.get_camera_intrinsics()
        self.camera.data.type = 'PERSP'
        self.camera.data.lens = fx  # Set lens to fx for now
        self.camera.data.sensor_width = metadata.width
        self.camera.data.sensor_height = metadata.height
        self.depth_scene.render.resolution_x = int(cx * 2)
        self.depth_scene.render.resolution_y = int(cy * 2)
        self.depth_scene.render.resolution_percentage = 100

        # depth rendering path
        depth_output_path = os.path.join(self.cfg.output_dir, id)
        self.depth_output_node.base_path = ''
        # print(list(self.depth_output_node.file_slots.keys()))
        self.depth_output_node.file_slots[0].path = depth_output_path

        # render depth map
        bpy.ops.render.render(write_still=True)
        os.replace(depth_output_path + '0001.exr', depth_output_path + '.exr')

def get_args():
    try:
        index = sys.argv.index('--') + 1
    except ValueError:
        index = len(sys.argv)
    argv = sys.argv[index:]

    parser = argparse.ArgumentParser(description='Render RGB and depth images in Blender')
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--base_lat', type=float, default=42.360472)
    parser.add_argument('--base_lon', type=float, default=-71.091528)
    parser.add_argument('--metadata_path', type=str, required=True)
    
    return parser.parse_known_args(argv)[0]

def render_batch(args):
    with open(args.metadata_path) as f:
        metadata = json.load(f)
    
    config = DepthRenderer.Config(
        base_lat=args.base_lat,
        base_lon=args.base_lon,
        input_path=args.input_path,
        output_dir=args.output_dir
    )
    renderer = DepthRenderer(config)
    
    for id, meta in tqdm(metadata.items()):
        if id == 'metadata':
            continue
        metadata = StreetViewMetaData(
            meta['lat'], meta['lon'], meta['alt'], meta['fov'], meta['heading'], meta['tilt'], meta['roll'],
            meta['width'], meta['height']
        )
        renderer.render(id, metadata)

if __name__ == '__main__':
    args = get_args()
    render_batch(args)  
    


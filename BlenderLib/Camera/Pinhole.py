from __future__ import annotations
import typing as T
import numpy as np
import bpy
bpy: T.Any

from ..BaseObject import BlenderObject
from ..Route.UseDevice import configure_cycles_devices


class PinholeCamera(BlenderObject):
    def __init__(self, camera, resolution: tuple[int, int], intrinsic: np.ndarray | None):
        assert camera is not None and camera.type == 'CAMERA'
        super().__init__(camera)
        
        self.__resolution = resolution
        self.__intrinsic  = intrinsic
        if self.__intrinsic is not None:
            self.configure(self.__resolution, self.__intrinsic)

    # Methods for generate / retrieve a mesh from blender #####################
    @classmethod
    def get_withName(cls, name: str, resolution: tuple[int, int], intrinsic: np.ndarray) -> PinholeCamera:
        return cls(bpy.data.objects[name], resolution, intrinsic)

    @classmethod
    def set_withName(cls, name: str, resolution: tuple[int, int], x_fov: float) -> PinholeCamera:
        cam_data = bpy.data.cameras.new(name)
        cam_data.angle_x = np.radians(x_fov)
        cam_obj = bpy.data.objects.new(name, cam_data)
        bpy.context.scene.collection.objects.link(cam_obj)
        return cls(cam_obj, resolution, None)
    
    @property
    def intrinsic(self) -> np.ndarray:
        focal_length_mm = self.data.lens
        sensor_width = self.data.sensor_width
        sensor_height = self.data.sensor_height
        resolution_x, resolution_y = self.__resolution

        f_x = (focal_length_mm / sensor_width) * resolution_x
        f_y = (focal_length_mm / sensor_height) * resolution_y
        c_x = resolution_x / 2
        c_y = resolution_y / 2

        K = np.array([[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]])
        return K


    def configure(self, resolution: tuple[int, int], intrinsic: np.ndarray):
        image_w, image_h  = resolution
        
        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]
        aspect_ratio = image_w / image_h
        
        if (fx / fy) > aspect_ratio:
            sensor_w = 36.0
            sensor_h = sensor_w / aspect_ratio
        else:
            sensor_h = 24.0
            sensor_w = sensor_h * aspect_ratio
        
        focal_length = (fx * sensor_w) / image_w
        lens_shift_x = (cx - (image_w / 2)) * (sensor_w / image_w)
        lens_shift_y = (cy - (image_h / 2)) * (sensor_h / image_h)
        
        self.object.data.sensor_width  = sensor_w
        self.object.data.sensor_height = sensor_h
        self.object.data.lens          = focal_length
        self.object.data.shift_x = lens_shift_x
        self.object.data.shift_y = lens_shift_y
        self.object.data.sensor_fit = "AUTO"

    def pre_render_setup(self):
        scene = bpy.context.scene
        scene.render.resolution_x = self.__resolution[0]
        scene.render.resolution_y = self.__resolution[1]
        scene.render.resolution_percentage = 100
        scene.camera = self.object
        
        # Update clipping range to improve rendering speed
        self.data.clip_start = 0.1
        self.data.clip_end = 30.0 
        
        view_layer = bpy.context.view_layer
        view_layer.use_pass_z = True
    
    @staticmethod
    def set_viewer_node_output():
        # switch on nodes
        bpy.context.scene.use_nodes = True
        tree = bpy.context.scene.node_tree
        links = tree.links
        
        for n in tree.nodes: tree.nodes.remove(n)
        
        rl = tree.nodes.new('CompositorNodeRLayers')
        v  = tree.nodes.new('CompositorNodeViewer')
        v.use_alpha = True
        links.new(rl.outputs[0], v.inputs[0])
    
    @staticmethod
    def set_depth_node_output():
        # switch on nodes
        scene = bpy.context.scene
        scene.render.engine = 'CYCLES'
        configure_cycles_devices(["CUDA", "OPTIX", "CPU"])
        bpy.context.scene.cycles.samples = 1  # Replace 16 with the desired sample number
        bpy.context.scene.cycles.use_adaptive_sampling = True  # Enable adaptive sampling for faster rendering

        scene.use_nodes = True
        tree  = scene.node_tree
        links = tree.links
        
        for n in tree.nodes: tree.nodes.remove(n)
        
        rl = tree.nodes.new('CompositorNodeRLayers')
        v  = tree.nodes.new('CompositorNodeViewer')
        v.use_alpha = True
        links.new(rl.outputs[2], v.inputs[0])
    
    @staticmethod
    def set_basecolor_node_output():
        scene = bpy.context.scene
        scene.render.engine = 'BLENDER_WORKBENCH'
        display = scene.display.shading
        display.light = 'FLAT'
        display.color_type = 'MATERIAL'
    
    def render(self) -> np.ndarray:
        self.pre_render_setup()
        self.set_viewer_node_output()
        
        bpy.ops.render.render()
        rendered_image = bpy.data.images['Viewer Node']

        width, height = rendered_image.size
        pixels = np.array(rendered_image.pixels[:])
        pixels = pixels.reshape((height, width, 4))
        pixels = np.flipud(pixels)
        return pixels

    def render_depth(self) -> np.ndarray:
        self.pre_render_setup()
        self.set_depth_node_output()
        
        bpy.ops.render.render()
        rendered_image = bpy.data.images['Viewer Node']

        width, height = rendered_image.size
        pixels = np.array(rendered_image.pixels[:])
        pixels = pixels.reshape((height, width, 4))
        pixels = np.flipud(pixels)
        return pixels[..., 0]

    def render_basecolor(self) -> np.ndarray:
        self.pre_render_setup()
        self.set_basecolor_node_output()
        
        bpy.ops.render.render()
        rendered_image = bpy.data.images['Viewer Node']

        width, height = rendered_image.size
        pixels = np.array(rendered_image.pixels[:])
        pixels = pixels.reshape((height, width, 4))
        pixels = np.flipud(pixels)
        return pixels

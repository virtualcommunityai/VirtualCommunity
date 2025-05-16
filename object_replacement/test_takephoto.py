import numpy as np
import open3d as o3d
import pyrender
import trimesh
import math
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

def get_mesh_center(input_dir):
    mesh = trimesh.load(input_dir)
    R = mesh.graph[mesh.graph.nodes_geometry[0]][0]
    x, y, z = R[:3, 3]
    return -x, z, -y

def get_mesh_bbox(input_dir):
    mesh = o3d.io.read_triangle_mesh(input_dir)
    aabb = mesh.get_axis_aligned_bounding_box()
    aabb = np.array([aabb.min_bound, aabb.max_bound])
    return aabb

def take_photo(input_dir, output_dir, camera_intrinsic, camera_pose):
    # 加载 glTF（glb）模型
    trimesh_scene = trimesh.load(input_dir, force='scene')
    scene = pyrender.Scene()
    for mesh in trimesh_scene.geometry.values():
        mesh_obj = pyrender.Mesh.from_trimesh(mesh)
        scene.add(mesh_obj)
    # mesh = pyrender.Mesh.from_trimesh(trimesh_scene)
    # scene.add(mesh)

    # 设置相机参数
    # 假设 camera_intrinsic 是一个 3x3 的内参矩阵，camera_pose 是一个 4x4 的外参矩阵

    camera = pyrender.IntrinsicsCamera(fx=camera_intrinsic[0, 0], fy=camera_intrinsic[1, 1], 
                                    cx=camera_intrinsic[0, 2], cy=camera_intrinsic[1, 2])
    camera_node = pyrender.Node(camera=camera, matrix=camera_pose)
    scene.add_node(camera_node)

    # 添加定向光源
    dir_light = pyrender.DirectionalLight(color=np.array([255, 255, 255]), intensity=4.0)
    dir_light_node = pyrender.Node(light=dir_light, matrix=np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 0, 1]
    ]))
    scene.add_node(dir_light_node)

    # 渲染图像
    r = pyrender.OffscreenRenderer(viewport_width=512, viewport_height=384)
    color, depth = r.render(scene)

    # 保存图像
    from PIL import Image
    img = Image.fromarray(color)
    img.save(os.path.join(output_dir,'photoB.png'))

    return color, depth

def uv2world(camera_intrinsic, camera_pose, depth, uv):
    K_inv = np.linalg.inv(camera_intrinsic)
    R = camera_pose[:3, :3]
    t = camera_pose[:3, 3]
    u, v = uv
    z = depth[v, u]  # 假设深度值

    # 转换到归一化相机坐标
    x_cam = z * (K_inv @ np.array([u, v, 1]))
    # 转换到世界坐标
    delta = R @ x_cam[:3]
    delta[1] = -delta[1]
    delta[2] = -delta[2]
    x_world = delta + t

    # print("world:", x_world)
    return x_world

def get_intrinsic(fov=50, pixel_height=384, pixel_width=512):
    fx = (pixel_width/2.0)/math.tan(math.radians(fov/2.0))
    cx = pixel_width/2.0
    fy = (pixel_height/2.0)/math.tan(math.radians(fov/2.0))
    cy = pixel_height/2.0
    camera_intrinsic = np.array([[fx, 0, cx],
                                [0, fy, cy],
                                [0, 0, 1]])
    return camera_intrinsic

def get_pose(mesh_center_x, mesh_center_y, mesh_center_z, offset_x, offset_z):
    camera_pose = np.array([[1, 0, 0, mesh_center_x+offset_x],
                            [0, 1, 0, mesh_center_y],
                            [0, 0, 1, mesh_center_z+offset_z],
                            [0, 0, 0, 1]])
    return  camera_pose

if __name__ == '__main__':
    output_dir = '/project/pi_chuangg_umass_edu/yian/robogen/ljg/Depth-Anything-V2/output/test3'
    input_dir = '/project/pi_chuangg_umass_edu/yian/robogen/ljg/Depth-Anything-V2/output/test3/flat_CMU_180.glb'
    # tx, ty, tz = -195, -57, 78
    # offset_x = -6.927089981231966
    # offset_z = -27.62432915707986

    # get_intrinsic()
    # get_pose(tx, ty, tz, offset_x, offset_z)
    
    # color, depth = take_photo(input_dir, output_dir, camera_intrinsic, camera_pose)

    # bbox_coords = [388, 63, 480, 184]
    # bbox_coords = [0, 76 ,100 ,180]
    # u, v = (bbox_coords[0]+bbox_coords[2]) // 2, (bbox_coords[1]+bbox_coords[3]) // 2
    # uv2world(camera_intrinsic, camera_pose, depth, [u, v])
    get_mesh_center(input_dir)

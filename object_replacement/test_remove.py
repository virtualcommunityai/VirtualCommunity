# import trimesh
# import numpy as np

# def filter_mesh(mesh, min_coords, max_coords):
#     # 计算哪些顶点在指定范围内
#     in_range = (mesh.vertices[:, 0] >= min_coords[0]) & (mesh.vertices[:, 0] <= max_coords[0]) & \
#                (mesh.vertices[:, 1] >= min_coords[1]) & (mesh.vertices[:, 1] <= max_coords[1]) & \
#                (mesh.vertices[:, 2] >= min_coords[2]) & (mesh.vertices[:, 2] <= max_coords[2])

#     # 获取在指定范围外的顶点索引
#     keep_vertices = np.where(~in_range)[0]

#     # 创建一个新的网格，只包含需要保留的顶点和面
#     new_mesh = mesh.submesh([keep_vertices])[0]

#     return new_mesh

# # 加载模型
# scene = trimesh.load('/project/pi_chuangg_umass_edu/yian/robogen/ljg/Depth-Anything-V2/output/test3/flat_CMU_180.glb')

# # 设定要删除的坐标范围
# min_coordinates = [-200, -57, 37]
# max_coordinates = [-197, -50, 40]

# # 如果加载的是一个场景，我们需要遍历其中的每个网格
# if isinstance(scene, trimesh.Scene):
#     for geom in scene.geometry.values():
#         filtered_mesh = filter_mesh(geom, min_coordinates, max_coordinates)
#         geom.vertices = filtered_mesh.vertices
#         geom.faces = filtered_mesh.faces


from mathutils import Vector
import bmesh
import bpy
import os

def transform_glb(filepath):
    # 设置文件路径
    output_file = filepath.replace('glb','blend')
    if os.path.isfile(output_file):
        return output_file
    # 导入GLB文件
    bpy.ops.import_scene.gltf(filepath=filepath)

    # 保存为BLEND文件
    bpy.ops.wm.save_as_mainfile(filepath=output_file)
    return output_file

# 定义.blend文件的路径
blend_filepath = '/project/pi_chuangg_umass_edu/yian/robogen/ljg/Depth-Anything-V2/output/test3/flat_CMU_180.blend'
if blend_filepath.endswith('glb'):
    blend_filepath = transform_glb(blend_filepath)
output_filepath = '/project/pi_chuangg_umass_edu/yian/robogen/ljg/Depth-Anything-V2/output/test3/delete.blend'
if os.path.exists(output_filepath):
    os.remove(output_filepath)

# 打开.blend文件
bpy.ops.wm.open_mainfile(filepath=blend_filepath)

# 切换到编辑模式
bpy.ops.object.mode_set(mode='EDIT')

# 获取活动对象的网格数据
obj = bpy.context.edit_object
me = obj.data

# 获取BMesh表示
bm = bmesh.from_edit_mesh(me)

# 定义坐标范围
min_coord = Vector((-202.6232429242877, -80.0, 35.18612521151145))
max_coord = Vector((-194.6232429242877, -20.0, 42.18612521151145))
#  x>-198.6232429242877&&x<-197.6232429242877&&z>38.18612521151145&&z<39.18612521151145
# 查找并选择在坐标范围内的顶点
min_x,min_y,min_z =999, 999, 999
max_x,max_y,max_z = -999, -999 ,-999
for vert in bm.verts:
    if min_x > vert.co.x:
        min_x = vert.co.x
    if max_x < vert.co.x:
        max_x = vert.co.x
    if min_y > vert.co.y:
        min_y = vert.co.y
    if max_y < vert.co.y:
        max_y = vert.co.y
    if min_z > vert.co.z:
        min_z = vert.co.z
    if max_z < vert.co.z:
        max_z = vert.co.z
    meshlab_coord = (vert.co.x, vert.co.z, -vert.co.y)
    if min_coord.x <= vert.co.x <= max_coord.x and \
       min_coord.z <= -vert.co.y <= max_coord.z:
        vert.select = True
    else:
        vert.select = False
print([min_x, max_x, min_y, max_y, min_z, max_z])
for vert in bm.verts:
    if vert.select == True:
        print("!!!!", vert.co.x, vert.co.y, vert.co.z)

# 删除选中的顶点
bpy.ops.mesh.delete(type='VERT')

# 更新BMesh
bmesh.update_edit_mesh(me)

bpy.ops.object.mode_set(mode='OBJECT')

bpy.ops.wm.save_as_mainfile(filepath=output_filepath)

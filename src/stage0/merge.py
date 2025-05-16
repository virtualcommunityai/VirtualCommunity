import bpy
import bmesh
import os
import sys
import json
import numpy as np
from tqdm import tqdm
import mathutils


def np_matmul_coords(coords, matrix, space=None):
    M = (space @ matrix @ space.inverted()
         if space else matrix).transposed()
    ones = np.ones((coords.shape[0], 1))
    coords4d = np.hstack((coords, ones))
    bounding = np.dot(coords4d, M)[:, :-1]
    new_bound = np.array([[bounding[0][0], bounding[4][0]], # x
                          [bounding[0][1], bounding[2][1]], # y
                          [bounding[0][2], bounding[1][2]],]) # z
    return new_bound


def apply_inverse_transformation(transformation_matrix, blender_obj):
    transformation_matrix_blender = mathutils.Matrix(transformation_matrix)
    inverse_transformation_matrix_blender = transformation_matrix_blender.inverted()

    bpy.ops.object.mode_set(mode='OBJECT')
    blender_mesh = blender_obj.data
    matrix_world_inv = blender_obj.matrix_world.inverted()

    for vertex in blender_mesh.vertices:
        global_co = blender_obj.matrix_world @ vertex.co
        vertex_4d = global_co.to_4d()
        transformed_vertex_4d = inverse_transformation_matrix_blender @ vertex_4d
        local_co = matrix_world_inv @ transformed_vertex_4d
        vertex.co = local_co.to_3d()


def transformation_to_blender(transformation_matrix, blender_obj):
    transformation_matrix_blender = mathutils.Matrix(transformation_matrix)
    bpy.ops.object.mode_set(mode='OBJECT')
    blender_mesh = blender_obj.data
    matrix_world_inv = blender_obj.matrix_world.inverted()
    for vertex in blender_mesh.vertices:
        global_co = blender_obj.matrix_world @ vertex.co
        vertex_4d = global_co.to_4d()
        transformed_vertex_4d = transformation_matrix_blender @ vertex_4d
        local_co = matrix_world_inv @ transformed_vertex_4d
        vertex.co = local_co.to_3d()


def calculate_rotation_matrix(from_vec, to_vec):
    from_vec = from_vec / np.linalg.norm(from_vec)
    to_vec = to_vec / np.linalg.norm(to_vec)
    v = np.cross(from_vec, to_vec)
    c = np.dot(from_vec, to_vec)
    s = np.linalg.norm(v)

    if s == 0:  # from_vec and to_vec are parallel
        return np.eye(3)

    kmat = np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

    rotation_matrix = np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s ** 2))
    return rotation_matrix


def merge_tiles(input_dir, offset):
    # Clear existing objects in the scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    # Check and prepare directories
    input_directory = bpy.path.abspath(input_dir)

    # Load all GLB files from the directory
    files = [f for f in os.listdir(input_directory) if f.endswith('.glb')]
    if not files:
        print("No GLB files found in the specified directory.")
        sys.exit(1)

    objects_before = set(bpy.data.objects.keys())
    json_data = {}
    for filename in files:
        filepath = os.path.join(input_directory, filename)
        bpy.ops.import_scene.gltf(filepath=filepath)
        objects_after = set(bpy.data.objects.keys())
        new_object_name = objects_after - objects_before
        new_object_name = new_object_name.pop()
        obj = bpy.data.objects[new_object_name]
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.remove_doubles(threshold=0.001)
        bmesh.update_edit_mesh(obj.data)
        bpy.ops.object.mode_set(mode='OBJECT')
        obj.select_set(False)
        objects_before = objects_after
        json_data[new_object_name] = json.load(open(os.path.join(input_directory, filename.replace("glb", "json"))))['box']

    json_data_array = [v for v in json_data.values()]
    json_data_array = np.stack(json_data_array)
    center = json_data_array[:, :3].mean(axis=0, dtype=np.float64)
    offset = np.array([offset[0], -offset[2], offset[1]])
    json_data_array[:, :3] -= center
    center -= offset

    translation_matrix = mathutils.Matrix.Translation(mathutils.Vector(-center))
    for blender_obj in bpy.context.scene.objects:
        if blender_obj.type != 'MESH':
            continue
        blender_obj.matrix_world = translation_matrix @ blender_obj.matrix_world
    directions = json_data_array[:, 3:]

    norm_data = np.zeros_like(directions)
    for i in range(directions.shape[0]):
        norm_data[i, 0:3] = directions[i, 0:3] / np.linalg.norm(directions[i, 0:3])
        norm_data[i, 3:6] = directions[i, 3:6] / np.linalg.norm(directions[i, 3:6])
        norm_data[i, 6:9] = directions[i, 6:9] / np.linalg.norm(directions[i, 6:9])

    x_axis_mean = np.mean(norm_data[:, 0:3], axis=0)
    y_axis_mean = np.mean(norm_data[:, 3:6], axis=0)

    rotation_matrix_x = calculate_rotation_matrix(x_axis_mean, np.array([1, 0, 0]))
    y_axis_rotated = rotation_matrix_x @ y_axis_mean

    rotation_matrix_y = calculate_rotation_matrix(y_axis_rotated, np.array([0, 1, 0]))
    rotation_matrix = rotation_matrix_y @ rotation_matrix_x
    for i in range(4):
        json_data_array[:, 3*i:3*(i+1)] = json_data_array[:, 3*i:3*(i+1)] @ rotation_matrix.T

    blender_matrix = mathutils.Matrix((
        (rotation_matrix[0][0], rotation_matrix[0][1], rotation_matrix[0][2], 0),
        (rotation_matrix[1][0], rotation_matrix[1][1], rotation_matrix[1][2], 0),
        (rotation_matrix[2][0], rotation_matrix[2][1], rotation_matrix[2][2], 0),
        (0, 0, 0, 1)
    ))

    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    for obj in mesh_objects:
        transformation_to_blender(blender_matrix, obj)

    y_thres = abs(json_data_array[:, 7]).max() - 1
    z_thres = abs(json_data_array[:, 11]).max() - 1
    basic_blocks = [arr for arr in json_data_array if abs(arr[7]) > y_thres and abs(arr[11]) > z_thres]

    basic_blocks_array = np.array(basic_blocks)

    y_centers = []
    still_alive = np.ones(len(basic_blocks), dtype=bool)
    _arange = np.arange(len(basic_blocks))
    while sum(still_alive) > 0:
        ymax_block = basic_blocks_array[still_alive, 1] > basic_blocks_array[still_alive, 1].max() - 2
        y_centers.append(basic_blocks_array[still_alive][ymax_block, 1].mean())
        still_alive[_arange[still_alive][ymax_block]] = False

    z_centers = []
    still_alive = np.ones(len(basic_blocks), dtype=bool)
    while sum(still_alive) > 0:
        zmax_block = basic_blocks_array[still_alive, 2] > basic_blocks_array[still_alive, 2].max() - 2
        z_centers.append(basic_blocks_array[still_alive][zmax_block, 2].mean())
        still_alive[_arange[still_alive][zmax_block]] = False

    y_centers = np.array([[0] + y_centers, y_centers + [0]])
    y_lines = y_centers.mean(axis=0)[1:-1]
    z_centers = np.array([[0] + z_centers, z_centers + [0]])
    z_lines = z_centers.mean(axis=0)[1:-1]
    # get the global coordinates of all object bounding box corners
    coords = np.stack(
        tuple(np_matmul_coords(np.array(o.bound_box), o.matrix_world.copy())
              for o in mesh_objects)
    )

    x_bound_list = coords[:, 0]
    max_size = (x_bound_list[:, 1] - x_bound_list[:, 0]).max()
    inside_number_list = []
    for lower_bound in x_bound_list[:, 0]:
        inside = (x_bound_list[:, 1] < lower_bound + max_size) & (x_bound_list[:, 0] > lower_bound)
        inside_number_list.append(inside.sum())
    max_value = max(inside_number_list)
    max_index = inside_number_list.index(max_value)
    x_lines = [x_bound_list[:, 0][max_index], x_bound_list[:, 0][max_index] + max_size]

    sequence_idx_dict = {}
    sequence_idx_dict_reverse = {}
    x_max = x_lines[1] * 10 - x_lines[0] * 9
    x_min = x_lines[0] * 10 - x_lines[1] * 9
    x_lines = np.arange(x_min, x_max, x_lines[1] - x_lines[0])
    x_means = (x_lines[:-1] + x_lines[1:]) / 2
    x, y, z = np.meshgrid(x_means, y_centers[0][1:], z_centers[0][1:])
    grid_points = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T
    for key in sequence_idx_dict:
        sequence_idx_dict[key] = sequence_idx_dict[key][0]
    from sklearn.neighbors import KDTree

    tree = KDTree(grid_points)
    dist, indices = tree.query(coords.mean(axis=-1), k=1)
    three_d_indices_z = indices % x.shape[2]
    three_d_indices_y = (indices // x.shape[2]) // x.shape[1]
    three_d_indices_x = (indices // x.shape[2]) % x.shape[1]
    three_d_indices = np.stack([three_d_indices_x, three_d_indices_y, three_d_indices_z]).T[0]
    for idx, key in enumerate(json_data):
        tup = " ".join([str(int(_id)) for _id in three_d_indices[idx][[1, 2, 0]]])
        sequence_idx_dict[tup] = idx
        sequence_idx_dict_reverse[idx] = tup

    bpy.ops.object.select_all(action='DESELECT')
    row_index_to_meshes = [[] for _ in range(len(y_centers[0][1:]))]
    sequence_idx_dict_by_row = [{} for _ in range(len(y_centers[0][1:]))]
    sequence_idx_dict_reverse_by_row = [{} for _ in range(len(y_centers[0][1:]))]
    counter_by_row = [0] * len(y_centers[0][1:])
    for idx, obj in enumerate(bpy.context.scene.objects):
        tup = sequence_idx_dict_reverse[idx]
        row_index = int(tup.split(" ")[0])
        row_index_to_meshes[row_index].append(obj)
        sequence_idx_dict_by_row[row_index][tup] = counter_by_row[row_index]
        sequence_idx_dict_reverse_by_row[row_index][counter_by_row[row_index]] = tup
        counter_by_row[row_index] += 1

    for row_index in tqdm(range(len(y_centers[0][1:])), desc="Processing rows"):
        bpy.ops.object.select_all(action='DESELECT')
        mesh_objects = row_index_to_meshes[row_index]

        verts_nums = [len(obj.data.vertices) for obj in mesh_objects]

        if len(mesh_objects) > 1:
            bpy.ops.object.select_all(action='DESELECT')
            for obj in mesh_objects:
                obj.select_set(True)
            bpy.context.view_layer.objects.active = mesh_objects[0]
            bpy.ops.object.join()
        else:
            mesh_objects[0].select_set(True)

        obj = bpy.context.view_layer.objects.active

        thres = 1.0
        bpy.ops.object.mode_set(mode='EDIT')
        bm = bmesh.from_edit_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        world_matrix = obj.matrix_world
        inv_world_matrix = world_matrix.inverted()

        # filter vertex near the bounding box
        nodes_list = []
        for idx, _ in enumerate(verts_nums):
            nodes = []
            tup = sequence_idx_dict_reverse_by_row[row_index][idx]
            y, z, x = [int(ele) for ele in tup.split(" ")]
            range_low = sum(verts_nums[:idx])
            if idx == len(verts_nums) - 1:
                range_high = len(bm.verts)
            else:
                range_high = sum(verts_nums[:idx + 1])
            if z < len(z_lines):
                for vid in range(range_low, range_high):
                    v = bm.verts[vid]
                    world_coord = world_matrix @ v.co
                    if z_lines[z] - world_coord.z < thres:
                        nodes.append([vid, obj.matrix_world @ v.co])
            nodes_list.append(nodes)

        # build KD tree for filtered vertex
        kd_dict = {}
        vid_dict = {}
        zero_idx = -1
        for idx, nodes in enumerate(nodes_list):
            if len(nodes) == 0:
                zero_idx = idx
            kd_dict[idx] = mathutils.kdtree.KDTree(len(nodes))
            vid_dict[idx] = {}
            for i, tup in enumerate(nodes):
                vid, node = tup
                kd_dict[idx].insert(node, i)
                vid_dict[idx][i] = vid
            kd_dict[idx].balance()

        # find vertex pairs
        for idx, _ in enumerate(verts_nums):
            tup = sequence_idx_dict_reverse_by_row[row_index][idx]
            y, z, x = [int(ele) for ele in tup.split(" ")]
            range_low = sum(verts_nums[:idx])
            if idx == len(verts_nums) - 1:
                range_high = len(bm.verts)
            else:
                range_high = sum(verts_nums[:idx + 1])
            for vid in range(range_low, range_high):
                v = bm.verts[vid]
                world_coord = world_matrix @ v.co
                zdiff = z_lines - world_coord.z
                if abs(zdiff).min() < thres:
                    if abs(zdiff).argmin() == z - 1:
                        neighbor = " ".join([str(y), str(z - 1), str(x)])
                        if neighbor in sequence_idx_dict_by_row[row_index]:
                            neighbor_id = sequence_idx_dict_by_row[row_index][neighbor]
                            if neighbor_id == zero_idx:
                                continue
                            co, index, distance = kd_dict[neighbor_id].find(world_coord)
                            if index is not None and distance < thres:
                                target_vid = vid_dict[neighbor_id][index]
                                target_vertex = bm.verts[target_vid]
                                midpoint = (world_matrix @ v.co + world_matrix @ target_vertex.co) / 2
                                v.co = inv_world_matrix @ midpoint
                                target_vertex.co = inv_world_matrix @ midpoint
        bpy.ops.mesh.remove_doubles(threshold=0.001)
        bmesh.update_edit_mesh(obj.data)
        bpy.ops.object.mode_set(mode='OBJECT')
        obj.name = f"Row_{str(row_index)}"
        obj.select_set(False)
    merged_mesh = bpy.data.objects.get(f"Row_0")
    merged_mesh.name = "Mesh_0"
    for col_index in tqdm(range(1, len(y_centers[0][1:])), desc="Processing columns"):
        merged_mesh = bpy.data.objects.get("Mesh_0")
        mesh_to_merge = bpy.data.objects.get(f"Row_{str(col_index)}")
        merged_mesh_vert_num = len(merged_mesh.data.vertices)
        merged_mesh.select_set(True)
        mesh_to_merge_vert_num = len(mesh_to_merge.data.vertices)
        mesh_to_merge.select_set(True)
        bpy.context.view_layer.objects.active = merged_mesh
        bpy.ops.object.join()
        merged_mesh = bpy.data.objects.get("Mesh_0")

        thres = 1.0
        bpy.ops.object.mode_set(mode='EDIT')
        bm = bmesh.from_edit_mesh(merged_mesh.data)
        bm.verts.ensure_lookup_table()
        world_matrix = merged_mesh.matrix_world
        inv_world_matrix = world_matrix.inverted()

        nodes = []
        y_line = y_lines[col_index-1]
        for vid in range(merged_mesh_vert_num, merged_mesh_vert_num + mesh_to_merge_vert_num):
            v = bm.verts[vid]
            world_coord = world_matrix @ v.co
            ydiff = y_line - world_coord.y
            if abs(ydiff) < thres:
                nodes.append([vid, world_matrix @ v.co])

        kd_dict = {"tgt": mathutils.kdtree.KDTree(mesh_to_merge_vert_num)}
        vid_dict = {"tgt": {}}
        for i, tup in enumerate(nodes):
            vid, node = tup
            kd_dict["tgt"].insert(node, i)
            vid_dict["tgt"][i] = vid
        kd_dict["tgt"].balance()

        for vid in range(merged_mesh_vert_num):
            v = bm.verts[vid]
            world_coord = world_matrix @ v.co
            ydiff = y_line - world_coord.y
            if abs(ydiff) < thres:
                co, index, distance = kd_dict["tgt"].find(world_coord)
                if index is not None:
                    target_vid = vid_dict["tgt"][index]
                    target_vertex = bm.verts[target_vid]
                    midpoint = (world_matrix @ v.co + world_matrix @ target_vertex.co) / 2
                    v.co = inv_world_matrix @ midpoint
                    target_vertex.co = inv_world_matrix @ midpoint
        bpy.ops.mesh.remove_doubles(threshold=0.001)
        bmesh.update_edit_mesh(merged_mesh.data)
        bpy.ops.object.mode_set(mode='OBJECT')
    apply_inverse_transformation(transformation_matrix=blender_matrix, blender_obj=bpy.data.objects[0])
    center += offset
    translation_matrix = mathutils.Matrix.Translation(mathutils.Vector(center))
    bpy.data.objects[0].matrix_world = translation_matrix @ bpy.data.objects[0].matrix_world

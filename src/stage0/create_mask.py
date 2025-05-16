import os
import pdb
import time
import bpy
import bmesh
import pickle
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm
import sys
import argparse
from .utils import export_scene
from shapely.geometry import Point


DISTANCES = [
    # name, width, y_diff
    ('primary', 12.5, 0.7),
    ('residential', 5, 0.7),
    ('tertiary', 5, 0.7),
    ('service', 4, 0.7),
    ('footway', 1, 0.7),
    ('steps', 1, 0.7),
    ('cycleway', 3, 0.7),
    # ('path', 3, 1.5),
]

DISTANCES_VEHICLE = [
    # name, width, y_diff
    ('primary', 10),
    ('residential', 5),
    ('tertiary', 5),
    ('service', 5),
    ('cycleway', 5),
    # ('path', 3, 1.5),
]


def near_vehicle_way(coord, tree):
    # if road_info_dict is not None and use_road_dict:
    # if road_info_dict is not None:
    for type, distance_thres in DISTANCES_VEHICLE:
        if type not in tree:
            continue
        distances, indices = tree[type].query((coord[0], coord[2]), k=1)
        if distances < distance_thres:
            return True
    return False


def get_nearest_y(coord, tree, ref_points, road_info_dict=None, use_road_dict=True, ignore_y_diff=False,
                  focus_road=False):
    y_diff = None
    road = None
    # if road_info_dict is not None and use_road_dict:
    # if road_info_dict is not None:
    if focus_road:
        selected_type = None
        for type, distance_thres, _y_diff in DISTANCES:
            if type not in tree:
                continue
            distances, indices = tree[type].query((coord[0], coord[2]), k=1)
            if distances < distance_thres and (abs(coord[1] - ref_points[type][indices][1]) < _y_diff or ignore_y_diff):
                selected_type = type
                y_diff = _y_diff
                break
        if selected_type is not None:
            distances, indices = tree[selected_type].query((coord[0], coord[2]), k=40)
            indices = indices[indices < len(ref_points[selected_type])]
            nearest_points = ref_points[selected_type][indices]
        else:
            distances, indices = tree['all'].query((coord[0], coord[2]), k=40)
            indices = indices[indices < len(ref_points['all'])]
            nearest_points = ref_points['all'][indices]
        close_points = [point for point in nearest_points if abs(point[1] - coord[1]) <= 2]
        if close_points:
            chosen_point = min(close_points, key=lambda point: np.linalg.norm(point[[0, 2]] -
                                                                              np.array([coord[0], coord[2]])))
        else:
            chosen_point = nearest_points[0]
        road = chosen_point[-1]
        if not use_road_dict:
            y_new = chosen_point[1]
        else:
            A, B, C, D, E, F = road_info_dict[road]["param"]
            y_new = B * coord[0] + D * coord[2] + F
        # y_new = (D * coord[0] - E * coord[2] - F) / B
    else:
        distances, indices = tree['all'].query((coord[0], coord[2]), k=40)
        nearest_points = ref_points['all'][indices]
        close_points = [point for point in nearest_points if abs(point[1] - coord[1]) <= 2]
        if close_points:
            chosen_point = min(close_points, key=lambda point: np.linalg.norm(point[[0, 2]] -
                                                                              np.array([coord[0], coord[2]])))
        else:
            chosen_point = nearest_points[0]
        y_new = chosen_point[1]
        if road_info_dict is not None:
            road = chosen_point[-1]
    return y_new, y_diff, road


def should_modify_y(coord, tree, ref_points, ydiff, road_info_dict, ydiff_force=None, focus_road=False):
    nearest_y, _ydiff, road = get_nearest_y(coord, tree, ref_points, road_info_dict=road_info_dict, use_road_dict=False,
                                            focus_road=focus_road)
    # return abs(nearest_y - coord[1]) <= ydiff, road
    if ydiff_force is not None:
        return coord[1] - nearest_y <= ydiff_force, road
    if focus_road:
        return abs(nearest_y - coord[1]) <= ydiff, road
    return (coord[1] - nearest_y <= 0.5) and (coord[1] - nearest_y >= -1), road


def auto_split_for_groups(obj):
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.context.tool_settings.use_mesh_automerge = True
    bpy.context.tool_settings.use_mesh_automerge_and_split = True

    # Get all vertex groups that start with "road"
    bpy.ops.object.mode_set(mode='EDIT')
    road_vertex_groups = [vg for vg in obj.vertex_groups if vg.name.startswith("road")]
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.mesh.select_all(action='SELECT')
    # Move vertices slightly to trigger auto merge
    bpy.ops.transform.translate(value=(0.0001, 0, 0))
    bpy.ops.transform.translate(value=(-0.0001, 0, 0))

    # Iterate through all "road" vertex groups with a progress bar
    for vertex_group in tqdm(road_vertex_groups, desc="Processing vertex groups"):
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.object.vertex_group_set_active(group=vertex_group.name)
        bpy.ops.object.vertex_group_select()
        bpy.ops.mesh.fill_holes(sides=5)

    # Return to object mode and update the object data
    bpy.ops.object.mode_set(mode='OBJECT')


def find_subgraphs(bm, min_island_size, tree, ref_points, world_matrix, height_thres=4, area_thres=150):
    bpy.ops.mesh.select_mode(type="FACE")
    bpy.ops.mesh.select_all(action='DESELECT')
    bm.select_flush(False)

    islands = []
    islands_new = []
    islands_large = []
    processed_states = np.zeros(len(bm.faces), dtype=bool)
    counter = 0
    for f in tqdm(bm.faces, desc="Find subgraphs"):
        if not processed_states[f.index]:
            f.select = True
            bpy.ops.mesh.select_linked(delimit={'SEAM'})
            subgraph = [f for f in bm.faces if f.select]
            processed_states[[f.index for f in subgraph]] = True
            if len(subgraph) > min_island_size * 10:
                bpy.ops.mesh.select_all(action='DESELECT')
                bm.select_flush(False)
                continue
            subgraph_coords = [[world_matrix @ v.co for v in face.verts] for face in subgraph]
            subgraph_coords = np.array(sum(subgraph_coords, []))
            height = np.max(subgraph_coords[:, 1]) - np.min(subgraph_coords[:, 1])
            area = ((np.max(subgraph_coords[:, 0]) - np.min(subgraph_coords[:, 0])) *
                    (np.max(subgraph_coords[:, 2]) - np.min(subgraph_coords[:, 2])))
            min_y = np.min(subgraph_coords[:, 1])
            y_new, y_diff, road = get_nearest_y(subgraph_coords[0], tree, ref_points)
            if (len(subgraph) < min_island_size or height <= height_thres) and area < area_thres and min_y <= y_new + 1:
                near_way = False
                for coord in subgraph_coords:
                    if near_vehicle_way(coord, tree):
                        near_way = True
                if near_way:
                    islands.extend(subgraph)
                    islands_new.append(subgraph)
            else:
                islands_large.append(subgraph)
            bpy.ops.mesh.select_all(action='DESELECT')
            bm.select_flush(False)

    return islands, islands_new, islands_large

def flat(reference_points, road_info, ground_info):
    ydiff = 0.75
    obj = bpy.context.scene.objects[-1]
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj.data)
    bm.select_mode = {'VERT', 'EDGE', 'FACE'}
    bm.select_flush_mode()
    bpy.ops.mesh.select_all(action='DESELECT')

    types2road = {}
    for road in road_info:
        if road_info[road]['type'] not in types2road:
            types2road[road_info[road]['type']] = [road]
        else:
            types2road[road_info[road]['type']].append(road)
    road_column = reference_points[:, 3]
    points_dict = {}

    for type in types2road:
        points = []
        for road in types2road[type]:
            points.append(reference_points[road_column == road, :])
        points = np.concatenate(points, axis=0)
        points_dict[type] = points
    tree = {type: cKDTree(points_dict[type][:, [0, 2]]) for type in points_dict}
    all_points = np.concatenate([points_dict[key] for key in points_dict], axis=0)
    tree['all'] = cKDTree(all_points[:, [0, 2]])
    reference_points = {type: points_dict[type] for type in points_dict}
    reference_points['all'] = np.concatenate([points_dict[key] for key in points_dict], axis=0)
    world_matrix = obj.matrix_world
    road2vert = [[] for _ in range(len(road_info))]
    ground2vert = [[] for _ in range(len(ground_info))]
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    for vid, v in tqdm(enumerate(bm.verts), desc="Processing road vertices I"):
        world_coord = world_matrix @ v.co
        modify, road = should_modify_y(world_coord, tree, reference_points, ydiff, road_info_dict=road_info,
                                       ydiff_force=0.5)
        if modify:
            v.select = True

    # Process grounds
    point_list = [world_matrix @ v.co for v in bm.verts]
    point_list = [Point(world_coord.x, world_coord.z) for world_coord in point_list]
    for ground_idx, ground in tqdm(ground_info.items(), desc="Processing grounds"):
        polygon = ground['polygon']
        for vid in range(len(bm.verts)):
            if polygon.contains(point_list[vid]):
                ground2vert[ground_idx].append(vid)

    for fid, face in tqdm(enumerate(bm.faces), desc="Processing road vertices II"):
        normal = world_matrix.to_3x3() @ face.normal
        should_select = True
        for v in face.verts:
            if not v.select:
                should_select = False
        if should_select:
            face.select = True
            continue
        if normal[1] > 0.95:
            should_select = True
            for v in face.verts:
                coord = world_matrix @ v.co
                modify, road = should_modify_y(coord, tree, reference_points, ydiff, road_info_dict=road_info,
                                               ydiff_force=0.8)
                if not modify:
                    should_select = False
            if should_select:
                face.select = True
                continue

    for vid, v in tqdm(enumerate(bm.verts), desc="Processing road vertices III"):
        world_coord = world_matrix @ v.co
        modify, road = should_modify_y(world_coord, tree, reference_points, ydiff, road_info_dict=road_info,
                                       focus_road=True)
        if modify:
            if road is not None:
                road2vert[int(road)].append(vid)

    bpy.ops.mesh.select_mode(use_extend=False, use_expand=False, type='FACE')
    bpy.ops.mesh.select_all(action='INVERT')
    material = bpy.data.materials.new(name="BlackMaterial")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    for node in nodes:
        nodes.remove(node)
    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    principled_node = nodes.new(type='ShaderNodeBsdfPrincipled')
    image_node = nodes.new(type='ShaderNodeTexImage')
    width, height = 1024, 1024
    black_image = bpy.data.images.new("BlackImage", width=width, height=height)
    black_image.generated_color = (0.0, 0.0, 0.0, 1.0)
    image_node.image = black_image
    principled_node.inputs['Roughness'].default_value = 1.0
    links.new(image_node.outputs['Color'], principled_node.inputs['Base Color'])
    links.new(principled_node.outputs['BSDF'], output_node.inputs['Surface'])
    if len(obj.material_slots) == 0:
        bpy.ops.object.material_slot_add()
    obj.material_slots[0].material = material
    for face in bm.faces:
        if face.select:
            face.material_index = 0
    bmesh.update_edit_mesh(obj.data)


def create_mask(reference_points, road_type_list, ground_info, output_path):
    flat(reference_points, road_type_list, ground_info)
    export_scene(output_path=output_path)

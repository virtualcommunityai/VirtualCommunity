import pdb
import numpy as np
import pickle
import math
import open3d as o3d
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
import argparse
from tqdm import tqdm
import os


# define Quadratic Surface function to fit y
def surface_fit_with_minimal_y_change(points):
    initial_guess = [0, 0, 0, 0, 0, points[:, 1].mean()]  # init with y + 1 = 0

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    def objective_function(params, x, z, y):
        A, B, C, D, E, F = params
        y_new = B * x + D * z + F
        # y_new = A * x ** 2 + B * x + C * z ** 2 + D * z + E * x * z + F
        y_change = abs(y_new - y) ** 2
        regularization = abs(B) + abs(D)
        return y_change + regularization * 2

    result = least_squares(objective_function, initial_guess, args=(x, z, y))
    A, B, C, D, E, F = result.x
    y_new = B * x + D * z + F
    adjusted_points = np.column_stack((x, y_new, z))

    return adjusted_points, (A, B, C, D, E, F)


def smooth_sampled_points(all_road_data, all_ground_data, ground_polygons, pcd_dir="debug_pcd", output_pcd=False):
    road_info_dict = {}
    ground_info_dict = {}

    output_data = []
    output_all_data = []

    road_idx = 0
    for highway, data in all_road_data:
        if len(data) > 0:
            adjusted_data, road_param = surface_fit_with_minimal_y_change(data)
            road_idx_column = np.full((adjusted_data.shape[0], 1), road_idx)

            adjusted_data = np.hstack((adjusted_data, road_idx_column))
            data = np.hstack((data, road_idx_column))

            output_data.append(adjusted_data)
            output_all_data.append(data)

            road_info_dict[road_idx] = {
                "type": highway,
                "param": road_param
            }
            road_idx += 1


    street_view_loc_clean_smooth = np.concatenate(output_data, axis=0)
    output_all_data = np.concatenate(output_all_data, axis=0)
    street_view_loc_clean_all = []
    tree = cKDTree(output_all_data[:, [0, 2]])
    for datapoint in tqdm(output_all_data):
        if tree.query(datapoint[[0, 2]], k=400):
            y_min = output_all_data[tree.query(datapoint[[0, 2]], k=300)[1]][:, 1].min()
            if datapoint[1] < y_min + 2.0:
                street_view_loc_clean_all.append(datapoint)

    street_view_loc_clean_all = np.array(street_view_loc_clean_all)
    if output_pcd:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(street_view_loc_clean_all[:, :3])
        o3d.io.write_point_cloud(os.path.join(pcd_dir, "street_view_loc_clean.ply"), pcd)

    # process grounds
    ground_idx = 0
    for ground_data, polygon in zip(all_ground_data, ground_polygons):
        if len(ground_data) > 0:
            _, ground_param = surface_fit_with_minimal_y_change(ground_data)
            ground_info_dict[ground_idx] = {
                "polygon": polygon,
                "param": ground_param
            }
            ground_idx += 1

    return road_info_dict, street_view_loc_clean_smooth, street_view_loc_clean_all, ground_info_dict

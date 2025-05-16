import json
import pdb
import pickle
import trimesh
import numpy as np
import os
from tqdm import tqdm


def find_mesh_upper_bound_y(input_mesh, x, y):
    """
    Find the y-axis upper bound of a mesh at a given x, z coordinate.

    Parameters:
    - mesh (trimesh.Trimesh): The mesh to query.
    - x (float): The x coordinate.
    - z (float): The z coordinate.

    Returns:
    - float or None: The upper bound on the y-axis if an upper bound is found, otherwise None.
    """
    ray_origins = np.array([[x, y, input_mesh.bounds[1][2] + 1]])
    ray_directions = np.array([[0, 0, -1]])

    locations, index_ray, index_tri = input_mesh.ray.intersects_location(
        ray_origins=ray_origins,
        ray_directions=ray_directions
    )

    if len(locations) > 0:
        z_upbound = np.max(locations[:, 2])
        return z_upbound, True
    else:
        return 0, False


def get_street_view_lat_lng(meta_data_dict):
    lat_lng_list = []
    for pano_id in meta_data_dict:
        if pano_id == 'meta.json':
            continue
        meta_data = meta_data_dict[pano_id]
        if 'lat' not in meta_data:
            continue
        lat = meta_data['lat']
        lng = meta_data['lng']
        if 'heading' not in meta_data:
            pdb.set_trace()
        heading = meta_data['heading']
        tilt = meta_data['tilt']
        roll = meta_data['roll']
        lat_lng_list.append([lat, lng, pano_id, heading, tilt, roll, pano_id])
    if len(lat_lng_list) == 0:
        print("Can not find any pano id!")
        assert False
    return lat_lng_list


def lat_lng_to_xy_matrix(lat0, lng0):
    R = 6371.0 * 1000

    delta_lat = 1.0
    delta_lat_km = (np.pi / 180) * R * delta_lat

    delta_lng = 1.0
    delta_lng_km = (np.pi / 180) * R * np.cos(np.radians(lat0)) * delta_lng

    conversion_matrix = np.array([[0, -delta_lat_km],
                                  [delta_lng_km, 0]])

    return conversion_matrix


def find_pos(lat_lng_list, original_lat, original_lng, input_mesh, output_path):
    """
    Input
        lat_lng_list: latitude, longitude of street views
        pca_lat_lng_diff: top-2 3d tiles pca vectors in LLA coordinates
        pca_xyz_diff: top-2 3d tiles pca vectors in final xyz coordinates
        original_lat: latitude of 3d tiles center
        original_lng: longitude of 3d tiles center
        mesh: input mesh
        output_streeview_glb_dir: output glb of street view (use small spheres to represent them)
        car_height: height of Google street view car
    """
    transform_matrix = lat_lng_to_xy_matrix(original_lat, original_lng)
    street_view_locs = {}
    for lat_lng in tqdm(lat_lng_list):
        lat, lng, pano_id, heading, tilt, roll, pano = lat_lng
        x_trans, y_trans = np.dot(np.array([lat - original_lat, lng - original_lng]), transform_matrix)
        z_trans, find = find_mesh_upper_bound_y(input_mesh, x_trans, y_trans)
        if find:
            street_view_locs[pano_id] = [x_trans, -z_trans, y_trans]
    pickle.dump(street_view_locs, open(output_path, "wb"))


def align_street_view(input_glb_path, lat, lng, meta_data_dict, output_path):
    street_view_lat_lng_list = get_street_view_lat_lng(meta_data_dict=meta_data_dict)
    mesh = trimesh.load(input_glb_path, force='mesh')
    find_pos(lat_lng_list=street_view_lat_lng_list, original_lat=lat, original_lng=lng, input_mesh=mesh,
             output_path=output_path)

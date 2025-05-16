import os
import pdb
import open3d as o3d
import pickle
import osmnx as ox
from shapely.strtree import STRtree
from shapely.geometry import Point, Polygon
import math
import requests
import trimesh
from tqdm import tqdm
import numpy as np
from scipy.spatial import cKDTree


def fetch_buildings(lat, lng, rad):
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json];
    (
      way["building"](around:{rad},{lat},{lng});
      relation["building"](around:{rad},{lat},{lng});
    );
    out body;
    >;
    out skel qt;
    """
    response = requests.get(overpass_url, params={'data': overpass_query})
    data = response.json()
    return data


# Step 2: Parse the building data and create Shapely polygons
def parse_buildings(data):
    # Step 1: Create a dictionary for nodes
    nodes = {element['id']: (element['lat'], element['lon']) for element in data['elements'] if
             element['type'] == 'node'}
    nodes_1 = {element['id']: element for element in data['elements'] if
             element['type'] != 'node'}
    buildings = []

    # Step 2: Create polygons for ways
    for element in data['elements']:
        if element['type'] == 'way' and 'members' not in element and len(element['nodes']) > 3:
            try:
                coords = [(nodes[node_id][1], nodes[node_id][0]) for node_id in element['nodes']]
                if len(coords) < 4:
                    pdb.set_trace()
                buildings.append(Polygon(coords))
            except KeyError:
                # Handle the case where node_id is not found in nodes
                continue

        # Step 3: Handle relations (multipolygon)
        elif element['type'] == 'relation' or 'members' in element:
            outer_coords = []
            for member in element['members']:
                if member['role'] == 'outer' and member['type'] == 'way':
                    way_id = member['ref']
                    if way_id in [way['id'] for way in data['elements'] if way['type'] == 'way']:
                        way = next(way for way in data['elements'] if way['id'] == way_id)
                        outer_coords.extend([(nodes[node_id][1], nodes[node_id][0]) for node_id in way['nodes']])

            if outer_coords:
                outer_polygon = Polygon(outer_coords)
                buildings.append(outer_polygon)

    return buildings


def are_points_outside_buildings(lat_lng_list, buildings):
    result = []
    for p in lat_lng_list:
        point = Point(p)
        possible_polygons = buildings.query(point)
        if not any(poly.contains(point) for poly in possible_polygons):
            result.append(True)
        else:
            result.append(False)
    return np.array(result)


def is_point_in_building(lat, lng, buildings):
    point = Point(lng, lat)
    for building in buildings:
        if building.contains(point):
            return True
    return False


def reorder_points(points_list, nodes_list):
    nodes_table = {}
    for idx, nodes in enumerate(nodes_list):
        assert len(nodes) > 1
        for node in [nodes[0], nodes[-1]]:
            if node not in nodes_table:
                nodes_table[node] = []
            nodes_table[node].append(idx)

    ordered_points = []
    ordered_nodes = []
    starting_nodes = []
    for k, v in nodes_table.items():
        if len(v) % 2 == 1:
            starting_nodes.append((k, v[0]))
            nodes_table[k] = v[1:]
    current_node = starting_nodes[0][0]
    current_idx = starting_nodes[0][1]
    ordered_points.append(points_list[current_idx])
    ordered_nodes.append(nodes_list[current_idx])
    processed_starting_nodes = [current_node]
    counter = 0
    while len(processed_starting_nodes) < len(starting_nodes):
        counter += 1
        if counter > 1000:
            return sum(points_list, [])
        start_n, end_n = nodes_list[current_idx][0], nodes_list[current_idx][-1]
        current_node = start_n if end_n == current_node else end_n
        if len(nodes_table[current_node]) == 2:
            current_idx = nodes_table[current_node][0] if nodes_table[current_node][1] == current_idx else (
                nodes_table)[current_node][1]
        elif len(nodes_table[current_node]) == 0:
            processed_starting_nodes.append(current_node)
            if len(processed_starting_nodes) < len(starting_nodes):
                for node in starting_nodes:
                    if node[0] not in processed_starting_nodes:
                        current_node = node[0]
                        current_idx = node[1]
                        break
                continue
        else:
            for j, idx in enumerate(nodes_table[current_node]):
                if idx == current_idx:
                    nodes_table[current_node].remove(current_idx)
                    current_idx = nodes_table[current_node][0]
                    nodes_table[current_node].remove(current_idx)
                    break
        ordered_points.append(points_list[current_idx])
        ordered_nodes.append(nodes_list[current_idx])
    ordered_points = sum(ordered_points, [])
    return np.array(ordered_points)


def get_roads(lat, lng, radius):
    overpass_url = "http://overpass-api.de/api/interpreter"

    overpass_query = f"""
    [out:json];
    (
      way["highway"](around:{radius},{lat},{lng});
    );
    out body;
    >;
    out skel qt;
    """

    response = requests.get(overpass_url, params={'data': overpass_query})

    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print(f"Error: {response.status_code}")
        return None


def get_ground_areas(lat, lng, radius):
    location_point = (lat, lng)
    tags = {
        'leisure': ['park', 'pitch', 'playground'],
        'landuse': ['grass', 'meadow', 'recreation_ground'],
        'natural': ['grassland', 'meadow'],
        'amenity': ['parking']
    }
    gdf = ox.geometries_from_point(location_point, tags, dist=radius)
    gdf_polygons = gdf[gdf.geometry.type == 'Polygon']
    return gdf_polygons


def sample_points_on_way(way_nodes, num_points=100):
    lats = [node['lat'] for node in way_nodes]
    lons = [node['lon'] for node in way_nodes]

    lat_samples = np.interp(np.linspace(0, len(lats) - 1, num_points), np.arange(len(lats)), lats)
    lon_samples = np.interp(np.linspace(0, len(lons) - 1, num_points), np.arange(len(lons)), lons)

    return np.column_stack((lat_samples, lon_samples))


def sample_points_on_polygon(polygon, density=0.00001):
    min_x, min_y, max_x, max_y = polygon.bounds
    x_coords = np.arange(min_x, max_x, density)
    y_coords = np.arange(min_y, max_y, density)
    points = np.array([(x, y) for x in x_coords for y in y_coords if polygon.contains(Point(x, y))])
    return points


def latlng_to_xyz(lat, lng, origin_lat, origin_lng, radius=6371000):
    lat_diff = -math.radians(lat - origin_lat)
    lng_diff = math.radians(lng - origin_lng)

    x = lng_diff * radius * math.cos(math.radians(origin_lat))
    z = lat_diff * radius
    return x, z


def polygon_to_xyz(polygon, origin_lat, origin_lng):
    xz_coords = [latlng_to_xyz(lat, lng, origin_lat, origin_lng) for lng, lat in polygon.exterior.coords]
    return Polygon(xz_coords)


def find_mesh_upper_bound_y(mesh, xz_list):
    bounds = mesh.bounds
    filtered_xz_list = np.array([[x, z, bounds[1][2] + 1] for x, z in xz_list if bounds[0][0] < x < bounds[1][0] and
                                  bounds[0][1] < z < bounds[1][1]])
    if len(filtered_xz_list) == 0:
        return None
    ray_directions = np.array([[0, 0, -1]] * len(filtered_xz_list))
    locations, index_ray, index_tri = mesh.ray.intersects_location(filtered_xz_list, ray_directions)
    if len(index_ray) == 0:
        return None
    unique_indices, inverse_indices = np.unique(index_ray, return_inverse=True)
    min_y = np.full(unique_indices.shape, np.inf)
    np.minimum.at(min_y, inverse_indices, locations[:, -1])
    valid_indices = min_y != np.inf
    result_xz = filtered_xz_list[unique_indices[valid_indices]]
    result_y = -min_y[valid_indices]
    return np.column_stack((result_xz[:, 0], result_y, result_xz[:, 1]))


def filter_anomalous_points(points, height_diff=0.5, height_diff_max=0.8, future_points_num=10, future_points_ratio=0.8):
    if len(points) == 0:
        return []
    loc_list = points[:, [0,2]]
    height_list = points[:, 1]
    lowest_point_index = np.argmin(height_list)
    lowest_height = np.min(height_list)
    lowest_loc = loc_list[lowest_point_index]
    left_height = lowest_height
    right_height = lowest_height
    left_loc = lowest_loc
    right_loc = lowest_loc
    points_valid = [False] * len(points)
    points_valid[lowest_point_index] = True
    for i in range(max(len(points) - lowest_point_index, lowest_point_index)):
        left_index = lowest_point_index - i - 1
        if left_index >= 0:
            new_left_height = height_list[left_index]
            new_left_loc = loc_list[left_index]
            # print(abs(new_left_height - left_height) / np.linalg.norm(left_loc - new_left_loc))
            if abs(new_left_height - left_height) / np.linalg.norm(left_loc - new_left_loc) <= height_diff and \
                    abs(new_left_height - left_height) < height_diff_max:
            # if abs(new_left_height - left_height) * 0.3 <= height_diff:
                left_height = new_left_height
                left_loc = new_left_loc
                points_valid[left_index] = True
            else:
                if abs(new_left_height - left_height) <= height_diff_max:
                    checked_future_points = 0
                    valid_future_points = 0
                    j = left_index
                    while checked_future_points < future_points_num and j > 0:
                        j -= 1
                        if abs(new_left_height - height_list[j]) / np.linalg.norm(loc_list[j] - loc_list[j+1]) <= height_diff:
                            valid_future_points += 1
                    if valid_future_points >= future_points_ratio * future_points_num:
                        left_height = new_left_height
                        left_loc = new_left_loc
                        points_valid[left_index] = True
        right_index = lowest_point_index + i + 1
        if right_index < len(points):
            new_right_height = height_list[right_index]
            new_right_loc = loc_list[right_index]
            if abs(new_right_height - right_height) / np.linalg.norm(right_loc - new_right_loc) <= height_diff and \
                    abs(new_right_height - right_height) < height_diff_max:
            # if abs(new_right_height - right_height) * 0.3 <= height_diff:
                right_height = new_right_height
                right_loc = new_right_loc
                points_valid[right_index] = True
            else:
                if abs(new_right_height - right_height) <= height_diff_max:
                    checked_future_points = 0
                    valid_future_points = 0
                    j = right_index
                    while checked_future_points < future_points_num and j < len(points) - 1:
                        j += 1
                        if abs(new_right_height - height_list[j]) / np.linalg.norm(loc_list[j] - loc_list[j-1]) <= height_diff:
                            valid_future_points += 1
                    if valid_future_points >= future_points_ratio * future_points_num:
                        right_height = new_right_height
                        right_loc = new_right_loc
                        points_valid[right_index] = True
    filtered_points = [points[i] for i in range(len(points)) if points_valid[i]]
    return filtered_points


def filter_ground_points(points, distance_threshold=1.0, neighbor_count=20):
    if len(points) == 0:
        return points
    min_y = np.min(points[:, 1])
    tree = cKDTree(points[:, [0, 2]])
    filtered_points = []
    for i, point in enumerate(points):
        indices = tree.query(point[[0, 2]], k=neighbor_count)[1]
        neighbors = points[indices]
        if np.any(neighbors[:, 1] < point[1] - distance_threshold) or point[1] - min_y > 3 * distance_threshold:
            continue
        filtered_points.append(point)
    return np.array(filtered_points)


def align_osm(input_glb_path, lat, lng, rad, pcd_dir, output_pcd=False):
    road_data = get_roads(lat, lng, rad)
    ground_data = get_ground_areas(lat, lng, rad)
    data = fetch_buildings(lat, lng, rad)
    buildings = parse_buildings(data)

    if not road_data and not ground_data:
        print("No data found.")
        return
    else:
        print("Data loaded")

    # Dictionary to store the result
    roads_dict = {}
    ground_polygons = []
    types = set()
    if road_data:
        # Create a dictionary of node ID to node coordinates
        node_dict = {node['id']: {'lat': node['lat'], 'lon': node['lon']} for node in road_data['elements'] if
                     node['type'] == 'node'}

        for element in road_data['elements']:
            types.update({element['type']})
            if element['type'] == 'way' and 'tags' in element:
                name = element['tags'].get('name', f"unnamed road {element['id']}")
                highway = element['tags'].get('highway', 'footway')
                covered = element['tags'].get('covered', False)
                tunnel = element['tags'].get('tunnel', '')
                layer = element['tags'].get('layer', 0)
                covered = covered or tunnel != '' or str(layer).startswith('-')

                # Get the node coordinates for this way
                way_nodes = [{'lat': node_dict[node_id]['lat'], 'lon': node_dict[node_id]['lon']} for node_id in
                             element['nodes'] if node_id in node_dict]

                # Sample points on this way
                sampled_points = sample_points_on_way(way_nodes, num_points=300)
                # Store the sampled points in the dictionary
                if name in roads_dict:
                    roads_dict[name][1].append(sampled_points.tolist())
                    roads_dict[name][3].append(element['nodes'])
                else:
                    roads_dict[name] = [highway, [sampled_points.tolist()], covered, [element['nodes']]]

    for name in tqdm(roads_dict):
        if len(roads_dict[name][1]) > 1:
            roads_dict[name][1] = reorder_points(roads_dict[name][1], roads_dict[name][3])
        else:
            roads_dict[name][1] = roads_dict[name][1][0]
    origin_lat, origin_lng = lat, lng
    all_valid_points = []
    all_points = []
    all_ground_points = []
    tmesh = trimesh.load(input_glb_path, force='mesh')
    buildings_str_tree = STRtree(buildings)

    for road_name, value in tqdm(roads_dict.items()):
        highway = value[0]
        points = value[1]
        covered = value[2]
        sampled_points = []
        sampled_lat_lng = []
        for point in points:
            x, z = latlng_to_xyz(point[0], point[1], origin_lat, origin_lng)
            sampled_points.append((x, z))
            sampled_lat_lng.append((point[0], point[1]))
        sampled_points = np.array(sampled_points)
        results = are_points_outside_buildings(sampled_lat_lng, buildings_str_tree)
        sampled_points = sampled_points[results]
        valid_points = find_mesh_upper_bound_y(tmesh, sampled_points)
        if valid_points is None:
            continue
        filtered_points = filter_anomalous_points(valid_points)
        if highway not in ['corridor', 'via_ferrata', 'steps'] and not covered:
            all_valid_points.append([highway, np.array(filtered_points)])
            all_points.append(valid_points)

    # Process ground data
    for idx, polygon in tqdm(enumerate(ground_data.geometry)):
        ground_points = sample_points_on_polygon(polygon, density=0.00001)
        sampled_points = []
        for point in ground_points:
            x, z = latlng_to_xyz(point[1], point[0], origin_lat, origin_lng)
            sampled_points.append((x, z))
        valid_points = find_mesh_upper_bound_y(tmesh, sampled_points)
        if valid_points is None or len(valid_points) <= 20:
            continue
        filtered_ground_points = filter_ground_points(valid_points)
        all_ground_points.append(filtered_ground_points)
        ground_polygons.append(polygon_to_xyz(polygon, origin_lat, origin_lng))

    if output_pcd:
        # Save all points and valid points
        pcd = o3d.geometry.PointCloud()
        points = [x for x in all_points if len(x) > 0]
        points = np.concatenate(points, axis=0)
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud(os.path.join(pcd_dir, "roadpoint_all.ply"), pcd)
        points = [x[1] for x in all_valid_points if len(x[1]) > 0]
        points = np.concatenate(points, axis=0)
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud(os.path.join(pcd_dir, "roadpoint_valid.ply"), pcd)
        # Save all ground points
        points = [x for x in all_ground_points if len(x) > 0]
        points = np.concatenate(points, axis=0)
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud(os.path.join(pcd_dir, "groundpoint.ply"), pcd)

    return all_valid_points, all_ground_points, ground_polygons

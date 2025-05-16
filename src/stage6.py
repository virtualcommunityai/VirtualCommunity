import os, sys
import json
import bpy
import mathutils
import numpy as np
import math
import requests

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from blenderlib import MeshObject, BoxAABBMesh


OSMAPI: str = 'http://overpass-api.de/api/map?bbox={w},{s},{e},{n}'
EARTH_RADIUS = 6371000  # Earth's radius in meters

def get_bounding_box(center_lat, center_lon, radius):
    # Convert latitude and longitude to radians
    lat = math.radians(center_lat)
    lon = math.radians(center_lon)
    
    # Angular distance in radians on a great circle
    angular_distance = radius / EARTH_RADIUS
    
    # Calculate min and max latitudes
    min_lat = lat - angular_distance
    max_lat = lat + angular_distance
    
    # Calculate min and max longitudes
    delta_lon = math.asin(math.sin(angular_distance) / math.cos(lat))
    min_lon = lon - delta_lon
    max_lon = lon + delta_lon
    
    # Convert back to degrees
    min_lat = math.degrees(min_lat)
    min_lon = math.degrees(min_lon)
    max_lat = math.degrees(max_lat)
    max_lon = math.degrees(max_lon)
    
    print(f"Translate Circle(({center_lat}, {center_lon}), r={radius}) => {min_lat, min_lon, max_lat, max_lon}")
    return min_lat, min_lon, max_lat, max_lon


def get_osm_data(min_lat, min_lon, max_lat, max_lon):
    overpass_url = "https://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json];
    (
      way["building"]({min_lat},{min_lon},{max_lat},{max_lon});
      relation["building"]({min_lat},{min_lon},{max_lat},{max_lon});
    );
    out body;
    >;
    out skel qt;
    """
    response = requests.get(overpass_url, params={'data': overpass_query})
    response.raise_for_status()
    data = response.json()
    
    clean_data = {
        building.get("tags", dict()).get("name", "") : building.get("tags", dict())
        for building in data["elements"]
        if building["type"] != "node"
    }
    
    return clean_data


def main(exclude_names: list[str], save_as: str, circle: tuple[float, float, float]):
    exclude_names = set(exclude_names)
    
    osm_buildings: list[MeshObject] = list(filter(
        lambda obj: obj.name not in exclude_names,
        [MeshObject(obj) for obj in bpy.data.objects
         if obj.type == "MESH"]
    ))
    
    osm = get_osm_data(*get_bounding_box(*circle))
    for building in osm_buildings:
        box = building.mesh_object.bound_box
        T_ow = mathutils.Matrix(building.T_obj2world)
        corners = list(map(
            lambda x: np.array(x).tolist(),
            [T_ow @ mathutils.Vector(corner) for corner in box]
        ))
        try:
            osm[building.name]["custom:bounding_box"] = corners
        except KeyError:
            osm[building.name] = {"custom:bounding_box" : corners}
    
    filtered_data = {k: v for k, v in osm.items() if "custom:bounding_box" in v}
    with open(save_as, 'w') as f:
        json.dump(filtered_data, f, indent=2)
    
    bpy.ops.wm.quit_blender()
    print("Done")


def init(building_file: str, exclude_names: list[str], save_as: str, circle: tuple[float, float, float]):
    bpy.ops.wm.open_mainfile(filepath=building_file)
    main(exclude_names, save_as, circle)


if __name__ == "__main__":
    import argparse, sys
    parser = argparse.ArgumentParser()
    parser.add_argument("--building_file", type=str, required=True, help="Path to the blender file that contains the terrain mesh")
    parser.add_argument("--exclude_names", type=str, required=True, nargs="*", help="Name of terrain mesh in terrain_file")
    parser.add_argument("--save_as"     , type=str, required=True, help="Name of aabb json file to save as")
    parser.add_argument('--circle'      , nargs=3, type=float, help="Circle parameters: center_lat center_lon radius")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])

    init(args.building_file, args.exclude_names, args.save_as, args.circle)

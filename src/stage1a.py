import bpy
import argparse
import sys
import os
import math
import numpy as np

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

def get_args():
    try:
        index = sys.argv.index('--') + 1
    except ValueError:
        index = len(sys.argv)
    argv = sys.argv[index:]
    
    if sys.version_info >= (3, 9):
        def set_boolean_optional_action(parser, dest, label=None, default=None, help=None):
            label = dest.replace('_', '-') if label is None else label
            parser.add_argument(f'--{label}', action=argparse.BooleanOptionalAction, dest=dest, default=default, help=help)
    else:
        def set_boolean_optional_action(parser, dest, label=None, default=None, help=None):
            label = dest.replace('_', '-') if label is None else label
            parser.add_argument(f'--{label}', action='store_true', dest=dest, help=help)
            parser.add_argument(f'--no-{label}', action='store_false', dest=dest, help=help)
            parser.set_defaults(**{dest: default})

    parser = argparse.ArgumentParser(description='Fetch data via BLOSM')
    parser.add_argument(
        '--addon', type=str, dest='addon_name', default='blosm', 
        help='Name of the BLOSM addon to use'
    )
    
    ###################
    # Export settings #
    ###################
    group = parser.add_argument_group(title='Export Setting')
    group.add_argument(
        '--export', '-e', type=str, dest='export_file', 
        help='Path to the file to export'
    )
    set_boolean_optional_action(
        group, dest='keep_non_mesh', default=True,
        help='Keep non-mesh objects in the scene'
    )
    set_boolean_optional_action(
        group, dest='keep_custom_props', default=True,
        help='Keep custom properties of the objects'
    )

    ###################################
    # BLOSM addon preference settings #
    ###################################
    group = parser.add_argument_group(title='BLOSM Preferences')
    group.add_argument(
        '--data-dir', '-d', type=str, dest='data_dir', 
        help='Directory to store downloaded OpenStreetMap and terrain files'
    )
    group.add_argument(
        '--assets-dir', type=str, dest='assets_dir', 
        help='Directory with assets (building_materials.blend, vegetation.blend)'
    )
    group.add_argument(
        '--enable-experimental', action='store_true', dest='enable_experimental',
        help='Enable export to the popular 3D formats. Experimental feature! Use it with caution!'
    )

    subparsers = parser.add_subparsers(title='BLOSM data type', dest='data_type')

    ############################
    # Subparser for OSM import #
    ############################
    osm_parser = subparsers.add_parser('osm', description='Fetch data from OpenStreetMap API')
    osm_parser.add_argument(
        '--source', choices=['server', 'file'], dest='osm_source', default='server',
        help='From where to import OpenStreetMap data: remote server or a file on the local disk'
    )
    osm_parser.add_argument(
        '--file', type=str, dest='osm_filepath',
        help='Path to an OpenStreetMap file for import'
    )
    osm_parser.add_argument(
        '--server', choices=['overpass-api.de', 'vk maps', 'kumi.systems'], dest='osm_server', default='overpass-api.de',
        help='OSM data server'
    )

    group = osm_parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-x', '--extent', nargs=4, type=float,
                        help='Extent bounds: minLat, minLon, maxLat, maxLon')
    group.add_argument('-c', '--circle', nargs=3, type=float,
                        help="Circle parameters: center_lat center_lon radius")

    group = osm_parser.add_argument_group(title='scene settings')
    group.add_argument(
        '--mode', '-m', type=str, choices=['3Drealistic', '3Dsimple', '2D'], dest='mode', default='3Dsimple',
        help='Import data with textures and 3D objects (3D realistic) or without them (3D simple) or 2D only'
    )
    
    set_boolean_optional_action(
        group, dest='import_for_export', default=False,
        help='Import OpenStreetMap buildings ready for export to the popular 3D formats'
    )
    set_boolean_optional_action(
        group, dest='single_object', default=True,
        help='Import OSM objects as a single Blender mesh objects instead of separate ones'
    )
    set_boolean_optional_action(
        group, dest='relative_to_initial_import', default=True,
        help='Import relative to the initial import if it is available'
    )
    set_boolean_optional_action(
        group, dest='buildings', default=True,
        help='Import buildings'
    )
    set_boolean_optional_action(
        group, dest='water', default=True,
        help='Import water objects (rivers and lakes)'
    )
    set_boolean_optional_action(
        group, dest='forests', default=True,
        help='Import forests and woods'
    )
    set_boolean_optional_action(
        group, dest='vegetation', default=True,
        help='Import other vegetation (grass, meadow, scrub)'
    )
    set_boolean_optional_action(
        group, dest='highways', default=True,
        help='Import roads and paths'
    )
    set_boolean_optional_action(
        group, dest='railways', default=False,
        help='Import railways'
    )

    group = osm_parser.add_argument_group(title='buildings settings')
    group.add_argument(
        '--roof-shape', choices=['flat', 'gabled'], dest='roof_shape', default='flat',
        help='Roof shape for a building if the roof shape is not set in OpenStreetMap'
    )
    group.add_argument(
        '--level-height', type=float, dest='level_height', default=3.,
        help='Average height of a level in meters to use for OSM tags building:levels and building:min_level'
    )
    group.add_argument(
        '--straight-angle-threshold', type=float, dest='straight_angle_threshold', default=175.5,
        help='Threshold for an angle of the building outline: when consider it as straight one. '+
            'It may be important for calculation of the longest side of the building outline for a gabled roof.'
    )
    osm_parser.add_argument(
        '--subdivision-size', type=float, dest='subdivision_size', default=10.,
        help='Subdivision size in meters'
    )
    set_boolean_optional_action(
        group, dest='load_missing_members', default=True,
        help='Load missing members of relations'
    )
    set_boolean_optional_action(
        group, dest='subdivide', default=True,
        help='Subdivide curves, flat layers'
    )

    ########################################
    # Subparser for Google 3D Tiles import #
    ########################################
    google_3d_tiles_parser = subparsers.add_parser('google-3d-tiles', description='Fetch data from Google 3D Tiles API')
    google_3d_tiles_parser.add_argument(
        '--key', '-k', type=str, dest='google_maps_key', 
        help='Google 3D Tiles Key'
    )
    
    group = google_3d_tiles_parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-x', '--extent', nargs=4, type=float,
                        help='Extent bounds: minLat, minLon, maxLat, maxLon')
    group.add_argument('-c', '--circle', nargs=3, type=float,
                        help="Circle parameters: center_lat center_lon radius")
    
    google_3d_tiles_parser.add_argument(
        '--lod', type=str, choices=['lod1', 'lod2', 'lod3', 'lod4', 'lod5', 'lod6'], dest='lod', default='lod6',
        help='Level of details (LoD)'
    )
    set_boolean_optional_action(
        google_3d_tiles_parser, dest='join_3d_tiles_objects', default=True,
        help='Join 3D Tiles objects and remove double vertices'
    )
    set_boolean_optional_action(
        google_3d_tiles_parser, dest='cache_json_files', default=False,
        help='Cache JSON Files that define tilesets'
    )
    set_boolean_optional_action(
        google_3d_tiles_parser, dest='cache_3d_files', default=False,
        help='Cache 3D Files (for example in .glb format)'
    )
    google_3d_tiles_parser.add_argument(
        '--replace-materials-with', choices=['export-ready', 'custom'], dest='replace_materials_with', default='export-ready',
        help='Replace materials for the selected objects'
    )
    google_3d_tiles_parser.add_argument(
        '--replacement-material', type=str, dest='replacement_material',
        help='A custom material to replace materials with for the selected objects'
    )

    ################################
    # Subparser for Terrain import #
    ################################
    terrain_parser = subparsers.add_parser('terrain', description='Fetch data of terrain')
    group = terrain_parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-x', '--extent', nargs=4, type=float,
                        help='Extent bounds: minLat, minLon, maxLat, maxLon')
    group.add_argument('-c', '--circle', nargs=3, type=float,
                        help="Circle parameters: center_lat center_lon radius")
    
    args = parser.parse_known_args(argv)[0]
    
    # Convert center-radius form to min-max lat-lon form
    if args.circle: 
        args.extent = get_bounding_box(*args.circle)
    
    return args

def set_blosm_preferences(args):
    addon_prefs = bpy.context.preferences.addons[args.addon_name].preferences
    if args.data_dir:
        addon_prefs.dataDir = os.path.realpath(args.data_dir)
    if args.assets_dir:
        addon_prefs.assetsDir = os.path.realpath(args.assets_dir)
    if args.enable_experimental:
        addon_prefs.enableExperimental = True

def set_osm_import_settings(args):
    addon = bpy.context.scene.blosm
    pref = bpy.context.preferences.addons[args.addon_name].preferences
    addon.dataType = 'osm'
    if args.osm_source == 'file':
        addon.osmSource = 'file'
        addon.osmFilepath = args.osm_filepath
    else:
        addon.osmSource = 'server'
        pref.osmServer = args.osm_server
    addon.mode = args.mode
    addon.minLat, addon.minLon, addon.maxLat, addon.maxLon = args.extent
    addon.importForExport = args.import_for_export
    addon.singleObject = args.single_object
    addon.relativeToInitialImport = args.relative_to_initial_import
    addon.buildings = args.buildings
    addon.water = args.water
    addon.forests = args.forests
    addon.vegetation = args.vegetation
    addon.highways = args.highways
    addon.railways = args.railways
    addon.roofShape = args.roof_shape
    addon.levelHeight = args.level_height
    addon.straightAngleThreshold = args.straight_angle_threshold
    addon.subdivisionSize = args.subdivision_size
    addon.loadMissingMembers = args.load_missing_members
    addon.subdivide = args.subdivide

def set_google_3d_tiles_import_settings(args):
    addon = bpy.context.scene.blosm
    pref = bpy.context.preferences.addons[args.addon_name].preferences
    addon.dataType = 'google-3d-tiles'
    pref.googleMapsApiKey = args.google_maps_key
    addon.minLat, addon.minLon, addon.maxLat, addon.maxLon = args.extent
    addon.lodOf3dTiles = args.lod
    addon.join3dTilesObjects = args.join_3d_tiles_objects
    addon.cacheJsonFiles = args.cache_json_files
    addon.cache3DFiles = args.cache_3d_files
    addon.replaceMaterialsWith = args.replace_materials_with
    if args.replacement_material:
        addon.replacementMaterial = args.replacement_material

def set_terrain_import_settings(args):
    addon = bpy.context.scene.blosm
    pref = bpy.context.preferences.addons[args.addon_name].preferences
    addon.dataType = 'terrain'
    
    addon.minLat, addon.minLon, addon.maxLat, addon.maxLon = args.extent

def import_data(args):
    set_blosm_preferences(args)
    if args.data_type == 'osm':
        set_osm_import_settings(args)
    elif args.data_type == 'google-3d-tiles':
        set_google_3d_tiles_import_settings(args)
    elif args.data_type == 'terrain':
        set_terrain_import_settings(args)
    else:
        raise NotImplementedError(f'Unsupported data type: {args.data_type}')
    bpy.ops.blosm.import_data()
    
def convert_objects_to_mesh():
    for obj in bpy.data.objects:
        if obj.type in {'MESH', 'LIGHT', 'CAMERA', 'EMPTY'}:
            continue
        # preserve custom properties
        custom_props = obj.items()
        # convert to mesh
        try:
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
            bpy.ops.object.convert(target='MESH')
            obj.select_set(False)
        except Exception as _:
            pass
        # restore custom properties
        for key, value in custom_props:
            obj[key] = value

def export_data(args):
    if not args.export_file: return
    path = os.path.realpath(args.export_file)
    extension = os.path.splitext(path)[1]
    if extension == '.blend':
        bpy.ops.wm.save_as_mainfile(filepath=path)
    elif extension == '.glb':
        if args.keep_non_mesh:
            convert_objects_to_mesh() # gltf exporter does not support exporting non-mesh objects
        bpy.ops.export_scene.gltf(filepath=path, export_format='GLB', use_renderable=True, export_extras=args.keep_custom_props)
    elif extension == '.fbx':
        bpy.ops.export_scene.fbx(filepath=path, use_custom_props=args.keep_custom_props)
    elif extension == '.x3d':
        if args.keep_custom_props:
            print('WARNING: X3D format does not support custom properties. Custom properties will be lost!')
        bpy.ops.export_scene.x3d(filepath=path)
    else:
        raise NotImplementedError(f'Unsupported file format: {extension}')

def filter_data():
    # T_Bad_Location - Key:location of OSM element to filter out from our scene
    # Reference: https://wiki.openstreetmap.org/wiki/Key:location
    T_Bad_Location = {
        "underground", "underwater", "overground", "overhead"
    }
    
    # T_Bad_Amenity - Key:amenity of OSM element to filter out from our scene
    # Reference: https://wiki.openstreetmap.org/wiki/Key:amenity
    T_Bad_Amenity = {
        "bicycle_parking", "bicycle_repair_station", "bicycle_rental", "bicycle_wash",
        "boat_rental", "boat_sharing", 
        "bus_station",
        "car_rental", "car_sharing", "car_wash", 
        "compressed_air", "vehicle_inspection", "charging_station", "driver_training", 
        "grit_bin"
    }
    
    # T_Bad_Building - Key:building of OSM element to filter out from our scene
    # Reference: https://wiki.openstreetmap.org/wiki/Key:building
    T_Bad_Building = { "roof" }
    
    # T_Minimum_volume (Unit: m^3)
    # Set minimum volume (h*w*l) for an OSM building to be considered in the scene (this can improve
    # the visual appearance as small buildings generally have terrible texture)
    T_Minimum_volume = 100.
    
    # Filtering Process
    for obj in bpy.data.objects:
        if obj.type != 'MESH':
            bpy.data.objects.remove(obj, do_unlink=True)
            continue
        
        location = None if "location" not in obj else obj["location"]
        amenity  = None if "amenity"  not in obj else obj["amenity"]
        building = None if "building" not in obj else obj["building"]
        
        if (location in T_Bad_Location) or (amenity in T_Bad_Amenity) or (building in T_Bad_Building):
            bpy.data.objects.remove(obj, do_unlink=True)
            continue

        # Estimate the volume of building
        try:
            vertices = np.empty(len(obj.data.vertices) * 3, dtype=np.float32)
            obj.data.vertices.foreach_get("co", vertices)
            vertices = vertices.reshape(-1, 3)
            min_coord = vertices.min(axis=0)
            max_coord = vertices.max(axis=0)
            aabb_size = (max_coord - min_coord)
            volume    = aabb_size[0] * aabb_size[1] * aabb_size[2]
        except:
            volume    = 0.
        finally:
            if volume < T_Minimum_volume:
                bpy.data.objects.remove(obj, do_unlink=True)
                continue

def main():
    args = get_args()
    import_data(args)
    filter_data()
    export_data(args)

if __name__ == '__main__':
    main()
    

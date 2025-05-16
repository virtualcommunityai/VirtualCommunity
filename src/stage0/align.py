import numpy as np
import pyproj
import bpy
from sklearn.decomposition import PCA
import mathutils
from .utils import export_scene


def estimate_normal_vector_via_pca(blender_obj, normal_vector_only=True):
    """
    Estimate the principal normal vector of a Blender mesh object using PCA.

    Parameters:
        blender_obj: The Blender object to analyze (must be a 'MESH').
        normal_vector_only: If True, return only the normal vector; otherwise, return all PCA components.

    Returns:
        numpy.ndarray: The estimated normal vector or all PCA components.
    """
    # Access mesh data in local coordinates
    vertices = np.array([blender_obj.matrix_world @ v.co for v in blender_obj.data.vertices])

    # Perform PCA on the vertices
    pca = PCA(n_components=3)
    pca.fit(vertices)

    # Return the normal vector or all components
    if normal_vector_only:
        normal_vector = pca.components_[-1]  # The last component is the principal normal vector
        return normal_vector
    return pca.components_


def llh_to_ecef(lat, lon, alt):
    """
    Convert latitude, longitude, altitude (LLH or LLA) to ECEF coordinates.

    Parameters:
        lat (float): Latitude in degrees.
        lon (float): Longitude in degrees.
        alt (float): Altitude in meters.

    Returns:
        tuple: ECEF coordinates (x, y, z) in meters.
    """
    transformer = pyproj.Transformer.from_crs(
        {"proj": 'latlong', "ellps": 'WGS84', "datum": 'WGS84'},
        {"proj": 'geocent', "ellps": 'WGS84', "datum": 'WGS84'},
    )

    x, y, z = transformer.transform(lon, lat, alt, radians=False)
    return x, y, z


def ecef_to_llh(x, y, z):
    """
    Convert latitude, longitude, altitude (LLH or LLA) to ECEF coordinates.

    Parameters:
        tuple: ECEF coordinates (x, y, z) in meters.

    Returns:
        lat (float): Latitude in degrees.
        lon (float): Longitude in degrees.
    """
    transformer = pyproj.Transformer.from_crs(
        {"proj": 'geocent', "ellps": 'WGS84', "datum": 'WGS84'},
        {"proj": 'latlong', "ellps": 'WGS84', "datum": 'WGS84'},
    )
    lon1, lat1, alt1 = transformer.transform(x, y, z, radians=False)
    return lat1, lon1, alt1


def rotate_to_enu(input_blender_obj, lat, lng):
    """
    Align an object in the ECEF coordinate system so its Y-axis points up.

    Parameters:
        obj: The Blender object to align.
        lat: Latitude in degrees.
        lng: Longitude in degrees.
    """
    latitude = np.radians(lat)
    longitude = np.radians(lng)

    # Create the rotation matrix from ECEF to ENU
    R = np.array([
        [-np.sin(longitude), np.cos(longitude), 0],
        [np.cos(latitude) * np.cos(longitude), np.cos(latitude) * np.sin(longitude), np.sin(latitude)],
        [np.sin(latitude) * np.cos(longitude), np.sin(latitude) * np.sin(longitude), -np.cos(latitude)]
    ])

    # Create a 4x4 transformation matrix
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = R
    transformation_matrix_blender = mathutils.Matrix(transform_matrix)
    bpy.ops.object.mode_set(mode='OBJECT')
    blender_mesh = input_blender_obj.data
    matrix_world_inv = input_blender_obj.matrix_world.inverted()
    for vertex in blender_mesh.vertices:
        global_co = input_blender_obj.matrix_world @ vertex.co
        vertex_4d = global_co.to_4d()
        transformed_vertex_4d = transformation_matrix_blender @ vertex_4d
        local_co = matrix_world_inv @ transformed_vertex_4d
        vertex.co = local_co.to_3d()


def align_tiles(lat, lng, tile_output_path, blender_name="Mesh_0"):
    PCA_SCALE = 1000
    # translate the mesh to zero
    if blender_name is not None:
        blender_obj = bpy.data.objects[blender_name]
    else:
        blender_obj = None
    # calculate LLA of the mesh
    _, _, alt = ecef_to_llh(blender_obj.location.x, blender_obj.location.y, blender_obj.location.z)
    datapoint_ecef = llh_to_ecef(lat, lng, alt)

    translation_matrix = mathutils.Matrix.Translation(mathutils.Vector((-datapoint_ecef[0],
                                                                        -datapoint_ecef[1],
                                                                        -datapoint_ecef[2])))
    blender_obj.matrix_world = translation_matrix @ blender_obj.matrix_world

    # calculate pca_lat_lng_diff
    initial_coords_homogeneous = estimate_normal_vector_via_pca(blender_obj, normal_vector_only=False)[:2]
    initial_coords_homogeneous = initial_coords_homogeneous * PCA_SCALE + np.expand_dims(datapoint_ecef, 0)
    lat_vec1, lng_vec1, _ = ecef_to_llh(initial_coords_homogeneous[0][0],
                                        initial_coords_homogeneous[0][1],
                                        initial_coords_homogeneous[0][2])
    lat_vec2, lng_vec2, _ = ecef_to_llh(initial_coords_homogeneous[1][0],
                                        -initial_coords_homogeneous[1][1],
                                        initial_coords_homogeneous[1][2])
    pca_lat_lng_diff = np.array([[lat_vec1 - lat, lng_vec1 - lng],
                                 [lat_vec2 - lat, lng_vec2 - lng]])
    rotate_to_enu(blender_obj, lat, lng)
    # Recalculate pca_xyz_diff accurately after all transformations
    pca_xyz_diff = estimate_normal_vector_via_pca(blender_obj, normal_vector_only=False) * PCA_SCALE
    export_scene(output_path=tile_output_path)

    return pca_lat_lng_diff, pca_xyz_diff[:2]

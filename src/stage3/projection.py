import pickle
import os
import math
import cv2
import numpy as np
import bpy
import bmesh
from mathutils import Vector, Matrix
from tqdm import tqdm
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation


def uv_unwrap_with_area_weights(
        obj_name,
        y_offset=5.0,
        high_res_factor=2.0,
        low_res_factor=1.0
    ):
    """
    Perform UV unwrapping for a specified object, allocating larger UV space to faces
    with Y-coordinates below a threshold, and smaller UV space to other faces, followed by packing.
    """
    # 1) Retrieve object and mesh
    obj = bpy.data.objects.get(obj_name)
    if not obj or obj.type != 'MESH':
        print(f"Object '{obj_name}' does not exist or is not a mesh.")
        return
    mesh = obj.data

    ymin = min(v.co.y for v in mesh.vertices)
    y_threshold = ymin + y_offset

    # 2) Ensure UV map exists
    if not mesh.uv_layers.active:
        mesh.uv_layers.new(name="UVMap")

    # 3) Enter edit mode for initial unwrap & pack (select all)
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')

    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=0.001)
    bpy.ops.uv.average_islands_scale()
    bpy.ops.uv.pack_islands(margin=0.001)

    # Return to object mode and refresh
    bpy.ops.object.mode_set(mode='OBJECT')
    mesh.update()

    # Re-acquire UV layer reference to avoid index invalidation
    uv_layer = mesh.uv_layers.active
    uv_data = uv_layer.data

    # 4) Iterate through faces and scale UVs based on Y-coordinates
    for poly in mesh.polygons:
        avg_y = sum(mesh.vertices[v].co.y for v in poly.vertices) / len(poly.vertices)
        scale_factor = high_res_factor if avg_y < y_threshold else low_res_factor

        # Calculate UV centroid of the face
        uv_centroid = Vector((0.0, 0.0))
        for loop_idx in poly.loop_indices:
            uv_centroid += uv_data[loop_idx].uv
        uv_centroid /= len(poly.loop_indices)

        # Scale UVs relative to the centroid
        for loop_idx in poly.loop_indices:
            uv = uv_data[loop_idx].uv
            uv_data[loop_idx].uv = uv_centroid + (uv - uv_centroid) * scale_factor

    # 5) Re-enter edit mode and pack
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.uv.select_all(action='SELECT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.pack_islands(margin=0.001)

    # Return to object mode
    bpy.ops.object.mode_set(mode='OBJECT')
    mesh.update()


def warp_polygon_array(image_array, src_points, dst_points, texture_array, texture_shape):
    """
    Map the polygon region in the input image from source vertices to target vertices,
    and mask out the region outside the target polygon.

    Parameters:
        image_array (np.array): Input image data with shape (H, W, C).
        src_points (np.array): Source polygon vertices with shape (4, 2).
        dst_points (np.array): Target polygon vertices with shape (4, 2).

    Returns:
        np.array: Transformed image data with the region outside the polygon masked.
    """
    # Check input types
    if not isinstance(image_array, np.ndarray):
        raise ValueError("image_array must be a numpy array.")
    if not isinstance(src_points, np.ndarray) or not isinstance(dst_points, np.ndarray):
        raise ValueError("src_points and dst_points must be numpy arrays.")
    if src_points.shape != (4, 2) or dst_points.shape != (4, 2):
        raise ValueError("src_points and dst_points must have shape (4, 2).")

    width, height = texture_shape

    # Compute perspective transform matrix
    matrix = cv2.getPerspectiveTransform(src_points.astype(np.float32), dst_points.astype(np.float32))

    # Apply perspective transform
    warped_image = cv2.warpPerspective(image_array, matrix, (width, height))

    # Create a black mask and draw the polygon
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [dst_points.astype(np.int32)], 255)  # Fill the interior of the polygon with white (255)

    # Apply mask to the image
    warped_image = np.concatenate((warped_image, texture_array[:, :, 3:]), axis=-1)
    mask[(warped_image == 0).sum(axis=-1)==3] = 0
    texture_array[mask != 0] = warped_image[mask != 0]

    return texture_array, mask != 0


def find_opposing_faces(mesh_name, face_index_list):
    """
    Find all faces of the mesh that are facing the camera direction
    (angle < 60 degrees with the normal vector).
    """
    obj = bpy.data.objects.get(mesh_name)
    if obj is None:
        raise ValueError(f"Mesh '{mesh_name}' not found in the scene.")
    if obj.type != 'MESH':
        raise TypeError(f"Object '{mesh_name}' is not a mesh.")
    faces = []
    for idx, face in enumerate(obj.data.polygons):
        if face.index in face_index_list:
            vertices = [obj.matrix_world @ obj.data.vertices[v_idx].co for v_idx in face.vertices]
            faces.append((vertices, face.index))
    return faces


def compute_focal_length(image_width, fov_degrees):
    """
    Compute the focal length in pixels based on image width and field of view.

    Args:
        image_width (int): The width of the image in pixels.
        fov_degrees (float): Field of view in degrees.

    Returns:
        float: Focal length in pixels.
    """
    fov_radians = math.radians(fov_degrees)
    return (image_width / 2) / math.tan(fov_radians / 2)


def project_to_camera(vertices, camera_position, camera_direction, fov, image_width, image_height):
    """
    Project 3D vertices to 2D image coordinates.
    """
    camera_dir = Vector(camera_direction).normalized()
    up_vector = Vector((0, 1, 0))

    # Define camera coordinate system
    right_vector = camera_dir.cross(up_vector).normalized()
    up_vector = right_vector.cross(camera_dir).normalized()

    # Compute focal length dynamically
    focal_length = compute_focal_length(image_width, fov)

    projected_points = []
    for vertex in vertices:
        # Transform vertex to camera space
        to_vertex = Vector(vertex) - Vector(camera_position)
        x_camera = to_vertex.dot(right_vector)
        y_camera = to_vertex.dot(up_vector)
        z_camera = to_vertex.dot(camera_dir)

        if z_camera <= 0:  # Behind the camera
            continue

        # Perspective projection
        u = focal_length * (x_camera / z_camera) + image_width / 2
        v = focal_length * (y_camera / z_camera) + image_height / 2
        projected_points.append((int(u), int(v)))

    return projected_points


def find_uv_positions(obj, mesh_name, face_idx_list):
    mesh = obj.data
    if not mesh.uv_layers.active:
        raise ValueError(f"Mesh '{mesh_name}' does not have an active UV layer.")
    uv_layer = mesh.uv_layers.active.data
    texture_image = None
    if obj.material_slots:  # Ensure materials exist
        for mat_slot in obj.material_slots:
            material = mat_slot.material
            if material and material.use_nodes:  # Ensure the material uses nodes
                for node in material.node_tree.nodes:
                    if node.type == 'TEX_IMAGE':  # Find the image texture node
                        texture_image = node.image
                        break
            if texture_image:
                break

    if not texture_image:
        raise ValueError(f"No texture image found for the mesh '{mesh_name}'.")
    texture_width, texture_height = texture_image.size
    uv_pixel_positions_list = []
    for face in obj.data.polygons:
        if face.index in face_idx_list:
            uv_pixel_positions = []
            for loop_index in face.loop_indices:
                uv_coord = uv_layer[loop_index].uv  # Get UV coordinates
                u, v = uv_coord.x, uv_coord.y

                pixel_x = int(u * texture_height)
                pixel_y = int(v * texture_height)  # Flip v for correct orientation

                uv_pixel_positions.append({
                    "uv_coord": (u, v),
                    "pixel_coord": (pixel_x, pixel_y)
                })
            uv_pixel_positions_list.append(uv_pixel_positions)
    return uv_pixel_positions_list


def project_image_on_faces(obj, texture_image, image_path, faces_2d, mesh_name, texture_array):
    """
    Draw polygons corresponding to the 2D faces on the image and save.
    """
    image = cv2.imread(image_path)
    texture_width, texture_height = texture_image.size
    if image is None:
        raise ValueError(f"Image not found: {image_path}")
    mesh = obj.data
    if not mesh.uv_layers.active:
        raise ValueError(f"Mesh '{mesh_name}' does not have an active UV layer.")
    bm_ori = bmesh.new()
    bm_ori.from_mesh(obj.data)  # Use original object's mesh
    image = cv2.flip(image, 0)
    face_idx_list = [idx for face, idx in faces_2d]
    uv_positions = find_uv_positions(obj, mesh_name, face_idx_list)
    cam_mask = np.zeros((texture_array.shape[0], texture_array.shape[1]), dtype=bool)

    for (face, idx), uv_dict in zip(faces_2d, uv_positions):
        uv_dict = [uv['pixel_coord'] for uv in uv_dict]
        pts = np.array(face, np.int32)
        pts = pts.reshape((-1, 1, 2))
        src_points = pts[:,0,:]
        if src_points.shape[0] != 4:
            continue
        dst_points = np.array(uv_dict, dtype=np.float32)  # Target vertices

        if not texture_image:
            raise ValueError(f"No texture image found for the mesh '{mesh_name}'.")
        texture_array, mask = warp_polygon_array(image, src_points, dst_points, texture_array,
                                                 (texture_width, texture_height))
        cam_mask |= mask
    return texture_array, cam_mask

def rotate_around_y(vec, angle_deg):
    angle_rad = math.radians(-angle_deg)
    rot_mat = Matrix.Rotation(angle_rad, 4, 'Y')
    return rot_mat @ vec


def align_textures(new_textures, original_textures):
    """
    Align the brightness and contrast of new_textures to match original_textures.

    Parameters:
        new_textures (numpy.ndarray): Input array of shape (X, 4) to be aligned.
        original_textures (numpy.ndarray): Reference array of shape (Y, 4).

    Returns:
        numpy.ndarray: Aligned array of shape (X, 4).
    """
    # Separate RGB channels (first three dimensions) and keep the fourth unchanged
    new_rgb = new_textures[:, :3]
    original_rgb = original_textures[:, :3]
    new_alpha = new_textures[:, 3:]  # Keep the fourth dimension as it is

    # Calculate brightness (mean) and contrast (std) for both RGB sets
    mean_new, std_new = new_rgb.mean(axis=0), new_rgb.std(axis=0)
    mean_original, std_original = original_rgb.mean(axis=0), original_rgb.std(axis=0)

    # Normalize new_textures' RGB to match the original textures' RGB
    aligned_rgb = (new_rgb - mean_new) / (std_new + 1e-8)  # Normalize
    aligned_rgb = aligned_rgb * std_original + mean_original  # Scale and shift

    # Clip values to valid range [0, 255]
    aligned_rgb = np.clip(aligned_rgb, 0, 255)

    # Combine aligned RGB with the unchanged fourth dimension
    aligned_textures = np.hstack((aligned_rgb, new_alpha))

    return aligned_textures


def cluster_face_by_normal(obj, angle_threshold=45):
    normal_list = []
    for idx, poly in enumerate(obj.data.polygons):
        normal_local = poly.normal
        normal_world = obj.matrix_world.to_3x3() @ normal_local
        normal_world.normalize()
        normal_list.append((np.array([normal_world.x, normal_world.y, normal_world.z]), idx))

    clusters = []
    angle_threshold_rad = np.radians(angle_threshold)
    for vec, idx in normal_list:
        added_to_cluster = False
        for cluster in clusters:
            if np.arccos(np.clip(np.dot(vec, cluster[0][0]), -1.0, 1.0)) < angle_threshold_rad:
                cluster.append((vec, idx))
                added_to_cluster = True
                break
        if not added_to_cluster:
            clusters.append([(vec, idx)])
    return [[idx for vec, idx in cluster] for cluster in clusters]


def create_face_mask(obj, face_list, texture_width, texture_height):
    """
    Create a mask for a texture image where specific faces of an object are marked as True.

    Parameters:
    - obj: bpy.types.Object, the target object.
    - face_list: list, a list of face indices to include in the mask.
    - texture_width: int, the width of the texture.
    - texture_height: int, the height of the texture.

    Returns:
    - mask: np.ndarray, a boolean mask array where True indicates the face is included.
    """
    mesh = obj.data

    # Ensure the object has an active UV map
    if not mesh.uv_layers.active:
        raise ValueError("The object has no active UV map.")

    uv_layer = mesh.uv_layers.active.data  # Access UV data
    mask = np.zeros((texture_height, texture_width), dtype=bool)  # Initialize the mask array as all False

    # Iterate through the target faces and mark their corresponding areas in the mask
    for poly in mesh.polygons:
        if poly.index not in face_list:
            continue  # Skip faces not in the list

        # Get the UV coordinates of the face
        uv_coords = [uv_layer[loop_idx].uv for loop_idx in poly.loop_indices]
        uv_coords = [(int(uv.x * texture_width), int(uv.y * texture_height)) for uv in uv_coords]

        # Draw the face area on the mask
        mask_image = Image.fromarray(mask.astype(np.uint8))  # Convert to PIL image for drawing
        draw = ImageDraw.Draw(mask_image)
        draw.polygon(uv_coords, fill=1)  # Fill the face area with True (value = 1)
        mask = np.array(mask_image, dtype=bool)  # Convert back to boolean array

    return mask


from scipy.ndimage import label
def fill_minimum_rectangle(mask):

    mask_dilation = binary_dilation(mask, iterations=3)
    
    # 假设已有二值数组 mask 和其 binary_dilation 结果 mask_dilation

    # 对 mask_dilation 进行连通区域标记
    labeled, num_features = label(mask_dilation)

    # 新建一个与 mask 同样形状的数组，用于存放矩形结果
    mask_rectangles = np.zeros_like(mask)

    # 针对每个连通区域，计算最小矩形并填充1
    for i in range(1, num_features + 1):
        # 获取第 i 个连通区域的所有坐标
        rows, cols = np.where((labeled == i) & (mask == 1))
        if rows.size == 0 or cols.size == 0:
            continue
        # 得到最小和最大的行列索引
        rmin, rmax = rows.min(), rows.max()
        cmin, cmax = cols.min(), cols.max()
        # 使用最小矩形覆盖该区域，将区域内全部设为1
        mask_rectangles[rmin:rmax+1, cmin:cmax+1] = 1

    # 此时 mask_rectangles 即为每个连通区域被最小矩形覆盖后的结果
    return mask_rectangles


def projection_street_view_mesh(obj, texture_image, mesh_name, solve_result, streetview_locs, y_offset, base_forward, input_gsv_dir, fov, image_width, image_height):
    texture_width, texture_height = texture_image.size
    texture_array = np.array(texture_image.pixels).reshape(texture_width, texture_height, 4)
    texture_array[:, :, :3] *= 255
    texture_array = texture_array[:, :, [2, 1, 0, 3]]
    mesh_mask = np.zeros((texture_width, texture_height), dtype=bool)
    for camera_id in solve_result[mesh_name]:
        pano_name = "_".join(camera_id.split("_")[:-1])
        pano_id = int(camera_id.split("_")[-1])
        camera_position = streetview_locs[pano_name].copy()
        camera_position[1] += y_offset
        camera_direction = rotate_around_y(base_forward, int(camera_id.split("_")[-1]))
        pano_img_path = os.path.join(input_gsv_dir, pano_name, f'heading_{pano_id}.jpg')
        face_id_list = solve_result[mesh_name][camera_id]
        target_faces = find_opposing_faces(mesh_name, face_id_list)
        faces_2d = [
            (project_to_camera(face, camera_position, camera_direction, fov, image_width, image_height), idx)
            for face, idx in target_faces]
        # import pdb; pdb.set_trace()
        texture_array, mask = project_image_on_faces(obj, texture_image, pano_img_path, faces_2d, mesh_name,
                                                        texture_array)
        mesh_mask |= mask

    face_clusters = cluster_face_by_normal(obj, angle_threshold=30)
    # import pdb; pdb.set_trace()
    # os.makedirs(boundary_dir, exist_ok=True)
    boundary_mask = None
    boundary_mask_rect = None
    for cluster_id, cluster in enumerate(face_clusters):
        cluster_mask = create_face_mask(obj, cluster, texture_width, texture_height)
        new_textures = texture_array[mesh_mask & cluster_mask]
        original_textures = texture_array[(~mesh_mask) & cluster_mask]
        original_textures = original_textures[original_textures[:, :3].sum(axis=-1) > 0]
        if boundary_mask is None:
            boundary_mask = mesh_mask & cluster_mask
            boundary_mask_rect = fill_minimum_rectangle(mesh_mask & cluster_mask)
        else:
            boundary_mask |= mesh_mask & cluster_mask
            boundary_mask_rect |= fill_minimum_rectangle(mesh_mask & cluster_mask)
        if new_textures.shape[0] > 0 and original_textures.shape[0] > 0:
            texture_array[mesh_mask & cluster_mask] = align_textures(new_textures, original_textures)
    
    texture_array[:, :, :3] /= 255
    texture_array = texture_array[:, :, [2, 1, 0, 3]].flatten()
    # boundary_image = Image.fromarray(boundary_mask.astype(np.uint8) * 255)
    # boundary_image.save(f"{boundary_dir}/{obj.name.replace('/', '_')}.png")
    
    # boundary_image_rect = Image.fromarray(boundary_mask_rect.astype(np.uint8) * 255)
    # boundary_image_rect.save(f"{boundary_dir}/{obj.name.replace('/', '_')}_rect.png")
    return texture_array, boundary_mask, boundary_mask_rect


def get_boundary(a_array, b_array, bb_array):
    # import pdb; pdb.set_trace()

    c_array = (a_array & ~b_array) | (bb_array & ~b_array)
    c_array = c_array | (binary_dilation(c_array, iterations=5) & b_array)
    c_array = c_array | binary_dilation(c_array)

    return c_array


def projection_street_view(solve_result, streetview_locs_path, input_gsv_dir, output_blend_path, boundary_mask_dir, fov=90, image_width=2048, image_height=1536, y_offset=1.6, y_offset_boundary=3, camera_step=60, work_dir="."):
    # Set parameters
    streetview_locs = pickle.load(open(streetview_locs_path, "rb"))
    base_forward = Vector((0, 0, -1))
    for mesh_name in tqdm(solve_result):
        # uv_unwrap_with_area_weights(obj_name=mesh_name, y_offset=7.0)
        
        obj = bpy.data.objects[mesh_name]
        
        texture_image = obj.active_material.node_tree.nodes['Image Texture'].image
        
        texture_array, boundary_mask, boundary_mask_rect = projection_street_view_mesh(obj, texture_image, mesh_name, solve_result, streetview_locs, y_offset, base_forward, input_gsv_dir, fov, image_width, image_height)
        
        _, boundary_mask_high, _ = projection_street_view_mesh(obj, texture_image, mesh_name, solve_result, streetview_locs, y_offset + y_offset_boundary, base_forward, input_gsv_dir, fov, image_width, image_height)
        
        final_boundary_mask = get_boundary(boundary_mask_high, boundary_mask, boundary_mask_rect)
        
        final_boundary_mask = Image.fromarray(final_boundary_mask.astype(np.uint8) * 255)
        final_boundary_mask.save(f"{boundary_mask_dir}/{obj.name.replace('/', '_')}.png")
        
        
        texture_image.pixels[:] = texture_array
        texture_image.update()
        texture_image.pack()
        texture_image.filepath_raw = f"//{work_dir}/updated_texture_{mesh_name}.png"
        texture_image.file_format = 'PNG'
        texture_image.save()

    
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_mainfile(filepath=output_blend_path)

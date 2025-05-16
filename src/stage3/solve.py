import bpy
import pickle
import math
import numpy as np
from tqdm import tqdm
from mathutils import Vector, Matrix
from scipy.spatial import KDTree


def rotate_around_y(vec, angle_deg):
    """
    Rotate a vector `vec` (mathutils.Vector) around the Y-axis by `angle_deg` degrees clockwise.
    To achieve clockwise rotation, use `-angle_deg` as the actual rotation angle.
    """
    angle_rad = math.radians(-angle_deg)
    rot_mat = Matrix.Rotation(angle_rad, 4, 'Y')
    return rot_mat @ vec


def shoelace_area_2d(points_2d):
    """
    Calculate the area of a 2D polygon using the Shoelace formula.
    points_2d: [(x0, y0), (x1, y1), ...] (at least 3 points)
    Returns the absolute value of the area.
    """
    n = len(points_2d)
    area = 0.0
    for i in range(n):
        x1, y1 = points_2d[i]
        x2, y2 = points_2d[(i + 1) % n]
        area += (x1 * y2 - x2 * y1)
    return abs(area) / 2.0


def set_temp_camera_transform(cam_obj, cam_pos, forward_vec, up_vec=Vector((0, 1, 0))):
    """
    Set the world transform of a temporary camera to place it at `cam_pos` with
    `forward_vec` as the forward direction and `up_vec` as the approximate up direction.
    Simplified approach: align the Z-axis to `-forward_vec`.
    """
    f = forward_vec.normalized()
    up_n = up_vec.normalized()
    r = f.cross(up_n).normalized()        # right vector
    u = r.cross(f).normalized()           # new up vector

    rot_mat = Matrix((
        (r.x, u.x, -f.x, 0.0),
        (r.y, u.y, -f.y, 0.0),
        (r.z, u.z, -f.z, 0.0),
        (0.0, 0.0, 0.0, 1.0)
    ))
    trans_mat = Matrix.Translation(cam_pos)

    cam_obj.matrix_world = trans_mat @ rot_mat


def is_blocked(scene, depsgraph, cam_pos, test_pos, obj_self=None, face_idx=None):
    """
    Check if the ray from `cam_pos` to `test_pos` is blocked by other objects.
    If the ray intersects an object before reaching `test_pos`, and it's not
    `(obj_self, face_idx)`, it's considered blocked.
    """
    direction = (test_pos - cam_pos)
    dist = direction.length
    if dist < 1e-8:
        return False

    direction_n = direction.normalized()

    hit, loc, normal, face_i, obj_i, matrix = scene.ray_cast(
        depsgraph=depsgraph,
        origin=cam_pos,
        direction=direction_n
    )
    if hit:
        hit_dist = (loc - cam_pos).length
        if hit_dist + 1e-6 < dist:
            if obj_i == obj_self and face_i == face_idx:
                return False
            else:
                return True
        else:
            return False
    else:
        return False


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


def angle_extent_deg(points_3d, camera_pos, cam_forward, cam_up, x_thres=45., y_thres=30.):
    """
    Calculate the horizontal and vertical angular extents (in degrees) for a set of 3D points
    relative to a camera's position and orientation. Also compute their product as the angular area.

    Args:
        points_3d: [(x0, y0, z0), ...] list of target points in world coordinates
        camera_pos: Vector(cx, cy, cz) camera position in world coordinates
        cam_forward: Vector(fx, fy, fz) forward direction unit vector
        cam_up: Vector(ux, uy, uz) up direction unit vector

    Returns:
        tuple: (horizontal extent, vertical extent, angular area in degrees²)
    """
    cam_right = cam_forward.cross(cam_up)
    cam_right.normalize()

    cam_forward = cam_forward.normalized()
    cam_up = cam_up.normalized()

    horiz_angles = []
    vert_angles = []

    for p in points_3d:
        p_vec = Vector(p)
        d = p_vec - camera_pos
        if d.length < 1e-8:
            continue

        angle_x_rad = math.atan2(d.dot(cam_right), d.dot(cam_forward))
        angle_y_rad = math.atan2(d.dot(cam_up), d.dot(cam_forward))

        angle_x_deg = math.degrees(angle_x_rad)
        angle_y_deg = math.degrees(angle_y_rad)

        horiz_angles.append(angle_x_deg)
        vert_angles.append(angle_y_deg)

    if not horiz_angles or not vert_angles:
        return 0.0, 0.0, 0.0

    min_x = max(-x_thres, min(horiz_angles))
    max_x = min(x_thres, max(horiz_angles))

    min_y = max(-y_thres, min(vert_angles))
    max_y = min(y_thres, max(vert_angles))

    angle_x_range = max_x - min_x
    angle_y_range = max_y - min_y
    angle_area_deg2 = angle_x_range * angle_y_range

    return angle_x_range, angle_y_range, angle_area_deg2


def polygon_area_2d(points):
    """
    Calculate the area of a 2D polygon using the Shoelace formula.
    Args:
        points: [(x0, y0), ..., (xn-1, yn-1)]

    Returns:
        float: Absolute area of the polygon.
    """
    n = len(points)
    area = 0.0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += (x1 * y2 - x2 * y1)
    return abs(area) / 2.0



def solve(input_blend_path="input.blend", streetview_locs_path="output-NY/streetview_locs.pkl", fov=90.0,
          normal_campos_cosine_thres=(0.5, 0.2), normal_cam_direction_cosine_thres=(0.85, 0.5),
          camera_angles=(0, 60, 120, 180, 240, 300), output_blend_path="output.blend", height_thres=15):
    """
    Main function to process a Blender scene and identify the best camera for each face in the meshes.

    Args:
        input_blend_path: Path to the Blender file to process.
        streetview_locs_path: Path to the pickle file containing camera positions.
        fov: Field of view for the temporary camera.
        normal_campos_cosine_thres: Thresholds for cosine similarity between face normals and camera positions.
        normal_cam_direction_cosine_thres: Thresholds for cosine similarity between face normals and camera directions.
        camera_angles: Camera rotation angles (in degrees) around the Y-axis for candidate orientations.

    Returns:
        dict: Mapping of object names to face indices and their best candidate cameras.
    """
    # Load the Blender file
    bpy.ops.wm.open_mainfile(filepath=input_blend_path)

    # Deselect and delete all non-mesh objects
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.context.scene.objects:
        if obj.type != 'MESH' or obj.name == 'Roof':
            obj.select_set(True)
        else:
            obj.select_set(False)
    bpy.ops.object.delete()

    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()

    # Load camera positions from the pickle file
    with open(streetview_locs_path, "rb") as f:
        camera_dict = pickle.load(f)

    camera_names = list(camera_dict.keys())
    camera_positions = [camera_dict[name] for name in camera_names]
    camera_positions_xz = np.array([[pos[0], pos[2]] for pos in camera_positions])
    kd_tree = KDTree(camera_positions_xz)

    # Create a temporary camera object
    temp_cam_data = bpy.data.cameras.new("TempCamera")
    temp_cam_data.angle = math.radians(fov)
    temp_cam_obj = bpy.data.objects.new("TempCameraObj", temp_cam_data)
    scene.collection.objects.link(temp_cam_obj)

    final_candidate_dict = {}
    for obj in tqdm(scene.objects):
        if obj.name in {"Roof", "Cube"}:
            continue
        if obj.type != 'MESH':
            continue

        # Subdivide the mesh for finer resolution
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.subdivide(number_cuts=2)
        bpy.ops.object.mode_set(mode='OBJECT')

        mesh = obj.data
        mesh.update()

        # Find the minimum Y-coordinate in the object (for filtering purposes)
        y_min = float('inf')
        for v in mesh.vertices:
            wv = obj.matrix_world @ v.co
            if wv.y < y_min:
                y_min = wv.y

        face_best_camera = {}  # Mapping of face index to best camera or None

        for poly in mesh.polygons:
            face_idx = poly.index
            face_vert_idxs = poly.vertices
            face_verts_world = [(obj.matrix_world @ mesh.vertices[vid].co) for vid in face_vert_idxs]
            face_y_min = min([vert[1] for vert in face_verts_world])
            if face_y_min > (y_min + height_thres):
                # Skip faces that are too high above the minimum Y
                continue
            # Calculate the world coordinates of the face center
            center_3d_world = obj.matrix_world @ poly.center

            # Get the world-space normal of the face
            normal_local = poly.normal
            normal_world = obj.matrix_world.to_3x3() @ normal_local
            normal_world.normalize()
            normal_np = np.array([normal_world.x, normal_world.y, normal_world.z])

            # Step 1: Find cameras within a radius of 25
            query_xz = [center_3d_world.x, center_3d_world.z]
            camera_indices_in_radius_25 = kd_tree.query_ball_point(query_xz, r=25.0)
            candidate_list_1 = [(camera_names[i], 0) for i in camera_indices_in_radius_25]
            camera_indices_in_radius_40 = kd_tree.query_ball_point(query_xz, r=40.0)
            candidate_list_1.extend([(camera_names[i], 1) for i in camera_indices_in_radius_40])

            # Step 2: Filter by cosine similarity with face normal
            candidate_list_2 = []
            for cam_name, score in candidate_list_1:
                cpos = camera_dict[cam_name]
                direction = cpos - np.array([center_3d_world.x, cpos[1], center_3d_world.z])
                dir_len = np.linalg.norm(direction)
                if dir_len < 1e-8:
                    continue
                dir_unit = direction / dir_len
                dot_val = np.dot(normal_np, dir_unit)
                if dot_val > normal_campos_cosine_thres[0]:
                    candidate_list_2.append((cam_name, score))
                elif dot_val > normal_campos_cosine_thres[1]:
                    candidate_list_2.append((cam_name, score + 1))

            # Step 3: Further filter by camera direction cosine similarity
            candidate_list_3 = []
            if candidate_list_2:
                base_forward = Vector((0, 0, -1))
                for cam_name, score in candidate_list_2:
                    cpos = Vector(camera_dict[cam_name])
                    for i, angle_deg in enumerate(camera_angles):
                        fwd_vec = rotate_around_y(base_forward, angle_deg)
                        direction = np.array(cpos) - np.array([center_3d_world.x, cpos[1], center_3d_world.z])
                        direction = direction / np.linalg.norm(direction)
                        dot_val = -direction.dot(fwd_vec)
                        if dot_val > normal_cam_direction_cosine_thres[0]:
                            candidate_list_3.append(((cam_name, i), score))
                        if dot_val > normal_cam_direction_cosine_thres[1]:
                            candidate_list_3.append(((cam_name, i), score + 2))

            if not candidate_list_3:
                continue

            # Step 4: Compute visibility and projection area, select the best candidate
            face_vert_idxs = poly.vertices
            face_verts_world = [(obj.matrix_world @ mesh.vertices[vid].co) for vid in face_vert_idxs]

            best_cam = None
            best_score = float('inf')
            best_distance = float('inf')

            for (cam_name, orient_idx), score in candidate_list_3:
                cpos = Vector(camera_dict[cam_name])
                evaluated_ratio = 10
                for ratio in range(5):
                    interpolated_verts = [(Vector(vw) * (10 - ratio) + center_3d_world * ratio) / 10 for vw in face_verts_world]
                    visible = all(not is_blocked(scene, depsgraph, cpos, Vector(vw), obj_self=obj, face_idx=face_idx)
                                  for vw in interpolated_verts)
                    if visible:
                        evaluated_ratio = ratio
                        break
                if evaluated_ratio > 5:
                    continue

                distance = np.linalg.norm(cpos - Vector([center_3d_world.x, center_3d_world.y, center_3d_world.z]))
                current_score = score + evaluated_ratio / 10
                if current_score < best_score or (current_score == best_score and distance < best_distance):
                    best_score = current_score
                    best_distance = distance
                    best_cam = (cam_name, orient_idx)

            if best_cam and best_distance < 100.0:
                face_best_camera[face_idx] = (best_cam, best_distance)

        if face_best_camera:
            face_best_camera_reversed = {}
            for face_id, (cam, _) in face_best_camera.items():
                cam_name = f"{cam[0]}_{cam[1] * 60}"
                face_best_camera_reversed.setdefault(cam_name, []).append(face_id)
            final_candidate_dict[obj.name] = face_best_camera_reversed
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_mainfile(filepath=output_blend_path)
    return final_candidate_dict

import os
import argparse
import json
import torchvision
import sys
import matplotlib
from plyfile import PlyData, PlyElement
 
# segment anything
# from segment_anything import (
#     sam_model_registry,
#     SamPredictor
# )
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as TS
import torch
import gzip
import pickle 
# import trimesh
from PIL import Image
current_directory = os.getcwd()
sys.path.append(os.path.join(current_directory, 'depth_anything_v2'))
def backproject_depth_to_pointcloud(K: np.ndarray, depth: np.ndarray, pose):
    """Convert depth image to pointcloud given camera intrinsics.
    Args:
        depth (np.ndarray): Depth image.
    Returns:
        np.ndarray: (x, y, z) Point cloud. [n, 4]
        np.ndarray: (r, g, b) RGB colors per point. [n, 3] or None
    """
    cv2.imwrite('/project/pi_chuangg_umass_edu/yian/robogen/ljg/Depth-Anything-V2/output/test/depth.png', depth*230)
    _fx = K[0, 0]
    _fy = K[1, 1]
    _cx = K[0, 2]
    _cy = K[1, 2]
    if len(depth.shape) == 3:
        depth = depth[:, :, 0]

    # Mask out invalid depth
    mask = np.where(depth > -1.0)  # all should be valid
    x, y = mask[1], mask[0]

    # Normalize pixel coordinates
    normalized_x = x.astype(np.float32) - _cx
    normalized_y = y.astype(np.float32) - _cy

    # Convert to world coordinates
    world_x = normalized_x * depth[y, x] / _fx # z is the depth
    world_y = normalized_y * depth[y, x] / _fy #
    world_z = depth[y, x] #

    pc = np.vstack((world_x, world_y, world_z)).T # format of [x,y,z]

    point_cloud_h = np.hstack((pc, np.ones((pc.shape[0], 1))))
    point_cloud_world = (pose @ point_cloud_h.T).T
    point_cloud_world = point_cloud_world[:, :3].reshape(depth.shape[0], depth.shape[1], 3)

    return point_cloud_world

def euler_to_rotation_matrix(heading, tilt, roll):
    # Convert degrees to radians
    heading = np.radians(heading)
    tilt = np.radians(tilt)
    roll = np.radians(roll)
    
    # Compute rotation matrices around each axis
    R_heading = np.array([
        [np.cos(heading), -np.sin(heading), 0],
        [np.sin(heading), np.cos(heading), 0],
        [0, 0, 1]
    ])
    
    R_tilt = np.array([
        [1, 0, 0],
        [0, np.cos(tilt), -np.sin(tilt)],
        [0, np.sin(tilt), np.cos(tilt)]
    ])
    
    R_roll = np.array([
        [np.cos(roll), 0, np.sin(roll)],
        [0, 1, 0],
        [-np.sin(roll), 0, np.cos(roll)]
    ])
    
    # Combined rotation matrix
    R = R_roll @ R_tilt @ R_heading
    return R
import math
def compute_extrinsic(lat=0, lng=0, altitude=0, heading=0, tilt=90, roll=0): #need debug
    # Placeholder function to convert lat, lng, altitude to world coordinates (X, Y, Z)
    # This conversion depends on your specific coordinate system and map projection
    # X, Y, Z = lat_lng_to_world_coordinates(lat, lng, altitude)
    X, Y, Z = 0,0,0
    
    # Compute rotation matrix from heading, tilt, roll
    R = euler_to_rotation_matrix(heading, tilt, roll)
    
    # Extrinsics matrix [R | T]
    extrinsics = np.eye(4)
    extrinsics[:3, :3] = R
    extrinsics[:3, 3] = [X, Y, Z]
    
    return extrinsics

def extrinsic_to_camera_pose(extrinsic_matrix): # need debug
    """
    Convert the extrinsic matrix to camera pose.
    
    Parameters:
    - extrinsic_matrix (np.array): The 4x4 extrinsic matrix in the form:
      [R | t]
      [0 | 1]
      where R is the 3x3 rotation matrix and t is the 3x1 translation vector.
    
    Returns:
    - np.array: The 4x4 camera pose matrix that transforms points from the camera
      coordinate system to the world coordinate system.
    """
    # Extract the rotation matrix and translation vector
    R = extrinsic_matrix[0:3, 0:3]
    t = extrinsic_matrix[0:3, 3]
    
    # Compute the transpose of the rotation matrix (inverse of rotation)
    R_transpose = np.transpose(R)
    
    # Compute the transformed translation vector (-R^T * t)
    transformed_t = -np.dot(R_transpose, t)
    
    # Construct the camera pose matrix
    camera_pose = np.eye(4)  # Initialize a 4x4 identity matrix
    camera_pose[0:3, 0:3] = R_transpose
    camera_pose[0:3, 3] = transformed_t
    
    return camera_pose

def pred_depth(image_dir, max_depth):
    # seed = int(time.time())
    input_image = Image.open(image_dir)
    color_map = "Spectral"
    # generator = torch.Generator(device=device)
    # generator.manual_seed(seed)
    pipe_out = pipe(
        input_image,
        denoising_steps=denoise_steps,
        ensemble_size=ensemble_size,
        processing_res=processing_res,
        match_input_res=match_input_res,
        batch_size=1,
        color_map=color_map,
        show_progress_bar=True,
        resample_method=resample_method,
        generator=None
    )
    depth_pred: np.ndarray = pipe_out.depth_np
    depth_colored: Image.Image = pipe_out.depth_colored
    scale = max_depth / np.max(depth_pred)
    depth_pred *= scale
    print("scale:", scale)
    return depth_pred, depth_colored

# Example usage
lat, lng, altitude = 0,0,0  # Los Angeles with 100m altitude
heading, tilt, roll = 335.38214-180, 86.80848, 1.5849512  #
heading, tilt, roll = 0, 90, 0
extrinsic_matrix = compute_extrinsic(lat, lng, altitude, heading, tilt, roll)
# print('extrinsic_matrix:', extrinsic_matrix)

pixel_height = 384
pixel_width = 512
fov = 100
f_x = (pixel_width/2.0)/math.tan(math.radians(fov/2.0))
c_x = pixel_width/2.0
f_y = (pixel_height/2.0)/math.tan(math.radians(fov/2.0))
c_y = pixel_height/2.0

image_dir = '/project/pi_chuangg_umass_edu/yian/robogen/ljg/Depth-Anything-V2/images/CMU6View/-JAeZEDMWWo6IrBUeRsniQ/gsv_2.jpg'
output_dir = '/project/pi_chuangg_umass_edu/yian/robogen/ljg/Depth-Anything-V2/output/test'
tags = 'pole, tree'
descriptions = None
camera_pose = extrinsic_to_camera_pose(extrinsic_matrix)

intrinsic_K = np.array([[f_x,   0.0,   c_x],
               [0.0,   f_y,   c_y],
               [0.0,   0.0,   1.0]])
reference_depth = None # not depth anything v2
reference_mask = None # not GroundingSAM
binary_mask = None
name_dict = None


from depth_anything_v2.dpt import DepthAnythingV2

def pred_depth(img_path, outdir='/project/pi_chuangg_umass_edu/yian/robogen/ljg/Depth-Anything-V2/output/test' ,input_size = 518, encoder = 'vitl', pred_only = False, grayscale = False): #debug
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**model_configs[encoder])
    depth_anything.load_state_dict(torch.load(f'Depth-Anything-V2/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    
    if os.path.isfile(img_path):
        if img_path.endswith('txt'):
            with open(img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [img_path]
    else:
        filenames = glob.glob(os.path.join(img_path, '**/*'), recursive=True)
    
    os.makedirs(outdir, exist_ok=True)
    
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    
    for k, filename in enumerate(filenames):
        print(f'Progress {k+1}/{len(filenames)}: {filename}')
        
        raw_image = cv2.imread(filename)
        
        depth = depth_anything.infer_image(raw_image, input_size)
        
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        
        if grayscale:
            depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        else:
            depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        
        if pred_only:
            cv2.imwrite(os.path.join(outdir, os.path.splitext(os.path.basename(filename))[0] + '.png'), depth)
        else:
            split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
            combined_result = cv2.hconcat([raw_image, split_region, depth])
            
            cv2.imwrite(os.path.join(outdir, os.path.splitext(os.path.basename(filename))[0] + '.png'), combined_result)
    return depth

def save_point_cloud_as_ply(points, filename):
    """
    Saves a point cloud stored in a NumPy array as a PLY file using the plyfile library.

    Args:
    - points (np.ndarray): A NumPy array of shape (N, 3) containing the point cloud, where N is the number of points.
    - filename (str): The filename of the output PLY file.
    """
    # Create a structured array for the plyfile
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    structured_array = np.array(list(map(tuple, points)), dtype=dtype)

    # Create a PlyElement from the structured array and write to file
    el = PlyElement.describe(structured_array, 'vertex')
    PlyData([el]).write(filename)

def filter_bbox(idx, bbox_list, pred_phrases):
    filtered = False
    box = bbox_list[idx]
    name = pred_phrases[idx].split('(')[0]
    for j in range(len(bbox_list)):
        if idx == j:
            continue
        box2 = bbox_list[j]
        name1 = pred_phrases[j].split('(')[0]
        if get_box_iou(box, box2) > 0.9 and get_area(box) > get_area(box2) and name == name1:
            filtered = True
            break
    return filtered

# from groundingdino.util.inference import load_model, load_image, predict, annotate, Model
# def get_2d_bbox(image_dir, output_dir, tags, bbox_treshold=0.35, text_threshold=0.25):
#     CONFIG_PATH = "./Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
#     CHECKPOINT_PATH = "./Grounded-Segment-Anything/groundingdino_swint_ogc.pth"
#     DEVICE = "cuda"
#     FP16_INFERENCE = True

#     image_source, image = load_image(image_dir)
#     model = load_model(CONFIG_PATH, CHECKPOINT_PATH)

#     if FP16_INFERENCE:
#         image = image.half()
#         model = model.half()

#     boxes, logits, phrases = predict(
#         model=model,
#         image=image,
#         caption=tags,
#         box_threshold=bbox_treshold,
#         text_threshold=text_threshold,
#         device=DEVICE,
#     )

#     image = 
#     cv2.imwrite(os.path.join(output_dir,"2d_bbox.jpg"))
#     name_dict = {}
#     rename_phrases = phrases
#     for i,item in enumerate(phrases):
#         if item not in name_dict.keys():
#             name_dict[item] = 0
#             rename_phrases[i] = f'{name}-{0}'
#         else:
#             name_dict[item] += 1
#             rename_phrases[i] = f'{name}-{name_dict[item]}'
    
#     result = []
#     for box, logit, phrase in zip(boxes, logits, phrases):
#         result.append({
#             'bbox': box,
#             'center': (box[0]+box[2]/2, box[1]+box[3]/2),
#             'logit': logit,
#             'phrase': phrase
#         })
#     tags = 'poles, tree'
#     masks, pred_phrases, box_list = mask_and_save(args.image_dir, args.output_dir, tags)
#     image = cv2.imread(args.image_dir)
#     masked_images_dir = os.path.join(args.output_dir, "masked_images")

#     return result

import open3d as o3d
def get_3d_bbox(image_dir, output_dir, tags, descriptions, camera_pose, intrinsic_K, max_depth=None,
             reference_depth=None, reference_mask=None, binary_mask=None, name_dict={}, filter=False):
    # print("done mask and save")

    depth_pred = pred_depth(image_dir, grayscale=True)
    depth_pred = depth_pred / np.argmax(depth_pred)
    # print("shape of reference", reference_depth.shape)
    # print("shape of depth predict" , depth_pred.shape)
    # depth_colored.save(os.path.join(output_dir, 'depth_color.jpg'))
    # # print("shape of reference mask", reference_mask.shape)
    # range_pred = depth_pred[reference_mask].max() - depth_pred[reference_mask].min()
    # print("shape of combined", depth_pred[reference_mask].shape)
    # range_real = reference_depth[reference_mask].max() - reference_depth[reference_mask].min()
    # depth_pred = depth_pred * range_real / range_pred
    # offset = reference_depth[reference_mask].mean() - depth_pred[reference_mask].mean()
    # print(offset)
    # depth_pred = depth_pred + offset

    # print("done predict depth")

    # pc = backproject_depth_to_pointcloud(intrinsic_K, reference_depth, camera_pose)
    # save_point_cloud_as_ply(pc.reshape(-1, 3), os.path.join(output_dir, f'pc_real.ply'))
    pc = backproject_depth_to_pointcloud(intrinsic_K, depth_pred, camera_pose)
    save_point_cloud_as_ply(pc.reshape(-1, 3), os.path.join(output_dir, f'pc.ply'))
    # if rotation is None:
    # pc = predict_pos(image_dir, output_dir, fov, max_depth)
    print("tags: ", tags)
    tags_list = tags.split(',')
    for i in range(len(tags_list)):
        tags_list[i] = tags_list[i].strip()

    
    pickle_path = '/project/pi_chuangg_umass_edu/yian/robogen/ljg/Depth-Anything-V2/output/test'

    with open(os.path.join(pickle_path,'masks.pkl'),'rb') as f:
        masks = pickle.load(f)
    with open(os.path.join(pickle_path,'pred_phrases.pkl'), 'rb') as f:
        pred_phrases = pickle.load(f)
    with open(os.path.join(pickle_path,'boxlist.pkl'), 'rb') as f:
        box_list = pickle.load(f)

    image = cv2.imread(image_dir)
    masked_images_dir = os.path.join(output_dir, "masked_images")
    if not os.path.exists(masked_images_dir):
        os.makedirs(masked_images_dir)
    padding = 5
    name_dict_mask = {}
    for i in range(len(masks)):
        # filter out large boxes
        if filter:
            if filter_bbox(i, box_list, pred_phrases):
                continue
        mask = masks[i, 0].cpu().numpy()
        mask_pos = np.where(mask)
        top, down = np.min(mask_pos[0]), np.max(mask_pos[0])
        left, right = np.min(mask_pos[1]), np.max(mask_pos[1])
        pred_phrase = pred_phrases[i]
        mask_expanded = mask[:, :, None]
        inverse_mask = 1 - mask_expanded
        white_image = np.ones_like(image) * 255

        # masked_image = image * mask_expanded + white_image * inverse_mask
        masked_image = image
        masked_image = masked_image[max(top - padding, 0): min(down + padding, mask_expanded.shape[0]),
                       max(left - padding, 0): min(right + padding, mask_expanded.shape[1])]

        name = pred_phrases[i].split("(")[0]
        if not name in tags_list:
            for tag in tags_list:
                if name in tag:
                    name = tag
                    break

        # if not name in name_dict_mask:
        #     name_dict_mask[name] = 0
        # cnt = name_dict_mask[name]
        # name_dict_mask[name] += 1
        # cv2.imwrite(os.path.join(masked_images_dir, f"{name}.png"), masked_image)
        # cv2.imwrite(os.path.join(masked_images_dir, f"{name}{cnt}.png"), masked_image)

    # print(f"pred phrases length: {len(pred_phrases)}, content: {pred_phrases}")
    # print(f"masks shape: {masks.shape}")

    x_min, y_min, z_min = np.min(pc, axis=(0, 1))
    x_max, y_max, z_max = np.max(pc, axis=(0, 1))

    bbox = []
    result = []
    result.append({'room_bbox': [[x_min, y_min, z_min], [x_max, y_max, z_max]]})
    for idx, mask in enumerate(masks):
        if filter:
            if filter_bbox(idx, box_list, pred_phrases):
                continue
        mask = mask.cpu().numpy()[0]
        if binary_mask is not None and np.sum(np.logical_and(mask, binary_mask)) / np.sum(mask) > 0.5:
            continue
        pcs = np.copy(pc[mask == True]).reshape(-1, 3)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcs)

        # Use DBSCAN clustering to remove noise and outliers
        labels = np.array(pcd.cluster_dbscan(eps=0.04, min_points=10, print_progress=True))
        max_label = labels.max()

        # Create a new point cloud containing only the points from the largest cluster
        # (assuming the largest cluster is the main object and the rest are outliers/noise)
        main_cluster = pcs[labels == np.argmax(np.bincount(labels[labels != -1]))]
        cleaned_pcd = o3d.geometry.PointCloud()
        cleaned_pcd.points = o3d.utility.Vector3dVector(main_cluster)
        bounding_box = cleaned_pcd.get_axis_aligned_bounding_box()
        min_bound = bounding_box.min_bound
        max_bound = bounding_box.max_bound

        cnt = 0
        name = pred_phrases[idx].split("(")[0]
        if name == "":
            continue
        if not name in tags_list:
            for tag in tags_list:
                if name in tag:
                    name = tag
                    break
        if name in name_dict:
            cnt = name_dict[name]
            name_dict[name] += 1
        else:
            name_dict[name] = 1

        print(pred_phrases[idx], bounding_box)

        if name not in tags_list:
            description = ""
            on_floor = None
        else:
            if descriptions is None:
                description = None
                on_floor = None
            else:
                description = descriptions[name]
                # on_floor = on_floors[name]

        result.append(
            {
                "bbox": [[float(min_bound[0]), float(min_bound[1]), float(min_bound[2])],
                         [float(max_bound[0]), float(max_bound[1]), float(max_bound[2])]],
                "name": name + '-' + str(cnt),
                "type": "rigid mesh",
                "description": description,
                # "on_floor": on_floor,
                # "assetId": select_objects(name),
                "confidence": pred_phrases[idx].split("(")[1].split(")")[0]
            }
        )
    
    return result

from functools import partial
from pyproj import Proj, transform
def proj_trans(lat,lng):
    p1 = Proj(init="epsg:4326")  # 定义数据地理坐标系
    p2 = Proj(init="epsg:3857")  # 定义转换投影坐标系
    # x1, y1 = p1(lng, lat)
    transformer = partial(transform, p1, p2)
    x2,y2 = transformer(lat, lng)
    return [x2,y2]

def get_camera_offset(lat_mesh,lng_mesh,lat_img,lng_img):
    x = (lng_img-lng_mesh)*111194.926644558737*math.cos(lat_img/180*math.pi)
    z = -(lat_img-lat_mesh)*111194.926644558737
    print('camera relative position:')
    print('x:',x) #经度，朝东，x正s向
    print('z:',z) #纬度，朝北，z反向
    return x,z


import math
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')
    
    parser.add_argument('--image_dir', type=str, default='/project/pi_chuangg_umass_edu/yian/robogen/ljg/Depth-Anything-V2/output')
    parser.add_argument('--output_dir', type=str, default='/project/pi_chuangg_umass_edu/yian/robogen/ljg/Depth-Anything-V2/output')

    args = parser.parse_args()
    result = get_bbox(args.image_dir, output_dir, tags, descriptions, camera_pose, intrinsic_K)
    # # print(result)
    for i in range(len(result)):
        if i == 0: continue
        print('x>',result[i]['bbox'][0][0],'&& x<',result[i]['bbox'][1][0],'&& y>',result[i]['bbox'][0][1],'&& y<',result[i]['bbox'][1][1])

    # lat_mesh,lng_mesh = 40.44432884930322 ,-79.94494090959955
    # lat_img,lng_img = 40.44433556665464,-79.94448122176674
    # lat_img,lng_img = 40.44457728086382, -79.94502276774394

    # tags = 'poles, tree'
    # masks, pred_phrases, box_list = mask_and_save(args.image_dir, args.output_dir, tags)
    # image = cv2.imread(args.image_dir)
    # masked_images_dir = os.path.join(args.output_dir, "masked_images")

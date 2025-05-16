import sys
sys.path.append('./Depth-Anything-V2')
sys.path.append('./Grounded-Segment-Anything')
sys.path.append('./Grounded-Segment-Anything/GroundingDINO')
sys.path.append('./segment_anything')
from metadata import get_image_metadata, get_image_metadata_with_latlng, get_image_metadata_with_panoId
from test_getbbox import get_3d_bbox, extrinsic_to_camera_pose, get_camera_offset, compute_extrinsic
from test_mask_and_save import get_2d_bbox
from test_takephoto import take_photo, get_intrinsic, get_pose, uv2world, get_mesh_center, get_mesh_bbox
from test_sam import segment_objects
import argparse
import json
import shutil
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')
    
    # parser.add_argument('--image_dir', type=str, default='/project/pi_chuangg_umass_edu/yian/robogen/ljg/Depth-Anything-V2/output')
    parser.add_argument('--mesh_dir', type=str, default="/project/pi_chuangg_umass_edu/yian/robogen/ljg/Depth-Anything-V2/output/test3/flat_CMU_180.glb")
    parser.add_argument('--output_dir', type=str, default="/project/pi_chuangg_umass_edu/yian/robogen/ljg/Depth-Anything-V2/output/test6")
    parser.add_argument('--tags', type=str, default='tree')
    parser.add_argument('--lat_mesh', type=float, default='40.44432884930322')
    parser.add_argument('--lng_mesh', type=float, default='-79.94494090959955')
    parser.add_argument('--heading', type=int, default=0)
    parser.add_argument('--panoId', type=str, default='-JAeZEDMWWo6IrBUeRsniQ')
    args = parser.parse_args()

    # assume that there is a mesh model, we've known it's lat,lng of the mesh center
    lat_mesh, lng_mesh = 40.44432884930322 , -79.94494090959955
    if args.lat_mesh is not None and args.lng_mesh is not None: 
        lat_mesh, lng_mesh = args.lat_mesh, args.lng_mesh
    mesh_center = [-195, -57, 78] 
    mesh_center = get_mesh_center(args.mesh_dir)
    mesh_bbox = get_mesh_bbox(args.mesh_dir)
    print("mesh_bbox:",mesh_bbox)

    # get panoId & image
    if len(args.panoId) > 0:
        args.output_dir = os.path.join(args.output_dir, args.panoId)
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir, exist_ok=True)
        lat_img, lng_img = get_image_metadata_with_panoId(args.panoId, args.output_dir, args.heading)
    else:
        lat_img, lng_img, panoId = get_image_metadata_with_latlng(lat_mesh, lng_mesh, args.output_dir, args.heading)
    image_dir = os.path.join(args.output_dir,'streetview.jpg')

    # get some image from the lat,lng; and get their metadata; get camera intrinsic and pose
    offset_x, offset_z = get_camera_offset(lat_mesh,lng_mesh,lat_img,lng_img)
    photoB_camera_intrinsic = get_intrinsic(fov=50)
    photoB_camera_pose = get_pose(mesh_center[0],mesh_center[1],mesh_center[2],offset_x,offset_z)

    camera_location = [mesh_center[0]+offset_x, mesh_center[1], mesh_center[2]+offset_z]
    if not (mesh_bbox[0, 0]*2/3 + mesh_bbox[1, 0]*1/3 < camera_location[0] < mesh_bbox[0, 0]*1/3 + mesh_bbox[1, 0]*2/3 and \
        mesh_bbox[0, 1]*2/3 + mesh_bbox[1, 1]*1/3 < camera_location[1] < mesh_bbox[0, 1]*1/3 + mesh_bbox[1, 1]*2/3 and \
        mesh_bbox[0, 2]*2/3 + mesh_bbox[1, 2]*1/3 < camera_location[2] < mesh_bbox[0, 2]*1/3 + mesh_bbox[1, 2]*2/3):
        print("Out of range")
        shutil.rmtree(args.output_dir)
        exit()
        

    # get bbox of the image A, get depth of the image A
    descriptions = None
    extrinsic_matrix = compute_extrinsic(heading = 0) #TODO:need change
    imageA_camera_pose = extrinsic_to_camera_pose(extrinsic_matrix)
    imageA_camera_intrinsic = get_intrinsic(fov=50)
    result= get_2d_bbox(image_dir, args.output_dir, args.tags)
    # result = get_3d_bbox(image_dir, args.output_dir, args.tags, descriptions, imageA_camera_pose, imageA_camera_intrinsic)
    # with open(os.path.join(args.output_dir, 'result.json'), 'w') as file:
    #     json.dump(file, result)
    # for i in range(len(result)):
    #     if i == 0: continue
    #     print('x>',result[i]['bbox'][0][0],'&& x<',result[i]['bbox'][1][0],'&& y>',result[i]['bbox'][0][1],'&& y<',result[i]['bbox'][1][1])
    print("2d_bbox_of_imageA:",result)

    # take the same point of view photo B on the mesh
    photoB_color, photoB_depth = take_photo(args.mesh_dir, args.output_dir, photoB_camera_intrinsic, photoB_camera_pose)

    # segment the new photo B from SAM( get prompt from the segment result from image A); project the image on the mesh
    photoB_dir = os.path.join(args.output_dir, 'photoB.png')
    all_objects = []
    for i,item in enumerate(result):
        
        prompt_point = item['center']
        bbox_coords = segment_objects(photoB_dir, args.output_dir, prompt_point, item['phrase'])

    # regularize and normalize the location of the 3D bbox
        u, v = (bbox_coords[0]+bbox_coords[2]) // 2, (bbox_coords[1]+bbox_coords[3]) // 2
        real_location = uv2world(photoB_camera_intrinsic, photoB_camera_pose, photoB_depth, [v, u])
        print(item['phrase'],f"x>{real_location[0]-0.5}&&x<{real_location[0]+0.5}&&z>{real_location[2]-0.5}&&z<{real_location[2]+0.5}")
        all_objects.append({
            'name': item['phrase'],
            'location': real_location.tolist()
        })
    all_objects.append({'camera_location': camera_location})
    with open(os.path.join(args.output_dir,'scene.json'), 'w') as file:
        json.dump(all_objects, file)

    # TODO:visualization


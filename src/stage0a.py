import json
import os
import subprocess
import sys
import argparse
import bpy
import pickle
from datetime import datetime
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from stage0 import (
    download_3d_tiles,
    pre_move_tiles,
    merge_tiles,
    export_glb,
    align_tiles,
    align_osm,
    smooth_sampled_points,
    create_mask,
    download_streetview_metadata,
    download_pano_images_mapillary,
    process_mapillary_images,
    align_street_view
)

if __name__ == "__main__":
    if "--" not in sys.argv:
        pass
    else:
        sys.argv = [""] + sys.argv[sys.argv.index("--") + 1:]

    parser = argparse.ArgumentParser("Stage0a Lat, lng to mask", add_help=True)
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lng", type=float, required=True)
    parser.add_argument("--rad", type=float, required=True)
    parser.add_argument("--api_key_file", type=str, required=True)
    parser.add_argument("--mapillary_token_file", type=str, required=True)
    parser.add_argument("--cache_root", type=str, required=True)
    parser.add_argument("--mask_output_path", type=str, required=True)
    parser.add_argument("--tile_output_path", type=str, required=True)
    parser.add_argument("--ref_ground_output_path", type=str, required=True)
    parser.add_argument("--tag", type=str, required=False, default="")
    args = parser.parse_args()

    api_key = open(args.api_key_file).readline().strip()

    if args.tag == "":
        time = datetime.now()
        tag = time.strftime("%d-%H-%M-%S")
    else:
        tag = args.tag

    work_dir = args.cache_root
    raw_tiles_dir = os.path.join(work_dir, "raw_3d_tiles")
    processed_tiles_dir = os.path.join(work_dir, "processed_3d_tiles")
    meshes_dir = os.path.join(work_dir, "meshes")
    json_dir = os.path.join(work_dir, "jsons")
    pcd_dir = os.path.join(work_dir, "pcd")
    mapillary_dir = os.path.join(work_dir, "mapillary")
    street_view_dir = os.path.join(work_dir, "street_view")
    tile_output_path = args.tile_output_path
    mask_output_path = args.mask_output_path

    for directory in [work_dir, raw_tiles_dir, processed_tiles_dir, meshes_dir, pcd_dir, json_dir, mapillary_dir,
                      street_view_dir]:
        os.makedirs(directory, exist_ok=True)

    if os.path.exists(tile_output_path):
        print(f"Find {tile_output_path}, start from this...")
        bpy.ops.wm.open_mainfile(filepath=tile_output_path)
    else:
        if not os.path.exists(os.path.join(work_dir, "download_finished.txt")):
            print("Start downloading...")
            download_3d_tiles(api_key, args.lat, args.lng, args.rad, raw_tiles_dir, json_dir)
            fout = open(os.path.join(work_dir, "download_finished.txt"), "w")
            fout.write("Finished")
            fout.close()
        else:
            print("Skip downloading...")

        print("Finished downloading, doing preprocessing...")
        offset = pre_move_tiles(raw_tiles_dir, processed_tiles_dir)

        print("Finished preprocessing, doing merging...")
        merge_tiles(processed_tiles_dir, offset)

        print("Finished merging, doing aligning...")
        align_tiles(args.lat, args.lng, tile_output_path=tile_output_path)

    if os.path.exists(mask_output_path):
        print(f"Find {mask_output_path}, start from this...")
        bpy.ops.wm.open_mainfile(filepath=mask_output_path)
    else:
        export_glb(
            mesh_name="Mesh_0",
            output_path=os.path.join(meshes_dir, "aligned.glb")
        )

        print("Finished aligning, doing aligning OSM data...")
        all_valid_points, all_ground_points, ground_polygons = align_osm(
            input_glb_path=os.path.join(meshes_dir, "aligned.glb"),
            lat=args.lat, lng=args.lng, rad=args.rad,
            pcd_dir=pcd_dir, output_pcd=False,
        )

        road_info_dict, street_view_loc_clean_smooth, street_view_loc_clean_all, ground_info_dict = smooth_sampled_points(
            all_road_data=all_valid_points,
            all_ground_data=all_ground_points,
            ground_polygons=ground_polygons,
            pcd_dir=pcd_dir,
            output_pcd=False,
        )

        pickle.dump(street_view_loc_clean_all, open(args.ref_ground_output_path, "wb"))

        print("Finished aligning OSM data, creating mask...")
        create_mask(
            reference_points=street_view_loc_clean_all,
            road_type_list=road_info_dict,
            ground_info=ground_info_dict,
            output_path=mask_output_path,
        )

    print("Fetching street view meta data...")
    # Use Mapillary
    # print("Fetching street view meta data...")
    # download_pano_images_mapillary(args.lat, args.lng, args.rad, args.mapillary_token_file,
    #                                mapillary_dir, os.path.join(mapillary_dir, "meta.json"))
    # full_metadata_dict = process_mapillary_images(mapillary_dir, street_view_dir, os.path.join(mapillary_dir,
    #                                                                                            "meta.json"))

    # # Use Google
    # full_metadata_dict = download_streetview_metadata(args.lat, args.lng, args.rad, os.path.join(work_dir, "pano.pkl"),
    #                                                   os.path.join(work_dir, "meta.json"), api_key, step_count=50,
    #                                                   out_2d_dir=street_view_dir)
    # print("Aligning street view data...")
    # align_street_view(input_glb_path=os.path.join(meshes_dir, "aligned.glb"),
    #                   lat=args.lat, lng=args.lng, meta_data_dict=full_metadata_dict,
    #                   output_path=os.path.join(work_dir, "streetview_locs.pkl"))

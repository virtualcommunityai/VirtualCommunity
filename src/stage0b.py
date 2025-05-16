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

    parser = argparse.ArgumentParser("Stage0b fetch street view meta data", add_help=True)
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lng", type=float, required=True)
    parser.add_argument("--rad", type=float, required=True)
    parser.add_argument("--api_key_file", type=str, required=True)
    parser.add_argument("--mapillary_token_file", type=str, required=True)
    parser.add_argument("--streetview_locs_output_path", type=str, required=True)
    parser.add_argument("--cache_root", type=str, required=True)
    args = parser.parse_args()

    api_key = open(args.api_key_file).readline().strip()

    work_dir = args.cache_root
    meshes_dir = os.path.join(work_dir, "meshes")
    mapillary_dir = os.path.join(work_dir, "mapillary")
    street_view_dir = os.path.join(work_dir, "street_view")
    meta_data_output = os.path.join(work_dir, "meta_data.pkl")
    streetview_locs_output = args.streetview_locs_output_path

    for directory in [work_dir, meshes_dir, mapillary_dir, street_view_dir]:
        os.makedirs(directory, exist_ok=True)

    if os.path.exists(meta_data_output):
        print(f"Street view meta data found...")
        full_metadata_dict = pickle.load(open(meta_data_output, "rb"))
    else:
        # print("Fetching street view meta data...")
        # download_pano_images_mapillary(args.lat, args.lng, args.rad, args.mapillary_token_file,
        #                                mapillary_dir, os.path.join(mapillary_dir, "meta.json"))
        print("Fetching street view meta data...")
        # Use Google
        full_metadata_dict = download_streetview_metadata(args.lat, args.lng, args.rad,
                                                          os.path.join(work_dir, "pano.pkl"),
                                                          os.path.join(work_dir, "meta.json"), api_key, step_count=50,
                                                          out_2d_dir=street_view_dir)
        pickle.dump(full_metadata_dict, open(meta_data_output, "wb"))
        # Use Mapillary
        # full_metadata_dict = process_mapillary_images(mapillary_dir, street_view_dir, os.path.join(mapillary_dir,
        #                                                                                            "meta.json"))

    print("Aligning street view data...")
    align_street_view(input_glb_path=os.path.join(meshes_dir, "aligned.glb"),
                      lat=args.lat, lng=args.lng, meta_data_dict=full_metadata_dict,
                      output_path=streetview_locs_output)


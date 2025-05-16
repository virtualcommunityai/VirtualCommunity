import argparse
import pickle
import sys
import os
import bpy
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from stage3 import projection_street_view


if __name__ == "__main__":
    if "--" not in sys.argv:
        pass
    else:
        sys.argv = [""] + sys.argv[sys.argv.index("--") + 1:]

    parser = argparse.ArgumentParser("Stage1 Mask to Projection", add_help=True)
    parser.add_argument("--input_blend_path", type=str, required=True)
    parser.add_argument("--streetview_locs_path", type=str, required=True)
    parser.add_argument("--solve_result_path", type=str, required=True)
    parser.add_argument("--input_gsv_dir", type=str, required=True)
    parser.add_argument("--cache_root", type=str, required=True)
    parser.add_argument("--blender_save_path", type=str, required=True)
    parser.add_argument("--boundary_mask_dir", type=str, required=True)
    parser.add_argument("--tag", type=str, required=False, default='test-exp')
    args = parser.parse_args()

    work_dir = os.path.join(args.cache_root, "projection")
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(args.boundary_mask_dir, exist_ok=True)
    bpy.ops.wm.open_mainfile(filepath=args.input_blend_path)
    solve_result = pickle.load(open(args.solve_result_path, "rb"))
    projection_street_view(solve_result=solve_result, streetview_locs_path=args.streetview_locs_path,
                           input_gsv_dir=args.input_gsv_dir, output_blend_path=args.blender_save_path, boundary_mask_dir=args.boundary_mask_dir,
                           # fov=90.0, image_width=2048, image_height=1536, y_offset=1.6, camera_step=60,
                           fov=90.0, image_width=512, image_height=384, y_offset=1.6, y_offset_boundary=3, camera_step=60,
                           work_dir=work_dir)

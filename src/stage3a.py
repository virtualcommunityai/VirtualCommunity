import argparse
import pickle
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from stage3 import projection, solve


if __name__ == "__main__":
    if "--" not in sys.argv:
        pass
    else:
        sys.argv = [""] + sys.argv[sys.argv.index("--") + 1:]

    parser = argparse.ArgumentParser("Stage1 Mask to Projection", add_help=True)
    parser.add_argument("--input_blend_path", type=str, required=True)
    parser.add_argument("--streetview_locs_path", type=str, required=True)
    parser.add_argument("--output_solve_result_path", type=str, required=True)
    parser.add_argument("--output_blend_path", type=str, required=True)

    args = parser.parse_args()

    solve_result = solve(input_blend_path=args.input_blend_path, streetview_locs_path=args.streetview_locs_path,
                         fov=90.0, normal_campos_cosine_thres=(0.5, 0.2), normal_cam_direction_cosine_thres=(0.85, 0.5),
                         camera_angles=(0, 60, 120, 180, 240, 300), output_blend_path=args.output_blend_path)
    pickle.dump(solve_result, open(args.output_solve_result_path, "wb"))

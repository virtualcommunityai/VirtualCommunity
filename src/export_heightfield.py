import pickle
import numpy as np

import os, sys
dir = os.path.dirname(__file__)
if not dir in sys.path: sys.path.append(dir)

from blenderlib import CoordSystem, AssertLiteralType

def main(ref_pts: str, save_as: str):
    coord: CoordSystem = "Y+"
    AssertLiteralType(coord, CoordSystem)
    
    if coord == "Y+":
        alt_axis = 1
        gnd_axis = [0, 2]
    else:
        alt_axis = 2
        gnd_axis = [0, 1]
    
    with open(ref_pts, "rb") as fb: ref_points = pickle.load(fb)
    
    plane_coord  = ref_points[..., gnd_axis]
    terrain_alt  = ref_points[..., alt_axis] + 100. 
    # We move everything above by 100.0 meter to (try to) ensure positive
    # height for all outdoor scenes. Negative heights are used for indoor scenes.
    
    np.savez(save_as, plane_coord=plane_coord, terrain_alt=terrain_alt)
    print("\aDone.")



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref"     , type=str, required=True, help="Name of heightfield to save as")
    parser.add_argument("--save_as"     , type=str, required=True, help="Name of heightfield to save as")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])

    main(args.ref, args.save_as)

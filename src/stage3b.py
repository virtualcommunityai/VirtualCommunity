import argparse
import pickle
import sys
import os
from tqdm import tqdm
import shutil
import google_streetview.api


def download_panos(camera_infos: list, save_to: str, apikey: str, fov: float = 90):
    api_cost = 0
    for pano_id in tqdm(camera_infos):
        headings = camera_infos[pano_id]
        pano_dir = os.path.join(save_to, pano_id)
        if os.path.isdir(pano_dir):
            all_exists = True
            for heading in headings:
                if not os.path.exists(os.path.join(pano_dir, f'heading_{heading}.jpg')):
                    all_exists = False
                    break
            if all_exists:
                continue  # This panorama is already downloaded!

        params = [
            {
                "size": "512x384",
                "pano": pano_id,
                "heading": heading,
                "pitch": "0",
                "key": apikey,
                "return_error_code": "true",
                "source": "outdoor",
                "fov": str(fov)
            } for heading in headings
        ]
        api_cost += len(headings)
        result = google_streetview.api.results(params)

        os.makedirs(pano_dir, exist_ok=True)
        result.download_links(str(pano_dir))
        for i in range(len(headings)):
            src_file = os.path.join(pano_dir, f'gsv_{i}.jpg')
            dst_file = os.path.join(pano_dir, f'heading_{headings[i]}.jpg')
            assert os.path.exists(src_file)
            shutil.move(src_file, dst_file)
    print(f"Actual total API cost {api_cost*7/1000}")

if __name__ == "__main__":
    if "--" not in sys.argv:
        pass
    else:
        sys.argv = [""] + sys.argv[sys.argv.index("--") + 1:]

    parser = argparse.ArgumentParser("Stage1 Mask to Projection", add_help=True)
    parser.add_argument("--solve_result_path", type=str, required=True)
    parser.add_argument("--api_key_path", type=str, required=True)
    parser.add_argument("--output_gsv_dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_gsv_dir, exist_ok=True)
    solve_result = pickle.load(open(args.solve_result_path, "rb"))
    camera_set = set()
    for building in solve_result:
        camera_set.update(set(solve_result[building].keys()))
    apikey = open(args.api_key_path).readline()
    camera_infos = {}
    for camera in camera_set:
        camera_id, heading = "_".join(camera.split("_")[:-1]), camera.split("_")[-1]
        if camera_id not in camera_infos:
            camera_infos[camera_id] = []
        camera_infos[camera_id].append(heading)
    download_panos(camera_infos=camera_infos, save_to=args.output_gsv_dir, apikey=apikey, fov=90)
    with open(os.path.join(args.output_gsv_dir, "done.txt"), "w") as f:
        f.write("Done")

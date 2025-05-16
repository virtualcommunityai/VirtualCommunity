import json
from .download_src.tile_api import TileApi
from .download_src.bounding_volume import Square
from pathlib import Path
from tqdm import tqdm
from .utils import calculate_bounding_box


def download_3d_tiles(api_key, lat, lng, radius, output_dir, working_dir):
    api = TileApi(key=api_key, working_dir=working_dir)
    print("Traversing tile hierarchy...")
    lat_min, lat_max, lng_min, lng_max = calculate_bounding_box(lat, lng, radius)
    tiles = list(tqdm(api.get(Square(lat_min, lat_max, lng_min, lng_max))))

    outdir_path = Path(output_dir)
    print("Downloading tiles...")
    for i, t in tqdm(enumerate(tiles), total=len(tiles)):
        t.basename = t.basename[:30] + str(i)
        with open(outdir_path / Path(f"{t.basename}.json"), "w") as f:
            json.dump(t.boundingVolume, f)
        with open(outdir_path / Path(f"{t.basename}.glb"), "wb") as f:
            f.write(t.data)

import json
import pdb
import pickle
import numpy as np
import google_streetview.api
import os
from tqdm import tqdm
from pathlib import Path
from .utils import calculate_bounding_box
import requests


def retrieve_patch_pano_info(long_rng, lat_rng, key, step_cnt=50):
    long_range = np.linspace(long_rng[0], long_rng[1], num=step_cnt)
    lat_range = np.linspace(lat_rng[0], lat_rng[1], num=step_cnt)
    
    metadata = []
    pano_ids = set()
    
    for lat in tqdm(lat_range):
        params = [{
            'size': '512x384',
            'location': f'{lat},{lng}',
            'heading': '0',
            'pitch': '0',
            'key': key,
            'return_error_code': 'true',
            'source': 'outdoor',
            'radius': '50'
        }
        for lng in long_range]
        results = google_streetview.api.results(params)
        metas = results.metadata
        metas = [m for m in metas if (m["status"] == "OK" and m["pano_id"] not in pano_ids)]
        pano_ids.update({m["pano_id"] for m in metas})
        metadata.extend(metas)
    
    return metadata

def download_panos(key, panoids, save_to, fov=90, headings=(0, 90, 180, 270)):
    assert Path(save_to).exists(), "Provided path tosave panos does not exist."
    
    for panoid in tqdm(panoids):
        pano_dir = Path(save_to, panoid)
        exist = True
        headings_new = []
        for idx in range(len(headings)):
            if not os.path.exists(os.path.join(pano_dir, f'gsv_{str(idx)}.jpg')):
                exist = False
                headings_new.append(headings[idx])
        if exist:
            continue
        params = [
            {
                "size": "512x384",
                "pano": panoid,
                "heading": str(heading),
                "pitch": "0",
                "key": key,
                "return_error_code": "true",
                "source": "outdoor",
                "fov": str(fov)
            }
            for heading in headings_new
        ]
        result = google_streetview.api.results(params)

        if not os.path.isdir(pano_dir):
            pano_dir.mkdir(parents=True)
        result.download_links(str(pano_dir))

def batch_convert_xyz(metadatas: list, cvt_fn) -> None:
    """Convert the location from latitude,longtitude in metadata to a normalized xyz coordinate system.
    
    Args:
        metadatas (list): the metadata for each panorama (streetview)

    Returns:
        None.
        
        The metadata are updated in-place for each panorama. x, y, z are provided in meta["location"]["x"], ...
        separately.
    """
    pos = np.empty((len(metadatas), 3))
    for idx in range(len(metadatas)):
        lat, lng = metadatas[idx]["location"]["lat"], metadatas[idx]["location"]["lng"]
        # x, y, z = latlon_to_xyz(lat, lng)
        x, y, z = cvt_fn(lat, lng, LATITUDE_RNG[0], LONGTITUDE_RNG[0])
        pos[idx, 0] = x
        pos[idx, 1] = y
        pos[idx, 2] = z
    pos -= pos.mean(axis=0, keepdims=True)
    
    for idx in range(len(metadatas)):
        metadatas[idx]["location"]["xyz"] = (pos[idx, 0], pos[idx, 1], pos[idx, 2])

def build_connectivity_graph(xyz_metadatas: list, distance_thresh=0.75):
    locs = np.array([meta["location"]["xyz"] for meta in xyz_metadatas])
    for idx in range(len(xyz_metadatas)):
        dist = np.linalg.norm(locs - locs[idx:idx+1], axis=1)
        nn_dist = min(distance_thresh, dist[dist > 0].min())
        is_neighbor = dist < nn_dist * 1.25
        is_neighbor[idx] = False
        neighbors,  = np.where(is_neighbor)
        xyz_metadatas[idx]["neighbors"] = [xyz_metadatas[nid]["pano_id"] for nid in neighbors]


def get_direction(pano_list, api_key, oup_dir):
    session_token = get_session_token(api_key)
    if not session_token:
        return
    meta_data_dict = {}
    for pano_id in tqdm(pano_list):
        pano_ids_url = f"https://tile.googleapis.com/v1/streetview/metadata?session={session_token}&key={api_key}&panoId={pano_id}"
        response = requests.get(pano_ids_url)
        meta_data_dict[pano_id] = response.json()
        if pano_id != "meta.json":
            json.dump(meta_data_dict[pano_id], open(f"{oup_dir}/{pano_id}/meta.json", "w"))
    return meta_data_dict


def get_session_token(key):
    url = f"https://tile.googleapis.com/v1/createSession?key={key}"
    payload = {
        "mapType": "streetview",
        "language": "en-US",
        "region": "US"
    }
    response = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
    assert response.status_code == 200
    session_data = response.json()
    return session_data['session']


def download_streetview_metadata(lat, lng, rad, out_pano_info, out_pano_meta, key, out_2d_dir, step_count=50):
    if os.path.isfile(out_pano_meta):
        with open(out_pano_meta, "r") as f:
            return json.load(f)
    if os.path.isfile(out_pano_info):
        with open(out_pano_info, "rb") as fb:
            metas = pickle.load(fb)
    else:
        lat_min, lat_max, lng_min, lng_max = calculate_bounding_box(lat, lng, rad)
        metas = retrieve_patch_pano_info((lng_min, lng_max), (lat_min, lat_max), key=key, step_cnt=step_count)
    # download_panos(key, [meta['pano_id'] for meta in metas], save_to=out_2d_dir, fov=90, headings=(0, 90, 180, 270))
    full_metadata_dict = {}
    session_token = get_session_token(key)
    for meta in tqdm(metas, desc='Fetching meta jsons...'):
        pano_ids_url = f"https://tile.googleapis.com/v1/streetview/metadata?session={session_token}&key={key}&panoId={meta['pano_id']}"
        response = requests.get(pano_ids_url)
        full_metadata_dict[meta['pano_id']] = response.json()
    with open(out_pano_meta, "w") as fout:
        json.dump(full_metadata_dict, fout)
    return full_metadata_dict
import argparse
import pdb

import requests
import os
import json
import time
from tqdm import tqdm
import math


def meters_to_degrees(lat, rad):
    """Convert radius in meters to degrees for latitude and longitude."""
    lat_deg = rad / 111000  # 1 degree latitude ≈ 111,000 meters
    lng_deg = rad / (111000 * abs(math.cos(math.radians(lat))))  # Adjust for longitude based on latitude
    return lat_deg, lng_deg


def calculate_bbox(lat, lng, rad):
    """Calculate bounding box based on lat, lng, and rad in meters."""
    lat_deg, lng_deg = meters_to_degrees(lat, rad)
    return f'{lng - lng_deg},{lat - lat_deg},{lng + lng_deg},{lat + lat_deg}'


def load_access_token(token_file):
    """Load access token from file."""
    with open(token_file, 'r') as f:
        return f.read().strip()


def download_pano_images_mapillary(lat, lng, rad, token_file, image_dir, geojson_path):
    """Download pano images and save metadata in geojson format."""
    access_token = load_access_token(token_file)
    bbox = calculate_bbox(lat, lng, rad)
    os.makedirs(image_dir, exist_ok=True)

    geojson_data = {"type": "FeatureCollection", "features": []}
    image_search_url = f'https://graph.mapillary.com/images?bbox={bbox}&is_pano=true&access_token={access_token}&fields=id,geometry,compass_angle,captured_at,altitude,computed_rotation,computed_geometry,computed_compass_angle,computed_altitude,camera_type,camera_parameters,thumb_original_url'

    while image_search_url:
        response = requests.get(image_search_url)

        if response.status_code == 200:
            data = response.json()
            pano_images = data.get('data', [])
            next_url = data.get('paging', {}).get('next')
            print(f"Retrieved {len(pano_images)} images. Next page: {bool(next_url)}")

            # Add current images to geojson
            for image in tqdm(pano_images):
                image_id = image['id']
                thumb_url = image.get('thumb_original_url')

                geojson_data["features"].append(image)

                # Download images
                if thumb_url:
                    image_path = os.path.join(image_dir, f'{image_id}.jpg')
                    if os.path.exists(image_path):
                        print(f'Image {image_id} already exists, skipping...')
                        continue

                    success = False
                    retries = 0
                    max_retries = 10

                    while not success and retries < max_retries:
                        try:
                            img_response = requests.get(thumb_url)
                            if img_response.status_code == 200:
                                with open(image_path, 'wb') as img_file:
                                    img_file.write(img_response.content)
                                success = True
                                print(f'Image {image_id} saved.')
                            else:
                                print(f'Failed to download {image_id}: {img_response.status_code}')
                        except Exception as e:
                            print(f'Error downloading {image_id}: {e}')

                        if not success:
                            retries += 1
                            print(f'Retrying {image_id} ({retries}/{max_retries})...')
                            time.sleep(1)

                    if retries == max_retries:
                        print(f'Failed to download {image_id} after {max_retries} retries.')

            # Update URL for next page
            image_search_url = next_url
        else:
            print(f'Failed to retrieve images: {response.status_code}')
            break

    # Save geojson data
    with open(geojson_path, 'w') as geojson_file:
        json.dump(geojson_data, geojson_file, indent=4)
        print(f'GeoJSON metadata saved to {geojson_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download panoramic images and metadata from Mapillary.")
    parser.add_argument('--lat', type=float, required=True, help="Latitude of the bounding box center.")
    parser.add_argument('--lng', type=float, required=True, help="Longitude of the bounding box center.")
    parser.add_argument('--rad', type=float, required=True, help="Bounding box radius in meters.")
    parser.add_argument('--token_file', type=str, required=True, help="File containing the Mapillary access token.")
    parser.add_argument('--image_dir', type=str, default='pano_images', help="Directory to save downloaded images.")
    parser.add_argument('--geojson_path', type=str, default='panoramic_images.geojson',
                        help="Path to save GeoJSON metadata.")
    args = parser.parse_args()

    download_pano_images_mapillary(args.lat, args.lng, args.rad, args.token_file, args.image_dir, args.geojson_path)

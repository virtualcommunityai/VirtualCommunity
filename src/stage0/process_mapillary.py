import os
import json
import pdb

import cv2
import argparse
from .Equirec2Perspec import Equirec2Perspec


def process_mapillary_images(input_dir, output_dir, geojson_path):
    # Load geojson file
    with open(geojson_path, 'r') as f:
        geojson_data = json.load(f)

    # Create a mapping of pano ID to metadata for quick lookup
    pano_metadata = {str(feature['id']): feature for feature in geojson_data['features']}

    # Get all JPG files in the input directory
    jpg_files = [f for f in os.listdir(input_dir) if f.endswith('.jpg')]

    for jpg_file in jpg_files:
        # Extract pano ID from the filename (assuming it's the name without extension)
        pano_id = os.path.splitext(jpg_file)[0]

        # Create a separate output folder for each input image
        output_folder = os.path.join(output_dir, pano_id)
        os.makedirs(output_folder, exist_ok=True)

        # Load the equirectangular image
        equ = Equirec2Perspec.Equirectangular(os.path.join(input_dir, jpg_file))

        # Iterate over theta values: 0, 60, 120, 180, 240, 300
        for idx, theta in enumerate([0, 60, 120, 180, 240, 300]):
            output_path = os.path.join(output_folder, f"{idx}.jpg")

            # Generate perspective image
            img = equ.GetPerspective(90, theta, 0, 1024, 1024)

            # Save the output image
            cv2.imwrite(output_path, img)
            print(f"Saved: {output_path}")

        # Retrieve metadata for the current pano ID
        if pano_id in pano_metadata:
            meta_path = os.path.join(output_folder, 'metadata.json')
            metadata = pano_metadata[pano_id]
            metadata = [{
                'location': {'lat': metadata['geometry']['coordinates'][1],
                             'lng': metadata['geometry']['coordinates'][0]},
                'heading': metadata['compass_angle'],
                '_file': f'gsv_{str(i)}.jpg'
            } for i in range(6)]

            # Write metadata to meta.json
            with open(meta_path, 'w') as meta_file:
                json.dump(metadata, meta_file, indent=4)
                print(f"Saved metadata: {meta_path}")
        else:
            print(f"Metadata not found for pano ID: {pano_id}")

    full_metadata_dict = {}
    for _dir in os.listdir(output_dir):
        if os.path.isfile(os.path.join(output_dir, _dir, 'metadata.json')):
            metadata = json.load(open(os.path.join(output_dir, _dir, 'metadata.json'), 'r'))[0]
            full_metadata_dict[_dir] = {
                'lat': metadata['location']['lat'],
                'lng': metadata['location']['lng'],
                'heading': metadata['heading'],
                'tilt': 0.,
                'roll': 0.}
        for _file in os.listdir(os.path.join(output_dir, _dir)):
            if _file.endswith("jpg") and len(_file) < 6:
                os.renames(os.path.join(output_dir, _dir, _file), os.path.join(output_dir, _dir, f"gsv_{_file}"))
    return full_metadata_dict


if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Generate perspective images from equirectangular images and save metadata.")
    parser.add_argument('--input_dir', type=str, required=True,
                        help="Path to the input directory containing .jpg files.")
    parser.add_argument('--output_dir', type=str, required=True, help="Path to the output directory.")
    parser.add_argument('--geojson_path', type=str, required=True, help="Path to the panoramic_images.geojson file.")
    args = parser.parse_args()

    # Process images and metadata
    process_mapillary_images(args.input_dir, args.output_dir, args.geojson_path)

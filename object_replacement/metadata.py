import requests
import argparse
import cv2
import json
import os

def get_image_metadata(panoId):
    key = 'AIzaSyCvYRt2o27spVSDXCTkqjbXfN0IiBNEPFE'
    url = f"https://tile.googleapis.com/v1/createSession?key={key}"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "mapType": "streetview",
        "language": "en-US",
        "region": "US"
    }
    response = requests.post(url, headers=headers, json=data)
    session_token = response.json()['session']
    metadata_url = f"https://tile.googleapis.com/v1/streetview/metadata?session={session_token}&key={key}&panoId={panoId}"

    response = requests.get(metadata_url)
    metadata = response.json()

    # print(metadata)
    return [metadata['lat'], metadata['lng']]

def get_image_metadata_with_latlng(lat, lng, output_dir, heading=0):
    key = 'AIzaSyCvYRt2o27spVSDXCTkqjbXfN0IiBNEPFE'
    url = f'https://maps.googleapis.com/maps/api/streetview/metadata?location={lat},{lng}&key={key}&size=512x384&heading={heading}'
    response = requests.get(url).content
    response = response.decode()
    print("image_metadata:",response)
    response = json.loads(response)
    lat, lng, panoId = response['location']['lat'], response['location']['lng'], response['pano_id']
    image_url = f'https://maps.googleapis.com/maps/api/streetview?location={lat},{lng}&key={key}&size=512x384&heading={heading}'
    image = requests.get(image_url).content
    with open(os.path.join(output_dir, "streetview.jpg"), 'wb') as file:
        file.write(image)
    return lat, lng, panoId
    
def get_image_metadata_with_panoId(panoId, output_dir, heading=0):
    key = 'AIzaSyCvYRt2o27spVSDXCTkqjbXfN0IiBNEPFE'
    url = f'https://maps.googleapis.com/maps/api/streetview/metadata?pano={panoId}&key={key}&size=512x384&heading={heading}'
    response = requests.get(url).content
    response = response.decode()
    print("image_metadata:",response)
    response = json.loads(response)
    lat, lng = response['location']['lat'], response['location']['lng']
    image_url = f'https://maps.googleapis.com/maps/api/streetview?pano={panoId}&key={key}&size=512x384&heading={heading}'
    image = requests.get(image_url).content
    with open(os.path.join(output_dir, "streetview.jpg"), 'wb') as file:
        file.write(image)
    return lat, lng

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Metadata of Image')
    parser.add_argument('--panoId', type=str, default='x4tG1WRPDYM-ENtwqdQfJg')
    args = parser.parse_args()


    location = get_image_metadata(args.panoId)
    print(location)

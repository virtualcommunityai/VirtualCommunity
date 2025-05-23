# Scene Generation Pipelie

## Install
* Install Blender - https://www.blender.org/download/
* Install Blosm on Blender - `install/install_blosm.py`
* Install Inpaint Anything under `inpainting/`: https://github.com/geekyutao/Inpaint-Anything/tree/main
* Install `upscayl`

## Run
* For example: `./pipeline_unity.sh --radius 400 --dataroot data/ --cacheroot cache/ --datapoint datapoint/MIT.txt`. (You can change `MIT.txt` to any another datapoint under `datapoints/`)
* Currently `Inpainting Street Views` stage can not run in the same env as main pipeline. Need to run it separately in another env.

## Stage 0 - Preprocessing

* Stage 0a - Fetch, merge, and align Google 3D tiles.
* Stage 0b - Fetch street view meta data.
* Stage 0c - Align 3D data, 2D data and OSM data.

## Stage 1 - Preliminary Steps

* Stage 1a - Fetch and filter OSM geometry primitive from the Blosm plugin using Overpass API
* Stage 1b - Construct terrain geometry based on the 3D data.

## Stage 2 - Texture Transfer

* Stage 2a - Transfer texture from masked 3D tiles to terrain geometry constructed at `stage 1b`.
* Stage 2b - Transfer texture from masked 3D tiles to OSM geometry fetch & constructed at `stage 1a`.

## Stage 3 - Texture Enhancement with Street Views

* Stage 3a - Process street view meta data and solve a mapping from each building face to street view image.
* Stage 3b - Fetch the required street view images according to the solve results.
* Inpainting Street Views - Using Lang_SAM to mask the trees / pedestrians / vehicles in street view images and doing inpainting.
* Stage 3c - Project inpainted street view images on buildings.
* Inpainting Textures - Inpaint the missing textures of buildings and ground.

## Stage 4 - Combine Building and Terrain
* Stage 4 - Combine buildings, roof, and terrain to a single blender file.

## Stage 5 - Upscale the scene
* Stage 5 - Applying super resolution to the textures of buildings and terrain.

## Stage 6 - Retrieve Map Annotations
* Stage 6 - Get building names and 3D AABB Json annotations from OSM.

## Stage 7 - Postprocess and Export
* Stage 7 - Convert the emissive texture to basic. Export the `glb` 3D models (`building.glb`, `terrain.glb`, and `roofs.glb`).
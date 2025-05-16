import bpy
import os
import urllib.request
import argparse

if __name__ == "__main__":
    # Step 1: Parse arguments
    parser = argparse.ArgumentParser(description="Install and configure blender-osm addon")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to the data directory")
    args = parser.parse_args()

    data_dir = args.data_dir

    # Step 2: Download blosm.zip
    blosm_path = "https://github.com/vvoovv/blosm/releases/download/v2.4.21/blender-osm.zip"
    addon_zip = os.path.join(data_dir, "blender-osm.zip")

    print("Downloading blosm addon...")
    if not os.path.exists(addon_zip):
        urllib.request.urlretrieve(blosm_path, addon_zip)
        print(f"Downloaded blosm to {addon_zip}")
    else:
        print(f"blosm already exists at {addon_zip}")

    # Step 3: Install the addon
    print("Installing blosm addon...")
    bpy.ops.preferences.addon_install(filepath=addon_zip)
    bpy.ops.preferences.addon_enable(module="blosm")

    # Save user preferences to ensure the addon remains enabled
    bpy.ops.wm.save_userpref()
    print("blosm installed and enabled!")

    # Step 4: Set data directory
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created data directory at: {data_dir}")

    addon_prefs = bpy.context.preferences.addons["blosm"].preferences
    addon_prefs.dataDir = data_dir
    bpy.ops.wm.save_userpref()

    print(f"blosm data directory set to: {data_dir}")

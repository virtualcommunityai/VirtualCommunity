import bpy
import math
import numpy as np
from pyproj import Transformer
import base64
import json
import os
from urllib.parse import urlparse, parse_qs
import requests


WGS84_A = 6378137.0  # semi-major axis
WGS84_B = 6356752.3  # semi-minor axis


def cartesian_from_radians(lon, lat, height=0.):
    WGS84_RADII_SQUARED = np.array([
        6378137.0,
        6378137.0,
        6356752.3142451793,
    ])**2
    cos_lat = np.cos(lat)
    N = np.array([
        cos_lat * np.cos(lon),
        cos_lat * np.sin(lon),
        np.sin(lat),
    ])
    N /= np.linalg.norm(N)
    K = WGS84_RADII_SQUARED * N
    gamma = np.sqrt(N.dot(K))
    K /= gamma
    N *= height

    return K + N


def cartesian_from_degrees(lon, lat, height=0.):
    return cartesian_from_radians(np.deg2rad(lon), np.deg2rad(lat), height)


def _parse(root, target_volume):
    assert "contents" not in root, "contents array not supported"

    if "children" in root:
        for child in root["children"]:
            bv = OrientedBoundingBox.from_tilespec(child["boundingVolume"])
            if target_volume.intersects(bv):
                yield from _parse(child, target_volume)
    elif "content" in root:
        yield (root["content"], root["boundingVolume"])


class TileApi:
    def __init__(self, key, working_dir, api="https://tile.googleapis.com"):
        self.key = key
        self.api = api
        self.session = None
        self.json_counter = 0
        self.working_dir = working_dir

    def get(self, target_volume, uri="/v1/3dtiles/root.json", boundingVolume=None):
        fetcher = lambda: requests.get(
            f"{self.api}{uri}",
            params={'key': self.key, 'session': self.session if uri != "/v1/3dtiles/root.json" else None},
        )

        # We got a glTF tile. Don't immediately download it, but end the recursion here.
        if uri.endswith(".glb"):
            yield Tile(uri=uri, download_thunk=fetcher, boundingVolume=boundingVolume)
            return

        response = fetcher()

        if not response.ok:
            raise RuntimeError(f"response not ok: {response.status_code}, {response.text}")

        content_type = response.headers.get("content-type")
        if content_type != "application/json":
            raise RuntimeError(f"expected JSON response but got {content_type}")

        data = response.json()
        json.dump(data, open(f"{self.working_dir}/{str(self.json_counter)}.json", "w"))
        self.json_counter += 1

        # Parse response
        for content, boundingVolume in _parse(data["root"], target_volume):
            if "uri" in content:
                uri = urlparse(content["uri"])
                # Update session ID from child URI (this is usually only returned as the URI of the
                # child of the initial root request)
                self.session = parse_qs(uri.query).get("session", [self.session])[0]
                # Recurse into child tiles
                yield from self.get(target_volume, uri.path, boundingVolume=boundingVolume if uri.path.endswith(".glb") else None)
            else:
                raise RuntimeError(f"unsupported content: {content}")


class Tile:
    def __init__(self, uri=None, data=None, download_thunk=None, boundingVolume=None):
        self.uri = uri
        self._data = data
        self.basename = uri.rsplit('/', 1)[-1][:-4]
        self.name = base64.decodebytes(f"{self.basename}==".encode()).decode("utf-8")
        self._download = download_thunk
        self.boundingVolume = boundingVolume

    def __repr__(self):
        is_downloaded = "pending" if data is None else "downloaded"
        return f"<Tile:{self.name}:{is_downloaded}>"

    def download(self):
        if not self._data:
            self._data = self._download().content

    @property
    def data(self):
        self.download()
        return self._data


class OrientedBoundingBox:
    def __init__(self, vertices):
        assert vertices.shape == (8, 3)
        self.vertices = vertices

    @staticmethod
    def from_tilespec(spec, eps=1e-2):
        assert "box" in spec
        center = np.array(spec["box"][:3])
        halfx = np.array(spec["box"][3:6])
        halfy = np.array(spec["box"][6:9])
        halfz = np.array(spec["box"][9:12])

        return OrientedBoundingBox(np.stack((
            center - halfx - halfy - halfz,
            center + halfx - halfy - halfz,
            center + halfx + halfy - halfz,
            center - halfx + halfy - halfz,
            center - halfx - halfy + halfz,
            center + halfx - halfy + halfz,
            center + halfx + halfy + halfz,
            center - halfx + halfy + halfz,
        )))


class Sphere:
    def __init__(self, center, r):
        self.center = np.array(center)
        self.r = r

    @staticmethod
    def from_obb(obb):
        return Sphere(
            0.5 * (obb.vertices[0] + obb.vertices[6]),
            0.5 * np.linalg.norm(obb.vertices[6] - obb.vertices[0])
        )

    def intersects(self, other):
        """Sphere intersection test. WARNING: If other is an OBB, then the OBB is first
        approximated using a sphere."""

        if isinstance(other, OrientedBoundingBox):
            return self.intersects(Sphere.from_obb(other))

        if not isinstance(other, Sphere):
            raise TypeError("unsupported type")

        return np.linalg.norm(other.center - self.center) < self.r + other.r


class Square:
    def __init__(self, lat_min, lat_max, lng_min, lng_max):
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lng_min = lng_min
        self.lng_max = lng_max

        # Initialize the transformers for coordinate conversion
        self.transformer_to_lla = Transformer.from_crs("epsg:4978", "epsg:4326")  # ECEF to LLA
        self.transformer_to_ecef = Transformer.from_crs("epsg:4326", "epsg:4978")  # LLA to ECEF

        center_lat = (self.lat_min + self.lat_max) / 2
        center_lng = (self.lng_min + self.lng_max) / 2
        ecef_center = self.to_ecef(center_lat, center_lng)

    def to_lla(self, x, y, z):
        lat, lon, alt = self.transformer_to_lla.transform(x, y, z)
        return np.array([lat, lon, alt])

    def to_ecef(self, lat, lon):
        x, y, z = self.transformer_to_ecef.transform(lat, lon, 0)
        return np.array([x, y, z])

    def intersects(self, obb):
        if not isinstance(obb, OrientedBoundingBox):
            raise TypeError("unsupported type")

        lla_vertices = np.array([self.to_lla(*vertex) for vertex in obb.vertices])
        max_elevation = np.max(np.abs(lla_vertices[:, 2]))

        if max_elevation > 5000:
            # Use ECEF coordinates for intersection check
            center_lat = (self.lat_min + self.lat_max) / 2
            center_lng = (self.lng_min + self.lng_max) / 2
            ecef_center = self.to_ecef(center_lat, center_lng)
            return self.point_in_obb(ecef_center, obb.vertices)
        else:
            # Use LLA coordinates for intersection check
            for lat, lon, _ in lla_vertices:
                if self.lat_min <= lat <= self.lat_max and self.lng_min <= lon <= self.lng_max:
                    return True
            lla_min = lla_vertices.min(axis=0)
            lla_max = lla_vertices.max(axis=0)
            if (lla_min[0] <= self.lat_min <= lla_max[0] and lla_min[1] <= self.lng_min <= lla_max[1]) or \
                    (lla_min[0] <= self.lat_min <= lla_max[0] and lla_min[1] <= self.lng_max <= lla_max[1]) or \
                    (lla_min[0] <= self.lat_max <= lla_max[0] and lla_min[1] <= self.lng_min <= lla_max[1]) or \
                    (lla_min[0] <= self.lat_max <= lla_max[0] and lla_min[1] <= self.lng_max <= lla_max[1]):
                return True

            return False

    def point_in_obb(self, point, obb_vertices):
        # Using Separating Axis Theorem (SAT) to check if a point is inside the OBB
        vertices = obb_vertices
        axes = [
            vertices[1] - vertices[0],
            vertices[3] - vertices[0],
            vertices[4] - vertices[0]
        ]

        point_rel = point - vertices[0]
        projections = [np.dot(point_rel, axis) / np.dot(axis, axis) for axis in axes]

        return all(0 <= projection <= 1 for projection in projections)


def calculate_bounding_box(lat, lng, rad):
    r = rad / 1000.0
    delta_lat = r / 111.32
    delta_lng = r / (111.32 * math.cos(math.radians(lat)))

    lat_max = lat + delta_lat
    lat_min = lat - delta_lat
    lng_max = lng + delta_lng
    lng_min = lng - delta_lng

    return lat_min, lat_max, lng_min, lng_max


def export_scene(output_path):
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(output_path))


def export_glb(mesh_name, output_path):
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH' and obj.name == mesh_name:
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            bpy.ops.export_scene.gltf(
                filepath=output_path,
                export_format='GLB',
                export_materials='NONE',
                use_selection=True
            )

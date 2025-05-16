import os, sys, json 
import utm 
import numpy as np

from typing import Optional

class TerrainMap:
    def __init__(
        self, height_field_file: str, 
        base_lat: Optional[float] = None, 
        base_lon: Optional[float] = None
    ):
        self.base_lat = base_lat
        self.base_lon = base_lon
        self.plane_coord = np.load(height_field_file)['plane_coord']
        self.terrain_alt = np.load(height_field_file)['terrain_alt']

        self.x_coords = self.plane_coord[:, 0]
        self.y_coords = self.plane_coord[:, 1]

    def _bilinear_interpolate(self, lu_idx: int, ru_idx: int, lb_idx: int, rb_idx: int, x: float, y: float):
        x00, y0 = self.x_coords[lu_idx], self.y_coords[lu_idx]
        x01, _ = self.x_coords[ru_idx], self.y_coords[ru_idx]
        x10, y1 = self.x_coords[lb_idx], self.y_coords[lb_idx]
        x11, _ = self.x_coords[rb_idx], self.y_coords[rb_idx]

        z00 = self.terrain_alt[lu_idx]
        z01 = self.terrain_alt[ru_idx]
        z10 = self.terrain_alt[lb_idx]
        z11 = self.terrain_alt[rb_idx]

        def lerp(a: float, b: float, t: float):
            return a + t * (b - a)
        return lerp(
            lerp(z00, z01, (x - x00) / (x01 - x00) if x01 != x00 else 0),
            lerp(z10, z11, (x - x10) / (x11 - x10) if x11 != x10 else 0),
            (y - y0) / (y1 - y0) if y1 != y0 else 0
        )
        
    def get_alt_by_xy(self, x: float, y: float):
        def find_closest_given_y(y_lower: int, x: float):
            y_upper = np.searchsorted(self.y_coords, self.y_coords[y_lower], side='right')
            x_lower = y_lower + np.searchsorted(self.x_coords[y_lower: y_upper], x, side='left')
            x_upper = y_lower + np.searchsorted(self.x_coords[y_lower: y_upper], x, side='right')
            return x_lower, x_upper

        y0_idx = np.searchsorted(self.y_coords, y, side='left')
        y1_idx = np.searchsorted(self.y_coords, y, side='right')
        x00_idx, x01_idx = find_closest_given_y(y0_idx, x)
        x10_idx, x11_idx = find_closest_given_y(y1_idx, x)
        return self._bilinear_interpolate(x00_idx, x01_idx, x10_idx, x11_idx, x, y)

    def get_alt_by_latlon(self, lat: float, lon: float):
        base_e, base_n, _, _ = utm.from_latlon(BASE_LAT, BASE_LON)
        e, n, _, _ = utm.from_latlon(lat, lon)
        return self.get_alt_by_xy(e - base_e, n - base_n)

BASE_LAT = 45.46171045434891
BASE_LON = 9.174386698817148
terrain_map = TerrainMap('height_field.npz', base_lat=BASE_LAT, base_lon=BASE_LON)

path = 'street_view/MILAN/'
metadata = {}

spots = os.listdir(path)
for spot_id in spots:
    if not os.path.isdir(os.path.join(path, spot_id)):
        continue
    for i in range(6):
        id = f'{spot_id}_{i}'
        with open(os.path.join(path, spot_id, f'meta.json')) as f:
            meta = json.load(f)[spot_id]
        metadata[id] = {
            'path': os.path.join(path, spot_id, f'gsv_{i}.jpg'),
            'lon': meta['lng'],
            'lat': meta['lat'],
            'alt': terrain_map.get_alt_by_latlon(meta['lat'], meta['lng']),
            'heading': [0, 60, 120, 180, 240, 300][i],
            'tilt': meta['tilt'],
            'roll': meta['roll'],
            'fov': 90,
            'height': 384,
            'width': 512
        }

with open('metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

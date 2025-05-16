import numpy as np
from pyproj import Transformer

WGS84_A = 6378137.0  # semi-major axis
WGS84_B = 6356752.3  # semi-minor axis


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

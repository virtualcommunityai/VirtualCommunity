"""
Utilities for creating different shape'd mesh for visualization and debugging
"""
from typing import Any
from typing_extensions import Self

import bpy
import mathutils
bpy: Any

from .MeshObject import MeshObject


class BoxAABBMesh(MeshObject):
    @classmethod
    def create(cls, name: str, co_min: tuple[float, float, float], co_max: tuple[float, float, float]) -> Self:
        mesh = bpy.data.meshes.new(name)
        obj = bpy.data.objects.new(name, mesh)
        bpy.context.collection.objects.link(obj)
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        
        # Define vertices based on min and max coordinates
        x_min, y_min, z_min = co_min
        x_max, y_max, z_max = co_max
        vertices = [
            (x_min, y_min, z_min),  # v0
            (x_max, y_min, z_min),  # v1
            (x_max, y_max, z_min),  # v2
            (x_min, y_max, z_min),  # v3
            (x_min, y_min, z_max),  # v4
            (x_max, y_min, z_max),  # v5
            (x_max, y_max, z_max),  # v6
            (x_min, y_max, z_max),  # v7
        ]

        # Define edges for a skeleton structure
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face edges
            (4, 5), (5, 6), (6, 7), (7, 4),  # Top face edges
            (0, 4), (1, 5), (2, 6), (3, 7)   # Side edges
        ]

        # Create mesh from vertices and edges
        mesh.from_pydata(vertices, edges, [])
        mesh.update()
        
        return cls(obj)

    @classmethod
    def aabb_of(cls, other_mesh: MeshObject, name: str | None = None) -> "BoxAABBMesh":
        box = other_mesh.object.bound_box
        cos = [mathutils.Vector(c) for c in box]
        xs  = [c.x for c in cos]
        ys  = [c.y for c in cos]
        zs  = [c.z for c in cos]
        co_min = (min(xs), min(ys), min(zs))
        co_max = (max(xs), max(ys), max(zs))
        if name is None: name = other_mesh.name + "_aabb"
        
        return cls.create(name, co_min, co_max)


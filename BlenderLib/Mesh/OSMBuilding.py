import bpy
from typing import Any
from typing_extensions import Self
bpy: Any

from .MeshObject import MeshObject
from .ShapeMesh  import BoxAABBMesh


bpy.types.Object.is_osm_building = bpy.props.BoolProperty(
    name="Is OSM Building",
    description="True if the object is an OSM building geometry",
    default=False
)
bpy.types.Object.is_osm_node = bpy.props.BoolProperty(
    name="Is OSM Node",
    description="True if the object is an OSM node place holder",
    default=False
)
bpy.types.Object.osm_amenity_tag = bpy.props.StringProperty(
    name="Amenity",
    description="Amenity tag of OpenStreetMap",
    default="No Data"
)
bpy.types.Object.osm_name_tag = bpy.props.StringProperty(
    name="Name",
    description="Name of OpenStreetMap Building",
    default="No Data"
)
bpy.types.Object.osmid = bpy.props.StringProperty(
    name="OSM ID",
    description="ID of OSM element",
    default="No Data"
)


class OSMBUILDING_PT_CustomPanel(bpy.types.Panel):
    bl_label = "OSM Properties"
    bl_idname = "OSMBUILDING_PT_CustomPanel"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "object"
    
    def draw(self, context):
        layout = self.layout
        obj = context.object

        # Display custom properties in the panel
        layout.prop(obj, "is_osm_building")
        layout.prop(obj, "is_osm_node")
        layout.prop(obj, "osmid")
        layout.prop(obj, "osm_amenity_tag")
        layout.prop(obj, "osm_name_tag")


class OSMObject:
    bpy.utils.register_class(OSMBUILDING_PT_CustomPanel)
    
    def __init__(self, mesh, osmid: str | None = None, name: str | None = None, amenity: str | None = None):
        if amenity is not None: mesh.osm_amenity_tag = amenity
        if name is not None   : mesh.osm_name_tag    = name
        if osmid is not None  : mesh.osmid           = osmid
        self.object = mesh

    @property
    def amenity(self) -> str: return self.object.osm_amenity_tag
    @amenity.setter
    def amenity(self, tag: str) -> None: self.object.osm_amenity_tag = tag
    
    @property
    def osm_name(self) -> str: return self.object.osm_name_tag
    @osm_name.setter
    def osm_name(self, tag: str) -> None: self.object.osm_name_tag = tag
    
    @property
    def osmid(self) -> str: return self.object.osmid


class OSMBuildingMesh(OSMObject, MeshObject):
    def __init__(self, mesh, osmid: str | None = None, name: str | None = None, amenity: str | None = None):
        OSMObject.__init__(self, mesh, osmid, name, amenity)
        MeshObject.__init__(self, mesh)
        mesh.is_osm_building = True
    
    @classmethod
    def proxy(cls, mesh) -> Self:
        return cls(mesh, osmid=mesh.osmid, name=mesh.osm_name_tag, amenity=mesh.osm_amenity_tag)

    @classmethod
    def retrieve_all(cls) -> list["OSMBuildingMesh"]:
        results = []
        for obj in bpy.data.objects:
            if obj.type != "MESH": continue
            if hasattr(obj, "is_osm_building") and obj.is_osm_building:
                results.append(cls(obj))
        return results


class OSMNodeMesh(OSMObject, MeshObject):
    Counter = 0
    
    def __init__(self, mesh, osmid: str | None = None, name: str | None = None, amenity: str | None = None):
        OSMObject.__init__(self, mesh, osmid, name, amenity)
        MeshObject.__init__(self, mesh)
        self.object.is_osm_node = True
    
    @classmethod
    def create(cls, loc: tuple[float, float, float], size: float, osmid: str, name: str | None = None, amenity: str | None = None) -> Self:
        obj_name = name
        if obj_name is None:
            obj_name = f"Node_{OSMNodeMesh.Counter}"
            OSMNodeMesh.Counter += 1
        assert obj_name is not None
        
        box_mesh = BoxAABBMesh.create(obj_name, *OSMNodeMesh.aabb_from_center_size(loc, size))
        return cls(box_mesh.object, osmid, name, amenity)
    
    @classmethod
    def proxy(cls, mesh) -> Self:
        return cls(mesh, osmid=mesh.osmid, name=mesh.osm_name_tag, amenity=mesh.osm_amenity_tag)
    
    @staticmethod
    def aabb_from_center_size(center: tuple[float, float, float], size: float):
        half_size = size / 2
        x, y, z   = center
        co_min = (x-half_size, y-half_size, z-half_size)
        co_max = (x+half_size, y+half_size, z+half_size)
        return co_min, co_max

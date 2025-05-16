import typing as T
import bpy
import math
import bmesh
import mathutils
import numpy as np
import numpy.typing as npt

from dataclasses import dataclass
from mathutils.bvhtree import BVHTree
from contextlib import contextmanager, ExitStack

# For custom expansion of blender (bpy.ops.custom.*)
from bpy.types import Operator
#


CoordSystem = T.Literal["Z+", "Y+"]
# Z+ => z-up coordinate system
# Y+ => y-up coordinate system


ArrayCoord = T.Annotated[npt.NDArray[np.float32], T.Literal["N", "N", 3]]
ArrayMask  = T.Annotated[npt.NDArray[np.bool_], T.Literal["N"]]

def AssertLiteralType(value: str | float | int | bool, type: T.Type):
    assert value in T.get_args(type), f"AssertLiteralType failed - expect `value` to be one of {T.get_args(type)}, but get {value}"


class MeshObject:
    def __init__(self, mesh):
        assert mesh is not None and mesh.type == 'MESH', f"Only a proxy for MESH object, but get {mesh.type}"
        self.mesh_object = mesh

    @classmethod
    def withName(cls, name: str) -> "MeshObject":
        return cls(bpy.data.objects[name])

    @classmethod
    def remoteAppend(cls, blender_file: str, mesh_name: str) -> "MeshObject":
        try:
            bpy.ops.object.mode_set(mode="OBJECT")
        except: pass
        mesh_path = blender_file + "\\" + "Object" + "\\" + mesh_name
        bpy.ops.wm.append(
            filepath  = mesh_path, 
            directory = blender_file + "\\" + "Object",
            filename  = mesh_name
        )
        return MeshObject.withName(mesh_name)

    @property
    def is_active(self) -> bool:
        return self.mesh_object == bpy.context.active_object

    def set_active(self) -> None:
        bpy.context.view_layer.objects.active = self.mesh_object

    @property
    def is_select(self) -> bool:
        return self.mesh_object.select_get()

    @is_select.setter
    def is_select(self, mode: bool) -> None:
        return self.mesh_object.select_set(mode)

    @property
    def mode(self) -> T.Literal["OBJECT", "EDIT"]:
        return self.mesh_object.mode

    @mode.setter
    def mode(self, mode: T.Literal["OBJECT", "EDIT"]) -> None:
        assert self.is_active, "Can only change mode of active object"
        bpy.ops.object.mode_set(mode=mode)

    @property
    def name(self) -> str:
        return self.mesh_object.name
    
    @name.setter
    def name(self, new_name: str) -> None:
        self.mesh_object.name = new_name
        self.data.name = new_name

    @property
    def data(self):
        """Convenient proxy property shorthand for object data"""
        return self.mesh_object.data

    @property
    def verts_Tobj(self) -> ArrayCoord:
        """
        Get vertex coordinates under object coordinate frame
        """
        mesh = self.mesh_object.data
        vertices = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", vertices)
        return vertices.reshape(-1, 3)

    @property
    def verts_Tworld(self) -> ArrayCoord:
        mesh = self.mesh_object.data
        world_matrix = self.mesh_object.matrix_world

        local_coords = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", local_coords)
        local_coords = local_coords.reshape(-1, 3)

        local_coords_homogeneous = np.column_stack((local_coords, np.ones(len(local_coords))))
        world_matrix_np = np.array(world_matrix)
        world_coords_homogeneous = np.dot(local_coords_homogeneous, world_matrix_np.T)

        w = world_coords_homogeneous[:, 3]
        world_coords = world_coords_homogeneous[:, :3] / w[:, np.newaxis]
        return world_coords

    @property
    def T_obj2world(self) -> np.ndarray:
        return np.array(self.mesh_object.matrix_world)

    @property
    def surface_area(self) -> float:
        total_area = 0.0
        for p in self.mesh_object.data.polygons: total_area += p.area
        return total_area

    @property
    def aabb_box(self) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
        """Returns (min xyz), (max xyz)"""
        if self.verts_Tworld.size == 0: return None
        min_coord = self.verts_Tworld.min(axis=0)
        max_coord = self.verts_Tworld.max(axis=0)
        return (min_coord[0], min_coord[1], min_coord[2]), (max_coord[0], max_coord[1], max_coord[2])

    def as_BVHTree(self, use_modifiers: bool) -> BVHTree:
        with self.scoped_BMesh(use_modifiers) as bm:
            tree = BVHTree.FromBMesh(bm)
        return tree

    @T_obj2world.setter
    def T_obj2world(self, new_transform: np.ndarray) -> None:
        blender_matrix = mathutils.Matrix(new_transform.tolist())
        self.mesh_object.matrix_world = blender_matrix
        bpy.context.view_layer.update()

    @contextmanager
    def scoped_active(self):
        original_active_obj = bpy.context.active_object
        self.set_active()
        try:
            yield
        finally:
            bpy.context.view_layer.objects.active = original_active_obj

    @contextmanager
    def scoped_mode(self, mode: T.Literal["OBJECT", "EDIT", "TEXTURE_PAINT"], scoped_active: bool):
        original_mode = self.mode
        if scoped_active:
            with self.scoped_active():
                self.mode = mode
                try: yield
                finally: self.mode = original_mode
        else:
            self.set_active()
            self.mode = mode
            try: yield
            finally: self.mode = original_mode

    @contextmanager
    def scoped_BMesh(self, use_modifiers: bool):
        if use_modifiers:
            depsgraph = bpy.context.evaluated_depsgraph_get()
            object_eval = self.mesh_object.evaluated_get(depsgraph)
            me = object_eval.to_mesh()
            bm = bmesh.new()
            bm.from_mesh(me)
        else:
            me = self.mesh_object.data
            bm = bmesh.new()
            bm.from_mesh(me)
        
        try: yield bm
        finally:
            bm.to_mesh(me)
            me.update()
            bm.free()
            if use_modifiers: object_eval.to_mesh_clear()

    @contextmanager
    def scoped_select(self, select: bool=True):
        original_select = self.is_select
        self.is_select = select
        try: yield
        finally:
            self.is_select = original_select

    def copy(self, keep_material: bool=True) -> "MeshObject":
        new_object = self.mesh_object.copy()
        new_object.data = self.mesh_object.data.copy()
        bpy.context.collection.objects.link(new_object)
        
        if not keep_material:
            new_object.data.materials.clear()
        
        return MeshObject(new_object)

    def delete(self):
        print("Remove mesh", self.mesh_object.name)
        bpy.data.objects.remove(self.mesh_object, do_unlink=True)

    def clear_material(self):
        with self.scoped_mode("OBJECT", True):
            for _ in range(1,len(self.mesh_object.material_slots)):
                self.mesh_object.active_material_index = 1
                bpy.ops.object.material_slot_remove()
            
            with self.scoped_mode("EDIT", True):
                bpy.ops.mesh.select_all(action = 'SELECT')
                bpy.ops.object.material_slot_assign()

            bpy.ops.object.material_slot_remove_unused()

    def displace(self, direction: T.Literal["X", "Y", "Z", "NORMAL", "CUSTOM_NORMAL", "RGB_TO_XYZ"], distance: float) -> None:
        with self.scoped_mode("OBJECT", True):
            disp_mod = self.mesh_object.modifiers.new(name=f"Disp", type='DISPLACE')
            disp_mod.direction = direction
            disp_mod.strength = distance
            bpy.ops.object.modifier_apply(modifier=disp_mod.name)

    def decimate(self, type: T.Literal["COLLAPSE", "UNSUBDIV", "DISSOLVE"], iteration: int) -> None:
        with self.scoped_mode("OBJECT", True):
            deci_mod = self.mesh_object.modifiers.new(name=f"Decimate", type="DECIMATE")
            deci_mod.decimate_type = type
            deci_mod.iterations = iteration
            bpy.ops.object.modifier_apply(modifier=deci_mod.name)
    
    def subdivide(self, iteration: int) -> None:
        with self.scoped_mode("OBJECT", True):
            subs_mod = self.mesh_object.modifiers.new(name=f"Subdivision", type="SUBSURF")
            subs_mod.levels = iteration
            subs_mod.subdivision_type = "SIMPLE"
            bpy.ops.object.modifier_apply(modifier=subs_mod.name)
    
    def triangulate(self, min_vert: int = 4) -> None:
        with self.scoped_mode("OBJECT", True):
            subs_mod = self.mesh_object.modifiers.new(name=f"Subdivision", type="TRIANGULATE")
            subs_mod.min_vertices = min_vert
            bpy.ops.object.modifier_apply(modifier=subs_mod.name)

    def random_sample_on_plane(self, coord: CoordSystem, n_sample: int, world_frame: bool=True) -> tuple[np.ndarray, np.ndarray]:
        from shapely import Point, MultiPoint
        AssertLiteralType(coord, CoordSystem)
        
        verts = self.verts_Tworld if world_frame else self.verts_Tobj
        if coord == "Y+":
            verts = verts[..., [0, 2]]
        else:
            verts = verts[..., :2]
        
        geom = MultiPoint(verts)
        minx, miny, maxx, maxy = geom.bounds 
        x = np.random.uniform(minx, maxx, n_sample)
        y = np.random.uniform(miny, maxy, n_sample)
        mask = np.array([geom.dwithin(Point(cx, cy), 0.0) for cx, cy in zip(x, y)])
        return x[mask], y[mask]

    def cast_ray_on(self, sources: np.ndarray, direction: tuple[float, float, float], distance: float, self_bvh: BVHTree | None, use_modifiers: bool=False) -> tuple[np.ndarray, np.ndarray, BVHTree]:
        """Cast ray ono the mesh from a sequence of sources along the specified direction

        Args:
            sources (np.ndarray): Nx3 np array as the source of ray
            direction (tuple[float, float, float]): direction of ray casting
            distance (float): distance threshold for ray casting, will be invalid after this distance threshold
            self_bvh (BVHTree | None): BVH acceleration structure for ray casting, if not provided, the function will create a new BVHTree
                automatically.
            
            use_modifiers (bool, optional): Whether to apply modifiers when creating BVHTree, will not be used if BVHTree is provided. Defaults to False.

        Returns:
            tuple[np.ndarray, np.ndarray, BVHTree]: 
                * Ray casting positions - Nx3 array
                * Validity Mask - N array with boolean value, True means valid
                * BVHTree - Acceleration structure (from input / newly built), can be used for future calls.
        """
        if self_bvh is None: self_bvh = self.as_BVHTree(use_modifiers)

        queries = [mathutils.Vector((sources[idx, 0], sources[idx, 1], sources[idx, 2]))
                   for idx in range(sources.shape[0])]
        
        result_position = np.zeros_like(sources)
        result_validity = np.zeros((sources.shape[0],), dtype=bool)
        
        for idx, query in enumerate(queries):
            on_mesh_pt, _, _, _ = self_bvh.ray_cast(
                query, mathutils.Vector(direction), distance
            )
            if on_mesh_pt is None:
                result_validity[idx] = False
            else:
                result_validity[idx] = True
                result_position[idx, 0] = on_mesh_pt.x
                result_position[idx, 1] = on_mesh_pt.y
                result_position[idx, 2] = on_mesh_pt.z
        
        return result_position, result_validity, self_bvh

    def apply_transform(self):
        with self.scoped_mode("OBJECT", scoped_active=True), self.scoped_select():
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    def mask(self, vertex_group: "VertexGroup", apply: bool = True, invert: bool = False):
        with self.scoped_select(True), self.scoped_mode("OBJECT", True):
            mask_modifier = self.mesh_object.modifiers.new(name="Mask", type='MASK')
            mask_modifier.vertex_group = vertex_group.name
            # mask_modifier.invert = invert
            if apply: bpy.ops.object.modifier_apply(modifier=mask_modifier.name)
            return mask_modifier

    def delete_loose(self):
        with self.scoped_select(True), self.scoped_mode("EDIT", True):
            bpy.ops.mesh.select_all(action = 'SELECT')
            bpy.ops.mesh.delete_loose()
            bpy.ops.mesh.select_all(action = 'DESELECT')


class VertexGroup:
    def __init__(self, mesh: MeshObject, vg):
        self.mesh = mesh
        self.vg   = vg      # bpy.vertex_group, no type annotation
    
    @classmethod
    def create(cls, mesh: MeshObject, name: str):
        with mesh.scoped_mode("OBJECT", True):
            new_group = mesh.mesh_object.vertex_groups.new(name=name)
        return cls(mesh, new_group)
    
    @property
    def name(self) -> str:
        return self.vg.name
    
    @property
    def verts_id(self) -> list[int]:
        vertex_indices = []
        for i, vertex in enumerate(self.mesh.data.vertices):
            for group_element in vertex.groups:
                if group_element.group == self.vg.index:
                    vertex_indices.append(i)
                    break  # We've found the group for this vertex, no need to continue inner loop
        return vertex_indices
    
    def add(self, pred: T.Callable[[ArrayCoord,], ArrayMask]):
        world_coords = self.mesh.verts_Tworld
        verts_idx    = np.array([v.index for v in self.mesh.data.vertices])
        
        # Apply the batched predicate and add to vertex group
        mask = pred(world_coords)
        vertices_to_add = verts_idx[mask]
        self.vg.add(vertices_to_add.tolist(), 1.0, 'ADD')
        print(f"Vertex group added {mask.sum()} vertices")
    
    def add_verts(self, verts: list[int]):
        with self.mesh.scoped_select(), self.mesh.scoped_mode("EDIT", True):
            bpy.ops.mesh.select_all(action='DESELECT')
        with self.mesh.scoped_select(), self.mesh.scoped_mode("OBJECT", True):
            for index in verts:
                self.mesh.data.vertices[index].select = True
        with self.mesh.scoped_select(), self.mesh.scoped_mode("EDIT", True):
            bpy.ops.object.vertex_group_assign()
            
    def expand(self, expand_iter: int):
        with self.mesh.scoped_select(True), self.mesh.scoped_mode("EDIT", True):
            bpy.ops.mesh.select_all(action='DESELECT')
            bpy.ops.mesh.select_mode(type="VERT")
            
            self.mesh.mesh_object.vertex_groups.active_index = self.vg.index
            bpy.ops.object.vertex_group_select()
            for _ in range(expand_iter):
                bpy.ops.mesh.select_more()
    
        with self.mesh.scoped_select(True), self.mesh.scoped_mode("OBJECT", True):
            selected_verts = [v.index for v in self.mesh.data.vertices if v.select]
            self.vg.add(selected_verts, 1.0, 'REPLACE')

    def clean_by_connected_component_size(self, numverts_bound: float):
        new_vg = VertexGroup.create(self.mesh, "Clean")
        
        with self.mesh.scoped_select(), self.mesh.scoped_BMesh(False) as bm:
            bm.verts.ensure_lookup_table()
            bm.edges.ensure_lookup_table()
            all_vert_ids = set(self.verts_id)
            ccs = DisjointSet.retrieve_ccs_bmesh(
                bm, lambda v1, v2: (v1 in all_vert_ids) == (v2 in all_vert_ids)
            )
            clean_idx = set()
        
            with self.mesh.scoped_mode("OBJECT", True):
                remove_counter = 0
                for idx, cc in enumerate(ccs):
                    # print(f"\rFiltering ccs: {idx}/{len(ccs)}", end="", flush=True)
                    if cc.peek() not in all_vert_ids: continue
                    if cc.size < numverts_bound:
                        print(cc.size)
                        self.vg.remove(list(cc.vertices))
                        remove_counter += 1
                    else:
                        print("+", cc.size)
                        clean_idx |= cc.vertices
        
        print(len(clean_idx))
        new_vg.add_verts(list(clean_idx))
        self.mesh.data.update()
        # print(f"Removed {remove_counter} connected components with area smaller than {numverts_bound} verts")
        return new_vg

class PinholeCam:
    def __init__(self, cam, size: tuple[int, int] | None = None):
        assert cam is not None and cam.type == "CAMERA", "PinholeCam must receive a camera, but get a " + cam.type
        self.cam  = cam
        self.size = size
    
    @property
    def name(self) -> str:
        return self.cam.name

    @classmethod
    def withName(cls, name: str, size: tuple[float, float] = (640., 640.)) -> "PinholeCam":
        return cls(bpy.data.objects[name], size)

    @classmethod
    def withSize(cls, name: str, wfov: float, width: int, height: int, loc: tuple[float, float, float]):
        cam_data = bpy.data.cameras.new(name)
        cam_object = bpy.data.objects.new(name, cam_data)
        bpy.context.scene.collection.objects.link(cam_object)
        
        cam_object.location = loc
        cam_data.angle_x = math.radians(wfov)
        return cls(cam_object, (width, height))

    def render_materialpreview(self, save_to: str, animate: bool = False):
        scene = bpy.context.scene
        if self.size is not None:
            scene.render.resolution_x = self.size[0]
            scene.render.resolution_y = self.size[1]
        
        # scene.render.engine = 'BLENDER_EEVEE'
        scene.render.film_transparent = True    # Enable transparent background
        
        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                for space in area.spaces:
                    if space.type == 'VIEW_3D':
                        space.shading.type = 'MATERIAL'
                        space.shading.use_scene_lights = False
                        space.shading.use_scene_world = False
        scene.display.shading.type = 'MATERIAL'
        scene.display.shading.use_scene_lights = False
        scene.display.shading.use_scene_world = False
        
        bpy.context.scene.render.filepath = save_to
        
        with self.scoped_active(), self.scoped_select():
            if animate:
                bpy.ops.render.render(animation=True)
            else:
                bpy.ops.render.render(write_still=True)

    @property
    def is_active(self) -> bool:
        return self.cam == bpy.context.scene.camera

    def set_active(self) -> None:
        bpy.context.scene.camera = self.cam

    @property
    def is_select(self) -> bool:
        return self.cam.select_get()

    @is_select.setter
    def is_select(self, mode: bool) -> None:
        return self.cam.select_set(mode)
    
    def apply_transform(self):
        with self.scoped_active(), self.scoped_select():
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    
    @contextmanager
    def scoped_active(self):
        original_active_obj = bpy.context.scene.camera
        self.set_active()
        try:
            yield
        finally:
            bpy.context.scene.camera = original_active_obj

    @contextmanager
    def scoped_select(self, select: bool=True):
        original_select = self.is_select
        self.is_select = select
        try: yield
        finally:
            self.is_select = original_select

    @classmethod
    def append_all_cameras_in(cls, blend_filename: str) -> "list[PinholeCam]":
        # Change the context to the scene where we want to append
        # original_context = bpy.context.area.type
        # bpy.context.area.type = 'VIEW_3D'
        
        # Deselect all objects
        bpy.ops.object.select_all(action='DESELECT')
        
        # Get the current scene
        scene = bpy.context.scene
        
        # Remember existing cameras to avoid duplicates
        existing_cameras = set(obj.name for obj in bpy.data.objects if obj.type == 'CAMERA')
        
        # Append all cameras from the other file
        with bpy.data.libraries.load(blend_filename) as (data_from, data_to):
            data_to.objects = [name for name in data_from.objects if name not in existing_cameras and bpy.data.objects.get(name) is None]
        
        # Link the appended cameras to the scene
        cams = []
        for obj in data_to.objects:
            if obj.type == 'CAMERA':
                scene.collection.objects.link(obj)
                cams.append(PinholeCam(obj, size=(512, 384)))
                obj.select_set(True)
        
        # Restore the original context
        # bpy.context.area.type = original_context
        
        print(f"Appended {len(data_to.objects)} cameras from {blend_filename}")
        return cams

    def occlusion_test(self, ratio: float=0.1, dist: float=100.):
        scene = bpy.context.scene
        depsgraph = bpy.context.evaluated_depsgraph_get()
        camera = self.cam
        
        res_x = int(self.size[0] * ratio)
        res_y = int(self.size[1] * ratio)
        
        # get vectors which define view frustum of camera
        top_right, _, bottom_left, top_left = camera.data.view_frame(scene=scene)

        camera_quaternion = camera.matrix_world.to_quaternion()
        camera_translation = camera.matrix_world.translation

        x_range = np.linspace(top_left[0], top_right[0], res_x)
        y_range = np.linspace(top_left[1], bottom_left[1], res_y)

        z_dir = top_left[2]

        hit_data = set()

        for x in x_range:
            for y in y_range:
                pixel_vector = mathutils.Vector((x, y, z_dir))
                pixel_vector.rotate(camera_quaternion)
                pixel_vector.normalized()

                is_hit, location, normal, face_id, hit_obj, _ = scene.ray_cast(depsgraph, camera_translation, pixel_vector, distance=dist)

                if is_hit:
                    dist = (location - camera_translation).length
                    hit_data.add((face_id, hit_obj.name, dist, pixel_vector.angle(normal), location.freeze()))

        return hit_data

    def __hash__(self) -> int:
        return hash(self.name)


class FaceData:
    def __init__(self, id: int, data: dict):
        self.id = id
        self.data = data
    
    def __hash__(self) -> int:
        return self.id
    
    def __eq__(self, other: "FaceData") -> bool:
        return self.id == other.id

    def __repr__(self) -> str:
        return f"Face({self.id}, data={self.data})"

FaceType      = tuple[str, FaceData]

class FaceGroup:
    def __init__(self, name: str):
        self.name : str                                  = name
        self.faces: dict[str, set[FaceData]] = dict()
        self.meshs: list[str]                            = []
    
    def add_face(self, face: FaceType) -> None:
        if face[0] not in self.faces:
            self.faces[face[0]] = { face[1] }
        else:
            self.faces[face[0]].add(face[1])
    
    def remove_face(self, face: FaceType):
        if face[0] not in self.faces: return
        if face[1] in self.faces[face[0]]:
            self.faces[face[0]].remove(face[1])
            print(f"Remove {face}")
        else:
            breakpoint()
        
    def add_entire_mesh(self, mesh: MeshObject):
        self.meshs.append(mesh.name)
    
    def mask(self) -> tuple[list[str], list[str]]:
        mask_modifiers = []
        
        for mesh_name, mesh_face_ids in self.faces.items():
            mesh_obj   = MeshObject.withName(mesh_name)
            mesh_verts = { vid 
                           for faceid in mesh_face_ids
                           for vid in mesh_obj.data.polygons[faceid.id].vertices }
            vg = VertexGroup.create(mesh_obj, "temporary_mask_FG__")
            vg.add_verts(list(mesh_verts))
            
            mesh_obj.mask(vg, apply=False, invert=True)
            mask_modifiers.append(mesh_name)
        
        for mesh_name in self.meshs:
            bpy.data.objects[mesh_name].hide_viewport = True
            bpy.data.objects[mesh_name].hide_render   = True
        
        return mask_modifiers, self.meshs

    def cancel_mask(self, modifiers: tuple[list[str], list[str]]) -> None:
        for meshname in modifiers[0]:
            obj = bpy.data.objects[meshname]
            obj.modifiers.remove(obj.modifiers["Mask"])
        
        for mesh_name in modifiers[1]:
            bpy.data.objects[mesh_name].hide_viewport = False
            bpy.data.objects[mesh_name].hide_render   = False
    
    @contextmanager
    def scoped_mask(self):
        mask_modifiers = []
        vgs            : list[VertexGroup] = []
        
        try:
            for mesh_name, mesh_face_ids in self.faces.items():
                mesh_obj   = MeshObject.withName(mesh_name)
                mesh_verts = { vid 
                            for faceid in mesh_face_ids
                            for vid in mesh_obj.data.polygons[faceid.id].vertices }
                vg = VertexGroup.create(mesh_obj, "temporary_mask_FG__")
                vg.add_verts(list(mesh_verts))
                vgs.append(vg)
                
                mesh_obj.mask(vg, apply=False, invert=True)
                mask_modifiers.append(mesh_name)
            
            for mesh_name in self.meshs:
                bpy.data.objects[mesh_name].hide_viewport = True
                bpy.data.objects[mesh_name].hide_render   = True
            
            yield
        
        finally:
            for meshname, vg in zip(mask_modifiers, vgs):
                obj = bpy.data.objects[meshname]
                obj.modifiers.remove(obj.modifiers["Mask"])
                obj.vertex_groups.remove(vg.vg)
            
            for mesh_name in self.meshs:
                bpy.data.objects[mesh_name].hide_viewport = False
                bpy.data.objects[mesh_name].hide_render   = False


class BakeServiceConfig(T.NamedTuple):
    margin: int = 16
    margin_type: T.Literal["EXTEND", "ADJACENT_FACES"] = "ADJACENT_FACES"
    ray_distance: float = 0.0
    
    use_pass_direct: bool = False
    use_pass_indirect: bool = False
    use_selected_to_active: bool = True
    use_cage: bool = False
    cage_object: MeshObject | None = None


class BakeService:
    CYCLES_GPUTYPES_PREFERENCE = [
        # key must be a valid cycles device_type
        # ordering indicate preference - earlier device types will be used over later if both are available
        #  - e.g most OPTIX gpus will also show up as a CUDA gpu, but we will prefer to use OPTIX due to this list's ordering
        "CUDA",
        "OPTIX",
        "METAL",  # untested
        "HIP",  # untested
        "ONEAPI",  # untested
        "CPU",
    ]
    
    def __init__(self):
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.cycles.device = 'GPU'
        bpy.context.scene.cycles.bake_type = 'COMBINED'
        self.configure_cycles_devices()
    
    @staticmethod
    def configure_cycles_devices():
        assert bpy.context.scene.render.engine == "CYCLES"
        bpy.context.scene.cycles.device = "GPU"
        prefs = bpy.context.preferences.addons["cycles"].preferences

        # Necessary to "remind" cycles that the devices exist? Not sure. Without this no devices are found.
        for dt in prefs.get_device_types(bpy.context):
            prefs.get_devices_for_type(dt[0])

        assert len(prefs.devices) != 0, prefs.devices

        types = list(d.type for d in prefs.devices)

        types = sorted(types, key=BakeService.CYCLES_GPUTYPES_PREFERENCE.index)
        use_device_type = types[0]

        if use_device_type == "CPU":
            print(f"Render will use CPU-only, only found {types=}")
            bpy.context.scene.cycles.device = "CPU"
            return

        bpy.context.preferences.addons[
            "cycles"
        ].preferences.compute_device_type = use_device_type
        use_devices = [d for d in prefs.devices if d.type == use_device_type]

        print(f"Cycles will use {use_device_type=}, {len(use_devices)=}")

        for d in prefs.devices:
            d.use = False
        for d in use_devices:
            d.use = True

        return use_devices

    @staticmethod
    def apply_config(cfg: BakeServiceConfig, scene=None):
        if scene is None:
            scene = bpy.context.scene
        bpy.context.scene.render.bake.margin = cfg.margin
        bpy.context.scene.render.bake.margin_type = cfg.margin_type
        bpy.context.scene.render.bake.max_ray_distance = cfg.ray_distance
        
        bpy.context.scene.render.bake.use_pass_direct = cfg.use_pass_direct
        bpy.context.scene.render.bake.use_pass_indirect = cfg.use_pass_indirect
        bpy.context.scene.render.bake.use_selected_to_active = cfg.use_selected_to_active
        bpy.context.scene.render.bake.use_cage = cfg.use_cage
        
        if cfg.cage_object is not None:
            bpy.context.scene.render.bake.cage_object = cfg.cage_object.mesh_object

    def core_bake(self, from_objects: list[MeshObject], to_object: MeshObject, cfg: BakeServiceConfig):
        self.apply_config(cfg)
        
        with ExitStack() as selection_scope_stack:
            # Deselect everything temporarily
            for obj in bpy.data.objects:
                if obj.type != "MESH": continue
                selection_scope_stack.enter_context(MeshObject(obj).scoped_select(select=False))
            
            # Select all objects of interest
            for obj in from_objects + [to_object]:
                selection_scope_stack.enter_context(obj.scoped_select(select=True))
            
            # Set active temporarily
            with to_object.scoped_active(), to_object.scoped_select():
                bpy.ops.object.bake(type='EMIT')
        
    def bake_with_cage(self, from_objects: list[MeshObject], to_object: MeshObject, cage_offset: float=4.0):
        cage = to_object.copy(keep_material=False)
        cage.displace("NORMAL", cage_offset)
        print("Start baking, this may take a while ...", end="", flush=True)
        
        self.core_bake(
            from_objects, to_object, BakeServiceConfig(
                margin=16, margin_type="EXTEND",
                use_pass_direct=False, use_pass_indirect=False,
                use_selected_to_active=True, 
                use_cage=True, cage_object=cage
            )
        )
        
        print("\rService.bake: Baking finished." + " " * 30)
        cage.delete()

    def bake_limited_dist(self, from_objects: list[MeshObject], to_object: MeshObject, 
                          out_offset: float=5.0, in_offset: float | None=-5.0, ray_distance: float=10.0):
        out_cage = to_object.copy(keep_material=False)
        out_cage.displace("NORMAL", out_offset)
        
        if in_offset is not None:
            int_cage = to_object.copy(keep_material=False)
            int_cage.displace("NORMAL",  in_offset)
        else:
            int_cage = None
        
        print("Start baking, this may take a while ...", end="", flush=True)
        bake_from = (from_objects + [int_cage]) if int_cage is not None else from_objects
        
        self.core_bake(
                bake_from, to_object, BakeServiceConfig(
                margin=1, margin_type="EXTEND",
                use_pass_direct=False, use_pass_indirect=False,
                use_selected_to_active=True, 
                use_cage=True, cage_object=out_cage,
                ray_distance=ray_distance
            )
        )
        
        print("\rService.bake: Baking finished." + " " * 30)
        out_cage.delete()
        if int_cage is not None: int_cage.delete()

    def bake_with_custom_cage(self, from_objects: list[MeshObject], to_object: MeshObject, cage_fn: T.Callable[[MeshObject,], None]):
        cage = to_object.copy(keep_material=False)
        cage_fn(cage)
        print("Start baking, this may take a while ...", end="", flush=True)
        
        self.core_bake(
            from_objects, to_object, BakeServiceConfig(
                margin=16, margin_type="EXTEND",
                use_pass_direct=False, use_pass_indirect=False,
                use_selected_to_active=True, 
                use_cage=True, cage_object=cage
            )
        )
        
        print("\rService.bake: Baking finished." + " " * 30)
        cage.delete()


@dataclass
class ConnectedComponent:
    bmesh: T.Any
    vertices: set[int]
    
    def peek(self) -> int: return next(iter(self.vertices))

    @property
    def size(self) -> int: return len(self.vertices)
    
    @property
    def frontier(self):
        for vertid in self.vertices:
            vert = self.bmesh.verts[vertid]
            if any([(e.other_vert(vert).index not in self.vertices) for e in vert.link_edges]) or vert.is_boundary:
                yield vert
    
    @property
    def adjacency_with_other_ccs(self):
        for vertid in self.vertices:
            vert = self.bmesh.verts[vertid]
            if any([(e.other_vert(vert).index not in self.vertices) for e in vert.link_edges]):
                yield vert

    def join(self, other: 'ConnectedComponent'):
        assert self.bmesh == other.bmesh, "Impossible to join components from different meshes"
        self.vertices |= other.vertices
    
    @property
    def area(self) -> float:
        return sum([f.calc_area() for f in self.bmesh.faces if all([v.index in self.vertices for v in f.verts])])
    
    @property
    def average_normal(self) -> np.ndarray:
        return self.normals.mean(axis=0)
    
    @property
    def normals(self) -> np.ndarray:
        return np.stack([self.bmesh.verts[v].normal for v in self.vertices], axis=0)
    
    @property
    def anistropic_score(self) -> float:
        """
        The higher the more anistropic this group of vertices is. Usually indicates these
        vertices are part of a surface instead of a cluster. (Cluster has lower anistropic score)
        """
        avg_norm = self.average_normal.reshape(1, 3)
        angle    = np.arccos(np.clip(np.dot(self.normals, avg_norm.T), -1, 1))
        return 1 / max(np.var(angle), 1e-5)


class DisjointSet: 
    def __init__(self, size): 
        self.parent = [i for i in range(size)] 

    def find(self, i): 
        if self.parent[i] != i: 
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i] 
  
    def union(self, i, j): 
        self.parent[self.find(i)] = self.find(j)

    def get_connected_components(self, bmesh) -> list[ConnectedComponent]:
        components = dict()
        for i in range(len(self.parent)):
            rep = self.find(i)
            if rep not in components: components[rep] = []
            components[rep].append(i)
        return [ConnectedComponent(bmesh, set(vertices)) for vertices in components.values()]
    
    def get_raw_connected_verts(self) -> list[list[int]]:
        components = dict()
        for i in range(len(self.parent)):
            rep = self.find(i)
            if rep not in components: components[rep] = []
            components[rep].append(i)
        return [vertices for vertices in components.values()]

    @staticmethod
    def retrieve_ccs_bmesh(bm, is_connected):
        raw_ccs = DisjointSet.abstract_ccs_retrieval(
            len(bm.verts),
            lambda vid: [e.other_vert(bm.verts[vid]).index for e in bm.verts[vid].link_edges],
            is_connected=lambda a, b: is_connected(bm.verts[a], bm.verts[b]),
        )
        return [ConnectedComponent(bm, set(cc)) for cc in raw_ccs]

    @staticmethod
    def abstract_ccs_retrieval(num_verts, get_adjacent_verts, is_connected=lambda v1, v2: True):
        disjointset = DisjointSet(num_verts)
        pb = range(num_verts)
        for vert in pb:
            for other in get_adjacent_verts(vert):
                if is_connected(vert, other): disjointset.union(vert, other)
        
        ccs = disjointset.get_raw_connected_verts()
        return ccs


class ProjectEdit(Operator):
    """Edit a snapshot of the 3D Viewport in an external image editor"""
    bl_idname = "custom.project_edit"
    bl_label = "Project Edit"
    bl_options = {'REGISTER'}

    _proj_hack = [""]

    def execute(self, context):
        print("Editing Start")
        import os

        EXT = "png"  # could be made an option but for now ok

        for image in bpy.data.images:
            image.tag = True

        # opengl buffer may fail, we can't help this, but best report it.
        try:
            bpy.ops.paint.image_from_view()
        except RuntimeError as ex:
            self.report({'ERROR'}, str(ex))
            return {'CANCELLED'}

        image_new = None
        for image in bpy.data.images:
            if not image.tag:
                image_new = image
                break

        if not image_new:
            self.report({'ERROR'}, "Could not make new image")
            return {'CANCELLED'}

        filepath = os.path.basename(bpy.data.filepath)
        filepath = os.path.splitext(filepath)[0]
        # fixes <memory> rubbish, needs checking
        # filepath = bpy.path.clean_name(filepath)

        if bpy.data.is_saved:
            filepath = "//" + filepath
        else:
            filepath = os.path.join(bpy.app.tempdir, "project_edit")

        obj = context.object

        if obj:
            filepath += "_" + bpy.path.clean_name(obj.name)

        filepath_final = filepath + "." + EXT
        i = 0

        while os.path.exists(bpy.path.abspath(filepath_final)):
            filepath_final = filepath + "{:03d}.{:s}".format(i, EXT)
            i += 1

        image_new.name = bpy.path.basename(filepath_final)
        ProjectEdit._proj_hack[0] = image_new.name

        image_new.filepath_raw = filepath_final  # TODO, filepath raw is crummy
        image_new.file_format = 'PNG'
        image_new.save()

        filepath_final = bpy.path.abspath(filepath_final)

        try:
            bpy.ops.custom.external_edit(filepath=filepath_final)
        except RuntimeError as ex:
            self.report({'ERROR'}, str(ex))

        return {'FINISHED'}

bpy.utils.register_class(ProjectEdit)


class BoxAABBMesh(MeshObject):
    def __init__(self, name: str, co_min: tuple[float, float, float], co_max: tuple[float, float, float]):
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
        
        super().__init__(obj)

    @classmethod
    def aabb_of(cls, other_mesh: MeshObject, name: str | None = None) -> "BoxAABBMesh":
        box = other_mesh.mesh_object.bound_box
        cos = [mathutils.Vector(c) for c in box]
        xs  = [c.x for c in cos]
        ys  = [c.y for c in cos]
        zs  = [c.z for c in cos]
        co_min = (min(xs), min(ys), min(zs))
        co_max = (max(xs), max(ys), max(zs))
        if name is None: name = other_mesh.name + "_aabb"
        
        return cls(name, co_min, co_max)

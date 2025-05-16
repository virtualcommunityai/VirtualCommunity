from typing import Any
import bpy
bpy: Any

from .Base import MaterialBase

class BSDF_UVGrid_Material(MaterialBase):
    def __init__(self, name: str):
        material = bpy.data.materials.new(name=name)
        material.use_nodes = True
        material_nodes = material.node_tree.nodes

        for node in material_nodes: material_nodes.remove(node)

        # Add Principled BSDF shader
        bsdf = material_nodes.new(type="ShaderNodeBsdfPrincipled")
        bsdf.location = (0, 0)

        # Add UV grid texture node
        uv_texture = material_nodes.new(type="ShaderNodeTexChecker")
        uv_texture.location = (-300, 0)
        uv_texture.inputs['Scale'].default_value = 10  # Adjust scale as desired

        # Add texture coordinate node to ensure the UV map is used
        tex_coord = material_nodes.new(type="ShaderNodeTexCoord")
        tex_coord.location = (-600, 0)

        # Connect the nodes
        material.node_tree.links.new(tex_coord.outputs['UV'], uv_texture.inputs['Vector'])
        material.node_tree.links.new(uv_texture.outputs['Color'], bsdf.inputs['Base Color'])

        # Output node
        output = material_nodes.new(type="ShaderNodeOutputMaterial")
        output.location = (300, 0)

        # Link BSDF to Material Output
        material.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
        
        super().__init__(material)

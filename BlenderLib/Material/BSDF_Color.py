from typing import Any
import bpy
bpy: Any

from .Base import MaterialBase


class BSDF_Color_Material(MaterialBase):
    """Create a BSDF material with RGB color of (R, G, B, A)
    """
    def __init__(self, name: str, color: tuple[float, float, float, float]) -> None:
        material = bpy.data.materials.new(name=name)
        material.use_nodes = True
        nodes = material.node_tree.nodes
        nodes.clear()

        node_bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
        node_bsdf.inputs['Base Color'].default_value = color
        
        # Set the default roughness and metallic values if desired
        node_bsdf.inputs['Roughness'].default_value = 0.5
        node_bsdf.inputs['Metallic'].default_value = 0.0
        

        node_output = nodes.new(type='ShaderNodeOutputMaterial')
        links = material.node_tree.links
        links.new(node_bsdf.outputs['BSDF'], node_output.inputs['Surface'])
        
        super().__init__(material)

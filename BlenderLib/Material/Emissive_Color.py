from typing import Any
import bpy
bpy: Any

from .Base import MaterialBase


class Emission_Color_Material(MaterialBase):
    """Create an Emissive material with RGB color of (R, G, B, A)
    """
    def __init__(self, name: str, color: tuple[float, float, float, float]) -> None:
        material = bpy.data.materials.new(name=name)
        material.use_nodes = True
        nodes = material.node_tree.nodes
        nodes.clear()
        node_emissive = nodes.new(type='ShaderNodeEmission')
        
        node_emissive.inputs['Color'].default_value=color
        
        node_output = nodes.new(type='ShaderNodeOutputMaterial')
        links = material.node_tree.links
        links.new(node_emissive.outputs['Emission'], node_output.inputs['Surface'])
        super().__init__(material)


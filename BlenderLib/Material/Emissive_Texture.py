from typing import Any
import bpy
bpy: Any

from .Base import MaterialBase


class Emission_Textured_Material(MaterialBase):
    """Create an Emissive material with some texture map in JPEG format
    """
    def __init__(self, name: str, tex_height: int, tex_width: int) -> None:
        material = bpy.data.materials.new(name=name)
        material.use_nodes = True
        nodes = material.node_tree.nodes
        
        # Clean up default nodes
        for node in nodes: nodes.remove(node)
        
        # Define new material nodes
        image_texture = nodes.new(type="ShaderNodeTexImage")
        bsdf = nodes.new(type='ShaderNodeEmission')
        material_output = nodes.new(type="ShaderNodeOutputMaterial")
        
        image = bpy.data.images.new(name=f"Tex_{name}", height=tex_height, width=tex_width)
        image.file_format = 'JPEG'
        image.pack()
        image_texture.image = image

        image_texture.location = (-600, 0)
        bsdf.location=(0, 0)
        material_output.location = (300, 0)
        
        material.node_tree.links.new(image_texture.outputs["Color"], bsdf.inputs["Color"])
        material.node_tree.links.new(bsdf.outputs["Emission"], material_output.inputs["Surface"])
        super().__init__(material)

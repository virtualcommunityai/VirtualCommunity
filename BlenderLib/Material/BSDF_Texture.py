from typing import Any
import bpy
bpy: Any

from .Base import MaterialBase

class BSDF_Textured_Material(MaterialBase):
    """Create a BSDF material with a texture map in JPEG format"""
    def __init__(self, name: str, tex_height: int, tex_width: int) -> None:
        material = bpy.data.materials.new(name=name)
        material.use_nodes = True
        nodes = material.node_tree.nodes
        
        for node in nodes: nodes.remove(node)
        
        # Define new material nodes
        image_texture = nodes.new(type="ShaderNodeTexImage")
        bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
        material_output = nodes.new(type="ShaderNodeOutputMaterial")
        
        # Create a new image for the texture
        image = bpy.data.images.new(name=f"Tex_{name}", height=tex_height, width=tex_width)
        image.file_format = 'JPEG'
        image.pack()  # Pack the image into the .blend file
        image_texture.image = image

        image_texture.location = (-600, 0)
        bsdf.location = (0, 0)
        material_output.location = (300, 0)
        
        material.node_tree.links.new(image_texture.outputs["Color"], bsdf.inputs["Base Color"])
        material.node_tree.links.new(bsdf.outputs["BSDF"], material_output.inputs["Surface"])
        
        super().__init__(material)

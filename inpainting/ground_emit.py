import bpy
import bmesh
from pathlib import Path
import numpy as np
from PIL import Image
import json
import os
import sys

sys.setrecursionlimit(16384)

def load_img_to_array(img_p):
    img = Image.open(img_p)
    if img.mode == "RGBA":
        img = img.convert("RGB")
    return np.array(img)

def valid(face):
    if face.material_index == 64:
        return False
    else:
        return True

def emit_texture_maps(blender_file_path: str) -> str:
    """Given a blender file with bundled texture maps internally, emit the texture maps to a directory and return the path
    of that directory.
    """
    bpy.ops.file.unpack_all(method="WRITE_LOCAL")
    return str(Path(Path(blender_file_path).parent, "textures"))

def save_mesh_texture_maps(mesh_name, output_directory):
    # Get the mesh object by name
    obj = bpy.data.objects.get(mesh_name)
    
    if obj is None or obj.type != 'MESH':
        print(f"Object '{mesh_name}' not found or is not a mesh.")
        return
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Iterate through the materials of the mesh
    ret={}
    for mat in obj.data.materials:
        if mat is None:
            continue
        
        # Check if the material uses nodes
        if mat.use_nodes:
            for node in mat.node_tree.nodes:
                if node.type == 'TEX_IMAGE' and node.image:
                    # image_name = f"{mat.name}_{node.name}.png"
                    image_name = f"{mat.name}.png"
                    image_path = os.path.join(output_directory, image_name)
                    node.image.save_render(image_path)
                    print(f"Saved texture: {image_path}")
                    ret[mat.name]=[1.0,1.0,0.0,0.0]
        else:
            # Check texture slots if not using nodes
            for texture_slot in mat.texture_slots:
                if texture_slot and texture_slot.texture and texture_slot.texture.type == 'IMAGE':
                    # image_name = f"{mat.name}_{texture_slot.texture.name}.png"
                    image_name = f"{mat.name}.png"
                    image_path = os.path.join(output_directory, image_name)
                    texture_slot.texture.image.save_render(image_path)
                    print(f"Saved texture: {image_path}")
                    ret[mat.name]=[1.0,1.0,0.0,0.0]
    return ret

def clear_transformations(bpy_obj):
    bpy_obj.location = (0.0, 0.0, 0.0)
    bpy_obj.rotation_quaternion = (1., 0., 0., 0.)
    bpy_obj.scale = (1.0, 1.0, 1.0)
    bpy.context.view_layer.update()    

dx,dy=[-1,0,1,0],[0,-1,0,1]
lx,ly,ux,uy=0,0,0,0
mp={}
blend_dir=""
def eq(x:float, y:float):
    return x-y<0.05 and y-x<0.05
def expand(face,x,y,prev,pdr):
    global lx,ux,ly,uy
    assert len(face.edges)==4
    if (face in mp) or (not valid(face)):
        return
    mp[face]=[x,y]
    lx,ux,ly,uy=min(lx,x),max(ux,x),min(y,ly),max(y,uy)
    link_faces = []
    dr=0
    for e in face.edges:
        # print(type(e),e.index)
        link_face = [f for f in e.link_faces if (f is not face) and (valid(f))]
        if len(link_face)==0 or len(link_face[0].edges)!= 4:
            link_faces.append(None)
            continue
        link_faces.append(link_face[0])
        # if face.index in [288,234,304,303,289,306]:
        #     print("**",len(link_faces)-1,link_face[0].index)
        if link_face[0]==prev:
            # if(len(link_faces)-1!=(pdr+2)%4):
            dr=((-(len(link_faces)-1)+(pdr+2))%4+4)%4
    # print(face.index,x,y,dr,(pdr+2)%4)
    mp[face].append(dr)
    for idx,nf in enumerate(link_faces):
        if nf==None:continue
        expand(nf,x+dx[(idx+dr)%4],y+dy[(idx+dr)%4],face,(idx+dr)%4)
def build_map(start_face,uvs,whs):
    expand(start_face,0,0,None,0)
    a=np.zeros((ux-lx+1,uy-ly+1),dtype=int)
    b=np.zeros((ux-lx+1,uy-ly+1),dtype=int)
    for key, val in mp.items():
        a[val[0]-lx][val[1]-ly]=key.index
        b[val[0]-lx][val[1]-ly]=(val[2] + uvs[key.index])%4
    return a,b

def main(blender_file_path: str, mesh_name: str, save_to: str):
    global blend_dir
    tile_size = 8
    blend_dir=Path(Path(blender_file_path).parent)
    bpy.ops.wm.open_mainfile(filepath=blender_file_path)

    mat_whs=save_mesh_texture_maps(mesh_name, save_to)

    bpy.ops.object.select_all(action='DESELECT')
    obj = bpy.data.objects.get(mesh_name)
    bpy.ops.object.mode_set(mode='EDIT')
    if obj is not None and obj.type == 'MESH':
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        mesh = obj.data
        bm = bmesh.from_edit_mesh(mesh)
        uv_layer = bm.loops.layers.uv.active
        faces = bm.faces
        materials = obj.data.materials
        uvs=[]
        whs=[]
        for face in faces:
            if not valid(face):continue
            # print(f"Face Index: {face.index}; Material Index: {face.material_index}")
            mx,my=0.0,0.0
            zero_point=-1
            uv_edges=[]
            for idx,loop in enumerate(face.loops):
                uv = loop[uv_layer].uv
                # print("uv:",uv)
                uv_edges.append([uv.x,uv.y])
                mx,my=max(mx,uv.x),max(my,uv.y)
                tmp_arr=[uv.x,uv.y,uv.x,uv.y]
                for i in range(0,2):
                    mat_whs[materials[face.material_index].name][i]=min(mat_whs[materials[face.material_index].name][i],tmp_arr[i])
                for i in range(2,4):
                    mat_whs[materials[face.material_index].name][i]=max(mat_whs[materials[face.material_index].name][i],tmp_arr[i])
            assert(len(uv_edges)==4)
            dx,dy=uv_edges[1][0]-uv_edges[0][0],uv_edges[1][1]-uv_edges[0][1]
            if eq(dx,0):
                if dy>0:
                    zero_point=0
                else:
                    zero_point=2
            else:
                if dx>0:
                    zero_point=3
                else:
                    zero_point=1
            whs.append([mx,my])
            uvs.append(zero_point)
        for face in faces:
            if not valid(face):continue
            a,b=build_map(face,uvs,whs)
            faces.ensure_lookup_table()
            mat_a=np.zeros(((a.shape[0])//tile_size,(a.shape[1])//tile_size),dtype=int)
            mat_b=np.zeros(((b.shape[0])//tile_size,(b.shape[1])//tile_size),dtype=int)
            for i in range(mat_a.shape[0]):
                for j in range(mat_a.shape[1]):
                    # print(a[i*16][j*16], len(faces))
                    mat_a[i][j]=faces[a[i*tile_size][j*tile_size]].material_index
                    mat_b[i][j]=b[i*tile_size][j*tile_size]
            with open(f"{save_to}/{blend_dir.stem}.json","w") as f:
                json.dump({"whs":mat_whs,"a":mat_a.tolist(),"b":mat_b.tolist()},f)
            break
    else:
        print(f"Object '{mesh_name}' not found or is not a mesh.")
    


if __name__ == "__main__":
    import argparse, sys
    parser = argparse.ArgumentParser()
    parser.add_argument("--blender_file", type=str, required=True, help="Path to the blender file that contains the mesh and building vertex groups")
    parser.add_argument("--mesh", type=str, required=True, help="Name of the mesh object in the blender file")
    parser.add_argument("--save_to", type=str, required=True, help="Save emitted image file to")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])

    main(args.blender_file, args.mesh, args.save_to)

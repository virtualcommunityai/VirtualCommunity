import torch
import sys
import argparse
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import cv2
from PIL import Image
import json
import time
from sam_segment import predict_masks_with_sam
from lama_inpaint import inpaint_img_with_lama
from utils import load_img_to_array, save_array_to_img, dilate_mask, \
    show_mask, show_points, get_clicked_point


def setup_args(parser):
    parser.add_argument(
        "--input_img", type=str, required=True,
        help="Path to a single input img",
    )
    parser.add_argument(
        "--coords_type", type=str, required=True,
        default="key_in", choices=["click", "key_in"], 
        help="The way to select coords",
    )
    parser.add_argument(
        "--point_coords", type=float, nargs='+', required=True,
        help="The coordinate of the point prompt, [coord_W coord_H].",
    )
    parser.add_argument(
        "--point_labels", type=int, nargs='+', required=True,
        help="The labels of the point prompt, 1 or 0.",
    )
    parser.add_argument(
        "--dilate_kernel_size", type=int, default=None,
        help="Dilate kernel size. Default: None",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output path to the directory with results.",
    )
    parser.add_argument(
        "--sam_model_type", type=str,
        default="vit_h", choices=['vit_h', 'vit_l', 'vit_b', 'vit_t'],
        help="The type of sam model to load. Default: 'vit_h"
    )
    parser.add_argument(
        "--sam_ckpt", type=str, required=True,
        help="The path to the SAM checkpoint to use for mask generation.",
    )
    parser.add_argument(
        "--lama_config", type=str,
        default="./lama/configs/prediction/default.yaml",
        help="The path to the config file of lama model. "
             "Default: the config of big-lama",
    )
    parser.add_argument(
        "--lama_ckpt", type=str, required=True,
        help="The path to the lama checkpoint.",
    )

def find_black_pixel(img):
    black_pixels = np.where((img[:,:,0] == 0) & 
                            (img[:,:,1] == 0) & 
                            (img[:,:,2] == 0))
    
    height, width, _ = img.shape
    if len(black_pixels[0]) *3 >img.size-10:
        return None
    def is_black(pixel):
        return np.all(pixel == [0, 0, 0])
    def is_isolated(y, x):
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    if not is_black(img[ny, nx]):
                        return False
        return True
    for y in range(height):
        for x in range(width):
            if is_black(img[y, x]) and is_isolated(y, x):
                return [x, y]
    if len(black_pixels[0]) > 0:
        return [black_pixels[1][0], black_pixels[0][0]]
    else:
        return None

def inpaint_one_image(img, model):
    latest_coords = find_black_pixel(img)
    iter_i = 0
    while latest_coords != None:
        # assert 0
        iter_i = iter_i + 1

        # masks, _, _ = predict_masks_with_sam(
        #     img,
        #     [latest_coords],
        #     args.point_labels,
        #     model_type=args.sam_model_type,
        #     ckpt_p=args.sam_ckpt,
        #     device=device,
        # )
        # masks = masks.astype(np.uint8) * 255

        # image_pil = Image.fromarray(img)
        # text_prompt = "car from aerial view, vehicle, cars, white car"
        # masks, boxes, phrases, logits = model.predict(image_pil, text_prompt)
        # def tensor_to_image(tensor):
        #     B=tensor.shape[0]
        #     numpy_image = tensor.cpu().numpy().astype(int)
        #     mask = np.zeros_like(numpy_image[0])
        #     for i in range(B):
        #         if np.sum(numpy_image[i].astype(np.uint8))*10 < numpy_image[i].size:
        #             mask |= numpy_image[i]
        #     return mask
        mask = cv2.inRange(img, (0, 0, 0), (0, 0, 0))
        # if len(masks)>0:
        #     mask |= tensor_to_image(masks).astype(np.uint8) * 255
        masks = [mask]


        # dilate mask to avoid unmasked edge effect
        if args.dilate_kernel_size is not None:
            masks = [dilate_mask(mask, args.dilate_kernel_size) for mask in masks]

        # visualize the segmentation results
        img_stem = Path(args.input_img).stem
        out_dir = Path(args.output_dir) / img_stem
        out_dir.mkdir(parents=True, exist_ok=True)
        for idx, mask in enumerate(masks):
            if idx != 0 : continue
            # path to the results
            mask_p = out_dir / f"mask_{idx}_{iter_i}.png"
            img_points_p = out_dir / f"with_points_{iter_i}.png"
            img_mask_p = out_dir / f"with_mask_{idx}_{iter_i}.png"

            # save the mask
            save_array_to_img(mask, mask_p)

            # save the pointed and masked image
            dpi = plt.rcParams['figure.dpi']
            height, width = img.shape[:2]
            plt.figure(figsize=(width/dpi/0.77, height/dpi/0.77))
            plt.imshow(img)
            plt.axis('off')
            show_points(plt.gca(), [latest_coords], args.point_labels,
                        size=(width*0.04)**2)
            plt.savefig(img_points_p, bbox_inches='tight', pad_inches=0)
            show_mask(plt.gca(), mask, random_color=False)
            plt.savefig(img_mask_p, bbox_inches='tight', pad_inches=0)
            plt.close()

        # inpaint the masked image
        for idx, mask in enumerate(masks):
            if idx != 0 : continue
            mask_p = out_dir / f"mask_{idx}_{iter_i}.png"
            img_inpainted_p = out_dir / f"inpainted_with_mask_{idx}_{iter_i}.png"
            img_inpainted = inpaint_img_with_lama(
                img, mask, args.lama_config, args.lama_ckpt, device=device)
            save_array_to_img(img_inpainted, img_inpainted_p)

        if np.array_equal(img, img_inpainted): break
        img  = img_inpainted
        latest_coords=find_black_pixel(img)
        break
    return img

if __name__ == "__main__":
    """Example usage:
    python remove_anything.py \
        --input_img FA_demo/FA1_dog.png \
        --coords_type key_in \
        --point_coords 750 500 \
        --point_labels 1 \
        --dilate_kernel_size 15 \
        --output_dir ./results \
        --sam_model_type "vit_h" \
        --sam_ckpt sam_vit_h_4b8939.pth \
        --lama_config lama/configs/prediction/default.yaml \
        --lama_ckpt big-lama 
    """
    parser = argparse.ArgumentParser()
    setup_args(parser)
    args = parser.parse_args(sys.argv[1:])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.coords_type == "click":
        latest_coords = get_clicked_point(args.input_img)
    elif args.coords_type == "key_in":
        latest_coords = args.point_coords
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for file in Path(args.input_img).iterdir():
        if not ("png" in str(file)):continue
        img = load_img_to_array(file)
        img_p = out_dir / f"{file.name}"
        save_array_to_img(img, img_p)
    # import pdb; pdb.set_trace()

    with open(Path(args.input_img)/f"{str(Path(args.input_img)).split('textures_')[-1]}.json","r") as f:
    #with open(Path(args.input_img)/f"EL_PASO.json","r") as f:
        json_data=json.load(f)
        whs=json_data["whs"]
        a=np.array(json_data["a"])
        b=np.array(json_data["b"])
    with open(out_dir/f"{Path(args.input_img).stem.split('_')[-1]}.json","w") as f:
    #with open(out_dir/f"E.json","w") as f:
        json.dump({"whs":whs,"a":a.tolist(),"b":b.tolist()},f)
    # a=np.array([[236, 200, 207, 241, 247, 246, 245, 240, 218, 197, 183, 239, 244, 243, 242, 238, 192, 229], [230, 237, 228, 208, 180, 187, 227, 234, 233, 232, 226, 184, 186, 164, 225, 231, 193, 172], [212, 217, 224, 223, 222, 216, 188, 161, 177, 215, 221, 220, 219, 214, 203, 157, 213, 202], [165, 173, 147, 153, 201, 211, 210, 209, 199, 168, 151, 150, 198, 206, 205, 204, 140, 196], [185, 195, 194, 182, 154, 148, 134, 181, 191, 190, 189, 179, 160, 145, 167, 178, 131, 158], [109, 117, 166, 176, 175, 174, 163, 135, 106, 92, 162, 171, 170, 169, 144, 132, 102, 159], [80, 89, 118, 149, 130, 139, 156, 128, 155, 86, 112, 146, 127, 124, 152, 103, 123, 122], [133, 143, 142, 141, 69, 97, 129, 108, 116, 138, 137, 136, 66, 72, 126, 105, 125, 81], [110, 111, 91, 101, 121, 120, 119, 44, 75, 107, 88, 96, 115, 114, 113, 36, 85, 93], [94, 104, 9, 61, 90, 71, 79, 100, 99, 98, 270, 43, 87, 68, 74, 95, 37, 73], [54, 65, 84, 83, 82, 8, 30, 70, 49, 60, 78, 77, 76, 27, 56, 55, 22, 67], [19, 18, 52, 51, 50, 35, 64, 63, 62, 6, 47, 46, 45, 28, 59, 58, 57, 250], [24, 21, 41, 40, 39, 16, 15, 14, 13, 11, 34, 33, 32, 4, 2, 1, 252, 0]])
    # b=np.array([[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0], [2, 2, 2, 2, 2, 0, 0, 2, 2, 2, 0, 0, 0, 0, 2, 2, 2, 2], [2, 2, 0, 0, 2, 2, 2, 2, 0, 0, 0, 2, 2, 2, 2, 2, 2, 0], [0, 2, 2, 2, 2, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 0, 0, 2, 2, 2, 2, 2], [2, 2, 0, 0, 0, 2, 2, 2, 2, 2, 0, 0, 2, 2, 2, 2, 2, 2], [0, 0, 0, 2, 0, 0, 2, 2, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]])
    # crop_box=(0,0,189,256)
    def open_image(i,j,a,b,whs,pth):
        img=Image.open(pth / f"Material_Ground{str(a[i][j]).zfill(3)}.png")
        w,h=img.width,img.height
        crop_box=(img.width*whs[f"Material_Ground{str(a[i][j]).zfill(3)}"][0],img.height*whs[f"Material_Ground{str(a[i][j]).zfill(3)}"][1],img.width*whs[f"Material_Ground{str(a[i][j]).zfill(3)}"][2],img.height*whs[f"Material_Ground{str(a[i][j]).zfill(3)}"][3])
        img = img.crop(crop_box)
        img = img.resize((512, 512))
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img = img.rotate((b[i][j]-1)*90)
        return [img,w,h]
    def open_row(i,j,a,b,whs,pth):
        img,w,h=open_image(i,j,a,b,whs,pth)
        sx=0
        if j:
            img1,_,_=open_image(i,j-1,a,b,whs,pth)
            stitched_image = Image.new('RGB', (img.width+img1.width, img.height))
            stitched_image.paste(img, (0, 0))
            stitched_image.paste(img1, (img.width, 0))
            img=stitched_image
        if j+1<a.shape[1]:
            img1,sx,_=open_image(i,j+1,a,b,whs,pth)
            sx=512
            stitched_image = Image.new('RGB', (img.width+img1.width, img.height))
            stitched_image.paste(img1, (0, 0))
            stitched_image.paste(img, (img1.width, 0))
            img=stitched_image
        return [img,w,h,sx]
    
    model = None
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            print(f"{i*a.shape[1]+j}/{a.size}")
            img,w,h,sx=open_row(i,j,a,b,whs,Path(args.input_img))
            sy=0
            if i:
                img1,_,_,_=open_row(i-1,j,a,b,whs,out_dir)
                stitched_image = Image.new('RGB',(img.width,img.height+img1.height))
                stitched_image.paste(img, (0, 0))
                stitched_image.paste(img1, (0, img.height))
                img=stitched_image
            if i+1<a.shape[0]:
                img1,_,sy,_=open_row(i+1,j,a,b,whs,out_dir)
                sy=512
                stitched_image = Image.new('RGB',(img.width,img.height+img1.height))
                stitched_image.paste(img1, (0, 0))
                stitched_image.paste(img, (0, img1.height))
                img=stitched_image
            img = Image.fromarray(inpaint_one_image(np.array(img),model))
            # img = img.crop((sx+0,sy+0,sx+w*whs[a[i][j]][0],sy+h*whs[a[i][j]][1]))
            img = img.crop((sx+0,sy+0,sx+512,sy+512))
            img = img.resize((w, h))
            img = img.rotate(-(b[i][j]-1)*90)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img_t = Image.new('RGB', (w, h))
            img_t.paste(img,(0,0))
            img_p = out_dir / f"Material_Ground{str(a[i][j]).zfill(3)}.png"
            save_array_to_img(np.array(img_t), img_p)
            time.sleep(0.1)
    # for file in Path(args.input_img).iterdir():
    #     print("Processing:",file)
    #     img = load_img_to_array(file)
    #     img = inpaint_one_image(img)
    #     img_p = out_dir / f"{file.name}"
    #     print("Save to:",img_p)
    #     save_array_to_img(img, img_p)

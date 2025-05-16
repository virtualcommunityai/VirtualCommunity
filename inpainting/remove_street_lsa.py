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
import os
import datetime
import math
import shutil
from tqdm import tqdm
from sam_segment import predict_masks_with_sam
from lama_inpaint import inpaint_img_with_lama
# from lang_sam import LangSAM
from Inpaint_Anything.utils import load_img_to_array, save_array_to_img, dilate_mask, \
    show_mask, show_points, get_clicked_point

import argparse
import os
import sys

import numpy as np
import json
import torch
from PIL import Image


# Grounding DINO
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap


# segment anything
from segment_anything import (
    sam_model_registry,
    sam_hq_model_registry,
    SamPredictor
)
import cv2
import numpy as np
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_config_path, model_checkpoint_path, bert_base_uncased_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    args.bert_base_uncased_path = bert_base_uncased_path
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases



def predict(image_pil, model, predictor, text_prompt, box_thres, text_thres):
    # Predict classes and hyper-param for GroundingDINO
    box_threshold = box_thres
    text_threshold = text_thres

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    
    # run grounding dino model
    boxes_filt, pred_phrases = get_grounding_output(
        model, image, text_prompt, box_threshold, text_threshold, device=device
    )
    # import pdb; pdb.set_trace()
    if boxes_filt.numel() == 0:
        return []
    # load image
    image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    predictor.set_image(image)

    size = image_pil.size
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)

    masks, _, _ = predictor.predict_torch(
        point_coords = None,
        point_labels = None,
        boxes = transformed_boxes.to(device),
        multimask_output = False,
    )
    
    # import pdb; pdb.set_trace()
    
    masks = masks.cpu().numpy()
    masks = np.squeeze(masks, axis=1)
    return masks

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
    parser.add_argument(
        "--job_num", type=int, required=True,
        help="The number of jobs to be divided into",
    )
    parser.add_argument(
        "--job_id", type=int, required=True,
        help="The job id",
    )
    
    
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--grounded_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_version", type=str, default="vit_h", required=False, help="SAM ViT version: vit_b / vit_l / vit_h"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, required=False, help="path to sam checkpoint file"
    )
    parser.add_argument(
        "--sam_hq_checkpoint", type=str, default=None, help="path to sam-hq checkpoint file"
    )
    parser.add_argument(
        "--use_sam_hq", action="store_true", help="using sam-hq for prediction"
    )
    parser.add_argument("--bert_base_uncased_path", type=str, required=False, help="bert_base_uncased model path, default=False")
    args = parser.parse_args()


def find_black_pixel(img):
    black_pixels = np.where((img[:, :, 0] == 0) &
                            (img[:, :, 1] == 0) &
                            (img[:, :, 2] == 0))

    height, width, _ = img.shape
    if len(black_pixels[0]) * 3 > img.size - 10:
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


def inpaint_one_epoch(img, model, predictor, box_thres, text_thres, lim, prompt):
    image_pil = Image.fromarray(img)
    # text_prompt = "people"
    # masks, boxes, phrases, logits = model.predict(image_pil, text_prompt, 0.20, 0.35)#80
    text_prompt = prompt
    #result = model.predict([image_pil], [text_prompt], box_thres, text_thres)  # 1
    masks = predict(image_pil, model, predictor, text_prompt, box_thres, text_thres)  # 1
    # masks = result[0]['masks']
    # masks = masks.numpy()

    # import pdb; pdb.set_trace()

    def array_to_image(array):
        B = array.shape[0]
        numpy_image = array.astype(np.uint)
        mask = np.zeros_like(numpy_image[0])
        for i in range(B):
            obj = numpy_image[i].astype(np.uint8)
            obj_size = np.sum(obj)
            if obj_size * lim < numpy_image[i].size:
                dilate_factor = int(math.sqrt(obj_size) * 0.15)
                obj = cv2.dilate(
                    obj,
                    np.ones((dilate_factor, dilate_factor), np.uint8),
                    iterations=1
                )
                mask |= obj
        return mask

    mask = cv2.inRange(img, (0, 0, 0), (0, 0, 0))
    if len(masks) == 0:
        return img, mask

    mask = array_to_image(masks).astype(np.uint8) * 255
    masks = [mask]
    mask_ret = mask

    # dilate mask to avoid unmasked edge effect
    if args.dilate_kernel_size is not None:
        masks = [dilate_mask(mask, args.dilate_kernel_size) for mask in masks]

    # visualize the segmentation results
    img_stem = Path(args.input_img).stem
    out_dir = Path(args.output_dir) / img_stem
    # out_dir.mkdir(parents=True, exist_ok=True)
    for idx, mask in enumerate(masks):
        if idx != 0: continue
        # path to the results
        mask_p = out_dir / f"mask_{idx}_{prompt}.png"
        img_points_p = out_dir / f"with_points_{prompt}.png"
        img_mask_p = out_dir / f"with_mask_{idx}_{prompt}.png"

        # save the mask
        # save_array_to_img(mask, mask_p)

        # save the pointed and masked image
        dpi = plt.rcParams['figure.dpi']
        height, width = img.shape[:2]
        plt.figure(figsize=(width / dpi / 0.77, height / dpi / 0.77))
        plt.imshow(img)
        plt.axis('off')
        show_points(plt.gca(), [latest_coords], args.point_labels,
                    size=(width * 0.04) ** 2)
        # plt.savefig(img_points_p, bbox_inches='tight', pad_inches=0)
        show_mask(plt.gca(), mask, random_color=False)
        # plt.savefig(img_mask_p, bbox_inches='tight', pad_inches=0)
        plt.close()

    # inpaint the masked image
    for idx, mask in enumerate(masks):
        if idx != 0: continue
        mask_p = out_dir / f"mask_{idx}_{prompt}.png"
        img_inpainted_p = out_dir / f"inpainted_with_mask_{idx}_{prompt}.png"
        img_inpainted = inpaint_img_with_lama(
            img, mask, args.lama_config, args.lama_ckpt, device=device)
        # save_array_to_img(img_inpainted, img_inpainted_p)

    return img_inpainted, mask_ret


def inpaint_one_image(img, model, predictor):
    with torch.inference_mode():
        img, mask_ret = inpaint_one_epoch(img=img, model=model, predictor=predictor, box_thres=0.20, text_thres=0.35, lim=50, prompt="people")
        img, mask_ret = inpaint_one_epoch(img=img, model=model, predictor=predictor, box_thres=0.30, text_thres=0.30, lim=1, prompt="car and its shadow")
        img, mask_ret = inpaint_one_epoch(img=img, model=model, predictor=predictor, box_thres=0.35, text_thres=0.35, lim=1, prompt="cars")
        img, mask_ret = inpaint_one_epoch(img=img, model=model, predictor=predictor, box_thres=0.45, text_thres=0.45, lim=1, prompt="tree")
    return img, mask_ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    setup_args(parser)
    args = parser.parse_args(sys.argv[1:])
    
    
    # cfg
    config_file = args.config  # change the path of the model config file
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    sam_version = args.sam_version
    sam_checkpoint = args.sam_checkpoint
    sam_hq_checkpoint = args.sam_hq_checkpoint
    use_sam_hq = args.use_sam_hq
    bert_base_uncased_path = args.bert_base_uncased_path
    
    
    model = load_model(config_file, grounded_checkpoint, bert_base_uncased_path, device=device)
    if use_sam_hq:
        predictor = SamPredictor(sam_hq_model_registry[sam_version](checkpoint=sam_hq_checkpoint).to(device))
    else:
        predictor = SamPredictor(sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device))
    

    if args.coords_type == "click":
        latest_coords = get_clicked_point(args.input_img)
    elif args.coords_type == "key_in":
        latest_coords = args.point_coords
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    start_time = datetime.datetime.now()

    assert args.job_id < args.job_num
    street_view_points = list(Path(args.input_img).iterdir())
    job_len = (len(street_view_points) - 1) // args.job_num + 1
    start_label = job_len * args.job_id
    end_label = min(start_label + job_len, len(street_view_points))
    print(len(street_view_points), start_label, end_label)

    for i in tqdm(range(start_label, end_label)):
        street_view_point = street_view_points[i]
        # if street_view_point.name != '-D8I6JXpunE0ZewLf1p_HQ':
        #     continue
        if not os.path.isdir(street_view_point): continue
        (out_dir / f"{street_view_point.name}").mkdir(parents=True, exist_ok=True)
        for file_path in (street_view_point).iterdir():
            filename = file_path.name
            new_file_path = out_dir / f"{street_view_point.name}" / f"{filename}"
            if Path.exists(new_file_path): continue
            # if filename != 'heading_60.jpg':
            #     continue
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                shutil.copy2(file_path, new_file_path)
                continue
            # if not filename.lower().startswith(('gsv')):continue
            img = load_img_to_array(file_path)
            # import pdb; pdb.set_trace()
            img, mask = inpaint_one_image(img, model, predictor)
            img_p = out_dir / f"{street_view_point.name}" / f"{filename}"
            save_array_to_img(img, img_p)
            save_array_to_img(mask, out_dir / f"{street_view_point.name}" / f"with_mask_{filename}")
            # import pdb; pdb.set_trace()
    end_time = datetime.datetime.now()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time}")

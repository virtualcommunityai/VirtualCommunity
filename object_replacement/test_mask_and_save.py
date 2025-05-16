import os
import argparse
import json
import torchvision
import sys

current_directory = os.getcwd()
# Grounding DINO
sys.path.append(os.path.join(current_directory, 'Grounded-Segment-Anything'))
sys.path.append(os.path.join(current_directory, 'Grounded-Segment-Anything', "GroundingDINO"))
sys.path.append(os.path.join(current_directory, 'Grounded-Segment-Anything', "segment_anything"))
# sys.path.append(os.path.join(current_directory, 'Marigold'))
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
# from marigold import MarigoldPipeline
# import supervision as sv

from plyfile import PlyData, PlyElement

# segment anything
from segment_anything import (
    sam_model_registry,
    SamPredictor
)
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as TS
import torch
import gzip
import pickle
import trimesh
from PIL import Image
current_directory = os.getcwd()
config_file = os.path.join(current_directory, "Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
ram_checkpoint = os.path.join(current_directory, "Grounded-Segment-Anything/ram_swin_large_14m.pth")
grounded_checkpoint = "/project/pi_chuangg_umass_edu/yian/robogen/ljg/architect/Grounded-Segment-Anything/groundingdino_swint_ogc.pth"
sam_checkpoint = os.path.join(current_directory, "Grounded-Segment-Anything/sam_vit_h_4b8939.pth")

def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    _ = model.eval()
    return model

def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image
    scale_factor = 1.0
    image_pil = image_pil.resize((int(scale_factor * image_pil.width), int(scale_factor * image_pil.height)))

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image
    
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
    ax.text(x0, y0, label)
    
def save_mask_data(output_dir, tags_chinese, mask_list, box_list, label_list):
    value = 0  # 0 for background

    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')

    np.save(os.path.join(output_dir, 'mask.npy'), mask_img.numpy())
    json_data = {
        'tags_chinese': tags_chinese,
        'mask': [{
            'value': value,
            'label': 'background'
        }]
    }

    for label, box in zip(label_list, box_list):
        show_box(box.numpy(), plt.gca(), label)
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1]  # the last is ')'
        json_data['mask'].append({'value': value, 'label': name, 'logit': float(logit), 'box': box.numpy().tolist()})

    plt.savefig(os.path.join(output_dir, 'mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

    with open(os.path.join(output_dir, 'label.json'), 'w') as f:
        json.dump(json_data, f)

model_dino = load_model(config_file, grounded_checkpoint, device="cuda")
predictor = SamPredictor(sam_model_registry["vit_h"](checkpoint=sam_checkpoint).to("cuda"))

def get_grounding_output(model, image, caption, box_threshold, text_threshold, device="cpu"):
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
    scores = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        scores.append(logit.max().item())

    return boxes_filt, torch.Tensor(scores), pred_phrases

def get_2d_bbox(image_path, output_dir, tags, box_threshold=0.35, text_threshold=0.25, iou_threshold=0.5,
                  device="cuda"):
    os.makedirs(output_dir, exist_ok=True)
    image_pil, image = load_image(image_path)

    image_pil.save(os.path.join(output_dir, "raw_image.jpg"))
    normalize = TS.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    transform = TS.Compose([
        TS.Resize((384, 384)),
        TS.ToTensor(), normalize
    ])
    boxes_filt, scores, pred_phrases = get_grounding_output(
        model_dino, image, tags, box_threshold, text_threshold, device=device
    )
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    size = image_pil.size
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()
    # use NMS to handle overlapped boxes
    print(f"Before NMS: {boxes_filt.shape[0]} boxes")
    nms_idx = torchvision.ops.nms(boxes_filt, scores, iou_threshold).numpy().tolist()
    boxes_filt = boxes_filt[nms_idx]
    print("phrases:", pred_phrases)
    pred_phrases = [pred_phrases[idx] for idx in nms_idx]
    print(f"After NMS: {boxes_filt.shape[0]} boxes")
    print("phrases:", pred_phrases)

    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)

    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes.to(device),
        multimask_output=False,
    )

    print("masks result shape:", masks.shape)
    print("input image shape:", size)

    # save_mask_data(output_dir, tags, masks, boxes_filt, pred_phrases)
    name_dict = {}
    rename_phrases = pred_phrases
    logits = []
    for i, item in enumerate(pred_phrases):
        name, logit = item.split('(')
        logit = float(logit[:-1])
        logits.append(logit)
        if name not in name_dict.keys():
            name_dict[name] = 0
            rename_phrases[i] = f'{name}-{0}'
        else:
            name_dict[name] += 1
            rename_phrases[i] = f'{name}-{name_dict[name]}'
    
    result = []
    for box, logit, phrase in zip(boxes_filt, logits, rename_phrases):
        result.append({
            'bbox': box,
            'center': [(box[0]+box[2])//2, (box[1]+box[3])//2],
            'logit': logit,
            'phrase': phrase
        })

    return result


# image_dir = '/project/pi_chuangg_umass_edu/yian/robogen/ljg/Depth-Anything-V2/output/test3/streetview.jpg'
# output_dir = '/project/pi_chuangg_umass_edu/yian/robogen/ljg/Depth-Anything-V2/output/test3'
# tags = 'pole, tree'
# masks, pred_phrases, box_list = mask_and_save(image_dir, output_dir, tags)
# with open(os.path.join(output_dir,'masks.pkl'),'wb') as f:
#     pickle.dump(masks, f)
# with open(os.path.join(output_dir,'pred_phrases.pkl'), 'wb') as f:
#     pickle.dump(pred_phrases, f )
# with open(os.path.join(output_dir,'boxlist.pkl'), 'wb') as f:
#     pickle.dump(box_list, f)

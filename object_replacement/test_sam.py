import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
import os
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor
import argparse

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

def predict_with_point(predictor,input_point,input_label):
    masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=False,
    )
    return masks, scores, logits
def predict_with_box(predictor,input_box):
    masks, _, _ = predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_box[None, :],
    multimask_output=False,
    )
    return masks
def predict_with_point_box(predictor,input_point,input_label,input_box):
    masks, _, _ = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    box=input_box,
    multimask_output=False,
    )
    return masks

def segment_objects(image_dir, output_dir, prompt_point,name, sam_checkpoint='Grounded-Segment-Anything/sam_vit_h_4b8939.pth', model_type='vit_h' ):
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    image = cv2.imread(image_dir)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor = SamPredictor(sam)
    predictor.set_image(image)

    input_point = np.array([prompt_point])
    input_label = np.array([1])
    masks, scores, logits = predict_with_point(predictor,input_point,input_label)
    # for i, (mask, score) in enumerate(zip(masks, scores)):
    #     plt.figure(figsize=(10,10))
    #     plt.imshow(image)
    #     show_mask(mask, plt.gca())
    #     show_points(input_point, input_label, plt.gca())
    #     plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    #     plt.axis('off')
    #     plt.savefig(os.path.join(output_dir,f'photoB_bbox_{name}_{i}.png'))
    mask, score, logit = masks[0], scores[0], logits[0]
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    show_mask(mask, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.title(f"Mask, Score: {score:.3f}", fontsize=18)
    plt.axis('off')
    plt.savefig(os.path.join(output_dir,f'photoB_bbox_{name}.png'))

    coords = np.column_stack(np.where(mask > 0))
    x, y, w, h = cv2.boundingRect(coords)
    x1 = x 
    y1 = y 
    x2 = x + w
    y2 = y + h
    bounding_boxes = [x1, y1, x2, y2]
    print(bounding_boxes)
    return bounding_boxes



if __name__ == '__main__' :

    parser = argparse.ArgumentParser(description='SAM')
    parser.add_argument('image_dir', type=str, default='vit_h')
    parser.add_argument('output_dir', type=str, default='"/project/pi_chuangg_umass_edu/yian/robogen/ljg/Depth-Anything-V2/output/test3')
    parser.add_argument('prompt_point', type=list, default=[417, 139])
    args = parser.parse_args()

    segment_objects(args.image_dir, args.output_dir, args.prompt_point, name)
    
    


    
    

    


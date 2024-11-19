import random
import numpy as np
from PIL import ImageDraw, ImageFont

import torch
from torch.nn import functional as F

def draw_boxes(image, annotations):
    colors = {}
    draw = ImageDraw.Draw(image)
    for annotation in annotations:
        x_min, y_min, width, height = annotation['bbox']
        x_max=x_min+width
        y_max=y_min+height
        box=[x_min, y_min, x_max,y_max]
        label = annotation['label']

        if label not in colors:
            colors[label] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        color = colors[label]
        draw.rectangle(box, outline=color, width=2)

        text_position = (box[0], box[1] - 10)
        font = ImageFont.load_default()

        text_bbox = font.getbbox(label)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        draw.rectangle(
            [text_position, (text_position[0] + text_width, text_position[1] + text_height)],
            fill=color
        )
        draw.text(text_position, label, fill=(255, 255, 255), font=font)

    return image

def yolo_iou(box1, box2):
    x_min=np.maximum(box1[..., 0]-box1[..., 2]/2, box2[..., 0]-box2[..., 2]/2)
    x_max=np.minimum(box1[..., 0]+box1[..., 2]/2, box2[..., 0]+box2[..., 2]/2)

    y_min=np.maximum(box1[..., 1]-box1[..., 3]/2, box2[..., 1]-box2[..., 3]/2)
    y_max=np.minimum(box1[..., 1]+box1[..., 3]/2, box2[..., 1]+box2[..., 3]/2)

    intersection = np.maximum(0, x_max-x_min)*np.maximum(0, y_max-y_min)
    union = box1[..., 2]*box1[..., 3]+box2[..., 2]*box2[..., 3] - intersection

    return intersection / np.maximum(union, 1e-9)

def format_prediction(predictions, anchors, grid_sizes):
    # prediction= list, scale, batch_size, num_anchors, w, h, num_classes+6
    refined=[]
    for scale_idx, scale in enumerate(predictions):
        for idx, anchor in enumerate(anchors[scale_idx]):
            pred=scale[:, idx]

            obj=F.sigmoid(pred[..., 0:1])
            x=F.sigmoid(pred[..., 1:2])+torch.arange(grid_sizes[scale_idx]).view(1, -1).expand(grid_sizes[scale_idx], -1).unsqueeze(-1)
            y=F.sigmoid(pred[..., 2:3])+torch.arange(grid_sizes[scale_idx]).view(-1,1).expand(-1, grid_sizes[scale_idx]).unsqueeze(-1)
            wh=torch.exp(pred[..., 3:5])*anchor
            prob, cls = torch.max(pred[..., 5:].softmax(dim=-1), dim=-1, keepdims=True)

            pred=torch.cat([obj, x, y, wh, cls, prob],dim=-1)
            refined.append(pred)

    return refined

def non_max_suppression(predictions, obj_threshold=0.5, iou_threshold=0.5):
    filtered_boxes=[[] for _  in range(predictions[0].shape[0])]
    for pred in predictions:
        for idx, sample in enumerate(pred):
            boxes=sample[sample[..., 0]>obj_threshold]
            filtered_boxes[idx].extend(boxes)

    result=[]
    for idx,batch in enumerate(filtered_boxes):
        if not batch:
            result.append([])
            continue

        boxes = torch.stack(batch, dim=0)
        boxes = boxes[torch.argsort(-boxes[..., 0])]

        selected=[]
        while len(boxes)>0:
            selected.append(boxes[0])
            ious=yolo_iou(selected[-1][1:5], boxes[..., 1:5])
            boxes = boxes[ious < iou_threshold]

        result.append(selected)

    return result

def get_true_boxes(targets, anchors, grid_sizes):
    refined=[]
    for scale_idx, scale in enumerate(targets):
        for idx, anchor in enumerate(anchors[scale_idx]):
            target=scale[:, idx]

            obj=target[..., 0:1]
            x=target[..., 1:2]+torch.arange(grid_sizes[scale_idx]).view(1, -1).expand(grid_sizes[scale_idx], -1).unsqueeze(-1)
            y=target[..., 2:3]+torch.arange(grid_sizes[scale_idx]).view(-1,1).expand(-1, grid_sizes[scale_idx]).unsqueeze(-1)
            wh=torch.exp(target[..., 3:5])*anchor
            cls=target[..., 5:]

            target=torch.cat([obj, x, y, wh, cls],dim=-1)
            refined.append(target)

    result=[[] for _ in range(targets[0].shape[0])]
    for target in refined:
        for idx, sample in enumerate(target):
            result[idx].extend(sample[sample[..., 0]==1])

    return result
    # list batch box(obj, x,y,w,h,cls)
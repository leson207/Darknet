import torch
import json
import random
import numpy as np
from PIL import ImageDraw, ImageFont

from src.entity.entity import InferencerConfig
from src.component.Preprocessor import Preprocessor
from src.utils.yolo import format_prediction, non_max_suppression

def yolo_to_pascal_voc_format(box):
    x1 = box[...,0]-box[...,2]/2
    y1 = box[...,1]-box[...,3]/2
    x2 = box[...,0]+box[...,2]/2
    y2 = box[...,1]+box[...,3]/2

    return np.array([x1,y1,x2,y2])

class Inferencer:
    def __init__(self, config: InferencerConfig):
        self.anchors=torch.tensor(config.ANCHORS)
        self.grid_sizes=config.GRID_SIZES
        with open(config.DATA_DIR+'/id_to_label.json') as file:
            self.id_to_label = json.load(file)
        
        self.transform=Preprocessor(config.PREPROCESSOR_CONFIG).test_transform

    def post_process(self, img, input,  output):
        output=format_prediction(output, self.anchors, self.grid_sizes)
        boxes=non_max_suppression(output)

        img_size=img.size
        scale_factor=max(img_size[0], img_size[1])/input.shape[-1]
        return self.draw_boxes(img, boxes[0], scale_factor)

    def draw_boxes(self, image, boxes, scale_factor):
        colors = {}
        draw = ImageDraw.Draw(image)
        for box in boxes:
            box=np.array(box)
            label_idx=str(int(box[5]))
            label_prob=box[6]
            label=self.id_to_label[label_idx]
            box=box[1:5]
            box=yolo_to_pascal_voc_format(box)
            box = [int(coord*scale_factor) for coord in box]
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
            draw.text(text_position, label+': '+str(label_prob), fill=(255, 255, 255), font=font)

        return image
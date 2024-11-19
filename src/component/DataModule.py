import json
import joblib
import numpy as np
from datasets import load_from_disk

import torch
from torch.utils.data import DataLoader

from src.entity.entity import DataModuleConfig
from src.logger import logger

class Collator:
    def __init__(self, config:DataModuleConfig):
        self.anchors=np.array(config.ANCHORS)
        self.image_size=config.IMAGE_SIZE
        self.num_scale=self.anchors.shape[0]
        self.num_anchors_per_scale=self.anchors.shape[1]
        self.grid_sizes=[self.image_size//32, self.image_size//16, self.image_size//8]

        self.train_transform=joblib.load(config.PREPROCESSOR_DIR+ '/train_transform.pkl')
        self.test_transform=joblib.load(config.PREPROCESSOR_DIR+ '/test_transform.pkl')

        self.transform_fn=self.train_transform
        with open(config.DATA_DIR+'/label_to_id.json') as file:
            self.label_to_id = json.load(file)

    def set_mode(self, split):
        if split=='train':
            self.transform_fn=self.train_transform
        else:
            self.transform_fn=self.test_transform

    def __call__(self, batch):
        images=[]
        target=[np.zeros([len(batch), self.num_anchors_per_scale, grid_size,grid_size, 6], dtype='float32') for grid_size in self.grid_sizes]
        for batch_idx, sample in enumerate(batch):
            image = sample['image']
            bboxes = [i['bbox'] for i in sample['label_bbox']]
            labels = [i['label'] for i in sample['label_bbox']]
            processed = self.transform_fn(image=image, bboxes=bboxes, labels=labels)
            image, bboxes, labels = processed['image'], processed['bboxes'], processed['labels']
            images.append(image)

            for box, label in zip(bboxes, labels):
                box=self.coco_to_yolo_format(box)
                for scale_idx, grid_size in enumerate(self.grid_sizes):
                    cell_size = self.image_size/grid_size
                    i=int(box[1]/cell_size)
                    j=int(box[0]/cell_size)
                    ious=self.anchor_iou(self.anchors[scale_idx], box)
                    selected_anchor_idx=np.argmax(ious)
                    selected_anchor = self.anchors[scale_idx][selected_anchor_idx]

                    x = (box[0]-j*cell_size)/cell_size
                    y = (box[1]-i*cell_size)/cell_size

                    # real_w = exp(pred_w)*anchor_w*image_size
                    # real_h = exp(pred_h)*anchor_h*image_size
                    w = np.log(box[2]/selected_anchor[0]/self.image_size)
                    h = np.log(box[3]/selected_anchor[1]/self.image_size)

                    target[scale_idx][batch_idx, selected_anchor_idx, i, j,0]= 1
                    target[scale_idx][batch_idx, selected_anchor_idx, i, j,1:5] = x,y,w,h
                    target[scale_idx][batch_idx, selected_anchor_idx, i, j, 5] = self.label_to_id[label]

        images=torch.stack(images, dim=0)
        target=[torch.tensor(i) for i in target]
        return images, target

    def coco_to_yolo_format(self,box):
        x,y,w, h = box

        x_center = x+w/2
        y_center = y+h/2

        return x_center,y_center,w,h

    @staticmethod
    def anchor_iou(anchors, box):
        anchors_area = anchors[..., 0]*anchors[..., 1]
        box_area = box[0]*box[1]
        intersection = np.minimum(anchors[..., 0], box[0]) * np.minimum(anchors[..., 1], box[1])
        iou = intersection/(anchors_area+box_area-intersection)
        return iou

class DataModule:
    def __init__(self, config:DataModuleConfig):
        self.dataset=load_from_disk(config.DATA_DIR).with_format("numpy")
        with open(config.DATA_DIR+'/label_to_id.json') as file:
            self.label_to_id = json.load(file)
        with open(config.DATA_DIR+'/id_to_label.json') as file:
            self.id_to_label = json.load(file)

        self.collator=Collator(config)

    def get_data_loader(self,split, batch_size=4, shuffle=False):
        self.collator.set_mode(split)
        logger.info(f'Creating dataloader for {split} set')
        return DataLoader(self.dataset[split], batch_size=batch_size, shuffle=shuffle, drop_last=True, collate_fn=self.collator)
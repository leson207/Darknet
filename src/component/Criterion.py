import torch
from src.utils.yolo import *
from src.entity.entity import LossConfig


class YOLOLoss:
    def __init__(self, config:LossConfig):
        self.anchors=torch.tensor(config.ANCHORS)
        self.lambda_obj=config.LAMBDA_OBJ
        self.lambda_noobj=config.LAMBDA_NOOBJ
        self.lambda_coord=config.LAMBDA_COORD
        self.lambda_class=config.LAMBDA_CLASS

        self.mse = torch.nn.MSELoss()
        self.ce = torch.nn.CrossEntropyLoss()
        self.bce = torch.nn.BCEWithLogitsLoss()

    def __call__(self, preds, targets):
        loss=0.0
        for pred, target, anchor in zip(preds, targets, self.anchors):
            loss=loss+self.single_loss(pred, target, anchor)

        return loss

    def single_loss(self, pred, target, anchors):
        obj = target[..., 0]==1
        noobj = target[..., 0]==0

        # no obj loss
        noobj_loss = self.bce(pred[..., 0][noobj], target[..., 0][noobj])

        # obj loss
        anchors = anchors.reshape(1,3,1,1,2)
        pred_boxes = torch.cat([torch.sigmoid(pred[...,1:3]), torch.exp(pred[...,3:5])*anchors], dim=-1)
        iou_scores = yolo_iou(pred_boxes[obj].detach(), target[..., 1:5][obj].detach())
        obj_loss = self.bce(pred[...,0][obj], iou_scores*target[...,0][obj])

        # box loss
        pred_boxes = torch.cat([torch.sigmoid(pred[..., 1:3]), pred[..., 3:5]], dim=-1)
        true_boxes = torch.cat([target[..., 1:3], target[...,3:5]], axis=-1)
        coord_loss = self.mse(pred_boxes[obj], true_boxes[obj])
        # class loss
        class_loss = self.ce(pred[...,5:][obj], target[..., 5][obj].long())

        return self.lambda_noobj*noobj_loss + self.lambda_obj*obj_loss + self.lambda_coord*coord_loss + self.lambda_class*class_loss
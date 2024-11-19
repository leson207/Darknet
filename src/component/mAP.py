import torch
import numpy as np

from src.entity.entity import mAPConfig
from src.utils.yolo import yolo_iou, format_prediction, non_max_suppression, get_true_boxes

class MAP:
    def __init__(self, config:mAPConfig):
        super().__init__()
        self.eps=config.EPS
        self.num_classes=config.NUM_CLASSES
        self.anchors=torch.tensor(config.ANCHORS)
        self.grid_sizes=[config.IMAGE_SIZE//32, config.IMAGE_SIZE//16, config.IMAGE_SIZE//8]

        self.iou_thresholds=config.IOU_THRESHOLD

        self.TP_FP=[[] for _ in range(self.num_classes)]
        self.TP_FN=np.zeros([self.num_classes])

    def box_count(self, pred_boxes,true_boxes):
        for instance_pred_boxes, instance_true_boxes in zip(pred_boxes, true_boxes):
            if len(instance_true_boxes)==0:
                   continue

            instance_pred_boxes = sorted(instance_pred_boxes, key=lambda x:x[5], reverse=True)

            for true_box in instance_true_boxes:
                self.TP_FN[int(true_box[4])]+=1

            matched=[False for _ in range(len(instance_true_boxes))]
            instance_pred_boxes = torch.stack(instance_pred_boxes, dim=0)
            instance_true_boxes = torch.stack(instance_true_boxes, dim=0)
            ious=yolo_iou(instance_pred_boxes.unsqueeze(1), instance_true_boxes.unsqueeze(0))
            best_scores,best_indices = torch.max(ious, dim=1)

            for idx, (best_score, best_idx) in enumerate(zip(best_scores, best_indices)):
                pred_box=instance_pred_boxes[idx]
                if matched[best_idx]==False:
                    self.TP_FP[int(pred_box[4])].append([best_score, 'TP'])
                    matched[best_idx]=True
                else:
                    self.TP_FP[int(pred_box[4])].append([best_score, 'FP'])

    def reset(self):
        self.TP_FP=[[] for _ in range(self.num_classes)]
        self.TP_FN=np.zeros([self.num_classes])

    def update(self, preds, targets):
        true_boxes=get_true_boxes(targets, self.anchors, self.grid_sizes)
        preds=format_prediction(preds, self.anchors, self.grid_sizes)
        pred_boxes=non_max_suppression(preds, iou_threshold=0.05, obj_threshold=0.05)

        self.box_count(pred_boxes,true_boxes)

    def compute(self):
        mAP=np.zeros([len(self.iou_thresholds), self.num_classes])
        res={}
        for cls_idx, P in enumerate(self.TP_FP):
            for threshold_idx, threshold in enumerate(self.iou_thresholds):
                P=sorted(P, key= lambda x: x[0], reverse=True)
                TP=[1 if threshold<=x[0] and x[1]=='TP' else 0 for x in P]
                FP=[1 if threshold>x[0] and x[1]=='FP' else 0 for x in P]
                TP_cumsum=torch.cumsum(torch.tensor(TP),dim=0)
                FP_cumsum=torch.cumsum(torch.tensor(FP),dim=0)
                recall=TP_cumsum/(self.TP_FN[cls_idx]+self.eps)
                recall=torch.cat([torch.tensor([0]), recall])

                precision=torch.divide(TP_cumsum, TP_cumsum+FP_cumsum+self.eps)
                precision=torch.cat([torch.tensor([1]), precision])

                mAP[threshold_idx][cls_idx]=torch.trapz(precision, recall)

            res['mAP@'+str(threshold)]=sum(mAP[threshold_idx])/len(mAP[threshold_idx])

        return res
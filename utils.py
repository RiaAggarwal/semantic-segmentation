# def iou(pred, target):
#     ious = []
#     for cls in range(n_class):
#         # Complete this function
#         intersection = # intersection calculation
#         union = #Union calculation
#         if union == 0:
#             ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
#         else:
#             # Append the calculated IoU to the list ious
#     return ious

import torch
import numpy as np
from dataloader import useful_ids

def pixel_acc(pred, Y):
    pred = pred.cpu()
    Y = Y.cpu()
    Y_num = Y.numpy()
    pred_num = pred.numpy()
    return np.count_nonzero(np.logical_and((pred_num == Y_num), np.isin(Y_num, useful_ids, invert=True))), np.count_nonzero(np.isin(Y_num, useful_ids, invert=True))

def iou(pred, Y, class_id):
    pred = pred.cpu()
    Y = Y.cpu()
    pred = pred.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    
    intersection = np.count_nonzero(np.logical_and(pred == class_id, Y == class_id)) 
    union = np.count_nonzero(np.logical_or(pred == class_id, Y == class_id))
    
    iou = (intersection) / (union)  # We smooth our devision to avoid 0/0

    return iou 
# def iou(preds, labels):
#     preds = 
#     imPred = np.asarray(imPred).copy()
#     imLab = np.asarray(imLab).copy()

#     imPred += 1
#     imLab += 1
#     # Remove classes from unlabeled pixels in gt image.
#     # We should not penalize detections in unlabeled portions of the image.
#     imPred = imPred * (imLab > 0)

#     # Compute area intersection:
#     intersection = imPred * (imPred == imLab)
#     (area_intersection, _) = np.histogram(
#         intersection, bins=numClass, range=(1, numClass))

#     # Compute area union:
#     (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
#     (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
#     area_union = area_pred + area_lab - area_intersection

#     return (area_intersection, area_union)
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

# (pred == Y) and

def pixel_acc(pred, Y):
    pred = pred.cpu()
    Y = Y.cpu()
    Y_num = Y.numpy()
    pred_num = pred.numpy()
    return np.sum(np.logical_and((pred_num == Y_num), np.isin(Y_num, useful_ids, invert=True)))/np.size(pred_num)
#     print(np.logical_not((np.isin(Y, useful_ids))))
#    return np.sum(1*(np.logical_and((pred == Y), np.logical_not(np.isin(Y, useful_ids)))))/np.numel(pred)

# def iou(pred, target):
#     pred = pred.cpu()
#     target = target.cpu()
#     pred = pred.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    
#     intersection = (pred & target).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
#     union = (pred | target).float().sum((1, 2))         # Will be zzero if both are 0
    
#     iou = (intersection) / (union)  # We smooth our devision to avoid 0/0

#     return iou 
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


# def pixel_acc(preds, label):
#     preds = preds.cpu()
#     label = label.cpu()
#     valid = (label >= 0)
#     acc_sum = (valid * (preds == label)).sum()
#     valid_sum = valid.sum()
#     acc = float(acc_sum) / (valid_sum + 1e-10)
#     return acc, valid_sum
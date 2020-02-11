import torch
import numpy as np
from dataloader import useful_ids

def pixel_acc(pred, Y):
    Y_num = Y.numpy()
    pred_num = pred.numpy()
    return np.count_nonzero(np.logical_and((pred_num == Y_num), np.isin(Y_num, useful_ids, invert=True))), np.count_nonzero(np.isin(Y_num, useful_ids, invert=True))

def iou(pred, Y, class_id):
    pred = pred.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    
    intersection = np.count_nonzero(np.logical_and(pred == class_id, Y == class_id)) 
    union = np.count_nonzero(np.logical_or(pred == class_id, Y == class_id))
    
    return intersection, union  # We smooth our devision to avoid 0/0
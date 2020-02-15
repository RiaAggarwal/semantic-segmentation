import torch
import torch.nn as nn
import numpy as np
from dataloader import useful_ids

def pixel_acc(pred, Y):
    Y_num = Y.numpy()
    pred_num = pred.numpy()
    return np.count_nonzero(np.logical_and((pred_num == Y_num), np.isin(Y_num, useful_ids, invert=False))), np.count_nonzero(np.isin(Y_num, useful_ids, invert=False))

def iou(pred, Y, class_id):
    pred = pred.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    
    intersection = np.count_nonzero(np.logical_and(pred == class_id, Y == class_id)) 
    union = np.count_nonzero(np.logical_or(pred == class_id, Y == class_id))
    
    return intersection, union  # We smooth our devision to avoid 0/0

def tensordot_pytorch(a, b, axes=2):
    # code adapted from numpy
    try:
        iter(axes)
    except Exception:
        axes_a = list(range(-axes, 0))
        axes_b = list(range(0, axes))
    else:
        axes_a, axes_b = axes
    try:
        na = len(axes_a)
        axes_a = list(axes_a)
    except TypeError:
        axes_a = [axes_a]
        na = 1
    try:
        nb = len(axes_b)
        axes_b = list(axes_b)
    except TypeError:
        axes_b = [axes_b]
        nb = 1
    
    # uncomment in pytorch >= 0.5
    # a, b = torch.as_tensor(a), torch.as_tensor(b)
    as_ = a.shape
    nda = a.dim()
    bs = b.shape
    ndb = b.dim()
    equal = True
    if na != nb:
        equal = False
    else:
        for k in range(na):
            if as_[axes_a[k]] != bs[axes_b[k]]:
                equal = False
                break
            if axes_a[k] < 0:
                axes_a[k] += nda
            if axes_b[k] < 0:
                axes_b[k] += ndb
    if not equal:
        raise ValueError("shape-mismatch for sum")

    # Move the axes to sum over to the end of "a"
    # and to the front of "b"
    notin = [k for k in range(nda) if k not in axes_a]
    newaxes_a = notin + axes_a
    N2 = 1
    for axis in axes_a:
        N2 *= as_[axis]
    newshape_a = (int(np.multiply.reduce([as_[ax] for ax in notin])), N2)
    olda = [as_[axis] for axis in notin]

    notin = [k for k in range(ndb) if k not in axes_b]
    newaxes_b = axes_b + notin
    N2 = 1
    for axis in axes_b:
        N2 *= bs[axis]
    newshape_b = (N2, int(np.multiply.reduce([bs[ax] for ax in notin])))
    oldb = [bs[axis] for axis in notin]

    at = a.permute(newaxes_a).reshape(newshape_a)
    bt = b.permute(newaxes_b).reshape(newshape_b)

    res = at.matmul(bt)
    return res.reshape(olda + oldb)

# def weighted_ce_loss(outputs, targets_one_hot, weighted=False):
#     cls_weights = torch.Tensor(np.ones(34)).cuda()
#     if weighted:
#         ss = targets_one_hot.sum()
#         cls_weights = ss/targets_one_hot.sum((0,2,3))
#         cls_weights[cls_weights > ss] = 0
    
#     logp = nn.functional.log_softmax(outputs,1)

#     return -torch.mean(tensordot_pytorch(targets_one_hot*logp, cls_weights, axes=[1,0]))

def weighted_ce_loss(outputs, targets_one_hot, loaded_cls_weights, weighted=False):
    sizes = outputs.size()
    if weighted:
        cls_weights = loaded_cls_weights
#         ss = targets_one_hot.sum()
#         cls_weights = ss/targets_one_hot.sum((0,2,3))
#         cls_weights[cls_weights > ss] = 0
    else:
        cls_weights = torch.Tensor(np.ones(sizes[1])).cuda()
    
    logp = nn.functional.log_softmax(outputs,1)
    appended_weightes = cls_weights.expand(sizes[3], sizes[1]).expand(sizes[2], sizes[3], sizes[1]).expand(sizes[0], sizes[2], sizes[3], sizes[1]).permute(0,3,1,2)
    
    return -torch.mean((targets_one_hot*logp)*appended_weightes)

def dice_loss(outputs, targets_one_hot, loaded_cls_weights, weighted=False):
    sizes = outputs.size()
    if weighted:
        cls_weights = loaded_cls_weights
#         ss = targets_one_hot.sum()
#         cls_weights = ss/targets_one_hot.sum((0,2,3))
#         cls_weights[cls_weights > ss] = 0
    else:
        cls_weights = torch.Tensor(np.ones(sizes[1])).cuda()
        
    soft_outputs = nn.functional.softmax(outputs, dim=1)
    appended_weightes = cls_weights.expand(sizes[3], sizes[1]).expand(sizes[2], sizes[3], sizes[1]).expand(sizes[0], sizes[2], sizes[3], sizes[1]).permute(0,3,1,2)
    
    weighted_targets = targets_one_hot*appended_weightes
#     weighted_targets = tensordot_pytorch(targets_one_hot, cls_weights, axes=[1,0])
    
    smooth = 1.
    iflat = soft_outputs.view(-1)
    tflat = weighted_targets.view(-1)
    intersection = (iflat * tflat).sum()
    
    return 1 - ((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth))
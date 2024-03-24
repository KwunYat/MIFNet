import torch
import torch.nn.functional as F
import numpy as np


def color(pred, target,FNC, FPC):
    #pred = pred.cpu().numpy()
    #target = target.cpu().numpy()
    result = np.zeros(pred.shape)
    for i, pred_i in np.ndenumerate(pred):
        if pred_i < target[i]:
            result[i] = FNC
        elif pred_i > target[i]:
            result[i] = FPC
        else:
            result[i] = target[i]
    return result


def evaluation(pred, target):
    eps = 0.0001
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    pred_oa = np.where(pred < 0.5, 0, 1)
    intersection = np.logical_and(target, pred)
    union = np.logical_or(target, pred)
    iou_score = np.sum(intersection) / (np.sum(union) + eps)
    oa = np.sum(np.equal(pred_oa, target)) / (target.shape[1] * target.shape[2])
    pred_tp = np.where(pred < 0.5, 0.5, 1)
    tp = np.sum(np.equal(pred_tp, target))
    recall = (tp + eps) / (np.sum(np.equal(target, 1)) + eps)
    fn = np.sum(np.equal(target, 1)) - tp
    fp = np.sum(np.equal(pred, 1)) - tp
    tn = np.sum(np.equal(pred_oa, target)) - tp
    precision = (tp + eps) / (np.sum(np.equal(pred, 1)) + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return oa, recall, precision, f1, iou_score, tp, fn, tn, fp


def eval_net(net, dataset, gpu=False):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    oa, recall, precision, f1, iou_score= 0, 0, 0, 0, 0
    for i, b in enumerate(dataset):
        img = b[0].astype(np.float32)
        true_mask = b[1].astype(np.float32)

        img = torch.from_numpy(img).unsqueeze(0)
        true_mask = torch.from_numpy(true_mask).unsqueeze(0)

        if gpu:
            img = img.cuda()
            true_mask = true_mask.cuda()

        mask_pred = net(img)[0]
        mask_pred = (mask_pred > 0.5).float()

        oa += evaluation(mask_pred, true_mask)[0].item()
        recall += evaluation(mask_pred, true_mask)[1].item()
        precision += evaluation(mask_pred, true_mask)[2].item()
        f1 += evaluation(mask_pred, true_mask)[3].item()
        iou_score += evaluation(mask_pred, true_mask)[4].item()
    return iou_score / (i + 1), oa / (i + 1), recall / (i + 1), precision / (i + 1), f1 / (i + 1)


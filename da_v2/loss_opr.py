import sys
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import shutil

from torch.autograd import Variable
from torch.utils import data
import os

from dataset import IC15_TT_Loader
from metrics import runningScore
import models
from util import Logger, AverageMeter
import time
import util

def ohem_single(score, gt_text, training_mask):
    pos_num = (int)(torch.sum(gt_text > 0.5)) - (int)(torch.sum((gt_text > 0.5) & (training_mask <= 0.5)))

    if pos_num == 0:
        selected_mask = training_mask
        selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).float()
        return selected_mask

    neg_num = (int)(torch.sum(gt_text <= 0.5))
    neg_num = (int)(min(pos_num * 3, neg_num))

    if neg_num == 0:
        selected_mask = training_mask
        selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).float()
        return selected_mask

    neg_score = score[gt_text <= 0.5]
    neg_score_sorted, _ = torch.sort(-neg_score)
    threshold = -neg_score_sorted[neg_num - 1]

    selected_mask = ((score >= threshold) | (gt_text > 0.5)) & (training_mask > 0.5)
    selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).float()
    return selected_mask


def ohem_batch(scores, gt_texts, training_masks):
    selected_masks = []
    for i in range(scores.shape[0]):
        selected_masks.append(ohem_single(scores[i, :, :], gt_texts[i, :, :], training_masks[i, :, :]))

    selected_masks = torch.cat(selected_masks, 0).float()

    return selected_masks


def dice_loss(input, target, mask):
    input = torch.sigmoid(input)

    input = input.contiguous().view(input.size()[0], -1)
    target = target.contiguous().view(target.size()[0], -1)
    mask = mask.contiguous().view(mask.size()[0], -1)

    input = input * mask
    target = target * mask

    a = torch.sum(input * target, 1)
    b = torch.sum(input * input, 1) + 0.001
    c = torch.sum(target * target, 1) + 0.001
    d = (2 * a) / (b + c)
    dice_loss = torch.mean(d)
    return 1 - dice_loss


def L2_loss(input, target):
    crit = torch.nn.MSELoss(reduction='mean')
    loss = crit(input, target)
    return loss


def cal_text_score(texts, gt_texts, training_masks, running_metric_text):
    # training_masks = training_masks.data.cpu().numpy()
    pred_text = torch.sigmoid(texts) * training_masks
    pred_text[pred_text <= 0.5] = 0
    pred_text[pred_text > 0.5] = 1
    pred_text = pred_text.int()
    pred_text = pred_text.data.cpu().numpy()

    gt_text = (gt_texts * training_masks).int()
    gt_text = gt_text.data.cpu().numpy()
    running_metric_text.update(gt_text, pred_text)
    score_text, _ = running_metric_text.get_scores()
    return score_text


def cal_kernel_score(kernels, gt_kernels, gt_texts, training_masks, running_metric_kernel):
    mask = (gt_texts * training_masks)
    kernel = kernels[:, -1, :, :]
    gt_kernel = gt_kernels[:, -1, :, :]
    pred_kernel = torch.sigmoid(kernel)
    pred_kernel[pred_kernel <= 0.5] = 0
    pred_kernel[pred_kernel > 0.5] = 1
    pred_kernel = (pred_kernel * mask).int()

    gt_kernel = (gt_kernel * mask).int()
    gt_kernel = gt_kernel.data.cpu().numpy()
    pred_kernel = pred_kernel.data.cpu().numpy()

    running_metric_kernel.update(gt_kernel, pred_kernel)
    score_kernel, _ = running_metric_kernel.get_scores()
    return score_kernel

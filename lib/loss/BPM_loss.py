# -*- coding: utf-8 -*-
import torch
import numpy as np
import torch.nn.functional as F




def bpm_loss_func(pred_action, pred_start, pred_end,gt_action, gt_start, gt_end):
    def bi_loss(pred_score, gt_label):
        pred_score = pred_score.view(-1)
        gt_label = gt_label.view(-1)
        pmask = (gt_label > 0.5).float()
        num_entries = len(pmask)
        num_positive = torch.sum(pmask)
        ratio = num_entries / num_positive
        coef_0 = 0.5 * ratio / (ratio - 1)
        coef_1 = 0.5 * ratio
        epsilon = 0.000001
        loss_pos = coef_1 * torch.log(pred_score + epsilon) * pmask
        loss_neg = coef_0 * torch.log(1.0 - pred_score + epsilon) * (1.0 - pmask)
        loss = -1 * torch.mean(loss_pos + loss_neg)
        return loss
    loss_action = bi_loss(pred_action, gt_action)
    loss_start = bi_loss(pred_start, gt_start)
    loss_end = bi_loss(pred_end, gt_end)
    loss = loss_start + loss_end + loss_action
    return loss ,loss_action, loss_start, loss_end


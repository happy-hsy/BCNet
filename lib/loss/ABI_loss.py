# -*- coding: utf-8 -*-
import torch
import numpy as np
import torch.nn.functional as F



def aem_loss_func(pred_a, gt_iou_map):
    pred_a_reg = pred_a[:, :, 0].contiguous()
    pred_a_cls = pred_a[:, :, 1].contiguous()
    aem_reg_loss = reg_loss_func(pred_a_reg, gt_iou_map)
    aem_cls_loss = cls_loss_func(pred_a_cls, gt_iou_map)
    loss = aem_reg_loss + aem_cls_loss
    return loss, aem_reg_loss, aem_cls_loss

def bem_loss_func(pred_bm, gt_iou_map):
    pred_bm_cls = pred_bm[:, :, 0].contiguous()
    bem_cls_loss = cls_loss_func(pred_bm_cls, gt_iou_map)
    loss = bem_cls_loss
    return loss, bem_cls_loss


def abi_loss_func(cls_p, cls_b ,confidence_p, confidence_b,match_score_action,match_score_start,match_score_end,match_score_background,gt_proposal,gt_background):
    gt_proposal = gt_proposal.cuda()
    gt_background = gt_background.cuda()
    match_score_action = match_score_action.cuda()
    match_score_start = match_score_start.cuda()
    match_score_end = match_score_end.cuda()
    match_score_background = match_score_background.cuda()
    
    loss_p, aem_reg_loss, aem_cls_loss = aem_loss_func(confidence_p,gt_proposal)
    loss_b, bem_cls_loss = bem_loss_func(confidence_b,gt_background)
    loss_frame ,loss_action, loss_background = frame_loss_func(cls_p, cls_b,match_score_action,match_score_start,match_score_end,match_score_background)
    loss = loss_p + loss_b + loss_frame

   
   
    return loss, loss_frame, loss_p, loss_b


def frame_loss_func(cls_p,pred_background,gt_action,gt_start,gt_end,gt_background):
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
    
    pred_action = cls_p[:, :, 0].contiguous()

    loss_action = bi_loss(pred_action, gt_action)

    loss_background = bi_loss(pred_background, gt_background)

    loss = loss_background + loss_action
    return loss ,loss_action, loss_background


def reg_loss_func(pred_score, gt_iou_map):
    u_hmask = (gt_iou_map > 0.7).float()
    u_mmask = ((gt_iou_map <= 0.7) & (gt_iou_map > 0.3)).float()
    u_lmask = ((gt_iou_map <= 0.3) & (gt_iou_map > 0.)).float()
    u_lmask = u_lmask #* mask

    num_h = torch.sum(u_hmask)
    num_m = torch.sum(u_mmask)
    num_l = torch.sum(u_lmask)

    r_m = num_h / num_m
    u_smmask = torch.Tensor(np.random.rand(*gt_iou_map.shape)).cuda()
    u_smmask = u_mmask * u_smmask
    u_smmask = (u_smmask > (1. - r_m)).float()

    r_l = num_h / num_l
    u_slmask = torch.Tensor(np.random.rand(*gt_iou_map.shape)).cuda()
    u_slmask = u_lmask * u_slmask
    u_slmask = (u_slmask > (1. - r_l)).float()

    weights = u_hmask + u_smmask + u_slmask

    loss = F.smooth_l1_loss(pred_score * weights, gt_iou_map * weights)
    loss = torch.sum(loss * torch.ones(*weights.shape).cuda()) / torch.sum(weights)

    return loss


def cls_loss_func(pred_score, gt_iou_map):
    pmask = (gt_iou_map > 0.8).float()
    nmask = (gt_iou_map <= 0.8).float()
    nmask = nmask# * mask

    num_positive = torch.sum(pmask)
    num_entries = num_positive + torch.sum(nmask)
    ratio = num_entries / num_positive
    coef_0 = 0.5 * ratio / (ratio - 1)
    coef_1 = 0.5 * ratio
    epsilon = 0.000001
    loss_pos = coef_1 * torch.log(pred_score + epsilon) * pmask
    loss_neg = coef_0 * torch.log(1.0 - pred_score + epsilon) * nmask
    loss = -1 * torch.sum(loss_pos + loss_neg) / num_entries
    return loss


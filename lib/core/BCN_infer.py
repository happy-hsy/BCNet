import sys
import os

sys.path.insert(0, os.getcwd())
print(os.getcwd())
import _init_paths
from lib.dataset.dataset_thu_abi import VideoDataSet
import json
import torch
import torch.nn.parallel
import torch.optim as optim
import numpy as np
from lib.utils import opts
import pandas as pd
from lib.utils.post_processing import BCN_post_processing
from lib.eval.thumos_AR.eval import evaluation_proposal
import math

from lib.model.bpm_model import get_bpm_model
from lib.model.abi_model import get_abi_model

import time
from lib.utils.log_save import create_logger

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
logger = create_logger("BCN_infer")


def iou_with_anchors(anchors_min, anchors_max, box_min, box_max):
    """Compute jaccard score between a box and the anchors.
    """
    len_anchors = anchors_max - anchors_min
    int_xmin = np.maximum(anchors_min, box_min)
    int_xmax = np.minimum(anchors_max, box_max)
    inter_len = np.maximum(int_xmax - int_xmin, 0.)
    union_len = len_anchors - inter_len + box_max - box_min
    # print inter_len,union_len
    jaccard = np.divide(inter_len, union_len)
    return jaccard


def BCN_infer(opt):
    results_path = opt['output_path'] + 'BCN_results'
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    model_bpm = get_bpm_model()
    model_abi = get_abi_model()
    model_bpm = torch.nn.DataParallel(model_bpm, device_ids=[0]).cuda()
    model_abi = torch.nn.DataParallel(model_abi, device_ids=[0]).cuda()
    bpm_check_str = "/" + str(opt["bpm_check_num"]) + "_BPM_checkpoint.pth.tar"
    abi_check_str = "/" + str(opt["abi_check_num"]) + "_ABI_checkpoint.pth.tar"

    bpm_checkpoint = torch.load(opt["checkpoint_path"] + bpm_check_str)
    abi_checkpoint = torch.load(opt["checkpoint_path"] + abi_check_str)
    
    model_bpm.load_state_dict(bpm_checkpoint['state_dict'], strict=False)
    model_bpm.eval()
    model_abi.load_state_dict(abi_checkpoint['state_dict'], strict=False)
    model_abi.eval()


    print_info = "the bpm_model epoch is {}".format(bpm_checkpoint["epoch"])
    logger.info(print_info)

    print_info = "the abi_model epoch is {}".format(abi_checkpoint["epoch"])
    logger.info(print_info)

    test_loader = torch.utils.data.DataLoader(VideoDataSet(opt, subset="val", mode='inference'),
                                              batch_size=1, shuffle=False,
                                              num_workers=8, pin_memory=True, drop_last=False)
    with torch.no_grad():

        for idx, input_data, sample_mask, can_proposal_s, can_proposal_e in test_loader:
            batch = sample_mask.shape[0]
            nums = sample_mask.shape[1]
            can_proposal_s = can_proposal_s[0]
            can_proposal_e = can_proposal_e[0]

            video_name = test_loader.dataset.video_list[idx[0]]

            offset = min(test_loader.dataset.data['indices'][idx[0]])
            video_name = video_name + '_{}'.format(math.floor(offset / 250))
            input_data = input_data.cuda()
            sample_mask = sample_mask.cuda().reshape(batch, nums, -1)
            se = model_bpm(input_data)
            _, _, pred_am, pred_bm = model_abi(input_data, sample_mask)


            start_scores = se[:, :, 1][0].detach().cpu().numpy()
            end_scores = se[:, :, 2][0].detach().cpu().numpy()
            # bem result
            pred_bm_cls = pred_bm[0, :, 0].detach().cpu().numpy()

            # pem result
            pred_am_reg = pred_am[0, :, 0].detach().cpu().numpy()
            pred_am_cls = pred_am[0, :, 1].detach().cpu().numpy()

            new_props = []
            for idx in range(opt["temporal_scale"]):
                start_index = idx
                if start_index == 0:
                    start_index = 1
                if start_index == 255:
                    start_index = 254
                if start_scores[start_index] >= opt['proposal_thres'] or (
                        start_scores[start_index] > start_scores[start_index - 1] and start_scores[start_index] >
                        start_scores[start_index + 1]):
                    for jdx in range(opt["max_D"]):
                        end_index = start_index + jdx + 1
                        if end_index < opt["temporal_scale"] - 1:
                            if end_scores[end_index] >= opt['proposal_thres'] or (
                                    end_scores[end_index] > end_scores[end_index - 1] and end_scores[end_index] >
                                    end_scores[end_index + 1]):
                                iou_anchor = iou_with_anchors(can_proposal_s, can_proposal_e, start_index, end_index)
                                max_idx = iou_anchor.argmax()
                                ##bem score
                                bm_clr_score = (1 - pred_bm_cls[max_idx])
                                ##pem score
                                pm_clr_score = pred_am_cls[max_idx]
                                pm_reg_score = pred_am_reg[max_idx]

                                if pm_clr_score > 0.9 and pm_reg_score > 0.8:
                                    xmin = (can_proposal_s[max_idx] + start_index) / 2. * opt['skip_videoframes'] + offset
                                    xmax = (can_proposal_e[max_idx] + end_index) / 2. * opt['skip_videoframes'] + offset
                                else:
                                    xmin = start_index * opt['skip_videoframes'] + offset  # map [0,99] to frames
                                    xmax = end_index * opt['skip_videoframes'] + offset
                                xmin_score = start_scores[start_index]
                                xmax_score = end_scores[end_index]
                                score = xmin_score * xmax_score * pm_clr_score * pm_reg_score * bm_clr_score 
                                new_props.append(
                                    [xmin, xmax, xmin_score, xmax_score, pm_clr_score, pm_reg_score, bm_clr_score,
                                     score])
            try:
                new_props = np.stack(new_props)
                col_name = ["xmin", "xmax", "xmin_score", "xmax_score", "pm_clr_score", "pm_reg_socre", "bm_clr_score",
                            "score"]
                new_df = pd.DataFrame(new_props, columns=col_name)
                new_df.to_csv("./result/output/BCN_results/" + video_name + ".csv", index=False)
                print(results_path + '/' + video_name + ".csv")
            except:
                print_info = video_name
                logger.info(print_info)

            
def main(opt):
    print("BCN inference start")
    BCN_infer(opt)
    BCN_post_processing(opt)

if __name__ == '__main__':
    opt = opts.parse_opt()
    opt = vars(opt)
    if not os.path.exists(opt["checkpoint_path"]):
        os.makedirs(opt["checkpoint_path"])
    opt_file = open(opt["checkpoint_path"] + "/infer_opts.json", "w")
    json.dump(opt, opt_file)
    opt_file.close()
    main(opt)

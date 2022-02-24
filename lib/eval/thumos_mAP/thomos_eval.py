import json
import os
import numpy as np

from lib.utils import opts

# from configuration import Logger, parse_base_args

label_id = [7, 9, 12, 21, 22, 23, 24, 26, 31, 33, 36, 40, 45, 51, 68, 79, 85, 92, 93, 97]
label_id = [i - 1 for i in label_id]
label_name = ['BaseballPitch', 'BasketballDunk', 'Billiards', 'CleanAndJerk', 'CliffDiving',
              'CricketBowling', 'CricketShot', 'Diving', 'FrisbeeCatch', 'GolfSwing', 'HammerThrow',
              'HighJump', 'JavelinThrow', 'LongJump', 'PoleVault', 'Shotput', 'SoccerPenalty', 'TennisSwing',
              'ThrowDiscus', 'VolleyballSpiking']


def iou_score(gt, anchor):
    gt_min, gt_max = gt
    an_min, an_max = anchor
    if (an_min >= gt_max) or (gt_min >= an_max):
        return 0.
    else:
        union = max(gt_max, an_max) - min(gt_min, an_min)
        inter = min(gt_max, an_max) - max(gt_min, an_min)
        return float(inter) / union


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)  # only difference


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def AddPreLabel(original, output):
    un = np.load('./lib/eval/thumos_mAP/data/uNet_test.npy')
    un_new = un[:, label_id]

    data = json.load(open(original))
    with open('./lib/eval/thumos_mAP/data/detclasslist.txt', 'r') as f:
        lines = f.readlines()

    out = {'version': 'THUMOS14', 'results': {}, 'external_data': '{}'}
    out['results'] = {vid: [] for vid in data['results'].keys()}

    for videoid, v in data['results'].items():
        # print(videoid)
        vid = int(videoid.split('_')[-1])

        tmp_list = []

        i = 0
        for result in v:
            # i += 1
            # if i > 200:
            #     break
            topK = 2
            class_tops = np.argsort(un_new[vid - 1])[::-1]
            class_scores = np.sort(softmax(un_new[vid - 1]))[::-1]

            for i in range(topK):
                score = result['score'] * class_scores[i]
                label = label_name[class_tops[i]]
                tmp_list.append({'score': score, 'label': label, 'segment': result['segment']})

        index = np.argsort([t['score'] for t in tmp_list])
        tmp_list = [tmp_list[idx] for idx in index[::-1]]

        if len(tmp_list) > 400:
            out['results'][videoid] = tmp_list[:400]
        else:
            out['results'][videoid] = tmp_list
    with open(output, 'w') as f:
        json.dump(out, f)


if __name__ == '__main__':
    opt = opts.parse_opt()
    opt = vars(opt)

    tapg_results_json = opt['output_path'] + '/result_proposal.json'   #os.path.join(save_path, 'proposals', 'results_softnms_n5_score_se_pem_epoch{}.json'.format(epoch))
    tad_results_json = os.path.join(opt['output_path'] + '/result_proposal_detect.json')
    AddPreLabel(tapg_results_json, tad_results_json)    
    from gtad_eval_detection import ANETdetection

    tious = [0.3, 0.4, 0.5, 0.6, 0.7]
    anet_detection = ANETdetection(
        ground_truth_filename='./lib/eval/thumos_mAP/data/thumos14.json',
        prediction_filename=tad_results_json,
        subset='test', tiou_thresholds=tious)
    mAPs, average_mAP = anet_detection.evaluate()
    for (tiou, mAP) in zip(tious, mAPs):
        print("mAP at tIoU {} is {}".format(tiou, mAP))


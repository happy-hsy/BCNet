# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import json
import multiprocessing as mp
import os
import time

def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data


def getDatasetDict(opt):
    df = pd.read_csv(opt["video_info"])
    json_data = load_json(opt["video_anno"])
    database = json_data
    video_dict = {}
    for i in range(len(df)):
        video_name = df.video.values[i]
        video_info = database[video_name]
        video_new_info = {}
        video_new_info['duration_frame'] = video_info['duration_frame']
        video_new_info['duration_second'] = video_info['duration_second']
        video_new_info["feature_frame"] = video_info['feature_frame']
        video_subset = df.subset.values[i]
        video_new_info['annotations'] = video_info['annotations']
        if video_subset == 'test':
            video_dict[video_name] = video_new_info
    return video_dict

def IOU(s1, e1, s2, e2):
    if (s2 > e1) or (s1 > e2):
        return 0
    Aor = max(e1, e2) - min(s1, s2)
    Aand = min(e1, e2) - max(s1, s2)
    return float(Aand) / Aor

def Soft_NMS(df, low,nms_range, nms_threshold=0.46, num_prop=1000):
    '''
    From BSN code
    :param df:
    :param nms_threshold:
    :return:
    '''
    df = df.sort_values(by="score", ascending=False)

    tstart = list(df.xmin.values[:])
    tend = list(df.xmax.values[:])
    tscore = list(df.score.values[:])

    rstart = []
    rend = []
    rscore = []


    for idx in range(0, len(tscore)):
        if tend[idx] - tstart[idx] >= 300:
            tscore[idx] = 0

    while len(tscore) > 1 and len(rscore) < num_prop and max(tscore)>0:
        max_index = tscore.index(max(tscore))
        for idx in range(0, len(tscore)):
            if idx != max_index:
                tmp_iou = IOU(tstart[max_index], tend[max_index], tstart[idx], tend[idx])
                tmp_width = tend[max_index] - tstart[max_index]
                #print('width',tmp_width)
                if tmp_iou > low+nms_range*tmp_width/300:
                    tscore[idx] = tscore[idx] * np.exp(-np.square(tmp_iou) / nms_threshold)

        rstart.append(tstart[max_index])
        rend.append(tend[max_index])
        rscore.append(tscore[max_index])
        tstart.pop(max_index)
        tend.pop(max_index)
        tscore.pop(max_index)

    newDf = pd.DataFrame()
    newDf['score'] = rscore
    newDf['xmin'] = rstart
    newDf['xmax'] = rend
    return newDf



def video_post_process(opt, video_list):
    thumos_gt = pd.read_csv(opt['video_info'] + "thumos14_test_groundtruth.csv")
    result = {
        video:
            {
                'fps': thumos_gt.loc[thumos_gt['video-name'] == video]['frame-rate'].values[0],
                'num_frames': thumos_gt.loc[thumos_gt['video-name'] == video]['video-frames'].values[0]
            }
        for video in video_list
    }
    for video_name in video_list:
        start = time.time()
        files = [opt['output_path'] + "BCN_results/" + f for f in os.listdir(opt['output_path'] + "BCN_results/") if
                 video_name in f]

        dfs = []  # merge pieces of video
        for snippet_file in files:
            snippet_df = pd.read_csv(snippet_file)
            snippet_df = Soft_NMS(snippet_df,0,0,nms_threshold=0.46,num_prop = 120)
            dfs.append(snippet_df)

        if len(files) == 0:
            df = pd.DataFrame({'score': [0], 'xmin': [0], 'xmax': [0]})
        else:
            df = pd.concat(dfs)

        if len(df) > 1:
            df = Soft_NMS(df, 0, 0, nms_threshold=0.46,num_prop = 1000)
        df = df.sort_values(by="score", ascending=False)

        # sort video classification
        fps = result[video_name]['fps']
        num_frames = result[video_name]['num_frames']
        proposal_list = []
        for j in range(min(1000, len(df))):
            tmp_proposal = {}
            tmp_proposal["score"] = float(round(df.score.values[j], 6))
            tmp_proposal["segment"] = [float(round(max(0, df.xmin.values[j]) / fps, 1)),
                                       float(round(min(num_frames, df.xmax.values[j]) / fps, 1))]
            proposal_list.append(tmp_proposal)
        result_dict[video_name] = proposal_list
        end = time.time()
        # print("finish", video_name,'cost time:',(end - start))



def BCN_post_processing(opt):
    thumos_test_anno = pd.read_csv(opt['video_info'] + "test_Annotation.csv")
    video_list = thumos_test_anno.video.unique()
    global result_dict
    result_dict = mp.Manager().dict()

    num_videos = len(video_list)
    num_videos_per_thread = num_videos // opt["post_process_thread"]
    processes = []
    for tid in range(opt["post_process_thread"] - 1):
        tmp_video_list = video_list[tid * num_videos_per_thread:(tid + 1) * num_videos_per_thread]
        p = mp.Process(target=video_post_process, args=(opt, tmp_video_list))
        p.start()
        processes.append(p)
    tmp_video_list = video_list[(opt["post_process_thread"] - 1) * num_videos_per_thread:]
    p = mp.Process(target=video_post_process, args=(opt, tmp_video_list))
    p.start()
    processes.append(p)
    for p in processes:
        p.join()

    result_dict = dict(result_dict)
    output_dict = {"version": "VERSION 1.3", "results": result_dict, "external_data": {}}
    outfile = open(opt["result_file"], "w")
    json.dump(output_dict, outfile)
    outfile.close()


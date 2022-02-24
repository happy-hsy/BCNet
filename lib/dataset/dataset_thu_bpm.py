import os
import numpy as np
import pandas as pd
import json, pickle
import torch.utils.data as data
import torch
import h5py

def ioa_with_anchors(anchors_min, anchors_max, box_min, box_max):
    # calculate the overlap proportion between the anchor and all bbox for supervise signal,
    # the length of the anchor is 0.01
    len_anchors = anchors_max - anchors_min
    int_xmin = np.maximum(anchors_min, box_min)
    int_xmax = np.minimum(anchors_max, box_max)
    inter_len = np.maximum(int_xmax - int_xmin, 0.)
    scores = np.divide(inter_len, len_anchors)
    return scores


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


def load_json(file):
    with open(file) as json_file:
        json_data = json.load(json_file)
        return json_data


class VideoDataSet(data.Dataset):  # thumos
    def __init__(self, opt, subset="train", mode="train"):
        self.temporal_scale = opt["temporal_scale"] #256   
        self.temporal_gap = 1. / self.temporal_scale  # 1/128
        self.subset = subset
        self.mode = mode
        self.feature_path = opt['feature_path'] 
        self.feature_dirs = [self.feature_path]
        self.video_info_path = opt['video_info'] 
        self.feat_dim = opt['feat_dim'] #2048
        self.num_sample = opt['num_sample'] #32
        self.num_sample_perbin = opt['num_sample_perbin'] #3
        #### THUMOS
        self.skip_videoframes = opt['skip_videoframes'] # 5
        self.num_videoframes =  opt['temporal_scale'] #256
        self.max_duration = opt['max_D'] #64
        self.min_duration = opt['min_D'] #0
        self._get_data()
        self.video_list = self.data['video_names']



    def _get_video_data(self, data, index):
        return data['video_data'][index]


    def __getitem__(self, index):
        video_data = self._get_video_data(self.data, index) # get one from 2793
        video_data = torch.tensor(video_data.transpose())
        if self.mode == "train":
            match_score_action, match_score_start, match_score_end = self._get_train_label(index)
            return video_data,match_score_action, match_score_start, match_score_end
        else:
            return index, video_data, self.mask_mat_vector, self.start_bins, self.end_bins

    def _get_train_label(self, index):
       
        gt_bbox = self.data['gt_bbox'][index]
        anchor_xmin = self.data['anchor_xmins'][index]
        anchor_xmax = self.data['anchor_xmaxs'][index]
        offset = int(min(anchor_xmin))

        # generate R_s and R_e
        gt_bbox = np.array(gt_bbox)
        gt_xmins = gt_bbox[:, 0]
        gt_xmaxs = gt_bbox[:, 1]
        gt_lens = gt_xmaxs - gt_xmins
        gt_len_small = 3 * self.skip_videoframes
        gt_start_bboxs = np.stack((gt_xmins - gt_len_small / 2, gt_xmins + gt_len_small / 2), axis=1)
        gt_end_bboxs = np.stack((gt_xmaxs - gt_len_small / 2, gt_xmaxs + gt_len_small / 2), axis=1)

        # calculate the ioa for all timestamp
        match_score_action = []
        for jdx in range(len(anchor_xmin)):
            match_score_action.append(
                np.max(ioa_with_anchors(anchor_xmin[jdx], anchor_xmax[jdx], gt_xmins, gt_xmaxs)))

        match_score_start = []
        for jdx in range(len(anchor_xmin)):
            match_score_start.append(np.max(
                ioa_with_anchors(anchor_xmin[jdx], anchor_xmax[jdx], gt_start_bboxs[:, 0], gt_start_bboxs[:, 1])))
        match_score_end = []
        for jdx in range(len(anchor_xmin)):
            match_score_end.append(np.max(
                ioa_with_anchors(anchor_xmin[jdx], anchor_xmax[jdx], gt_end_bboxs[:, 0], gt_end_bboxs[:, 1])))
        match_score_start = torch.Tensor(match_score_start)
        match_score_end = torch.Tensor(match_score_end)
        match_score_action = torch.Tensor(match_score_action)
        return match_score_action, match_score_start, match_score_end

    def __len__(self):
        return len(self.video_list)

    def _get_data(self):
        if 'train' in self.subset:
            anno_df = pd.read_csv(self.video_info_path + 'val_Annotation.csv')
        elif 'val' in self.subset:
            anno_df = pd.read_csv(self.video_info_path + 'test_Annotation.csv')

        video_name_list = sorted(list(set(anno_df.video.values[:])))

        video_info_dir = '/'.join(self.video_info_path.split('/')[:-1])
        saved_data_path = os.path.join(video_info_dir, 'saved.%s.%s.nf%d.sf%d.num%d.%s.pkl' % (
            self.feat_dim, self.subset, self.num_videoframes, self.skip_videoframes,
            len(video_name_list), self.mode)
                                       )
        print(saved_data_path)
        if not False and os.path.exists(saved_data_path):
            print('Got saved data.')
            with open(saved_data_path, 'rb') as f:
                self.data, self.durations = pickle.load(f)
            print('Size of data: ', len(self.data['video_names']), flush=True)
            return

        if self.feature_path:
            list_data = []

        list_anchor_xmins = []
        list_anchor_xmaxs = []
        list_gt_bbox = []
        list_videos = []
        list_indices = []

        num_videoframes = self.num_videoframes
        skip_videoframes = self.skip_videoframes
        start_snippet = int((skip_videoframes + 1) / 2)
        stride = int(num_videoframes / 2)
        #窗口爲256  stride 128
        self.durations = {}

        self.flow_val = h5py.File(self.feature_path + 'flow_val.h5', 'r')
        self.rgb_val = h5py.File(self.feature_path + 'rgb_val.h5', 'r')
        self.flow_test = h5py.File(self.feature_path + 'flow_test.h5', 'r')
        self.rgb_test = h5py.File(self.feature_path + 'rgb_test.h5', 'r')

        for num_video, video_name in enumerate(video_name_list):
            print(video_name)
            print('Getting video %d / %d' % (num_video, len(video_name_list)), flush=True)
            anno_df_video = anno_df[anno_df.video == video_name]
            if self.mode == 'train':
                gt_xmins = anno_df_video.startFrame.values[:]
                gt_xmaxs = anno_df_video.endFrame.values[:]

            if 'val' in video_name:
                feature_h5s = [
                    self.flow_val[video_name][::self.skip_videoframes,...],
                    self.rgb_val[video_name][::self.skip_videoframes,...]
                ]
            elif 'test' in video_name:
                feature_h5s = [
                    self.flow_test[video_name][::self.skip_videoframes,...],
                    self.rgb_test[video_name][::self.skip_videoframes,...]
                ]
            print("video name:",video_name)
            print('fea shape',feature_h5s[0].shape)
            num_snippet = min([h5.shape[0] for h5 in feature_h5s])

            df_data = np.concatenate([h5[:num_snippet, :]
                                      for h5 in feature_h5s],
                                     axis=1)

            df_snippet = [skip_videoframes * i for i in range(num_snippet)]
            num_windows = int((num_snippet + stride - num_videoframes) / stride)
            windows_start = [i * stride for i in range(num_windows)]

            if num_snippet < num_videoframes:
                windows_start = [0]
                # Add on a bunch of zero data if there aren't enough windows.
                tmp_data = np.zeros((num_videoframes - num_snippet, self.feat_dim))
                df_data = np.concatenate((df_data, tmp_data), axis=0)
                df_snippet.extend([
                    df_snippet[-1] + skip_videoframes * (i + 1)
                    for i in range(num_videoframes - num_snippet)
                ])
            elif num_snippet - windows_start[-1] - num_videoframes > int(num_videoframes / skip_videoframes):
                windows_start.append(num_snippet - num_videoframes)

            for start in windows_start:
                tmp_data = df_data[start:start + num_videoframes, :]

                tmp_snippets = np.array(df_snippet[start:start + num_videoframes])
                if self.mode == 'train':
                    tmp_anchor_xmins = tmp_snippets - skip_videoframes / 2.
                    tmp_anchor_xmaxs = tmp_snippets + skip_videoframes / 2.
                    tmp_gt_bbox = []
                    tmp_ioa_list = []
                    for idx in range(len(gt_xmins)):
                        tmp_ioa = ioa_with_anchors(gt_xmins[idx], gt_xmaxs[idx],
                                                   tmp_anchor_xmins[0],
                                                   tmp_anchor_xmaxs[-1])
                        tmp_ioa_list.append(tmp_ioa)
                        if tmp_ioa > 0:
                            tmp_gt_bbox.append([gt_xmins[idx], gt_xmaxs[idx]])

                    if len(tmp_gt_bbox) > 0 and max(tmp_ioa_list) > 0.9:
                        list_gt_bbox.append(tmp_gt_bbox)
                        list_anchor_xmins.append(tmp_anchor_xmins)
                        list_anchor_xmaxs.append(tmp_anchor_xmaxs)
                        list_videos.append(video_name)
                        list_indices.append(tmp_snippets)
                        if self.feature_dirs:
                            list_data.append(np.array(tmp_data).astype(np.float32))
                elif "infer" in self.mode:
                    list_videos.append(video_name)
                    list_indices.append(tmp_snippets)
                    list_data.append(np.array(tmp_data).astype(np.float32))

        print("List of videos: ", len(set(list_videos)), flush=True)
        self.data = {
            'video_names': list_videos,
            'indices': list_indices
        }
        if self.mode == 'train':
            self.data.update({
                'gt_bbox': list_gt_bbox,
                'anchor_xmins': list_anchor_xmins,
                'anchor_xmaxs': list_anchor_xmaxs,
            })
        if self.feature_dirs:
            self.data['video_data'] = list_data
        print('Size of data: ', len(self.data['video_names']), flush=True)
        with open(saved_data_path, 'wb') as f:
            pickle.dump([self.data, self.durations], f)
        print('Dumped data...')


if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.getcwd())
    print(os.getcwd())
    import _init_paths
    from lib.utils import opts

    opt = opts.parse_opt()
    opt = vars(opt)
    train_loader = torch.utils.data.DataLoader(VideoDataSet(subset="train"),
                                               batch_size=1, shuffle=True,
                                               num_workers=8, pin_memory=True)
    for a, b, c, d in train_loader:
        print(a.shape,b.shape,c.shape,d.shape)
        break

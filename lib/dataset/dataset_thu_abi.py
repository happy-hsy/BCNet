import os
import numpy as np
import pandas as pd
import json, pickle
import torch.utils.data as data
import torch
import math
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
        self.gama = opt['gama']
        self._get_data()
        self.video_list = self.data['video_names']
        self._get_sample_matrix()


    def _get_video_data(self, data, index):
        return data['video_data'][index]


    def __getitem__(self, index):
        video_data = self._get_video_data(self.data, index) # get one from 2793
        video_data = torch.tensor(video_data.transpose())
        if self.mode == "train":
            match_score_action,match_score_start,match_score_end,match_score_background,gt_proposal,gt_background = self._get_train_label(index)
            return video_data,self.mask_mat_vector,match_score_action,match_score_start,match_score_end,match_score_background,gt_proposal,gt_background,index
        else:
            return index, video_data, self.mask_mat_vector, self.start_bins, self.end_bins

    def _get_interp1d_bin_mask(self,seg_xmin, seg_xmax, tscale, num_sample, num_sample_perbin):
        plen = float(seg_xmax - seg_xmin)
        plen_sample = plen / (num_sample * num_sample_perbin - 1.0)
        total_samples = [
            seg_xmin + plen_sample * ii
            for ii in range(num_sample * num_sample_perbin)
        ]
        p_mask = []
        for idx in range(num_sample):
            bin_samples = total_samples[idx * num_sample_perbin:(idx + 1) * num_sample_perbin]
            bin_vector = np.zeros([tscale])
            for sample in bin_samples:
                sample_upper = math.ceil(sample)
                sample_decimal, sample_down = math.modf(sample)
                if int(sample_down) <= (tscale - 1) and int(sample_down) >= 0:
                    bin_vector[int(sample_down)] += 1 - sample_decimal
                if int(sample_upper) <= (tscale - 1) and int(sample_upper) >= 0:
                    bin_vector[int(sample_upper)] += sample_decimal
            bin_vector = 1.0 / num_sample_perbin * bin_vector
            p_mask.append(bin_vector)
        p_mask = np.stack(p_mask, axis=1)
        return p_mask

    def _get_proposal_bins(self):
        start_bins = []
        end_bins = []
        window_size_list = []
        init_size = 2
        flag = True
        while flag:
            if init_size < self.max_duration:
                window_size_list.append(init_size)
                init_size = init_size + 3
            else:
                flag = False
        print('window size is:', window_size_list)
        for window_size in window_size_list:
            stride = math.floor(math.sqrt(window_size)) + math.floor(window_size/self.gama)
            start_window_bin = np.array([ i*stride for i in range(math.ceil(self.temporal_scale/stride)) if i * stride + window_size <= self.temporal_scale])
            end_window_bin = start_window_bin + window_size
            start_window_bin.tolist()
            end_window_bin.tolist()
            start_bins.extend(start_window_bin)
            end_bins.extend(end_window_bin)
        self.start_bins = np.stack(start_bins)
        self.end_bins = np.stack(end_bins)
        print('proposal num is:', self.start_bins.shape[0])

    def _get_sample_matrix(self):
        self._get_proposal_bins()
        mask_mat_vector = []
        for idx in range(len(self.start_bins)):
            p_mask = self._get_interp1d_bin_mask(self.start_bins[idx],self.end_bins[idx],self.temporal_scale,self.num_sample,self.num_sample_perbin)
            mask_mat_vector.append(p_mask)
        self.mask_mat_vector = torch.Tensor(np.stack(mask_mat_vector, axis=2))

    def _get_train_label(self, index):
        # change the measurement from second to percentage

        gt_bbox = self.data['gt_bbox'][index]
        anchor_xmin = self.data['anchor_xmins'][index]
        anchor_xmax = self.data['anchor_xmaxs'][index]
        offset = int(min(anchor_xmin))
        ############frame-level label#################
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
        match_score_background = 1 - match_score_action
        ############frame-level label#################
        ############clip-level label#################
        gt_proposal = []
        gt_background = []
        for j in range(len(gt_bbox)):
            # tmp_info = video_labels[j]
            tmp_start = max(min(1, (gt_bbox[j][0]-offset)*self.temporal_gap/self.skip_videoframes), 0)
            tmp_end =   max(min(1, (gt_bbox[j][1]-offset)*self.temporal_gap/self.skip_videoframes), 0)
            #***
            tmp_gt_proposal = iou_with_anchors(self.start_bins*self.temporal_gap, self.end_bins*self.temporal_gap, tmp_start, tmp_end)
            tmp_gt_background = iou_with_anchors(self.start_bins*self.temporal_gap, self.end_bins*self.temporal_gap, tmp_start, tmp_end)
            gt_proposal.append(tmp_gt_proposal)
            gt_background.append(tmp_gt_background)

        gt_proposal = np.array(gt_proposal)
        gt_proposal = np.max(gt_proposal, axis=0)
        gt_proposal = torch.Tensor(gt_proposal)
        

        gt_background = np.array(gt_background)
        gt_background = 1 - np.max(gt_background, axis=0)
        gt_background = torch.Tensor(gt_background)

        ############clip-level label#################

        return match_score_action,match_score_start,match_score_end,match_score_background,gt_proposal,gt_background

    def __len__(self):
        return len(self.video_list)

    def _get_data(self):
        if 'train' in self.subset:
            anno_df = pd.read_csv(self.video_info_path+'val_Annotation.csv')
        elif 'val' in self.subset:
            anno_df = pd.read_csv(self.video_info_path+'test_Annotation.csv')

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
            #print('fea',feature_h5s.shape)
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
    train_loader = torch.utils.data.DataLoader(VideoDataSet(subset="train"),
                                               batch_size=1, shuffle=True,
                                               num_workers=8, pin_memory=True)
    for a, b, c, d in train_loader:
        # print(max(c))
        print(a.shape,b.shape,c.shape,d.shape)
        break

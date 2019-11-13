#!/usr/bin/env python
# coding=utf-8

import os
import sys
import numpy as np
import scipy.io as sio
# import open3d as o3d
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

sys.path.append('utils')
import dataset_utils
# import visual_utils

ROOT_DIR = './'
PROPOSALS_DIR = '/home/yanghaitao/Projects/cvpr2020/votenet18cls/classification_net/update_cues_with18class/'
PC_DIR = '/home/yanghaitao/Dataset/scannet_train_detection_data/' 

MAX_NUM_OBJECT = 64
NEAR_THRESHOLD = 0.3


class ScannetDatasetConfig(object):
    def __init__(self):
        self.num_class = 18
        self.type2class = {'cabinet':0, 'bed':1, 'chair':2, 'sofa':3, 'table':4, 'door':5,
            'window':6,'bookshelf':7,'picture':8, 'counter':9, 'desk':10, 'curtain':11,
            'refrigerator':12, 'showercurtrain':13, 'toilet':14, 'sink':15, 'bathtub':16, 'garbagebin':17}  
        self.class2type = {self.type2class[t]:t for t in self.type2class}


class SunrgbdDatasetConfig(object):
    def __init__(self):
        self.num_class = 10
        self.type2class={'bed':0, 'table':1, 'sofa':2, 'chair':3, 'toilet':4, 'desk':5, 'dresser':6, 'night_stand':7, 'bookshelf':8, 'bathtub':9}
        self.class2type = {self.type2class[t]:t for t in self.type2class}


class ClassificationDataset(Dataset):
    '''
    @Returns:
    '''
    def __init__(self, split_set='train', num_points=40000):
        self.proposals_dir = PROPOSALS_DIR
        self.pc_dir = PC_DIR
        self.num_points = num_points

        all_scan_names = list(set([os.path.basename(x)[0:12] \
            for x in os.listdir(self.pc_dir) if x.startswith('scene')]))
        assert len(all_scan_names) == 1513
        if split_set=='all':            
            self.scan_names = all_scan_names
        elif split_set in ['train', 'val', 'test']:
            split_filenames = os.path.join(ROOT_DIR, 'scannet/meta_data',
                'scannetv2_{}.txt'.format(split_set))
            with open(split_filenames, 'r') as f:
                self.scan_names = f.read().splitlines()   
            # remove unavailiable scans
            num_scans = len(self.scan_names)
            self.scan_names = [sname for sname in self.scan_names \
                if sname in all_scan_names]
            print('kept {} scans out of {}'.format(len(self.scan_names), num_scans))
            num_scans = len(self.scan_names)
        elif split_set in ['debug']:
            split_filenames = os.path.join(ROOT_DIR, 'scannet/meta_data',
                'scannetv2_{}.txt'.format('train'))
            with open(split_filenames, 'r') as f:
                self.scan_names = f.read().splitlines()   
            # remove unavailiable scans
            num_scans = len(self.scan_names)
            self.scan_names = [sname for sname in self.scan_names \
                if sname in all_scan_names]
            print('kept {} scans out of {}'.format(len(self.scan_names), num_scans))
            self.scan_names = self.scan_names[:10]
        else:
            print('illegal split name')
            return

    def __len__(self):
        return len(self.scan_names) * 256

    def __getitem__(self, idx):
        scan_id = idx // 256
        object_id = idx % 256
        scan_name = self.scan_names[scan_id]

        # load 256 initial proposals from the scan_id-th scan
        bboxes_init_pps, bboxes_gt = dataset_utils.load_pps_and_gt(self.proposals_dir, scan_name)
        point_cloud = np.load(os.path.join(self.pc_dir, scan_name + '_vert.npy'))
        point_cloud = point_cloud[:, :3]
        point_cloud, choices = dataset_utils.random_sampling(point_cloud, 
                self.num_points, return_choices=True)

        # select the object_id-th object from the scan
        bbox_init_pps = bboxes_init_pps[object_id]
        center = bbox_init_pps[:3]
        mask = dataset_utils.judge_points_in_bbox(point_cloud, dataset_utils.bboxParams2bbox(bbox_init_pps))
        inputs = np.hstack([point_cloud, mask[:, None]]) # num_points * 4

        # online generate label
        dist_min = np.min(np.linalg.norm(center - bboxes_gt[:, :3], axis=1))
        label = (dist_min < NEAR_THRESHOLD).astype(np.int)

        ret_dict = {}

        # change 1~18 class to 0~17 class
        bbox_init_pps[7] = bbox_init_pps[7] - 1
        bboxes_gt[:, 7] = bboxes_gt[:, 7] - 1

        ret_dict['scan_id'] = scan_id
        ret_dict['object_id'] = object_id

        ret_dict['inputs'] = inputs.astype(np.float32)
        ret_dict['label'] = label

        ret_dict['bbox_init_pps'] = bbox_init_pps  # (8,)
        ret_dict['bboxes_gt'] = np.zeros([MAX_NUM_OBJECT, 8]).astype(np.float32)
        ret_dict['bboxes_gt_mask'] = np.zeros([MAX_NUM_OBJECT]).astype(np.float32)
        ret_dict['bboxes_gt'][:bboxes_gt.shape[0]] = bboxes_gt
        ret_dict['bboxes_gt_mask'][:bboxes_gt.shape[0]] = 1.0
        
        return ret_dict


if __name__ == '__main__':
    dset = ClassificationDataset(split_set='all')
    label_list = []
    for i_example in range(256):
        scan_name = dset.scan_names[i_example]
        print(i_example, scan_name)
        example = dset.__getitem__(i_example)
        inputs = example['inputs']
        mask = inputs[:, 3]
        idx = np.where(mask)
        # print(idx[0].shape)
        bboxes_gt = example['bboxes_gt']
        bbox_init_pps = example['bbox_init_pps']
        label_list.append(example['label'])
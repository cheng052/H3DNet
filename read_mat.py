import scipy.io as sio
import scipy
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import pc_util

import numpy as np

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--path', default=None)
parser.add_argument('--dataset')
parser.add_argument('--dump_dir')
FLAGS = parser.parse_args()

all_scan_names = list(set([os.path.basename(x)
                           for x in os.listdir(FLAGS.path) if x.endswith('gt.mat')]))

dump_dir = FLAGS.dump_dir

if not os.path.exists(dump_dir):
    os.mkdir(dump_dir)

for name in all_scan_names:
    scene_name = name[3:15]
    if scene_name != 'scene0488_00':
        continue
    obbs=sio.loadmat(os.path.join(FLAGS.path,name))['gt']
    obbs=np.array(obbs)
    print(obbs[:, -1])
    boxes = obbs[:, :-1]
    sem_cls = obbs[:, -1] - 1
    print('(%d, %d)' % (obbs.shape[0], obbs.shape[1]))
    case_dump_dir = os.path.join(dump_dir, scene_name)
    if not os.path.exists(case_dump_dir):
        os.mkdir(case_dump_dir)
    for l in np.unique(sem_cls):
        mask = (sem_cls == l)
        if np.sum(mask)>0:
            pc_util.write_oriented_bbox(boxes[mask, :], os.path.join(case_dump_dir, '%d_pred_confident_nms_bbox.ply'%(l)))

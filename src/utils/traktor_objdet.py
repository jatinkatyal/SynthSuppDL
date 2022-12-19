import pickle
import re
import string

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset,ConcatDataset
from torchvision.io import read_image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, Resize, Normalize
import os

import utils as utils
from engine import train_one_epoch

from matplotlib import pyplot as plt
from matplotlib import patches


IMAGE_RESOLUTION = (540, 1024)  # Constant Resolution to maintain for inputs
MEAN = [0.485, 0.456, 0.406]  # Mean for normalizing
STD = [0.225, 0.225, 0.225]  # STD for normalizing
NUM_CLASSES = 2

class UAVDTObjDet(Dataset):
    """
    Class for UAVDT dataset object for images and annotations in format as per the pytorch tutorial for object detection.
    """
    num_classes = 2

    def __init__(self, path, mode, transform=None):
        """
        Take in important info like the root path of dataset and
        record metainfo for later access
        """
        self.path = path
        self.transform = transform
        self.meta = []
        seq_dir = 'UAV-benchmark-M'
        anot_dir = 'UAV-benchmark-MOTD_v1.0/GT'
        if mode == 'train':
            seqs = ['M0101', 'M0201', 'M0202', 'M0204', 'M0206',
                    'M0207', 'M0210', 'M0301', 'M0401', 'M0402',
                    'M0501', 'M0603', 'M0604', 'M0605', 'M0702',
                    'M0703', 'M0704', 'M0901', 'M0902', 'M1002',
                    'M1003', 'M1005', 'M1006', 'M1008', 'M1102',
                    'M1201', 'M1202', 'M1304', 'M1305', 'M1306']
        elif mode == 'test':
            seqs = ['M0203', 'M0205', 'M0208', 'M0209', 'M0403', 'M0601', 'M0602', 'M0606', 'M0701', 'M0801', 'M0802',
                    'M1001', 'M1004', 'M1007', 'M1009', 'M1101', 'M1301', 'M1302', 'M1303', 'M1401']
        elif mode == 'train5-a':
            print('Skipping some real sequences to balance the total number of samples')
            seqs = ['M0101', 'M0201', 'M0202', 'M0204', 'M0206',
                    #'M0207', 'M0210', 'M0301', 'M0401', 'M0402',
                    #'M0501', 'M0603', 'M0604', 'M0605', 'M0702',
                    #'M0703', 'M0704', 'M0901', 'M0902', 'M1002',
                    #'M1003', 'M1005', 'M1006', 'M1008', 'M1102',
                    #'M1201', 'M1202', 'M1304', 'M1305', 'M1306'
                    ]
        elif mode == 'train5-b':
            print('Skipping some real sequences to balance the total number of samples')
            seqs = [#'M0101', 'M0201', 'M0202', 'M0204', 'M0206',
                    'M0207', 'M0210', 'M0301', 'M0401', 'M0402',
                    #'M0501', 'M0603', 'M0604', 'M0605', 'M0702',
                    #'M0703', 'M0704', 'M0901', 'M0902', 'M1002',
                    #'M1003', 'M1005', 'M1006', 'M1008', 'M1102',
                    #'M1201', 'M1202', 'M1304', 'M1305', 'M1306'
                    ]
        elif mode == 'train5-c':
            print('Skipping some real sequences to balance the total number of samples')
            seqs = [#'M0101', 'M0201', 'M0202', 'M0204', 'M0206',
                    #'M0207', 'M0210', 'M0301', 'M0401', 'M0402',
                    'M0501', 'M0603', 'M0604', 'M0605', 'M0702',
                    #'M0703', 'M0704', 'M0901', 'M0902', 'M1002',
                    #'M1003', 'M1005', 'M1006', 'M1008', 'M1102',
                    #'M1201', 'M1202', 'M1304', 'M1305', 'M1306'
                    ]
        elif mode == 'train5-d':
            print('Skipping some real sequences to balance the total number of samples')
            seqs = [#'M0101', 'M0201', 'M0202', 'M0204', 'M0206',
                    #'M0207', 'M0210', 'M0301', 'M0401', 'M0402',
                    #'M0501', 'M0603', 'M0604', 'M0605', 'M0702',
                    'M0703', 'M0704', 'M0901', 'M0902', 'M1002',
                    #'M1003', 'M1005', 'M1006', 'M1008', 'M1102',
                    #'M1201', 'M1202', 'M1304', 'M1305', 'M1306'
                    ]
        elif mode == 'train5-e':
            print('Skipping some real sequences to balance the total number of samples')
            seqs = [#'M0101', 'M0201', 'M0202', 'M0204', 'M0206',
                    #'M0207', 'M0210', 'M0301', 'M0401', 'M0402',
                    #'M0501', 'M0603', 'M0604', 'M0605', 'M0702',
                    #'M0703', 'M0704', 'M0901', 'M0902', 'M1002',
                    'M1003', 'M1005', 'M1006', 'M1008', 'M1102',
                    #'M1201', 'M1202', 'M1304', 'M1305', 'M1306'
                    ]
        elif mode == 'train5-f':
            print('Skipping some real sequences to balance the total number of samples')
            seqs = [#'M0101', 'M0201', 'M0202', 'M0204', 'M0206',
                    #'M0207', 'M0210', 'M0301', 'M0401', 'M0402',
                    #'M0501', 'M0603', 'M0604', 'M0605', 'M0702',
                    #'M0703', 'M0704', 'M0901', 'M0902', 'M1002',
                    #'M1003', 'M1005', 'M1006', 'M1008', 'M1102',
                    'M1201', 'M1202', 'M1304', 'M1305', 'M1306'
                    ]
        elif mode == 'train10-a':
            print('Skipping some real sequences to balance the total number of samples')
            seqs = ['M0101', 'M0201', 'M0202', 'M0204', 'M0206',
                    'M0207', 'M0210', 'M0301', 'M0401', 'M0402',
                    #'M0501', 'M0603', 'M0604', 'M0605', 'M0702',
                    #'M0703', 'M0704', 'M0901', 'M0902', 'M1002',
                    #'M1003', 'M1005', 'M1006', 'M1008', 'M1102',
                    #'M1201', 'M1202', 'M1304', 'M1305', 'M1306'
                    ]
        elif mode == 'train10-b':
            print('Skipping some real sequences to balance the total number of samples')
            seqs = [#'M0101', 'M0201', 'M0202', 'M0204', 'M0206',
                    #'M0207', 'M0210', 'M0301', 'M0401', 'M0402',
                    'M0501', 'M0603', 'M0604', 'M0605', 'M0702',
                    'M0703', 'M0704', 'M0901', 'M0902', 'M1002',
                    #'M1003', 'M1005', 'M1006', 'M1008', 'M1102',
                    #'M1201', 'M1202', 'M1304', 'M1305', 'M1306'
                    ]
        elif mode == 'train10-c':
            print('Skipping some real sequences to balance the total number of samples')
            seqs = [#'M0101', 'M0201', 'M0202', 'M0204', 'M0206',
                    #'M0207', 'M0210', 'M0301', 'M0401', 'M0402',
                    #'M0501', 'M0603', 'M0604', 'M0605', 'M0702',
                    #'M0703', 'M0704', 'M0901', 'M0902', 'M1002',
                    'M1003', 'M1005', 'M1006', 'M1008', 'M1102',
                    'M1201', 'M1202', 'M1304', 'M1305', 'M1306']
        elif mode == 'train15-a':
            print('Skipping some real sequences to balance the total number of samples')
            seqs = ['M0101', 'M0201', 'M0202', 'M0204', 'M0206',
                    #'M0207', 'M0210', 'M0301', 'M0401', 'M0402',
                    #'M0501', 'M0603', 'M0604', 'M0605', 'M0702',
                    #'M0703', 'M0704', 'M0901', 'M0902', 'M1002',
                    'M1003', 'M1005', 'M1006', 'M1008', 'M1102',
                    'M1201', 'M1202', 'M1304', 'M1305', 'M1306']
        elif mode == 'train15-b':
            print('Skipping some real sequences to balance the total number of samples')
            seqs = [#'M0101', 'M0201', 'M0202', 'M0204', 'M0206',
                    #'M0207', 'M0210', 'M0301', 'M0401', 'M0402',
                    #'M0501', 'M0603', 'M0604', 'M0605', 'M0702',
                    #'M0703', 'M0704', 'M0901', 'M0902', 'M1002',
                    'M1003', 'M1005', 'M1006', 'M1008', 'M1102',
                    'M1201', 'M1202', 'M1304', 'M1305', 'M1306']
       	elif mode == 'train20-a':
            print('Skipping some real sequences to balance the total number of samples')
            seqs = ['M0101', 'M0201', 'M0202', 'M0204', 'M0206',
                    'M0207', 'M0210', 'M0301', 'M0401', 'M0402',
                    'M0501', 'M0603', 'M0604', 'M0605', 'M0702',
                    'M0703', 'M0704', 'M0901', 'M0902', 'M1002',
                    #'M1003', 'M1005', 'M1006', 'M1008', 'M1102',
                    #'M1201', 'M1202', 'M1304', 'M1305', 'M1306'
                    ]
        elif mode == 'train20-b':
            seqs = ['M0101', 'M0201', 'M0202', 'M0204', 'M0206',
                    'M0207', 'M0210', 'M0301', 'M0401', 'M0402',
                    #'M0501', 'M0603', 'M0604', 'M0605', 'M0702',
                    #'M0703', 'M0704', 'M0901', 'M0902', 'M1002',
                    'M1003', 'M1005', 'M1006', 'M1008', 'M1102',
                    'M1201', 'M1202', 'M1304', 'M1305', 'M1306']
        elif mode == 'train20-c':
            print('Skipping some real sequences to balance the total number of samples')
            seqs = [#'M0101', 'M0201', 'M0202', 'M0204', 'M0206',
                    #'M0207', 'M0210', 'M0301', 'M0401', 'M0402',
                    'M0501', 'M0603', 'M0604', 'M0605', 'M0702',
                    'M0703', 'M0704', 'M0901', 'M0902', 'M1002',
                    'M1003', 'M1005', 'M1006', 'M1008', 'M1102',
                    'M1201', 'M1202', 'M1304', 'M1305', 'M1306']
        elif mode == 'train25-a':
            print('Skipping some real sequences to balance the total number of samples')
            seqs = ['M0101', 'M0201', 'M0202', 'M0204', 'M0206',
                    'M0207', 'M0210', 'M0301', 'M0401', 'M0402',
                    'M0501', 'M0603', 'M0604', 'M0605', 'M0702',
                    'M0703', 'M0704', 'M0901', 'M0902', 'M1002',
                    'M1003', 'M1005', 'M1006', 'M1008', 'M1102',
                    #'M1201', 'M1202', 'M1304', 'M1305', 'M1306'
                    ]
        elif mode == 'train25-b':
            print('Skipping some real sequences to balance the total number of samples')
            seqs = ['M0101', 'M0201', 'M0202', 'M0204', 'M0206',
                    'M0207', 'M0210', 'M0301', 'M0401', 'M0402',
                    'M0501', 'M0603', 'M0604', 'M0605', 'M0702',
                    'M0703', 'M0704', 'M0901', 'M0902', 'M1002',
                    #'M1003', 'M1005', 'M1006', 'M1008', 'M1102',
                    'M1201', 'M1202', 'M1304', 'M1305', 'M1306']
        elif mode == 'train25-c':
            print('Skipping some real sequences to balance the total number of samples')
            seqs = ['M0101', 'M0201', 'M0202', 'M0204', 'M0206',
                    'M0207', 'M0210', 'M0301', 'M0401', 'M0402',
                    'M0501', 'M0603', 'M0604', 'M0605', 'M0702',
                    #'M0703', 'M0704', 'M0901', 'M0902', 'M1002',
                    'M1003', 'M1005', 'M1006', 'M1008', 'M1102',
                    'M1201', 'M1202', 'M1304', 'M1305', 'M1306']
        elif mode == 'train25-d':
            print('Skipping some real sequences to balance the total number of samples')
            seqs = ['M0101', 'M0201', 'M0202', 'M0204', 'M0206',
                    'M0207', 'M0210', 'M0301', 'M0401', 'M0402',
                    #'M0501', 'M0603', 'M0604', 'M0605', 'M0702',
                    'M0703', 'M0704', 'M0901', 'M0902', 'M1002',
                    'M1003', 'M1005', 'M1006', 'M1008', 'M1102',
                    'M1201', 'M1202', 'M1304', 'M1305', 'M1306']
        elif mode == 'train25-e':
            print('Skipping some real sequences to balance the total number of samples')
            seqs = ['M0101', 'M0201', 'M0202', 'M0204', 'M0206',
                    #'M0207', 'M0210', 'M0301', 'M0401', 'M0402',
                    'M0501', 'M0603', 'M0604', 'M0605', 'M0702',
                    'M0703', 'M0704', 'M0901', 'M0902', 'M1002',
                    'M1003', 'M1005', 'M1006', 'M1008', 'M1102',
                    'M1201', 'M1202', 'M1304', 'M1305', 'M1306']
        elif mode == 'train25-f':
            print('Skipping some real sequences to balance the total number of samples')
            seqs = [#'M0101', 'M0201', 'M0202', 'M0204', 'M0206',
                    'M0207', 'M0210', 'M0301', 'M0401', 'M0402',
                    'M0501', 'M0603', 'M0604', 'M0605', 'M0702',
                    'M0703', 'M0704', 'M0901', 'M0902', 'M1002',
                    'M1003', 'M1005', 'M1006', 'M1008', 'M1102',
                    'M1201', 'M1202', 'M1304', 'M1305', 'M1306']
        # seqs = os.listdir(os.path.join(self.path,seq_dir))
        seqs.sort()
        for seq in seqs:
            frames = os.listdir(os.path.join(self.path, seq_dir, seq))
            frames.sort()
            for frame in frames:
                meta = {
                    'framepath': os.path.join(self.path, seq_dir, seq, frame),
                    'anotpath': os.path.join(self.path, anot_dir, seq + '_gt.txt'),
                    'frameno': frames.index(frame) + 1
                }
                self.meta.append(meta)

    def __len__(self):
        """Returns total number of frames as in meta info"""
        return len(self.meta)

    def __getitem__(self, idx):
        """
        Returns a single frame and annotations for the meta info at index idx.
        takes input:
            idx: index to fetch from meta

        outputs:
            frame: image of shape (C,H,W)
            target: dictionary with annotations
        """
        frame = read_image(self.meta[idx]['framepath']) / 255
        if self.transform:
            frame = self.transform(frame)

        # Annotations
        # anotation file format: Frame No., Object Id., bbox x1, bbox y1, bbox x2, bbox y2
        anots = np.loadtxt(self.meta[idx]['anotpath'], delimiter=',')
        anots = torch.tensor(anots)
        anots = anots[anots[:, 0] == self.meta[idx]['frameno']]
        bbboxes = [anots[:, 2], anots[:, 3],  # x1,y1
                   anots[:, 2] + anots[:, 4],  # x2
                   anots[:, 3] + anots[:, 5]]  # y2
        bbboxes = torch.stack(bbboxes).T
        labels = torch.ones(len(bbboxes), dtype=torch.int64)
        area = torch.tensor([(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in bbboxes])
        iscrowd = torch.tensor(anots[:, 6])

        target = {
            'boxes': bbboxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'area': area,
            'iscrowd': iscrowd
        }
        return frame, target

class VisDroneObjDet(Dataset):
    """
    Class for Visdrone dataset object for images and annotations in format as per the pytorch tutorial for object detection.
    """
    num_classes = 2

    def __init__(self, path, mode, transform=None):
        """
        Take in important info like the root path of dataset and
        record metainfo for later access
        """
        self.path = path
        self.transform = transform
        self.meta = []
        seq_dir = 'sequences'
        anot_dir = 'annotations'
        if mode == 'train':
            seqs = ['uav0000124_00944_v.txt', 'uav0000126_00001_v.txt', 'uav0000140_01590_v.txt', 'uav0000143_02250_v.txt', 'uav0000145_00000_v.txt',
                    'uav0000218_00001_v.txt', 'uav0000222_03150_v.txt', 'uav0000239_03720_v.txt', 'uav0000239_12336_v.txt', 'uav0000243_00001_v.txt',
                    'uav0000244_01440_v.txt', 'uav0000248_00001_v.txt', 'uav0000263_03289_v.txt', 'uav0000264_02760_v.txt', 'uav0000270_00001_v.txt',
                    'uav0000273_00001_v.txt', 'uav0000278_00001_v.txt', 'uav0000279_00001_v.txt', 'uav0000281_00460_v.txt', 'uav0000289_00001_v.txt',
                    'uav0000289_06922_v.txt', 'uav0000295_02300_v.txt', 'uav0000307_00000_v.txt', 'uav0000315_00000_v.txt', 'uav0000323_01173_v.txt', #'uav0000300_00000_v.txt', excluded for less cars to other ratio
                    'uav0000326_01035_v.txt', 'uav0000342_04692_v.txt', 'uav0000352_05980_v.txt', 'uav0000361_02323_v.txt', 'uav0000366_00001_v.txt'
                    ]
        elif mode == 'train5-a':
            seqs = ['uav0000124_00944_v.txt', 'uav0000126_00001_v.txt', 'uav0000140_01590_v.txt', 'uav0000143_02250_v.txt', 'uav0000145_00000_v.txt',
                    #'uav0000218_00001_v.txt', 'uav0000222_03150_v.txt', 'uav0000239_03720_v.txt', 'uav0000239_12336_v.txt', 'uav0000243_00001_v.txt',
                    #'uav0000244_01440_v.txt', 'uav0000248_00001_v.txt', 'uav0000263_03289_v.txt', 'uav0000264_02760_v.txt', 'uav0000270_00001_v.txt',
                    #'uav0000273_00001_v.txt', 'uav0000278_00001_v.txt', 'uav0000279_00001_v.txt', 'uav0000281_00460_v.txt', 'uav0000289_00001_v.txt',
                    #'uav0000289_06922_v.txt', 'uav0000295_02300_v.txt', 'uav0000307_00000_v.txt', 'uav0000315_00000_v.txt', 'uav0000323_01173_v.txt', # 'uav0000300_00000_v.txt', excluded for less cars to other ratio
                    #'uav0000326_01035_v.txt', 'uav0000342_04692_v.txt', 'uav0000352_05980_v.txt', 'uav0000361_02323_v.txt', 'uav0000366_00001_v.txt'
                    ]
        elif mode == 'train5-b':
            seqs = [
                # 'uav0000124_00944_v.txt', 'uav0000126_00001_v.txt', 'uav0000140_01590_v.txt', 'uav0000143_02250_v.txt', 'uav0000145_00000_v.txt',
                'uav0000218_00001_v.txt', 'uav0000222_03150_v.txt', 'uav0000239_03720_v.txt', 'uav0000239_12336_v.txt', 'uav0000243_00001_v.txt',
                # 'uav0000244_01440_v.txt', 'uav0000248_00001_v.txt', 'uav0000263_03289_v.txt', 'uav0000264_02760_v.txt', 'uav0000270_00001_v.txt',
                # 'uav0000273_00001_v.txt', 'uav0000278_00001_v.txt', 'uav0000279_00001_v.txt', 'uav0000281_00460_v.txt', 'uav0000289_00001_v.txt',
                # 'uav0000289_06922_v.txt', 'uav0000295_02300_v.txt', 'uav0000307_00000_v.txt', 'uav0000315_00000_v.txt', 'uav0000323_01173_v.txt', # 'uav0000300_00000_v.txt', excluded for less cars to other ratio
                # 'uav0000326_01035_v.txt', 'uav0000342_04692_v.txt', 'uav0000352_05980_v.txt', 'uav0000361_02323_v.txt', 'uav0000366_00001_v.txt'
            ]
        elif mode == 'train5-c':
            seqs = [
                # 'uav0000124_00944_v.txt', 'uav0000126_00001_v.txt', 'uav0000140_01590_v.txt', 'uav0000143_02250_v.txt', 'uav0000145_00000_v.txt',
                # 'uav0000218_00001_v.txt', 'uav0000222_03150_v.txt', 'uav0000239_03720_v.txt', 'uav0000239_12336_v.txt', 'uav0000243_00001_v.txt',
                'uav0000244_01440_v.txt', 'uav0000248_00001_v.txt', 'uav0000263_03289_v.txt', 'uav0000264_02760_v.txt', 'uav0000270_00001_v.txt',
                # 'uav0000273_00001_v.txt', 'uav0000278_00001_v.txt', 'uav0000279_00001_v.txt', 'uav0000281_00460_v.txt', 'uav0000289_00001_v.txt',
                # 'uav0000289_06922_v.txt', 'uav0000295_02300_v.txt', 'uav0000307_00000_v.txt', 'uav0000315_00000_v.txt', 'uav0000323_01173_v.txt', # 'uav0000300_00000_v.txt', excluded for less cars to other ratio
                # 'uav0000326_01035_v.txt', 'uav0000342_04692_v.txt', 'uav0000352_05980_v.txt', 'uav0000361_02323_v.txt', 'uav0000366_00001_v.txt'
            ]
        elif mode == 'train5-d':
            seqs = [
                # 'uav0000124_00944_v.txt', 'uav0000126_00001_v.txt', 'uav0000140_01590_v.txt', 'uav0000143_02250_v.txt', 'uav0000145_00000_v.txt',
                # 'uav0000218_00001_v.txt', 'uav0000222_03150_v.txt', 'uav0000239_03720_v.txt', 'uav0000239_12336_v.txt', 'uav0000243_00001_v.txt',
                # 'uav0000244_01440_v.txt', 'uav0000248_00001_v.txt', 'uav0000263_03289_v.txt', 'uav0000264_02760_v.txt', 'uav0000270_00001_v.txt',
                'uav0000273_00001_v.txt', 'uav0000278_00001_v.txt', 'uav0000279_00001_v.txt', 'uav0000281_00460_v.txt', 'uav0000289_00001_v.txt',
                # 'uav0000289_06922_v.txt', 'uav0000295_02300_v.txt', 'uav0000307_00000_v.txt', 'uav0000315_00000_v.txt', 'uav0000323_01173_v.txt', # 'uav0000300_00000_v.txt', excluded for less cars to other ratio
                # 'uav0000326_01035_v.txt', 'uav0000342_04692_v.txt', 'uav0000352_05980_v.txt', 'uav0000361_02323_v.txt', 'uav0000366_00001_v.txt'
            ]
        elif mode == 'train5-e':
            seqs = [
                # 'uav0000124_00944_v.txt', 'uav0000126_00001_v.txt', 'uav0000140_01590_v.txt', 'uav0000143_02250_v.txt', 'uav0000145_00000_v.txt',
                # 'uav0000218_00001_v.txt', 'uav0000222_03150_v.txt', 'uav0000239_03720_v.txt', 'uav0000239_12336_v.txt', 'uav0000243_00001_v.txt',
                # 'uav0000244_01440_v.txt', 'uav0000248_00001_v.txt', 'uav0000263_03289_v.txt', 'uav0000264_02760_v.txt', 'uav0000270_00001_v.txt',
                # 'uav0000273_00001_v.txt', 'uav0000278_00001_v.txt', 'uav0000279_00001_v.txt', 'uav0000281_00460_v.txt', 'uav0000289_00001_v.txt',
                'uav0000289_06922_v.txt', 'uav0000295_02300_v.txt', 'uav0000307_00000_v.txt', 'uav0000315_00000_v.txt', 'uav0000323_01173_v.txt', # 'uav0000300_00000_v.txt', excluded for less cars to other ratio
                # 'uav0000326_01035_v.txt', 'uav0000342_04692_v.txt', 'uav0000352_05980_v.txt', 'uav0000361_02323_v.txt', 'uav0000366_00001_v.txt'
            ]
        elif mode == 'train5-f':
            seqs = [
                # 'uav0000124_00944_v.txt', 'uav0000126_00001_v.txt', 'uav0000140_01590_v.txt', 'uav0000143_02250_v.txt', 'uav0000145_00000_v.txt',
                # 'uav0000218_00001_v.txt', 'uav0000222_03150_v.txt', 'uav0000239_03720_v.txt', 'uav0000239_12336_v.txt', 'uav0000243_00001_v.txt',
                # 'uav0000244_01440_v.txt', 'uav0000248_00001_v.txt', 'uav0000263_03289_v.txt', 'uav0000264_02760_v.txt', 'uav0000270_00001_v.txt',
                # 'uav0000273_00001_v.txt', 'uav0000278_00001_v.txt', 'uav0000279_00001_v.txt', 'uav0000281_00460_v.txt', 'uav0000289_00001_v.txt',
                # 'uav0000289_06922_v.txt', 'uav0000295_02300_v.txt', 'uav0000307_00000_v.txt', 'uav0000315_00000_v.txt', 'uav0000323_01173_v.txt', # 'uav0000300_00000_v.txt', excluded for less cars to other ratio
                'uav0000326_01035_v.txt', 'uav0000342_04692_v.txt', 'uav0000352_05980_v.txt', 'uav0000361_02323_v.txt', 'uav0000366_00001_v.txt'
            ]
        elif mode == 'train10-a':
            seqs = [
                'uav0000124_00944_v.txt', 'uav0000126_00001_v.txt', 'uav0000140_01590_v.txt', 'uav0000143_02250_v.txt', 'uav0000145_00000_v.txt',
                'uav0000218_00001_v.txt', 'uav0000222_03150_v.txt', 'uav0000239_03720_v.txt', 'uav0000239_12336_v.txt', 'uav0000243_00001_v.txt',
                # 'uav0000244_01440_v.txt', 'uav0000248_00001_v.txt', 'uav0000263_03289_v.txt', 'uav0000264_02760_v.txt', 'uav0000270_00001_v.txt',
                # 'uav0000273_00001_v.txt', 'uav0000278_00001_v.txt', 'uav0000279_00001_v.txt', 'uav0000281_00460_v.txt', 'uav0000289_00001_v.txt',
                # 'uav0000289_06922_v.txt', 'uav0000295_02300_v.txt', 'uav0000307_00000_v.txt', 'uav0000315_00000_v.txt', 'uav0000323_01173_v.txt', # 'uav0000300_00000_v.txt', excluded for less cars to other ratio
                # 'uav0000326_01035_v.txt', 'uav0000342_04692_v.txt', 'uav0000352_05980_v.txt', 'uav0000361_02323_v.txt', 'uav0000366_00001_v.txt'
            ]
        elif mode == 'train10-b':
            seqs = [
                # 'uav0000124_00944_v.txt', 'uav0000126_00001_v.txt', 'uav0000140_01590_v.txt', 'uav0000143_02250_v.txt', 'uav0000145_00000_v.txt',
                # 'uav0000218_00001_v.txt', 'uav0000222_03150_v.txt', 'uav0000239_03720_v.txt', 'uav0000239_12336_v.txt', 'uav0000243_00001_v.txt',
                'uav0000244_01440_v.txt', 'uav0000248_00001_v.txt', 'uav0000263_03289_v.txt', 'uav0000264_02760_v.txt', 'uav0000270_00001_v.txt',
                'uav0000273_00001_v.txt', 'uav0000278_00001_v.txt', 'uav0000279_00001_v.txt', 'uav0000281_00460_v.txt', 'uav0000289_00001_v.txt',
                # 'uav0000289_06922_v.txt', 'uav0000295_02300_v.txt', 'uav0000307_00000_v.txt', 'uav0000315_00000_v.txt', 'uav0000323_01173_v.txt', # 'uav0000300_00000_v.txt', excluded for less cars to other ratio
                # 'uav0000326_01035_v.txt', 'uav0000342_04692_v.txt', 'uav0000352_05980_v.txt', 'uav0000361_02323_v.txt', 'uav0000366_00001_v.txt'
            ]
        elif mode == 'train10-c':
            seqs = [
                # 'uav0000124_00944_v.txt', 'uav0000126_00001_v.txt', 'uav0000140_01590_v.txt', 'uav0000143_02250_v.txt', 'uav0000145_00000_v.txt',
                # 'uav0000218_00001_v.txt', 'uav0000222_03150_v.txt', 'uav0000239_03720_v.txt', 'uav0000239_12336_v.txt', 'uav0000243_00001_v.txt',
                # 'uav0000244_01440_v.txt', 'uav0000248_00001_v.txt', 'uav0000263_03289_v.txt', 'uav0000264_02760_v.txt', 'uav0000270_00001_v.txt',
                # 'uav0000273_00001_v.txt', 'uav0000278_00001_v.txt', 'uav0000279_00001_v.txt', 'uav0000281_00460_v.txt', 'uav0000289_00001_v.txt',
                'uav0000289_06922_v.txt', 'uav0000295_02300_v.txt', 'uav0000307_00000_v.txt', 'uav0000315_00000_v.txt', 'uav0000323_01173_v.txt', # 'uav0000300_00000_v.txt', excluded for less cars to other ratio
                'uav0000326_01035_v.txt', 'uav0000342_04692_v.txt', 'uav0000352_05980_v.txt', 'uav0000361_02323_v.txt', 'uav0000366_00001_v.txt'
            ]
        elif mode == 'train15-a':
            seqs = [
                'uav0000124_00944_v.txt', 'uav0000126_00001_v.txt', 'uav0000140_01590_v.txt', 'uav0000143_02250_v.txt', 'uav0000145_00000_v.txt',
                'uav0000218_00001_v.txt', 'uav0000222_03150_v.txt', 'uav0000239_03720_v.txt', 'uav0000239_12336_v.txt', 'uav0000243_00001_v.txt',
                'uav0000244_01440_v.txt', 'uav0000248_00001_v.txt', 'uav0000263_03289_v.txt', 'uav0000264_02760_v.txt', 'uav0000270_00001_v.txt',
                # 'uav0000273_00001_v.txt', 'uav0000278_00001_v.txt', 'uav0000279_00001_v.txt', 'uav0000281_00460_v.txt', 'uav0000289_00001_v.txt',
                # 'uav0000289_06922_v.txt', 'uav0000295_02300_v.txt', 'uav0000307_00000_v.txt', 'uav0000315_00000_v.txt', 'uav0000323_01173_v.txt', # 'uav0000300_00000_v.txt', excluded for less cars to other ratio
                # 'uav0000326_01035_v.txt', 'uav0000342_04692_v.txt', 'uav0000352_05980_v.txt', 'uav0000361_02323_v.txt', 'uav0000366_00001_v.txt'
            ]
        elif mode == 'train15-b':
            seqs = [
                # 'uav0000124_00944_v.txt', 'uav0000126_00001_v.txt', 'uav0000140_01590_v.txt', 'uav0000143_02250_v.txt', 'uav0000145_00000_v.txt',
                # 'uav0000218_00001_v.txt', 'uav0000222_03150_v.txt', 'uav0000239_03720_v.txt', 'uav0000239_12336_v.txt', 'uav0000243_00001_v.txt',
                # 'uav0000244_01440_v.txt', 'uav0000248_00001_v.txt', 'uav0000263_03289_v.txt', 'uav0000264_02760_v.txt', 'uav0000270_00001_v.txt',
                'uav0000273_00001_v.txt', 'uav0000278_00001_v.txt', 'uav0000279_00001_v.txt', 'uav0000281_00460_v.txt', 'uav0000289_00001_v.txt',
                'uav0000289_06922_v.txt', 'uav0000295_02300_v.txt', 'uav0000307_00000_v.txt', 'uav0000315_00000_v.txt', 'uav0000323_01173_v.txt', # 'uav0000300_00000_v.txt', excluded for less cars to other ratio
                'uav0000326_01035_v.txt', 'uav0000342_04692_v.txt', 'uav0000352_05980_v.txt', 'uav0000361_02323_v.txt', 'uav0000366_00001_v.txt'
            ]
       	elif mode == 'train20-a':
            seqs = [
                'uav0000124_00944_v.txt', 'uav0000126_00001_v.txt', 'uav0000140_01590_v.txt', 'uav0000143_02250_v.txt', 'uav0000145_00000_v.txt',
                'uav0000218_00001_v.txt', 'uav0000222_03150_v.txt', 'uav0000239_03720_v.txt', 'uav0000239_12336_v.txt', 'uav0000243_00001_v.txt',
                'uav0000244_01440_v.txt', 'uav0000248_00001_v.txt', 'uav0000263_03289_v.txt', 'uav0000264_02760_v.txt', 'uav0000270_00001_v.txt',
                'uav0000273_00001_v.txt', 'uav0000278_00001_v.txt', 'uav0000279_00001_v.txt', 'uav0000281_00460_v.txt', 'uav0000289_00001_v.txt',
                # 'uav0000289_06922_v.txt', 'uav0000295_02300_v.txt', 'uav0000307_00000_v.txt', 'uav0000315_00000_v.txt', 'uav0000323_01173_v.txt', # 'uav0000300_00000_v.txt', excluded for less cars to other ratio
                # 'uav0000326_01035_v.txt', 'uav0000342_04692_v.txt', 'uav0000352_05980_v.txt', 'uav0000361_02323_v.txt', 'uav0000366_00001_v.txt'
            ]
        elif mode == 'train20-b':
            seqs = [
                'uav0000124_00944_v.txt', 'uav0000126_00001_v.txt', 'uav0000140_01590_v.txt', 'uav0000143_02250_v.txt', 'uav0000145_00000_v.txt',
                'uav0000218_00001_v.txt', 'uav0000222_03150_v.txt', 'uav0000239_03720_v.txt', 'uav0000239_12336_v.txt', 'uav0000243_00001_v.txt',
                # 'uav0000244_01440_v.txt', 'uav0000248_00001_v.txt', 'uav0000263_03289_v.txt', 'uav0000264_02760_v.txt', 'uav0000270_00001_v.txt',
                # 'uav0000273_00001_v.txt', 'uav0000278_00001_v.txt', 'uav0000279_00001_v.txt', 'uav0000281_00460_v.txt', 'uav0000289_00001_v.txt',
                'uav0000289_06922_v.txt', 'uav0000295_02300_v.txt', 'uav0000307_00000_v.txt', 'uav0000315_00000_v.txt', 'uav0000323_01173_v.txt', # 'uav0000300_00000_v.txt', excluded for less cars to other ratio
                'uav0000326_01035_v.txt', 'uav0000342_04692_v.txt', 'uav0000352_05980_v.txt', 'uav0000361_02323_v.txt', 'uav0000366_00001_v.txt'
            ]
        elif mode == 'train20-c':
            seqs = [
                # 'uav0000124_00944_v.txt', 'uav0000126_00001_v.txt', 'uav0000140_01590_v.txt', 'uav0000143_02250_v.txt', 'uav0000145_00000_v.txt',
                # 'uav0000218_00001_v.txt', 'uav0000222_03150_v.txt', 'uav0000239_03720_v.txt', 'uav0000239_12336_v.txt', 'uav0000243_00001_v.txt',
                'uav0000244_01440_v.txt', 'uav0000248_00001_v.txt', 'uav0000263_03289_v.txt', 'uav0000264_02760_v.txt', 'uav0000270_00001_v.txt',
                'uav0000273_00001_v.txt', 'uav0000278_00001_v.txt', 'uav0000279_00001_v.txt', 'uav0000281_00460_v.txt', 'uav0000289_00001_v.txt',
                'uav0000289_06922_v.txt', 'uav0000295_02300_v.txt', 'uav0000307_00000_v.txt', 'uav0000315_00000_v.txt', 'uav0000323_01173_v.txt', # 'uav0000300_00000_v.txt', excluded for less cars to other ratio
                'uav0000326_01035_v.txt', 'uav0000342_04692_v.txt', 'uav0000352_05980_v.txt', 'uav0000361_02323_v.txt', 'uav0000366_00001_v.txt'
            ]
        elif mode == 'train25-a':
            seqs = [
                'uav0000124_00944_v.txt', 'uav0000126_00001_v.txt', 'uav0000140_01590_v.txt', 'uav0000143_02250_v.txt', 'uav0000145_00000_v.txt',
                'uav0000218_00001_v.txt', 'uav0000222_03150_v.txt', 'uav0000239_03720_v.txt', 'uav0000239_12336_v.txt', 'uav0000243_00001_v.txt',
                'uav0000244_01440_v.txt', 'uav0000248_00001_v.txt', 'uav0000263_03289_v.txt', 'uav0000264_02760_v.txt', 'uav0000270_00001_v.txt',
                'uav0000273_00001_v.txt', 'uav0000278_00001_v.txt', 'uav0000279_00001_v.txt', 'uav0000281_00460_v.txt', 'uav0000289_00001_v.txt',
                'uav0000289_06922_v.txt', 'uav0000295_02300_v.txt', 'uav0000307_00000_v.txt', 'uav0000315_00000_v.txt', 'uav0000323_01173_v.txt', # 'uav0000300_00000_v.txt', excluded for less cars to other ratio
                # 'uav0000326_01035_v.txt', 'uav0000342_04692_v.txt', 'uav0000352_05980_v.txt', 'uav0000361_02323_v.txt', 'uav0000366_00001_v.txt'
            ]
        elif mode == 'train25-b':
            seqs = [
                'uav0000124_00944_v.txt', 'uav0000126_00001_v.txt', 'uav0000140_01590_v.txt', 'uav0000143_02250_v.txt', 'uav0000145_00000_v.txt',
                'uav0000218_00001_v.txt', 'uav0000222_03150_v.txt', 'uav0000239_03720_v.txt', 'uav0000239_12336_v.txt', 'uav0000243_00001_v.txt',
                'uav0000244_01440_v.txt', 'uav0000248_00001_v.txt', 'uav0000263_03289_v.txt', 'uav0000264_02760_v.txt', 'uav0000270_00001_v.txt',
                'uav0000273_00001_v.txt', 'uav0000278_00001_v.txt', 'uav0000279_00001_v.txt', 'uav0000281_00460_v.txt', 'uav0000289_00001_v.txt',
                #'uav0000289_06922_v.txt', 'uav0000295_02300_v.txt', 'uav0000307_00000_v.txt', 'uav0000315_00000_v.txt', 'uav0000323_01173_v.txt', # 'uav0000300_00000_v.txt', excluded for less cars to other ratio
                'uav0000326_01035_v.txt', 'uav0000342_04692_v.txt', 'uav0000352_05980_v.txt', 'uav0000361_02323_v.txt', 'uav0000366_00001_v.txt'
            ]
        elif mode == 'train25-c':
            seqs = [
                'uav0000124_00944_v.txt', 'uav0000126_00001_v.txt', 'uav0000140_01590_v.txt', 'uav0000143_02250_v.txt', 'uav0000145_00000_v.txt',
                'uav0000218_00001_v.txt', 'uav0000222_03150_v.txt', 'uav0000239_03720_v.txt', 'uav0000239_12336_v.txt', 'uav0000243_00001_v.txt',
                'uav0000244_01440_v.txt', 'uav0000248_00001_v.txt', 'uav0000263_03289_v.txt', 'uav0000264_02760_v.txt', 'uav0000270_00001_v.txt',
                # 'uav0000273_00001_v.txt', 'uav0000278_00001_v.txt', 'uav0000279_00001_v.txt', 'uav0000281_00460_v.txt', 'uav0000289_00001_v.txt',
                'uav0000289_06922_v.txt', 'uav0000295_02300_v.txt', 'uav0000307_00000_v.txt', 'uav0000315_00000_v.txt', 'uav0000323_01173_v.txt', # 'uav0000300_00000_v.txt', excluded for less cars to other ratio
                'uav0000326_01035_v.txt', 'uav0000342_04692_v.txt', 'uav0000352_05980_v.txt', 'uav0000361_02323_v.txt', 'uav0000366_00001_v.txt'
            ]
        elif mode == 'train25-d':
            seqs = [
                'uav0000124_00944_v.txt', 'uav0000126_00001_v.txt', 'uav0000140_01590_v.txt', 'uav0000143_02250_v.txt', 'uav0000145_00000_v.txt',
                'uav0000218_00001_v.txt', 'uav0000222_03150_v.txt', 'uav0000239_03720_v.txt', 'uav0000239_12336_v.txt', 'uav0000243_00001_v.txt',
                # 'uav0000244_01440_v.txt', 'uav0000248_00001_v.txt', 'uav0000263_03289_v.txt', 'uav0000264_02760_v.txt', 'uav0000270_00001_v.txt',
                'uav0000273_00001_v.txt', 'uav0000278_00001_v.txt', 'uav0000279_00001_v.txt', 'uav0000281_00460_v.txt', 'uav0000289_00001_v.txt',
                'uav0000289_06922_v.txt', 'uav0000295_02300_v.txt', 'uav0000307_00000_v.txt', 'uav0000315_00000_v.txt', 'uav0000323_01173_v.txt', # 'uav0000300_00000_v.txt', excluded for less cars to other ratio
                'uav0000326_01035_v.txt', 'uav0000342_04692_v.txt', 'uav0000352_05980_v.txt', 'uav0000361_02323_v.txt', 'uav0000366_00001_v.txt'
            ]
        elif mode == 'train25-e':
            seqs = [
                'uav0000124_00944_v.txt', 'uav0000126_00001_v.txt', 'uav0000140_01590_v.txt', 'uav0000143_02250_v.txt', 'uav0000145_00000_v.txt',
                # 'uav0000218_00001_v.txt', 'uav0000222_03150_v.txt', 'uav0000239_03720_v.txt', 'uav0000239_12336_v.txt', 'uav0000243_00001_v.txt',
                'uav0000244_01440_v.txt', 'uav0000248_00001_v.txt', 'uav0000263_03289_v.txt', 'uav0000264_02760_v.txt', 'uav0000270_00001_v.txt',
                'uav0000273_00001_v.txt', 'uav0000278_00001_v.txt', 'uav0000279_00001_v.txt', 'uav0000281_00460_v.txt', 'uav0000289_00001_v.txt',
                'uav0000289_06922_v.txt', 'uav0000295_02300_v.txt', 'uav0000307_00000_v.txt', 'uav0000315_00000_v.txt', 'uav0000323_01173_v.txt', # 'uav0000300_00000_v.txt', excluded for less cars to other ratio
                'uav0000326_01035_v.txt', 'uav0000342_04692_v.txt', 'uav0000352_05980_v.txt', 'uav0000361_02323_v.txt', 'uav0000366_00001_v.txt'
            ]
        elif mode == 'train25-f':
            seqs = [
                #'uav0000124_00944_v.txt', 'uav0000126_00001_v.txt', 'uav0000140_01590_v.txt', 'uav0000143_02250_v.txt', 'uav0000145_00000_v.txt',
                'uav0000218_00001_v.txt', 'uav0000222_03150_v.txt', 'uav0000239_03720_v.txt', 'uav0000239_12336_v.txt', 'uav0000243_00001_v.txt',
                'uav0000244_01440_v.txt', 'uav0000248_00001_v.txt', 'uav0000263_03289_v.txt', 'uav0000264_02760_v.txt', 'uav0000270_00001_v.txt',
                'uav0000273_00001_v.txt', 'uav0000278_00001_v.txt', 'uav0000279_00001_v.txt', 'uav0000281_00460_v.txt', 'uav0000289_00001_v.txt',
                'uav0000289_06922_v.txt', 'uav0000295_02300_v.txt', 'uav0000307_00000_v.txt', 'uav0000315_00000_v.txt', 'uav0000323_01173_v.txt', #'uav0000300_00000_v.txt', excluded for less cars to other ratio
                'uav0000326_01035_v.txt', 'uav0000342_04692_v.txt', 'uav0000352_05980_v.txt', 'uav0000361_02323_v.txt', 'uav0000366_00001_v.txt'
            ]
        elif mode == 'whatever-you-got':
            seqs = os.listdir(os.path.join(self.path,seq_dir))

        seqs.sort()
        for seq in seqs:
            frames = os.listdir(os.path.join(self.path, seq_dir, seq))
            frames.sort()
            for frame in frames:
                meta = {
                    'framepath': os.path.join(self.path, seq_dir, seq, frame),
                    'anotpath': os.path.join(self.path, anot_dir, seq + '.txt'),
                    'frameno': frames.index(frame) + 1
                }
                self.meta.append(meta)

    def __len__(self):
        """Returns total number of frames as in meta info"""
        return len(self.meta)

    def __getitem__(self, idx):
        """
        Returns a single frame and annotations for the meta info at index idx.
        takes input:
            idx: index to fetch from meta

        outputs:
            frame: image of shape (C,H,W)
            target: dictionary with annotations
        """
        frame = read_image(self.meta[idx]['framepath']) / 255
        if self.transform:
            frame = self.transform(frame)

        # Annotations
        # anotation file format: Frame No., Object Id., bbox x1, bbox y1, bbox x2, bbox y2
        anots = np.loadtxt(self.meta[idx]['anotpath'], delimiter=',')
        anots = anots[(anots[:,7]==4) | (anots[:,7]==5) | (anots[:,7]==6) | (anots[:,7]==9)] # Filtering vehicles
        anots = anots[anots[:, 0] == self.meta[idx]['frameno']]     # Filter annotations to current frame
        anots = torch.tensor(anots)
        bbboxes = [anots[:, 2], anots[:, 3],  # x1,y1
                   anots[:, 2] + anots[:, 4],  # x2
                   anots[:, 3] + anots[:, 5]]  # y2
        bbboxes = torch.stack(bbboxes).T
        labels = torch.ones(len(bbboxes), dtype=torch.int64)
        area = torch.tensor([(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in bbboxes])
        iscrowd = torch.tensor(anots[:, 6])

        target = {
            'boxes': bbboxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'area': area,
            'iscrowd': iscrowd
        }
        return frame, target

class AirSimObjDet(Dataset):
    """
    Class for synthetic dataset object for images and annotations in format as per the pytorch tutorial for object detection.
    """
    num_classes = 2

    def __init__(self, path, transform=None,mode='15'):
        """
        Take in important info like the root path of dataset and
        record metainfo for later access
        """
        self.path = path
        self.transform = transform
        self.meta = []
        seq_dir = 'Sequences'
        anot_dir = 'Annotations'
        if mode=='15':
            seqs = ['0', '1', '2', '3', '4', 
                    '5', '6', '7', '8', '9', 
                    '10', '11', '12', '13', '14']
        elif mode=='10a':
            seqs = ['0', '1', '2', '3', '4',
                    '5', '6', '7', '8', '9', 
                    #'10', '11', '12', '13', '14'
                    ]
        elif mode=='10b':
            seqs = ['0', '1', '2', '3', '4', 
                    #'5', '6', '7', '8', '9', 
                    '10', '11', '12', '13', '14']
        elif mode=='10c':
            seqs = [#'0', '1', '2', '3', '4',
                    '5', '6', '7', '8', '9', 
                    '10', '11', '12', '13', '14']
        elif mode=='5a':
            seqs = ['0', '1', '2', '3', '4',
                    #'5', '6', '7', '8', '9',
                    #'10', '11', '12', '13', '14'
                    ]
        elif mode=='5b':
            seqs = [#'0', '1', '2', '3', '4',
                    '5', '6', '7', '8', '9',
                    #'10', '11', '12', '13', '14'
                    ]
        elif mode=='5c':
            seqs = [#'0', '1', '2', '3', '4',
                    #'5', '6', '7', '8', '9',
                    '10', '11', '12', '13', '14']
        elif mode=='20':
            seqs = ['0', '1', '2', '3', '4',
                    '5', '6', '7', '8', '9',
                    '10', '11', '12', '13', '14',
                    '15', '16', '17', '18', '19'
                    ]
        elif mode=='25':
            seqs = ['0', '1', '2', '3', '4',
                    '5', '6', '7', '8', '9',
                    '10', '11', '12', '13', '14',
                    '15', '16', '17', '18', '19',
                    '20', '21', '22', '23', '24'
                    ]
        seqs.sort()
        for seq in seqs:
            frames = os.listdir(os.path.join(self.path, seq_dir, seq))
            frames.sort()
            for frame in frames:
                meta = {
                    'framepath': os.path.join(self.path, seq_dir, seq, frame),
                    'anotpath': os.path.join(self.path, anot_dir, seq + '.pkl'),
                    'frameno': frames.index(frame) + 1
                }
                self.meta.append(meta)

    def __len__(self):
        """Returns total number of frames as in meta info"""
        return len(self.meta)

    def __getitem__(self, idx):
        """
        Returns a single frame and annotations for the meta info at index idx.
        takes input:
            idx: index to fetch from meta

        outputs:
            frame: image of shape (C,H,W)
            target: dictionary with annotations
        """
        frame = read_image(self.meta[idx]['framepath'],mode=torchvision.io.ImageReadMode.RGB) / 255
        if self.transform:
            frame = self.transform(frame)

        # Annotations
        # anotation file format: Frame No., Object Id., bbox x1, bbox y1, bbox x2, bbox y2
        with open(self.meta[idx]['anotpath'],'rb') as anot_fs:
            anots = pickle.load(anot_fs)[self.meta[idx]['frameno']-1]
        try:
            pass
        except RuntimeError:
            # runtime error expected for DetectionInfo datatype
            pass
        bbboxes = []
        for object in anots:
            if re.match('Car_[0-9]+',object.name):
                x1,y1 = object.box2D.min.x_val, object.box2D.min.y_val
                x2,y2 = object.box2D.max.x_val, object.box2D.max.y_val
                if x1<x2 and y1<y2:
                    bbboxes.append(torch.tensor((x1,y1,x2,y2)))
        if len(bbboxes)>0:
            bbboxes = torch.stack(bbboxes)
        else:
            bbboxes =torch.empty((0,4))
        labels = torch.ones(len(bbboxes), dtype=torch.int64)
        area = torch.tensor([(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in bbboxes])
        #iscrowd = torch.tensor(anots[:, 6])

        target = {
            'boxes': bbboxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'area': area,
            #'iscrowd': iscrowd
        }
        return frame, target

def get_detection_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.roi_heads.nms_thresh = 0.3
    return model

def evaluate_and_write_result_files(model, data_loader):
    print(f'EVAL {data_loader.dataset}')
    model.eval()
    results = {}
    for i, (imgs, targets) in enumerate(data_loader):
        imgs = [img.to(device) for img in imgs]

        with torch.no_grad():
            preds = model(imgs)

        for pred, target in zip(preds, targets):
            results[target['image_id'].item()] = {'boxes': pred['boxes'].cpu(),
                                                  'scores': pred['scores'].cpu()}
    # data_loader.dataset.write_results_files(results, output_dir)
    data_loader.dataset.print_eval(results)

if __name__ == '__main__':
    import argparse

    argsparser = argparse.ArgumentParser(description='Just passing some parameters')
    argsparser.add_argument('--dataset', help='path to dataset')
    argsparser.add_argument('--dataset_class',help='name of dataset i.e. UAVDT or VisDrone')
    argsparser.add_argument('--synth_dataset', help='path to synthetic dataset')
    argsparser.add_argument('--model', help='path to model weights', default=None)
    argsparser.add_argument('--model_prefix', help='Name prefix for model', default=None)
    argsparser.add_argument('--mode', help='train for full training data, train15-(a/b) for half training data', default=None)
    argsparser.add_argument('--synthmode', help='train for synthetic (10/15/20)-(a/b..)', default=None)
    args = argsparser.parse_args()

    transform = Compose([Resize(IMAGE_RESOLUTION), Normalize(mean=MEAN, std=STD)])

    if args.dataset_class=='UAVDT':
        print('UAVDT selected')
        dataset = UAVDTObjDet(path=args.dataset, mode=args.mode, transform=transform)
    elif args.dataset_class=='VisDrone':
        print('VisDrone selected')
        dataset = VisDroneObjDet(path=args.dataset, mode=args.mode, transform=transform)
    else:
        print('Unknown dataset class entered, please verify')

    print(f'Data: {args.dataset}')
    if args.synth_dataset is not None:
        print(f'Synthetic data: {args.synth_dataset}')
        synth_dataset = AirSimObjDet(path=args.synth_dataset,transform=transform,mode=args.synthmode)
        dataset = ConcatDataset([dataset,synth_dataset])
    else:
        print('No synthetic dataset provided, working with real data only.')
    #dataset_test = UAVDTObjDet(path=args.dataset, mode='test')

    torch.manual_seed(1)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4,
                                              collate_fn=utils.collate_fn)
    #data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=2, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = get_detection_model(NUM_CLASSES)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.00001, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    num_epochs = 30
    # evaluate_and_write_result_files(model, data_loader_test)

    if args.model is not None:
        print(f'Resuming model: {args.model}')
        model.load_state_dict(torch.load(args.model))
        starting_epoch = int(re.findall('epoch_([0-9]+)', args.model)[0]) + 1
    else:
        print('Starting from scratch')
        starting_epoch = 0

    for epoch in range(starting_epoch, num_epochs + 1):
        print(f'TRAIN {data_loader.dataset}')
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=200)

        # update the learning rate
        lr_scheduler.step()

        # evaluate on the test dataset
        if epoch % 2 == 0:
            # evaluate_and_write_result_files(model, data_loader_test)
            torch.save(model.state_dict(), f"models/{args.model_prefix}_epoch_{epoch}.model")

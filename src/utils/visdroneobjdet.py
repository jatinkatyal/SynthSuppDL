import torch
from torch.utils.data import Dataset,DataLoader
from torchvision.io import read_image
from torchvision.transforms import Compose,Resize,Normalize

import utils

import numpy as np
import os

import cv2



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
            seqs = ['uav0000124_00944_v', 'uav0000126_00001_v', 'uav0000140_01590_v', 'uav0000143_02250_v', 'uav0000145_00000_v',
                    'uav0000218_00001_v', 'uav0000222_03150_v', 'uav0000239_03720_v', 'uav0000239_12336_v', 'uav0000243_00001_v',
                    'uav0000244_01440_v', 'uav0000248_00001_v', 'uav0000263_03289_v', 'uav0000264_02760_v', 'uav0000270_00001_v',
                    'uav0000273_00001_v', 'uav0000278_00001_v', 'uav0000279_00001_v', 'uav0000281_00460_v', 'uav0000289_00001_v',
                    'uav0000289_06922_v', 'uav0000295_02300_v', 'uav0000307_00000_v', 'uav0000315_00000_v', 'uav0000323_01173_v', #'uav0000300_00000_v', excluded for less cars to other ratio
                    'uav0000326_01035_v', 'uav0000342_04692_v', 'uav0000352_05980_v', 'uav0000361_02323_v', 'uav0000366_00001_v'
                    ]
        elif mode == 'test':
            seqs = ['uav0000009_03358_v', 'uav0000077_00720_v',  'uav0000119_02301_v', 'uav0000201_00000_v', 'uav0000249_00001_v',
                   'uav0000249_02688_v', 'uav0000297_02761_v', 'uav0000355_00001_v', 'uav0000370_00001_v']
        elif mode == 'train5-a':
            seqs = ['uav0000124_00944_v', 'uav0000126_00001_v', 'uav0000140_01590_v', 'uav0000143_02250_v', 'uav0000145_00000_v',
                    #'uav0000218_00001_v', 'uav0000222_03150_v', 'uav0000239_03720_v', 'uav0000239_12336_v', 'uav0000243_00001_v',
                    #'uav0000244_01440_v', 'uav0000248_00001_v', 'uav0000263_03289_v', 'uav0000264_02760_v', 'uav0000270_00001_v',
                    #'uav0000273_00001_v', 'uav0000278_00001_v', 'uav0000279_00001_v', 'uav0000281_00460_v', 'uav0000289_00001_v',
                    #'uav0000289_06922_v', 'uav0000295_02300_v', 'uav0000307_00000_v', 'uav0000315_00000_v', 'uav0000323_01173_v', # 'uav0000300_00000_v', excluded for less cars to other ratio
                    #'uav0000326_01035_v', 'uav0000342_04692_v', 'uav0000352_05980_v', 'uav0000361_02323_v', 'uav0000366_00001_v'
                    ]
        elif mode == 'train5-b':
            seqs = [
                # 'uav0000124_00944_v', 'uav0000126_00001_v', 'uav0000140_01590_v', 'uav0000143_02250_v', 'uav0000145_00000_v',
                'uav0000218_00001_v', 'uav0000222_03150_v', 'uav0000239_03720_v', 'uav0000239_12336_v', 'uav0000243_00001_v',
                # 'uav0000244_01440_v', 'uav0000248_00001_v', 'uav0000263_03289_v', 'uav0000264_02760_v', 'uav0000270_00001_v',
                # 'uav0000273_00001_v', 'uav0000278_00001_v', 'uav0000279_00001_v', 'uav0000281_00460_v', 'uav0000289_00001_v',
                # 'uav0000289_06922_v', 'uav0000295_02300_v', 'uav0000307_00000_v', 'uav0000315_00000_v', 'uav0000323_01173_v', # 'uav0000300_00000_v', excluded for less cars to other ratio
                # 'uav0000326_01035_v', 'uav0000342_04692_v', 'uav0000352_05980_v', 'uav0000361_02323_v', 'uav0000366_00001_v'
            ]
        elif mode == 'train5-c':
            seqs = [
                # 'uav0000124_00944_v', 'uav0000126_00001_v', 'uav0000140_01590_v', 'uav0000143_02250_v', 'uav0000145_00000_v',
                # 'uav0000218_00001_v', 'uav0000222_03150_v', 'uav0000239_03720_v', 'uav0000239_12336_v', 'uav0000243_00001_v',
                'uav0000244_01440_v', 'uav0000248_00001_v', 'uav0000263_03289_v', 'uav0000264_02760_v', 'uav0000270_00001_v',
                # 'uav0000273_00001_v', 'uav0000278_00001_v', 'uav0000279_00001_v', 'uav0000281_00460_v', 'uav0000289_00001_v',
                # 'uav0000289_06922_v', 'uav0000295_02300_v', 'uav0000307_00000_v', 'uav0000315_00000_v', 'uav0000323_01173_v', # 'uav0000300_00000_v', excluded for less cars to other ratio
                # 'uav0000326_01035_v', 'uav0000342_04692_v', 'uav0000352_05980_v', 'uav0000361_02323_v', 'uav0000366_00001_v'
            ]
        elif mode == 'train5-d':
            seqs = [
                # 'uav0000124_00944_v', 'uav0000126_00001_v', 'uav0000140_01590_v', 'uav0000143_02250_v', 'uav0000145_00000_v',
                # 'uav0000218_00001_v', 'uav0000222_03150_v', 'uav0000239_03720_v', 'uav0000239_12336_v', 'uav0000243_00001_v',
                # 'uav0000244_01440_v', 'uav0000248_00001_v', 'uav0000263_03289_v', 'uav0000264_02760_v', 'uav0000270_00001_v',
                'uav0000273_00001_v', 'uav0000278_00001_v', 'uav0000279_00001_v', 'uav0000281_00460_v', 'uav0000289_00001_v',
                # 'uav0000289_06922_v', 'uav0000295_02300_v', 'uav0000307_00000_v', 'uav0000315_00000_v', 'uav0000323_01173_v', # 'uav0000300_00000_v', excluded for less cars to other ratio
                # 'uav0000326_01035_v', 'uav0000342_04692_v', 'uav0000352_05980_v', 'uav0000361_02323_v', 'uav0000366_00001_v'
            ]
        elif mode == 'train5-e':
            seqs = [
                # 'uav0000124_00944_v', 'uav0000126_00001_v', 'uav0000140_01590_v', 'uav0000143_02250_v', 'uav0000145_00000_v',
                # 'uav0000218_00001_v', 'uav0000222_03150_v', 'uav0000239_03720_v', 'uav0000239_12336_v', 'uav0000243_00001_v',
                # 'uav0000244_01440_v', 'uav0000248_00001_v', 'uav0000263_03289_v', 'uav0000264_02760_v', 'uav0000270_00001_v',
                # 'uav0000273_00001_v', 'uav0000278_00001_v', 'uav0000279_00001_v', 'uav0000281_00460_v', 'uav0000289_00001_v',
                'uav0000289_06922_v', 'uav0000295_02300_v', 'uav0000307_00000_v', 'uav0000315_00000_v', 'uav0000323_01173_v', # 'uav0000300_00000_v', excluded for less cars to other ratio
                # 'uav0000326_01035_v', 'uav0000342_04692_v', 'uav0000352_05980_v', 'uav0000361_02323_v', 'uav0000366_00001_v'
            ]
        elif mode == 'train5-f':
            seqs = [
                # 'uav0000124_00944_v', 'uav0000126_00001_v', 'uav0000140_01590_v', 'uav0000143_02250_v', 'uav0000145_00000_v',
                # 'uav0000218_00001_v', 'uav0000222_03150_v', 'uav0000239_03720_v', 'uav0000239_12336_v', 'uav0000243_00001_v',
                # 'uav0000244_01440_v', 'uav0000248_00001_v', 'uav0000263_03289_v', 'uav0000264_02760_v', 'uav0000270_00001_v',
                # 'uav0000273_00001_v', 'uav0000278_00001_v', 'uav0000279_00001_v', 'uav0000281_00460_v', 'uav0000289_00001_v',
                # 'uav0000289_06922_v', 'uav0000295_02300_v', 'uav0000307_00000_v', 'uav0000315_00000_v', 'uav0000323_01173_v', # 'uav0000300_00000_v', excluded for less cars to other ratio
                'uav0000326_01035_v', 'uav0000342_04692_v', 'uav0000352_05980_v', 'uav0000361_02323_v', 'uav0000366_00001_v'
            ]
        elif mode == 'train10-a':
            seqs = [
                'uav0000124_00944_v', 'uav0000126_00001_v', 'uav0000140_01590_v', 'uav0000143_02250_v', 'uav0000145_00000_v',
                'uav0000218_00001_v', 'uav0000222_03150_v', 'uav0000239_03720_v', 'uav0000239_12336_v', 'uav0000243_00001_v',
                # 'uav0000244_01440_v', 'uav0000248_00001_v', 'uav0000263_03289_v', 'uav0000264_02760_v', 'uav0000270_00001_v',
                # 'uav0000273_00001_v', 'uav0000278_00001_v', 'uav0000279_00001_v', 'uav0000281_00460_v', 'uav0000289_00001_v',
                # 'uav0000289_06922_v', 'uav0000295_02300_v', 'uav0000307_00000_v', 'uav0000315_00000_v', 'uav0000323_01173_v', # 'uav0000300_00000_v', excluded for less cars to other ratio
                # 'uav0000326_01035_v', 'uav0000342_04692_v', 'uav0000352_05980_v', 'uav0000361_02323_v', 'uav0000366_00001_v'
            ]
        elif mode == 'train10-b':
            seqs = [
                # 'uav0000124_00944_v', 'uav0000126_00001_v', 'uav0000140_01590_v', 'uav0000143_02250_v', 'uav0000145_00000_v',
                # 'uav0000218_00001_v', 'uav0000222_03150_v', 'uav0000239_03720_v', 'uav0000239_12336_v', 'uav0000243_00001_v',
                'uav0000244_01440_v', 'uav0000248_00001_v', 'uav0000263_03289_v', 'uav0000264_02760_v', 'uav0000270_00001_v',
                'uav0000273_00001_v', 'uav0000278_00001_v', 'uav0000279_00001_v', 'uav0000281_00460_v', 'uav0000289_00001_v',
                # 'uav0000289_06922_v', 'uav0000295_02300_v', 'uav0000307_00000_v', 'uav0000315_00000_v', 'uav0000323_01173_v', # 'uav0000300_00000_v', excluded for less cars to other ratio
                # 'uav0000326_01035_v', 'uav0000342_04692_v', 'uav0000352_05980_v', 'uav0000361_02323_v', 'uav0000366_00001_v'
            ]
        elif mode == 'train10-c':
            seqs = [
                # 'uav0000124_00944_v', 'uav0000126_00001_v', 'uav0000140_01590_v', 'uav0000143_02250_v', 'uav0000145_00000_v',
                # 'uav0000218_00001_v', 'uav0000222_03150_v', 'uav0000239_03720_v', 'uav0000239_12336_v', 'uav0000243_00001_v',
                # 'uav0000244_01440_v', 'uav0000248_00001_v', 'uav0000263_03289_v', 'uav0000264_02760_v', 'uav0000270_00001_v',
                # 'uav0000273_00001_v', 'uav0000278_00001_v', 'uav0000279_00001_v', 'uav0000281_00460_v', 'uav0000289_00001_v',
                'uav0000289_06922_v', 'uav0000295_02300_v', 'uav0000307_00000_v', 'uav0000315_00000_v', 'uav0000323_01173_v', # 'uav0000300_00000_v', excluded for less cars to other ratio
                'uav0000326_01035_v', 'uav0000342_04692_v', 'uav0000352_05980_v', 'uav0000361_02323_v', 'uav0000366_00001_v'
            ]
        elif mode == 'train15-a':
            seqs = [
                'uav0000124_00944_v', 'uav0000126_00001_v', 'uav0000140_01590_v', 'uav0000143_02250_v', 'uav0000145_00000_v',
                'uav0000218_00001_v', 'uav0000222_03150_v', 'uav0000239_03720_v', 'uav0000239_12336_v', 'uav0000243_00001_v',
                'uav0000244_01440_v', 'uav0000248_00001_v', 'uav0000263_03289_v', 'uav0000264_02760_v', 'uav0000270_00001_v',
                # 'uav0000273_00001_v', 'uav0000278_00001_v', 'uav0000279_00001_v', 'uav0000281_00460_v', 'uav0000289_00001_v',
                # 'uav0000289_06922_v', 'uav0000295_02300_v', 'uav0000307_00000_v', 'uav0000315_00000_v', 'uav0000323_01173_v', # 'uav0000300_00000_v', excluded for less cars to other ratio
                # 'uav0000326_01035_v', 'uav0000342_04692_v', 'uav0000352_05980_v', 'uav0000361_02323_v', 'uav0000366_00001_v'
            ]
        elif mode == 'train15-b':
            seqs = [
                # 'uav0000124_00944_v', 'uav0000126_00001_v', 'uav0000140_01590_v', 'uav0000143_02250_v', 'uav0000145_00000_v',
                # 'uav0000218_00001_v', 'uav0000222_03150_v', 'uav0000239_03720_v', 'uav0000239_12336_v', 'uav0000243_00001_v',
                # 'uav0000244_01440_v', 'uav0000248_00001_v', 'uav0000263_03289_v', 'uav0000264_02760_v', 'uav0000270_00001_v',
                'uav0000273_00001_v', 'uav0000278_00001_v', 'uav0000279_00001_v', 'uav0000281_00460_v', 'uav0000289_00001_v',
                'uav0000289_06922_v', 'uav0000295_02300_v', 'uav0000307_00000_v', 'uav0000315_00000_v', 'uav0000323_01173_v', # 'uav0000300_00000_v', excluded for less cars to other ratio
                'uav0000326_01035_v', 'uav0000342_04692_v', 'uav0000352_05980_v', 'uav0000361_02323_v', 'uav0000366_00001_v'
            ]
       	elif mode == 'train20-a':
            seqs = [
                'uav0000124_00944_v', 'uav0000126_00001_v', 'uav0000140_01590_v', 'uav0000143_02250_v', 'uav0000145_00000_v',
                'uav0000218_00001_v', 'uav0000222_03150_v', 'uav0000239_03720_v', 'uav0000239_12336_v', 'uav0000243_00001_v',
                'uav0000244_01440_v', 'uav0000248_00001_v', 'uav0000263_03289_v', 'uav0000264_02760_v', 'uav0000270_00001_v',
                'uav0000273_00001_v', 'uav0000278_00001_v', 'uav0000279_00001_v', 'uav0000281_00460_v', 'uav0000289_00001_v',
                # 'uav0000289_06922_v', 'uav0000295_02300_v', 'uav0000307_00000_v', 'uav0000315_00000_v', 'uav0000323_01173_v', # 'uav0000300_00000_v', excluded for less cars to other ratio
                # 'uav0000326_01035_v', 'uav0000342_04692_v', 'uav0000352_05980_v', 'uav0000361_02323_v', 'uav0000366_00001_v'
            ]
        elif mode == 'train20-b':
            seqs = [
                'uav0000124_00944_v', 'uav0000126_00001_v', 'uav0000140_01590_v', 'uav0000143_02250_v', 'uav0000145_00000_v',
                'uav0000218_00001_v', 'uav0000222_03150_v', 'uav0000239_03720_v', 'uav0000239_12336_v', 'uav0000243_00001_v',
                # 'uav0000244_01440_v', 'uav0000248_00001_v', 'uav0000263_03289_v', 'uav0000264_02760_v', 'uav0000270_00001_v',
                # 'uav0000273_00001_v', 'uav0000278_00001_v', 'uav0000279_00001_v', 'uav0000281_00460_v', 'uav0000289_00001_v',
                'uav0000289_06922_v', 'uav0000295_02300_v', 'uav0000307_00000_v', 'uav0000315_00000_v', 'uav0000323_01173_v', # 'uav0000300_00000_v', excluded for less cars to other ratio
                'uav0000326_01035_v', 'uav0000342_04692_v', 'uav0000352_05980_v', 'uav0000361_02323_v', 'uav0000366_00001_v'
            ]
        elif mode == 'train20-c':
            seqs = [
                # 'uav0000124_00944_v', 'uav0000126_00001_v', 'uav0000140_01590_v', 'uav0000143_02250_v', 'uav0000145_00000_v',
                # 'uav0000218_00001_v', 'uav0000222_03150_v', 'uav0000239_03720_v', 'uav0000239_12336_v', 'uav0000243_00001_v',
                'uav0000244_01440_v', 'uav0000248_00001_v', 'uav0000263_03289_v', 'uav0000264_02760_v', 'uav0000270_00001_v',
                'uav0000273_00001_v', 'uav0000278_00001_v', 'uav0000279_00001_v', 'uav0000281_00460_v', 'uav0000289_00001_v',
                'uav0000289_06922_v', 'uav0000295_02300_v', 'uav0000307_00000_v', 'uav0000315_00000_v', 'uav0000323_01173_v', # 'uav0000300_00000_v', excluded for less cars to other ratio
                'uav0000326_01035_v', 'uav0000342_04692_v', 'uav0000352_05980_v', 'uav0000361_02323_v', 'uav0000366_00001_v'
            ]
        elif mode == 'train25-a':
            seqs = [
                'uav0000124_00944_v', 'uav0000126_00001_v', 'uav0000140_01590_v', 'uav0000143_02250_v', 'uav0000145_00000_v',
                'uav0000218_00001_v', 'uav0000222_03150_v', 'uav0000239_03720_v', 'uav0000239_12336_v', 'uav0000243_00001_v',
                'uav0000244_01440_v', 'uav0000248_00001_v', 'uav0000263_03289_v', 'uav0000264_02760_v', 'uav0000270_00001_v',
                'uav0000273_00001_v', 'uav0000278_00001_v', 'uav0000279_00001_v', 'uav0000281_00460_v', 'uav0000289_00001_v',
                'uav0000289_06922_v', 'uav0000295_02300_v', 'uav0000307_00000_v', 'uav0000315_00000_v', 'uav0000323_01173_v', # 'uav0000300_00000_v', excluded for less cars to other ratio
                # 'uav0000326_01035_v', 'uav0000342_04692_v', 'uav0000352_05980_v', 'uav0000361_02323_v', 'uav0000366_00001_v'
            ]
        elif mode == 'train25-b':
            seqs = [
                'uav0000124_00944_v', 'uav0000126_00001_v', 'uav0000140_01590_v', 'uav0000143_02250_v', 'uav0000145_00000_v',
                'uav0000218_00001_v', 'uav0000222_03150_v', 'uav0000239_03720_v', 'uav0000239_12336_v', 'uav0000243_00001_v',
                'uav0000244_01440_v', 'uav0000248_00001_v', 'uav0000263_03289_v', 'uav0000264_02760_v', 'uav0000270_00001_v',
                'uav0000273_00001_v', 'uav0000278_00001_v', 'uav0000279_00001_v', 'uav0000281_00460_v', 'uav0000289_00001_v',
                #'uav0000289_06922_v', 'uav0000295_02300_v', 'uav0000307_00000_v', 'uav0000315_00000_v', 'uav0000323_01173_v', # 'uav0000300_00000_v', excluded for less cars to other ratio
                'uav0000326_01035_v', 'uav0000342_04692_v', 'uav0000352_05980_v', 'uav0000361_02323_v', 'uav0000366_00001_v'
            ]
        elif mode == 'train25-c':
            seqs = [
                'uav0000124_00944_v', 'uav0000126_00001_v', 'uav0000140_01590_v', 'uav0000143_02250_v', 'uav0000145_00000_v',
                'uav0000218_00001_v', 'uav0000222_03150_v', 'uav0000239_03720_v', 'uav0000239_12336_v', 'uav0000243_00001_v',
                'uav0000244_01440_v', 'uav0000248_00001_v', 'uav0000263_03289_v', 'uav0000264_02760_v', 'uav0000270_00001_v',
                # 'uav0000273_00001_v', 'uav0000278_00001_v', 'uav0000279_00001_v', 'uav0000281_00460_v', 'uav0000289_00001_v',
                'uav0000289_06922_v', 'uav0000295_02300_v', 'uav0000307_00000_v', 'uav0000315_00000_v', 'uav0000323_01173_v', # 'uav0000300_00000_v', excluded for less cars to other ratio
                'uav0000326_01035_v', 'uav0000342_04692_v', 'uav0000352_05980_v', 'uav0000361_02323_v', 'uav0000366_00001_v'
            ]
        elif mode == 'train25-d':
            seqs = [
                'uav0000124_00944_v', 'uav0000126_00001_v', 'uav0000140_01590_v', 'uav0000143_02250_v', 'uav0000145_00000_v',
                'uav0000218_00001_v', 'uav0000222_03150_v', 'uav0000239_03720_v', 'uav0000239_12336_v', 'uav0000243_00001_v',
                # 'uav0000244_01440_v', 'uav0000248_00001_v', 'uav0000263_03289_v', 'uav0000264_02760_v', 'uav0000270_00001_v',
                'uav0000273_00001_v', 'uav0000278_00001_v', 'uav0000279_00001_v', 'uav0000281_00460_v', 'uav0000289_00001_v',
                'uav0000289_06922_v', 'uav0000295_02300_v', 'uav0000307_00000_v', 'uav0000315_00000_v', 'uav0000323_01173_v', # 'uav0000300_00000_v', excluded for less cars to other ratio
                'uav0000326_01035_v', 'uav0000342_04692_v', 'uav0000352_05980_v', 'uav0000361_02323_v', 'uav0000366_00001_v'
            ]
        elif mode == 'train25-e':
            seqs = [
                'uav0000124_00944_v', 'uav0000126_00001_v', 'uav0000140_01590_v', 'uav0000143_02250_v', 'uav0000145_00000_v',
                # 'uav0000218_00001_v', 'uav0000222_03150_v', 'uav0000239_03720_v', 'uav0000239_12336_v', 'uav0000243_00001_v',
                'uav0000244_01440_v', 'uav0000248_00001_v', 'uav0000263_03289_v', 'uav0000264_02760_v', 'uav0000270_00001_v',
                'uav0000273_00001_v', 'uav0000278_00001_v', 'uav0000279_00001_v', 'uav0000281_00460_v', 'uav0000289_00001_v',
                'uav0000289_06922_v', 'uav0000295_02300_v', 'uav0000307_00000_v', 'uav0000315_00000_v', 'uav0000323_01173_v', # 'uav0000300_00000_v', excluded for less cars to other ratio
                'uav0000326_01035_v', 'uav0000342_04692_v', 'uav0000352_05980_v', 'uav0000361_02323_v', 'uav0000366_00001_v'
            ]
        elif mode == 'train25-f':
            seqs = [
                #'uav0000124_00944_v', 'uav0000126_00001_v', 'uav0000140_01590_v', 'uav0000143_02250_v', 'uav0000145_00000_v',
                'uav0000218_00001_v', 'uav0000222_03150_v', 'uav0000239_03720_v', 'uav0000239_12336_v', 'uav0000243_00001_v',
                'uav0000244_01440_v', 'uav0000248_00001_v', 'uav0000263_03289_v', 'uav0000264_02760_v', 'uav0000270_00001_v',
                'uav0000273_00001_v', 'uav0000278_00001_v', 'uav0000279_00001_v', 'uav0000281_00460_v', 'uav0000289_00001_v',
                'uav0000289_06922_v', 'uav0000295_02300_v', 'uav0000307_00000_v', 'uav0000315_00000_v', 'uav0000323_01173_v', #'uav0000300_00000_v', excluded for less cars to other ratio
                'uav0000326_01035_v', 'uav0000342_04692_v', 'uav0000352_05980_v', 'uav0000361_02323_v', 'uav0000366_00001_v'
            ]
        elif mode == 'whatever-you-got':
            seqs = os.listdir(os.path.join(self.path,seq_dir))

        seqs.sort()
        #print(self.path, seq_dir, seqs)
        for seq in seqs:
            print()
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
            shape_og = frame.shape
            frame = self.transform(frame)
            shape_new = frame.shape

        # Annotations
        # anotation file format: Frame No., Object Id., bbox x1, bbox y1, bbox x2, bbox y2
        anots = np.loadtxt(self.meta[idx]['anotpath'], delimiter=',')
        anots = anots[anots[:, 0] == self.meta[idx]['frameno']]     # Filter annotations to current frame
        anots = anots[(anots[:,7]==4) | (anots[:,7]==5) | (anots[:,7]==6) | (anots[:,7]==9)] # Filtering vehicles
        anots = anots[anots[:,9]!=2]    # Filtering out heavily occluded objects

        anots = torch.tensor(anots)
        bbboxes = [anots[:, 2], anots[:, 3],  # x1,y1
                   anots[:, 2] + anots[:, 4],  # x2
                   anots[:, 3] + anots[:, 5]]  # y2
        bbboxes = torch.stack(bbboxes).T
        if self.transform is not None:
            c_og, y_og, x_og = shape_og
            c_new,y_new,x_new = shape_new
            bbboxes = bbboxes*torch.tensor([x_new/x_og,y_new/y_og,x_new/x_og,y_new/y_og])
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

if __name__=='__main__':
    IMAGE_RESOLUTION = (540, 1024)  # Constant Resolution to maintain for inputs
    MEAN = [0.485, 0.456, 0.406]  # Mean for normalizing
    STD = [0.225, 0.225, 0.225]  # STD for normalizing

    transform = Compose([Resize(IMAGE_RESOLUTION)])#, Normalize(mean=MEAN, std=STD)])
    dataset = VisDroneObjDet(path="/home/jatin/HDD2TB/datasets/VisDrone/VisDrone2019-MOT-test-dev", mode="test", transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=20, shuffle=True, num_workers=4, collate_fn=utils.collate_fn)

    for i,(X,y) in enumerate(data_loader):
        for j in range(len(X)):
            frame = X[j].squeeze().permute(1,2,0).numpy().copy()
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            boxes = y[j]['boxes']
            for box in boxes:
                x1,y1,x2,y2 = box
                x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
                cv2.rectangle(frame,(x1,y1),(x2,y2),(1,0,0),2)
            cv2.imshow('seq',frame)
            cv2.waitKey(0)
        #break
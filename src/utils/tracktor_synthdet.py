import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.io import read_image

import utils

import os
import pickle
import re


class AirSimObjDet(Dataset):
    """
    Class for synthetic dataset object for images and annotations in format as per the pytorch tutorial for object detection.
    """
    num_classes = 2

    def __init__(self, path, transform=None):
        """
        Take in important info like the root path of dataset and
        record metainfo for later access
        """
        self.path = path
        self.transform = transform
        self.meta = []
        seq_dir = 'Sequences'
        anot_dir = 'Annotations'
        seqs = os.listdir(os.path.join(self.path,seq_dir))
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

if __name__ =='__main__':
    dataset = AirSimObjDet('/home/jatin/HDD2TB/datasets/Synthetic/AirSim-UAV-Synth/data')
    dataloader = data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4,
                                              collate_fn=utils.collate_fn)

    syn_data_path = '/home/jatin/HDD2TB/datasets/Synthetic/AirSim-UAV-Synth/data'
    annos = os.listdir(os.path.join(syn_data_path,'Annotations'))
    for anno in annos:
        with open(os.path.join(syn_data_path,'Annotations',anno), 'rb') as anno_fs:
            anno = pickle.load(anno_fs)
        print((anno[0]))
        break
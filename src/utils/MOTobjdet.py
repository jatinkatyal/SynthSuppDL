import torch
import torchvision.io
from torch.utils.data import Dataset,DataLoader
from torchvision.io import read_image
from torchvision.transforms import Compose,Resize,Normalize

import utils

import numpy as np
import pandas as pd
import os
from configparser import ConfigParser

import cv2

class MOTSynthObjDet(Dataset):
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
        anot_dir = 'mot_annotations'
        memo = {}
        if mode == 'train-3-all':
            seq_dirs = ['MOTSynth_1', 'MOTSynth_2', 'MOTSynth_3']
            for seq_dir in seq_dirs:
                memo[seq_dir] = []
                for seq in os.listdir(os.path.join(self.path, seq_dir)):
                    memo[seq_dir].append(seq[:-4])
        elif mode == 'train-21':
            seq_dirs = ['MOTSynth_1'] #, 'MOTSynth_2', 'MOTSynth_3']
            for seq_dir in seq_dirs:
                memo[seq_dir] = []
                seqs = os.listdir(os.path.join(self.path, seq_dir))
                seqs.sort()
                seqs = seqs[:21]
                for seq in seqs:
                    memo[seq_dir].append(seq[:-4])

        for seq_dir in memo.keys():
            for seq in memo[seq_dir]:
                config = ConfigParser()
                info_path = os.path.join(self.path,anot_dir,seq,'seqinfo.ini')
                config.read(info_path)
                for frameno in range(1,int(config['Sequence']['seqLength'])+1):
                    frame_no = frameno
                    if frame_no > 999:
                        frame_no = str(frame_no)
                    elif frame_no > 99:
                        frame_no = '0'.join(str(frame_no))
                    elif frame_no > 9:
                        frame_no = '00'.join(str(frame_no))
                    else:
                        frame_no = '000'.join(str(frame_no))
                    meta = {
                        'subset': seq_dir,
                        'seq': seq,
                        'frame': frame_no,
                        'frameno': frameno
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
        meta = self.meta[idx]
        sequence_path = os.path.join(self.path,meta['subset'],meta['seq']+'.mp4')
        cap = cv2.VideoCapture(sequence_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES,meta['frameno']-1)
        _,frame = cap.read()
        frame = torch.tensor(frame) / 255
        frame = torch.permute(frame,[2,0,1])

        if self.transform:
            shape_og = frame.shape
            frame = self.transform(frame)
            shape_new = frame.shape

        # Annotations
        # anotation file format: Frame No., Object Id., bbox x1, bbox y1, bbox x2, bbox y2
        anot_dir = 'MOTSynth_mot_annotations/mot_annotations'
        #anots = pd.read_csv(os.path.join(self.path,anot_dir,meta['seq'],'gt/gt.txt'), names=['frame','objid','x','y','w','h','conf','rx','ry','rz'])
        anots = np.loadtxt(os.path.join(self.path,anot_dir,meta['seq'],'gt/gt.txt'), delimiter=',')
        anots = anots[anots[:,0]==meta['frameno']]      # Filter annotations to current frame
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
        iscrowd = torch.ones(len(bbboxes), dtype=torch.int64)

        target = {
            'boxes': bbboxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'area': area,
            'iscrowd': iscrowd
        }
        return frame, target

class MOT17ObjDet(Dataset):
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
        self.images_dir = ''
        self.anot_dir = ''
        #anot_dir = 'MOTSynth_mot_annotations/mot_annotations'
        memo = {}
        if mode == 'test':
            split = 'test'
            selector = range(len(os.listdir(os.path.join(self.path, self.images_dir, split))))
        elif mode == 'train':
            split = 'train'
            selector = range(len(os.listdir(os.path.join(self.path, self.images_dir, split))))
        elif mode == 'train-3a':
            split = 'train'
            selector = range(3)
        elif mode == 'train-3b':
            split = 'train'
            selector = range(3,6)
        elif mode == 'train-3c':
            split = 'train'
            selector = range(6,9)
        elif mode == 'train-3d':
            split = 'train'
            selector = range(9,12)
        elif mode == 'train-3e':
            split = 'train'
            selector = range(12,15)
        elif mode == 'train-3f':
            split = 'train'
            selector = range(15,18)
        elif mode == 'train-3g':
            split = 'train'
            selector = range(18,21)
        elif mode == 'train-14b':
            split = 'train'
            unselector = range(7, 14)
            selector = [i for i in range(21)]
            for i in unselector:
                selector.remove(i)
        elif mode == 'train-14c':
            split = 'train'
            selector = range(7, 21)
        elif mode == 'train-7a':
            split = 'train'
            selector = range(7)
        elif mode == 'train-7b':
            split = 'train'
            selector = range(7,14)
        elif mode == 'train-7c':
            split = 'train'
            selector = range(14,21)
        elif mode == 'train-14a':
            split = 'train'
            selector = range(14)
        elif mode == 'train-14b':
            split = 'train'
            unselector = range(7,14)
            selector = [i for i in range(21)]
            for i in unselector:
                selector.remove(i)
        elif mode == 'train-14c':
            split = 'train'
            selector = range(7,21)
        elif mode == 'train-18a':
            split = 'train'
            selector = range(18)
        elif mode == 'train-18b':
            split = 'train'
            unselector = range(15,18)
            selector = [i for i in range(21)]
            for i in unselector:
                selector.remove(i)
        elif mode == 'train-18c':
            split = 'train'
            unselector = range(12,15)
            selector = [i for i in range(21)]
            for i in unselector:
                selector.remove(i)
        elif mode == 'train-18d':
            split = 'train'
            unselector = range(9,12)
            selector = [i for i in range(21)]
            for i in unselector:
                selector.remove(i)
        elif mode == 'train-18e':
            split = 'train'
            unselector = range(6,9)
            selector = [i for i in range(21)]
            for i in unselector:
                selector.remove(i)
        elif mode == 'train-18f':
            split = 'train'
            unselector = range(3,6)
            selector = [i for i in range(21)]
            for i in unselector:
                selector.remove(i)
        elif mode == 'train-18g':
            split = 'train'
            selector = range(3,21)
        print(f'selected {[i for i in selector]}')

        seqs = os.listdir(os.path.join(self.path, self.images_dir, split))
        seqs = [seqs[i] for i in selector]
        memo[split] = seqs

        for seq_dir in memo.keys():
            for seq in memo[seq_dir]:
                frames = os.listdir(os.path.join(self.path, self.images_dir, seq_dir, seq, 'img1'))
                frames.sort()
                for frame in frames:
                    meta = {
                        'seq_dir': seq_dir,
                        'seq': seq,
                        'frame': frame,
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
        meta = self.meta[idx]
        frame = os.path.join(self.path, self.images_dir, meta['seq_dir'], meta['seq'],'img1',meta['frame'])
        frame = torchvision.io.read_image(frame)
        frame = torch.tensor(frame) / 255

        if self.transform:
            shape_og = frame.shape
            frame = self.transform(frame)
            shape_new = frame.shape

        # Annotations
        # names=['frame','objid','x','y','w','h','conf','rx','ry','rz']
        anots = np.loadtxt(os.path.join(self.path, self.anot_dir, meta['seq_dir'], meta['seq'], 'gt/gt.txt'), delimiter=',')
        anots = anots[anots[:, 0] == int(meta['frame'][:-4])]  # Filter annotations to current frame
        anots = torch.tensor(anots)
        bbboxes = [anots[:, 2], anots[:, 3],  # x1,y1
                   anots[:, 2] + anots[:, 4],  # x2
                   anots[:, 3] + anots[:, 5]]  # y2
        bbboxes = torch.stack(bbboxes).T
        if self.transform is not None:
            c_og, y_og, x_og = shape_og
            c_new, y_new, x_new = shape_new
            bbboxes = bbboxes * torch.tensor([x_new / x_og, y_new / y_og, x_new / x_og, y_new / y_og])
        labels = torch.ones(len(bbboxes), dtype=torch.int64)
        area = torch.tensor([(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in bbboxes])
        iscrowd = torch.ones(len(bbboxes), dtype=torch.int64)

        target = {
            'boxes': bbboxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'area': area,
            'iscrowd': iscrowd
        }
        return frame, target

class MOTSynthFramesObjDet(Dataset):
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
        self.images_dir = 'frames'
        self.anot_dir = 'mot_annotations'
        #anot_dir = 'MOTSynth_mot_annotations/mot_annotations'
        memo = {}
        if mode == 'train':
            selector = range(len(os.listdir(os.path.join(self.path, self.images_dir, 'rgb'))))
        elif mode == 'train-3a':
            selector = range(3)
        elif mode == 'train-3b':
            selector = range(3,6)
        elif mode == 'train-3c':
            selector = range(6,9)
        elif mode == 'train-3d':
            selector = range(9,12)
        elif mode == 'train-3e':
            selector = range(12,15)
        elif mode == 'train-3f':
            selector = range(15,18)
        elif mode == 'train-3g':
            selector = range(18,21)
        elif mode == 'train-14b':
            unselector = range(7, 14)
            selector = [i for i in range(21)]
            for i in unselector:
                selector.remove(i)
        elif mode == 'train-14c':
            selector = range(7, 21)
        elif mode == 'train-7a':
            selector = range(7)
        elif mode == 'train-7b':
            selector = range(7,14)
        elif mode == 'train-7c':
            selector = range(14,21)
        elif mode == 'train-14a':
            selector = range(14)
        elif mode == 'train-14b':
            unselector = range(7,14)
            selector = [i for i in range(21)]
            for i in unselector:
                selector.remove(i)
        elif mode == 'train-14c':
            selector = range(7,21)
        elif mode == 'train-18a':
            selector = range(18)
        elif mode == 'train-18b':
            unselector = range(15,18)
            selector = [i for i in range(21)]
            for i in unselector:
                selector.remove(i)
        elif mode == 'train-18c':
            unselector = range(12,15)
            selector = [i for i in range(21)]
            for i in unselector:
                selector.remove(i)
        elif mode == 'train-18d':
            unselector = range(9,12)
            selector = [i for i in range(21)]
            for i in unselector:
                selector.remove(i)
        elif mode == 'train-18e':
            unselector = range(6,9)
            selector = [i for i in range(21)]
            for i in unselector:
                selector.remove(i)
        elif mode == 'train-18f':
            unselector = range(3,6)
            selector = [i for i in range(21)]
            for i in unselector:
                selector.remove(i)
        elif mode == 'train-18g':
            selector = range(3,21)
        print(f'selected {[i for i in selector]}')

        seqs = ['001','002','011','019','073','038','072',
                '249','243','241','236','228','225','223',
                '196','192','181','168','143','121','112']
        #print(len(seqs))
        seqs = [seqs[i] for i in selector]

        for seq in seqs:
            frames = os.listdir(os.path.join(self.path, self.images_dir, seq, 'rgb'))
            frames.sort()
            for frame in frames:
                meta = {
                    'seq': seq,
                    'frame': frame,
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
        meta = self.meta[idx]
        frame = os.path.join(self.path, self.images_dir, meta['seq'],'rgb',meta['frame'])
        frame = torchvision.io.read_image(frame)
        frame = torch.tensor(frame) / 255

        if self.transform:
            shape_og = frame.shape
            frame = self.transform(frame)
            shape_new = frame.shape

        # Annotations
        # names=['frame_id','object_id','x','y','w','h','_1','_2','_3','rx','ry','rz'])
        anots = np.loadtxt(os.path.join(self.path, self.anot_dir, meta['seq'], 'gt/gt.txt'), delimiter=',')
        anots = anots[anots[:, 0] == int(meta['frame'][:-4])]  # Filter annotations to current frame
        anots = torch.tensor(anots)
        bbboxes = [anots[:, 2], anots[:, 3],  # x1,y1
                   anots[:, 2] + anots[:, 4],  # x2
                   anots[:, 3] + anots[:, 5]]  # y2
        bbboxes = torch.stack(bbboxes).T
        if self.transform is not None:
            c_og, y_og, x_og = shape_og
            c_new, y_new, x_new = shape_new
            bbboxes = bbboxes * torch.tensor([x_new / x_og, y_new / y_og, x_new / x_og, y_new / y_og])
        labels = torch.ones(len(bbboxes), dtype=torch.int64)
        area = torch.tensor([(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in bbboxes])
        iscrowd = torch.ones(len(bbboxes), dtype=torch.int64)

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
    #dataset = MOTSynthObjDet(path="/home/jatin/HDD2TB/datasets/MOT Challenge/Synth", mode="train-21", transform=transform)
    #dataset = MOT17ObjDet(path="/home/jatin/HDD2TB/datasets/MOT Challenge/", mode="train-18f", transform=transform)
    #dataset = MOTSynthFramesObjDet(path="/home/jatin/HDD2TB/datasets/MOT Challenge/Synth", mode="train-18g", transform=transform)

    #exit(0)
    #data = dataset[0]
    #data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4, collate_fn=utils.collate_fn)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)

    for i,(X,y) in enumerate(data_loader):
        print(X[0].shape)
        for j in range(len(X)):
            frame = X[j].squeeze().permute(1,2,0).numpy().copy()
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            boxes = y[j]['boxes']
            for box in boxes:
                x1,y1,x2,y2 = box
                x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
                cv2.rectangle(frame,(x1,y1),(x2,y2),(1,0,0),2)
            cv2.imshow('seq',frame)
            cv2.waitKey(50)
        #break

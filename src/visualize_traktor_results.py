import cv2
import pickle
import os

from src.datasets import UAVDTSeq, Images
from torch.utils.data import DataLoader

import torch
dataset_path = '/home/jatin/HDD2TB/datasets/UAVDT'
results_path = '/home/jatin/HDD2TB/tracking analysis/results/UAV15_n_synth15-det_N_UAV-reid'
seqs = os.listdir(os.path.join(dataset_path,'UAV-benchmark-M'))
seqs.sort()
for seq in seqs:
    frames = os.listdir(os.path.join(dataset_path,'UAV-benchmark-M',seq))
    frames.sort()
    for frame in frames:
        frame = cv2.imread(os.path.join(dataset_path,'UAV-benchmark-M',seq,frame))
        cv2.imshow(seq,frame)
        cv2.waitKey(20)

'''
dataset = UAVDTSeq("/home/jatin/HDD2TB/datasets/UAVDT",framelimit=100)
dataloader = DataLoader(dataset,batch_size=1)

for i,(X,y) in enumerate(dataloader):
    #with open(os.path.join("/home/jatin/HDD2TB/tracking analysis/results/UAV_n_synth15-det_N_UAV_n_synth15-reid",str(i)+'.json'),'rb') as anot_f:
    #with open(os.path.join("/home/jatin/HDD2TB/tracking analysis/results/UAV_n_synth15-det_N_UAV-reid",str(i)+'.json'),'rb') as anot_f:
    with open(os.path.join("/home/jatin/HDD2TB/tracking analysis/results/UAV-det_N_UAV-reid", str(i) + '.json'),'rb') as anot_f:
        tracks = pickle.load(anot_f)
    for frame in range(len(X)):
        img1 = X[frame].squeeze().permute([1,2,0]).numpy().copy()
        boxes = []
        for track in tracks.keys():
            if frame in tracks[track].keys():
                x1,y1,xl2,y2,alpha = tracks[track][frame]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(img1,(x1,y1),(x2,y2),(1,0,0),2)
        cv2.imshow('Model',img1)
        cv2.waitKey(100)
'''

import cv2
import pickle
import os
import argparse

from src.datasets import Images
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize
from tracking_wo_bnw.src.tracktor.frcnn_fpn import FRCNN_FPN
import torch


argparser = argparse.ArgumentParser()
argparser.add_argument('--dataset',help='path to dataset')
argparser.add_argument('--dataset_type',help='UAVDT / VisDrone')
argparser.add_argument('--results',help='path to results')
argparser.add_argument('--objdet',help='path to object detection model')
argparser.add_argument('--reid',help='path to reidentification model')
args = argparser.parse_args()

#dataset = '/home/jatin/HDD2TB/datasets/UAVDT/UAV-benchmark-M'
#results = '/home/jatin/HDD2TB/tracking analysis/results/UAV15_n_synth15-det_N_UAV-reid'

#seqs = os.listdir(os.path.join(args.dataset,'sequences'))
if args.results:
    seqs = [seq[:-5] for seq in os.listdir(args.results)]

IMAGE_RESOLUTION = (540, 1024)  # Constant Resolution to maintain for inputs
transform = None #Compose([Resize(IMAGE_RESOLUTION)])

if args.objdet:
    det_model = FRCNN_FPN(2).to('cuda')
    det_model.load_state_dict(torch.load(args.objdet))
    det_model.eval()

seqs.sort()
for seq in seqs:
    dataset = Images(args.dataset,seq,args.dataset_type,transform)
    dataloader = DataLoader(dataset,batch_size=1)
    if args.results:
        with open(os.path.join(args.results,seq+'.json'),'rb') as resf:
            res = pickle.load(resf)
    for batch,(X,y) in enumerate(dataloader):
        frame = X.squeeze().permute([1,2,0]).numpy().copy()
        annot = y.squeeze()
        #GT
        for bbox in annot:
            x1,y1,x2,y2 = bbox
            x1, y1, x2, y2 = int(x1),int(y1), int(x2), int(y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 1, 0), 2)
        # Predicted _resfile
        if args.results:
            for oid in res.keys():
                if batch in res[oid].keys():
                    x1, y1, x2, y2, _ = res[oid][batch]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    # if _ > 0.2:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (1, 0, 0), 2)
                    cv2.putText(frame,str(oid),(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),4,cv2.LINE_AA)
        # RT detection
        if args.objdet:
            preds = det_model(X.cuda().float())
            for bbox in preds[0]['boxes']:
                x1, y1, x2, y2, = bbox
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 1), 2)
        cv2.imshow(seq,frame)
        cv2.waitKey(50)

'''
--dataset="/home/jatin/HDD2TB/datasets/VisDrone/VisDrone2019-MOT-test-dev"
--dataset_type="VisDrone"
--results="/home/jatin/HDD2TB/tracking analysis/results/results-VisDrone/results/tracktor/VisDrone/R10aS20-det_n_R10aS20-reid"
--objdet="/home/jatin/HDD2TB/tracking analysis models/R10aS20_epoch_30.model"
'''
'''
--dataset="/home/jatin/HDD2TB/datasets/UAVDT"
--dataset_type="UAVDT"
--results="/home/jatin/HDD2TB/tracking analysis/results/UAV15_n_synth15-det_N_UAV-reid"
--objdet="/home/jatin/HDD2TB/tracking analysis models/R10aS20_epoch_30.model"
'''
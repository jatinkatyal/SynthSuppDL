import os.path

import torch
from torch.utils.data import DataLoader

from datasets import UAVDTSeq, plot, Images, collate_fn

import torchreid

import yaml

import argparse

import numpy as np
import cv2

import pickle
from tqdm import tqdm

import sys
sys.path.append('/home/jatin/work/tracking_analysis') #your path her  instead
from tracking_wo_bnw.src.tracktor.tracker import Tracker
from tracking_wo_bnw.src.tracktor.frcnn_fpn import FRCNN_FPN

#'''
def warp_pos(pos, warp_matrix):
    p1 = torch.Tensor([pos[0, 0], pos[0, 1], 1]).view(3, 1)
    p2 = torch.Tensor([pos[0, 2], pos[0, 3], 1]).view(3, 1)
    p1_n = torch.mm(warp_matrix, p1.type(warp_matrix.dtype)).view(1, 2)
    p2_n = torch.mm(warp_matrix, p2.type(warp_matrix.dtype)).view(1, 2)
    return torch.cat((p1_n, p2_n), 1).view(1, -1).cuda()
#'''

class MyTracker(Tracker):
    def align(self, blob):
        """Aligns the positions of active and inactive tracks depending on camera motion."""
        if self.im_index > 0:
            im1 = np.transpose(self.last_image.cpu().numpy(), (1, 2, 0))
            im2 = np.transpose(blob['img'][0].cpu().numpy(), (1, 2, 0))
            im1_gray = cv2.cvtColor(im1.astype(np.float32), cv2.COLOR_RGB2GRAY)
            im2_gray = cv2.cvtColor(im2.astype(np.float32), cv2.COLOR_RGB2GRAY)
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, self.number_of_iterations,  self.termination_eps)
            cc, warp_matrix = cv2.findTransformECC(im1_gray, im2_gray, warp_matrix, self.warp_mode, criteria)
            warp_matrix = torch.from_numpy(warp_matrix).type(torch.DoubleTensor)

            for t in self.tracks:
                #print(t.pos)
                t.pos = warp_pos(t.pos.type(torch.DoubleTensor), warp_matrix)
                # t.pos = clip_boxes(Variable(pos), blob['im_info'][0][:2]).data

            if self.do_reid:
                for t in self.inactive_tracks:
                    t.pos = warp_pos(t.pos, warp_matrix)

            if self.motion_model_cfg['enabled']:
                for t in self.tracks:
                    for i in range(len(t.last_pos)):
                        t.last_pos[i] = warp_pos(t.last_pos[i], warp_matrix)

argsparser = argparse.ArgumentParser(description='Just passing some parameters')
argsparser.add_argument('--reid_model', help='path to reid model weights')
argsparser.add_argument('--det_model', help='path to det model weights')
argsparser.add_argument('--config', help="path to config file")
argsparser.add_argument('--dataset',help='path to dataset')
argsparser.add_argument('--output',help='directory to store results')
argsparser.add_argument('--dataset_type',help='what dataset it is UAVDT, VisDrone etc.')
args = argsparser.parse_args()

det_model = FRCNN_FPN(2).to('cuda')
print(f'Loading detection model from {args.det_model}')
det_model.load_state_dict(torch.load(args.det_model))
det_model.eval()
det_model.double()

print(f'Loading re-identification model from {args.reid_model}')
reid_extractor = torchreid.utils.FeatureExtractor(model_name='resnet50_fc512', model_path=args.reid_model, device='cuda')

with open(args.config,'r') as f:
    config = yaml.safe_load(f)
    tracker_config = config['tracker']

#tracker = Tracker(det_model, reid_extractor, tracker_config)
tracker = MyTracker(det_model, reid_extractor, tracker_config)

if not os.path.exists(args.output):
    os.mkdir(args.output)
print(f'Results will be stored in {args.output}')

'''
dataset = UAVDTSeq(args.dataset,framelimit=100)
dataloader = DataLoader(dataset,batch_size=1)

for i,(seq,boxes) in enumerate(dataloader):
    print(f'Running for sequence {i} now.')
    tracker.reset()
    blobs = [{'img':seq[j],'dets':boxes[j]} for j in range(len(seq))]
    for blob in blobs:
        with torch.no_grad():
            tracker.step(blob)
    # print(tracker.get_results())
    with open(os.path.join(args.output,f'{i}.json'),'wb') as result_file:
        pickle.dump(tracker.get_results(), result_file)
    print(f'Results ready for the sequence {i} at {os.path.join(args.output,f"{i}.json")}')
'''

#All
#seqs = os.listdir(os.path.join(args.dataset,'UAV-benchmark-M'))
#Test only
if args.dataset_type=='UAVDT':
    seqs = ['M0203', 'M0205', 'M0208', 'M0209', 'M0403', 'M0601', 'M0602', 'M0606', 'M0701', 'M0801',
        'M0802', 'M1001', 'M1004', 'M1007', 'M1009', 'M1101', 'M1301', 'M1302', 'M1303', 'M1401']
elif args.dataset_type == 'VisDrone':
    seqs = ['uav0000009_03358_v', 'uav0000077_00720_v', 'uav0000119_02301_v', 'uav0000201_00000_v', 
        'uav0000249_00001_v', 'uav0000249_02688_v', 'uav0000297_02761_v', 'uav0000355_00001_v', 'uav0000370_00001_v']
seqs.sort()
for seq in seqs:
    tracker.reset()
    print(f'Running for sequence {seq} now.')
    dataset = Images(args.dataset,seq,args.dataset_type)
    dataloader = DataLoader(dataset, batch_size=1)#, collate_fn=collate_fn)
    for batch, (frame, annotation) in enumerate(tqdm(dataloader)):
        #for i in range(len(frame)):
        #frame = frame.unsqueeze(0)
        blob = {'img': frame, 'dets': annotation}
        with torch.no_grad():
            tracker.step(blob)
    #print(tracker.get_results())
    with open(os.path.join(args.output,f'{seq}.json'),'wb') as result_file:
        pickle.dump(tracker.get_results(), result_file)
    print(f'Results ready for the sequence {seq} at {os.path.join(args.output,f"{seq}.json")}')

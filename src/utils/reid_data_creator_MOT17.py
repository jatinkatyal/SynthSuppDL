import argparse
import os
import re

import pickle
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import cv2

IMG_DIR = 'train'
#ANT_DIR = 'MOTSynth_mot_annotations/mot_annotations'

if __name__=='__main__':
    argsparser = argparse.ArgumentParser(description='Parameters for reid data creation script')
    argsparser.add_argument('--dataset', help='path to dataset', required=True)
    argsparser.add_argument('--output', help='path to output directory', required=True)
    args = argsparser.parse_args()

    print(f'Input Dataset: {args.dataset}')
    print(f'Output Directory: {args.output}')

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    vids_path = os.path.join(args.dataset, IMG_DIR)
    #annot_dir = os.path.join(args.dataset, ANT_DIR)

    #seqs = os.listdir(vids_path)           # all
    '''
    #Train
    seqs = ['uav0000124_00944_v', 'uav0000126_00001_v', 'uav0000140_01590_v', 'uav0000143_02250_v', 'uav0000145_00000_v',
            'uav0000218_00001_v', 'uav0000222_03150_v', 'uav0000239_03720_v', 'uav0000239_12336_v', 'uav0000243_00001_v',
            'uav0000244_01440_v', 'uav0000248_00001_v', 'uav0000263_03289_v', 'uav0000264_02760_v', 'uav0000270_00001_v',
            'uav0000273_00001_v', 'uav0000278_00001_v', 'uav0000279_00001_v', 'uav0000281_00460_v', 'uav0000289_00001_v',
            'uav0000289_06922_v', 'uav0000295_02300_v', 'uav0000307_00000_v', 'uav0000315_00000_v', 'uav0000323_01173_v',  # 'uav0000300_00000_v', excluded for less cars to other ratio
            'uav0000326_01035_v', 'uav0000342_04692_v', 'uav0000352_05980_v', 'uav0000361_02323_v', 'uav0000366_00001_v'
            ]
    '''
    #validation
    seqs = os.listdir(vids_path)
    seqs.sort()
    for seq in seqs:
        print(seq)
        if not os.path.exists(os.path.join(args.output,seq)):
            os.mkdir(os.path.join(args.output,seq))
        #annot_path = os.path.join(annot_dir, f'{seq}.pkl')
        #with open(annot_path,'rb') as annot_fs:
        #    anots = pickle.load(annot_fs)
        #try:
        #    pass
        #except RuntimeError:
        #    # runtime error expected for DetectionInfo datatype
        #    pass
        annot_file = os.path.join(vids_path, seq,'gt/gt.txt')
        annot = pd.read_csv(annot_file, delimiter=',', header=None,
                            names=['frame_id','object_id','x','y','w','h','_1','_2','_3','rx','ry','rz'])
        #annot['frame_id'] = annot['frame_id'] - 1

        frames_nm = os.listdir(os.path.join(vids_path,seq,'img1'))
        frames_nm.sort()

        '''for frame_no in range(len(anots)):
            frame_pth = os.path.join(vids_path, seq, frames_nm[frame_no])
            frame = Image.open(frame_pth)
            frame = np.array(frame)

            objs = anots[frame_no]
            for obj in objs:
                if re.match('Car_[0-9]+',obj.name):
                    if not os.path.exists(os.path.join(args.output, seq, str(obj.name))):
                        os.mkdir(os.path.join(args.output, seq, str(obj.name)))
                    x1, y1 = int(obj.box2D.min.x_val), int(obj.box2D.min.y_val)
                    x2, y2 = int(obj.box2D.max.x_val), int(obj.box2D.max.y_val)
                    if x1<x2 and y1<y2:
                        patch = frame[y1:y2,x1:x2]
                        patch = Image.fromarray(patch)
                        patch.save(os.path.join(args.output, seq, obj.name, str(frame_no) + '.png'))'''
        for obj in annot['object_id'].unique():
            if not os.path.exists(os.path.join(args.output, seq, str(obj))):
                os.mkdir(os.path.join(args.output, seq, str(obj)))

        frame_ids = annot['frame_id'].unique()
        frame_ids.sort()
        for frame_id in tqdm(frame_ids):
            #print(frame_id)
            frame_pth = os.path.join(vids_path, seq,'img1', frames_nm[frame_id-1])
            frame = Image.open(frame_pth)
            frame = np.array(frame)
            this_frame_annot = annot[annot['frame_id'] == frame_id]
            for obj in this_frame_annot.values:
                #print(obj)
                patch = frame[int(obj[3]):int(obj[3]+obj[5]), int(obj[2]):int(obj[2] + obj[4])]
                w,h,c = patch.shape
                if w>0  and  h>0:
                    patch = Image.fromarray(patch)
                    patch.save(os.path.join(args.output, seq, str(int(obj[1])), str(int(obj[0]))) + '.jpg')
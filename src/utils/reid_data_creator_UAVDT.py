import argparse
import os

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from PIL import Image

IMG_DIR = 'UAV-benchmark-M'
ANT_DIR = 'UAV-benchmark-MOTD_v1.0/GT'

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
    annot_dir = os.path.join(args.dataset, ANT_DIR)

    seqs = os.listdir(vids_path)
    seqs.sort()
    for seq in seqs:
        if not os.path.exists(os.path.join(args.output,seq)):
            os.mkdir(os.path.join(args.output,seq))
        annot_file = os.path.join(annot_dir, f'{seq}_gt.txt')
        annot = pd.read_csv(annot_file, delimiter=',', header=None,
                names=['frame_id', 'object_id', 'x','y','w','h','score','in_view','occlusinon'])
        annot['frame_id'] = annot['frame_id']-1

        frames_nm = os.listdir(os.path.join(vids_path,seq))
        frames_nm.sort()

        for obj in annot['object_id'].unique():
            if not os.path.exists(os.path.join(args.output, seq, str(obj))):
                os.mkdir(os.path.join(args.output, seq, str(obj)))

        frame_ids = annot['frame_id'].unique()
        for frame_id in frame_ids:
            frame_pth = os.path.join(vids_path,seq,frames_nm[frame_id])
            frame = Image.open(frame_pth)
            frame = np.array(frame)
            this_frame_annot = annot[annot['frame_id'] == frame_id]
            for obj in this_frame_annot.values:
                #print(obj)
                patch = frame[obj[3]:obj[3]+obj[5], obj[2]:obj[2]+obj[4]]
                patch = Image.fromarray(patch)
                patch.save(os.path.join(args.output,seq,str(obj[1]),str(obj[0]))+'.jpg')
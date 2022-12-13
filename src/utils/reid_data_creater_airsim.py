import argparse
import os
import re

import pickle
import numpy as np
import pandas as pd
from PIL import Image

IMG_DIR = 'Sequences'
ANT_DIR = 'Annotations'

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

    # seqs = os.listdir(vids_path)           # all
    # seqs = [str(i) for i in range(15,21)]   # additional sequences for later experiments
    seqs = [str(i) for i in range(21, 27)]  # additional sequences for later experiments
    seqs.sort()
    for seq in seqs:
        if not os.path.exists(os.path.join(args.output,seq)):
            os.mkdir(os.path.join(args.output,seq))
        annot_path = os.path.join(annot_dir, f'{seq}.pkl')
        with open(annot_path,'rb') as annot_fs:
            anots = pickle.load(annot_fs)
        try:
            pass
        except RuntimeError:
            # runtime error expected for DetectionInfo datatype
            pass

        frames_nm = os.listdir(os.path.join(vids_path,seq))
        frames_nm.sort()

        for frame_no in range(len(anots)):
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
                        patch.save(os.path.join(args.output, seq, obj.name, str(frame_no) + '.png'))
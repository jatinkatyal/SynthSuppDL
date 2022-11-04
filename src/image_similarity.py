import os
import cv2

import itertools
from tqdm import tqdm
from image_similarity_measures.quality_metrics import ssim,fsim

import argparse

def returnSeq(seqPath,seqName):
    frames_nm = os.listdir(os.path.join(seqPath,seqName))
    frames_nm.sort()
    seq = []
    for frame_nm in frames_nm:
        frame = cv2.imread(os.path.join(seqPath,seqName,frame_nm))
        seq.append(frame)
    return seq

if __name__ == "__main__":
    argparser = argparse.ArgumentParser('for commandline arguements')
    argparser.add_argument('--real',help='path to directory real sequences')
    argparser.add_argument('--synthetic', help='path to directory with synthetic sequences')
    args = argparser.parse_args()

    synseqs_nm = os.listdir(args.synthetic)
    synseqs_nm.sort()
    seqs_nm = os.listdir(args.real)
    seqs_nm.sort()

    print('Calculating similarities for all sequence pairs')
    for seq_nm,synseq_nm in tqdm(list(itertools.product(seqs_nm,synseqs_nm))):
        seq = returnSeq(args.real,seq_nm)
        synseq = returnSeq(args.synthetic,synseq_nm)
        for frame in seq:
            for synframe in synseq:
                frame = cv2.resize(frame,(540,1024))
                synframe = cv2.resize(synframe, (540, 1024))
                print(fsim(frame,synframe))
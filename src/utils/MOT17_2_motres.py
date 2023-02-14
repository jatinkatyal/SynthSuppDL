import pickle
import csv
import os
import argparse
import shutil

argparser = argparse.ArgumentParser('Just parsing some arguments,')
argparser.add_argument('--results',help='directory for results')
argparser.add_argument('--output',help='directory to store converted results')
args = argparser.parse_args()

seqs = os.listdir(args.results)
for seq in seqs:
    src_path = os.path.join(args.results,seq,'seqinfo.ini')
    dest_path = os.path.join(args.output,seq)
    if not os.path.exists(dest_path):
        os.mkdir(dest_path)
    shutil.copy(src_path, dest_path)

    #src_path = os.path.join(args.results, seq, 'det', 'det.txt')
    src_path = os.path.join(args.results, seq, 'gt', 'gt.txt')
    dest_path = os.path.join(dest_path,'gt')
    if not os.path.exists(dest_path):
        os.mkdir(dest_path)
    dest_path = os.path.join(dest_path,'gt.txt')
    shutil.copy(src_path,dest_path)

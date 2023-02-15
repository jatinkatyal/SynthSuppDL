import os
import argparse
import subprocess
import re
import pickle

argparser = argparse.ArgumentParser()
argparser.add_argument("--source")
argparser.add_argument("--target")
argparser.add_argument("--type")
argparser.add_argument("--dims")
argparser.add_argument("--output")
args = argparser.parse_args()

src_seqs = os.listdir(args.source)
tgt_seqs = os.listdir(args.target)

dict = {}

for src_seq in src_seqs:
    subdict = {}
    for tgt_seq in tgt_seqs:
        cmd = f'python -m pytorch_fid --device=cuda:0 --batch-size=4 --dims={args.dims}'
        cmd = cmd.split(' ')
        if args.type == 'MOT':
            src_seq_path = f'{os.path.join(args.source,src_seq,"img1")}'
            tgt_seq_path = f'{os.path.join(args.target,tgt_seq,"rgb")}'
        if args.type in ['UAVDT','VisDrone']:
            src_seq_path = f'{os.path.join(args.source, src_seq)}'
            tgt_seq_path = f'{os.path.join(args.target, tgt_seq)}'
        cmd.append(src_seq_path)
        cmd.append(tgt_seq_path)
        print(cmd)

        result = subprocess.run(cmd, stdout=subprocess.PIPE)
        fid = re.findall('[0-9.]+',str(result.stdout))
        subdict[tgt_seq] = fid
    dict[src_seq] = subdict

print(dict)
with open(os.path.join(args.output,f"fids-{args.type}_{args.dims}.pkl"),"wb") as fp:
    pickle.dump(dict,fp)

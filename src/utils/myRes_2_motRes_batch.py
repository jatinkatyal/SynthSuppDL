import pickle
import csv
import os
import argparse

argparser = argparse.ArgumentParser('Just parsing some arguments,')
argparser.add_argument('--results',help='directory for results to be converted to desired format')
argparser.add_argument('--output',help='directory to store converted results')
args = argparser.parse_args()

for exp in os.listdir(args.results):
    print(f'converting experiment {exp}')
    res_files = os.listdir(os.path.join(args.results,exp))
    res_files.sort()

    for res_file in res_files:
        file_path = os.path.join(args.results,exp,res_file)
        with open(file_path,'rb') as f:
            result = pickle.load(f)
        output = []
        for track in result.keys():
            for frame in result[track].keys():
                x1,y1,x2,y2,score = result[track][frame]
                w,h = x2-x1, y2-y1
                output.append([frame+1,track+1,x1,y1,w,h,score,-1,-1,-1])

        seq = res_file.split('.')[0]

        if not os.path.exists(os.path.join(args.output,exp)):
            os.mkdir(os.path.join(args.output,exp))
            os.mkdir(os.path.join(args.output,exp,'data'))
        file_path2 = os.path.join(args.output,exp,'data',seq)+'.txt'
        with open(file_path2, "w", newline="") as f2:
            writer = csv.writer(f2)
            writer.writerows(output)
        print(file_path2+' is ready')
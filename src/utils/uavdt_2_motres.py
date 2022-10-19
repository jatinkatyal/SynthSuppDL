import pickle
import csv
import os
import argparse

argparser = argparse.ArgumentParser('Just parsing some arguments,')
argparser.add_argument('--results',help='directory for results')
args = argparser.parse_args()


def keep_anot_files(files, suffix):
    files_to_keep = []
    for file in files:
        if file.endswith(suffix):
            files_to_keep.append(file)
    return files_to_keep

res_files = os.listdir(args.results)
res_files = keep_anot_files(res_files,'_gt.txt')
res_files.sort()

for res_file in res_files:
    output = []
    file_path = os.path.join(args.results,res_file)
    with open(file_path,newline='') as f:
        reader = csv.reader(f,delimiter=',')
        for row in reader:
            frame,id,x,y,w,h,score,inview,occlusion = row
            output.append([frame,id,x,y,w,h,score,-1,-1,-1])
    with open(file_path+'.csv', "w", newline="") as f2:
        writer = csv.writer(f2)
        writer.writerows(output)
    print(file_path+'.csv is ready')
import pickle
import csv
import os
import argparse

argparser = argparse.ArgumentParser('Just parsing some arguments,')
argparser.add_argument('--results',help='directory for results')
argparser.add_argument('--output',help='directory to store converted results')
args = argparser.parse_args()


def keep_anot_files(files, suffix):
    files_to_keep = []
    for file in files:
        if file.endswith(suffix):
            files_to_keep.append(file)
    return files_to_keep

res_files = os.listdir(args.results)
#res_files = keep_anot_files(res_files,'_gt.txt')
res_files.sort()

for res_file in res_files:
    output = []
    file_path = os.path.join(args.results,res_file)
    with open(file_path,newline='') as f:
        reader = csv.reader(f,delimiter=',')
        for row in reader:
            #frame,id,x,y,w,h,score,inview,occlusion = row
            frame_id, object_id, x, y, w, h, cls, category, in_view, occlusinon = row
            #if category in ['4','5','6','9']:
            output.append([frame_id,object_id,x,y,w,h,cls,-1,-1,-1])
            #output.append([frame_id, object_id, x, y, w, h, score, -1, -1, -1])

    if not os.path.exists(os.path.join(args.output,res_file[:-4])):
        os.mkdir(os.path.join(args.output,res_file[:-4]))
        os.mkdir(os.path.join(args.output, res_file[:-4], 'gt'))

    with open(os.path.join(args.output, res_file[:-4], 'gt','gt.txt'), "w", newline="") as f2:
        writer = csv.writer(f2)
        writer.writerows(output)
    print(file_path+'.txt is ready')
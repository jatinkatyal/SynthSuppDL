import json

import numpy as np
import os
import argparse
import configparser
from tqdm import tqdm


argparser = argparse.ArgumentParser('Just parsing some arguments,')
argparser.add_argument('--data',help='directory for results')
argparser.add_argument('--output',help='directory to store converted results')
argparser.add_argument('--set',help='training set eg 3a,5b,10a etc.',default=None)
args = argparser.parse_args()


seqs = ["MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-05-FRCNN", "MOT17-09-FRCNN", "MOT17-10-FRCNN", "MOT17-11-FRCNN", "MOT17-13-FRCNN"]

print('Seqs: ',seqs)
images,annotations,videos = [],[],[]
img_idx = 1
det_idx = 1
for seq in tqdm(seqs):
    #print(f'sequence: {seq}')
    video_data = {
        'id':seqs.index(seq)+1,
        'file_name':seq
    }
    videos.append(video_data)

    config = configparser.ConfigParser()
    config.read(os.path.join(args.data,seq,'seqinfo.ini'))
    height, width = config.get('Sequence','imHeight'), config.get('Sequence','imwidth')

    imgs = os.listdir(os.path.join(args.data,seq,'img1'))
    imgs.sort()
    #print('reading images:')
    for img in imgs:
        img_data = {
            'file_name':os.path.join(seq,'img1',img),
            'id':img_idx,
            'frame_id':imgs.index(img)+1,
            'prev_image_id': -1 if imgs.index(img)==0 else images[-1]['id'],
            'next_image_id': -1 if imgs.index(img)==len(imgs)-1 else img_idx+1,
            'video_id':seqs.index(seq)+1,
            'height':int(height),
            'width':int(width)
        }
        images.append(img_data)
        img_idx += 1

    #print('reading anotations:')
    # frame, trackid, x,y,w,h ....
    anot_file = np.loadtxt(os.path.join(args.data,seq,'gt/gt.txt'),delimiter=',')
    for row in anot_file:
        frame,trackid,x,y,w,h = row[:6]
        anot_data = {
            'id':det_idx,
            'category_id':1,
            'image_id':img_idx-1-len(imgs)+int(frame),
            'track_id':int(trackid)+1,
            'bbox':[x,y,w,h],
            'conf':1,
            'iscrowd':0,
            'area':w*h
        }
        annotations.append(anot_data)
        det_idx += 1

    #print(images)
    #print(annotations[-1])
    #print(videos)

final = {
    'images':images,
    'annotations':annotations,
    'videos':videos,
    "categories": [{"id": 1, "name": "pedestrian"}]
}
with open(args.output,'w') as fp:
    json.dump(final,fp)

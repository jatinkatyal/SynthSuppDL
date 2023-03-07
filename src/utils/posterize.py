from torchvision.io import read_image, write_jpeg, ImageReadMode
from torchvision.transforms.functional import posterize
#import cv2
import os
from tqdm import tqdm

# FID768
#avg 5a=2.315, 5b=2.07, 5c=2.09, 5d=1.907, 5e=2.08, 5f=2.5

#5d
#dataset_path = '/home/jatin/HDD2TB/datasets/VisDrone/VisDrone2019-MOT-train/sequences'
#out_path = '/home/jatin/HDD2TB/datasets/VisDrone/Visdrone-posterize/'
#seqs = ['uav0000273_00001_v', 'uav0000278_00001_v', 'uav0000279_00001_v', 'uav0000281_00460_v', 'uav0000289_00001_v']

#dataset_path = '/home/jatin/HDD2TB/datasets/Synthetic/AirSim-UAV-Synth/data/Sequences'
#out_path = '/home/jatin/HDD2TB/datasets/Synthetic/AirSim-UAV-Synth-posterize/'
#seqs = os.listdir(dataset_path)

dataset_path = '/home/jatin/HDD2TB/datasets/MOT-Challenge/Synth/frames'
out_path= '/home/jatin/HDD2TB/datasets/MOT-Challenge/Synth/frames_posterized'
seqs = os.listdir(dataset_path)

for seq in seqs:
    print(f'In {seq}')

    if not os.path.exists(os.path.join(out_path, 'post_2', seq)):
        os.mkdir(os.path.join(out_path, 'post_2', seq))
        os.mkdir(os.path.join(out_path, 'post_2', seq,'rgb'))
    '''
    if not os.path.exists(os.path.join(out_path, 'post_3', seq)):
        os.mkdir(os.path.join(out_path, 'post_3', seq))
    if not os.path.exists(os.path.join(out_path, 'post_4', seq)):
        os.mkdir(os.path.join(out_path, 'post_4', seq))
    '''

    img_nms = os.listdir(os.path.join(dataset_path,seq,'rgb'))
    for img_nm in tqdm(img_nms):
        img = read_image(os.path.join(dataset_path,seq,'rgb',img_nm),mode=ImageReadMode.RGB)

        img2 = posterize(img, bits=2)
        #img3 = posterize(img, bits=3)
        #img4 = posterize(img, bits=4)

        write_jpeg(img2,os.path.join(out_path,'post_2',seq,'rgb',img_nm[:-4]+'.jpg'))
        #write_jpeg(img3, os.path.join(out_path, 'post_3',seq, img_nm[:-4]+'.jpg'))
        #write_jpeg(img4, os.path.join(out_path, 'post_4',seq, img_nm[:-4]+'.jpg'))



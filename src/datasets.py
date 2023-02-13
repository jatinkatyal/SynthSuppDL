import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import os

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patches

class Images(Dataset):
    def __init__(self,path,seq,dataset_type):
        if dataset_type == 'UAVDT':
            seq_pattern = 'UAV-benchmark-M/'+seq
            annot_pattern = 'UAV-benchmark-MOTD_v1.0/GT/'+seq+'_gt.txt'
        elif dataset_type == 'VisDrone':
            seq_pattern = 'sequences/'+seq
            annot_pattern = 'annotations/'+seq+'.txt'
        elif dataset_type == 'MOT-test':
            seq_pattern = 'test/'+seq+'/img1'
            annot_pattern = 'test/'+seq+'/det/det.txt'
        elif dataset_type == 'MOT-train':
            seq_pattern = 'train/'+seq+'/img1'
            annot_pattern = 'train/'+seq+'/gt/gt.txt'
        self.path = path
        self.seq_path = os.path.join(self.path,seq_pattern)
        self.img_list = os.listdir(self.seq_path)
        self.img_list.sort()

        annotation_path = os.path.join(self.path,annot_pattern)
        self.annotations = np.loadtxt(annotation_path, delimiter=',')

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, item):
        img_path = os.path.join(self.seq_path,self.img_list[item])
        img = read_image(img_path)/255
        img = img.type(torch.DoubleTensor)

        anots_ith_frame = self.annotations[self.annotations[:, 0] == item+1]
        bboxes = [
            anots_ith_frame[:, 2],
            anots_ith_frame[:, 3],
            anots_ith_frame[:, 2] + anots_ith_frame[:, 4],
            anots_ith_frame[:, 3] + anots_ith_frame[:, 5]
        ]
        bboxes = np.array(bboxes)
        bboxes = torch.tensor(bboxes).T
        return img, bboxes

class UAVDTSeq(Dataset):
    """
    Class for UAVDT dataset object for each sequence as 1 batch
    """
    def __init__(self,path,transform=None,framelimit=500):
        self.imgs_path = os.path.join(path,'UAV-benchmark-M')
        self.annot_path = os.path.join(path,'UAV-benchmark-MOTD_v1.0','GT')
        self.transform = transform
        self.framelimit = framelimit

    def __len__(self):
        return len(os.listdir(self.imgs_path))

    def __getitem__(self, idx):
        """
        Returns frames and annotations for video sequence at idx.
        seq: tensor with shape (N,C,H,W) holding each frame of a
        anot: dictionary with annotations
        """
        #Annotations
        # anotation file format: Frame No., Object Id., bbox x, bbox y, bbox width, bbox, width
        anot_files = os.listdir(self.annot_path)
        anot_files.sort()
        anot_files = self.keep_anot_files(anot_files,suffix='gt.txt')
        anots = np.loadtxt(os.path.join(self.annot_path,anot_files[idx]),delimiter=',')
        anots = torch.tensor(anots)

        #limiting to framelimit
        anots = anots[anots[:,0]<=self.framelimit]

        bbboxes = []
        for i in range(1,int(anots[:,0].max()+1)):
            anots_ith_frame = anots[anots[:,0]==i]
            bboxes_ith_frame = [anots_ith_frame[:,2],
                                anots_ith_frame[:,3],
                                anots_ith_frame[:,2] + anots_ith_frame[:,4],
                                anots_ith_frame[:,3] + anots_ith_frame[:,5]]
            bboxes_ith_frame = torch.stack(bboxes_ith_frame).T
            bbboxes.append(bboxes_ith_frame)

        # Read frames from image sequence
        seq_dirs = os.listdir(self.imgs_path)
        seq_dirs.sort()
        seq_dir_path = os.path.join(self.imgs_path,seq_dirs[idx])
        seq_dir = os.listdir(seq_dir_path)
        seq_dir.sort()

        # limiting to framelimit to save memory
        seq_dir = seq_dir[:self.framelimit]

        seq = []
        for img_nm in seq_dir:
            frame = read_image(os.path.join(seq_dir_path,img_nm))/255
            frame = frame.type(torch.DoubleTensor)
            if self.transform:
                frame=self.transform(frame)
            frame.req_grads=False
            seq.append(frame)
        #Note required by the new model
        #seq = torch.stack(seq)
        return seq,bbboxes

    def keep_anot_files(self,files,suffix):
        files_to_keep = []
        for file in files:
            if file.endswith(suffix):
                files_to_keep.append(file)
        return files_to_keep

def plot(img,boxes,color='red'):
    matplotlib.use('TkAgg')
    plt.figure()
    plt.imshow(torch.permute(img, (1, 2, 0)))
    ref = plt.gca()
    for x1,y1,x2,y2 in boxes:
        bbox = patches.Rectangle((x1,y1),x2-x1,y2-y1,fill=False,color=color)
        ref.add_patch(bbox)
    plt.show()

def collate_fn(batch):
    image_list, boxes_list = [], []
    for image, boxes in batch:
        image_list.append(image)
        boxes_list.append(boxes)
    return image_list, boxes_list

if __name__ == '__main__':
    import argparse
    argsparser = argparse.ArgumentParser(description='Just passing some parameters')
    argsparser.add_argument('--dataset', help='path to dataset')
    args = argsparser.parse_args()

    from torchvision.transforms import Compose, Normalize
    transform = Compose([
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.225, 0.225, 0.225])
    ])
    dataset = UAVDTSeq(args.dataset, transform=transform)

    import random
    j = random.randint(0, len(dataset))
    img,target = dataset[j]
    print(img[0].shape)
    plot(img,target['boxes'])


    from torch.utils.data import DataLoader
    dataset = UAVDTObjDet(args.dataset)
    loader = DataLoader(dataset, batch_size=32,collate_fn=collate_fn)

    for i,(frames,targets) in enumerate(loader):
        print(f'batch:{i}\tframes:{frames.shape}')
        print(f'target: {targets}')
        break


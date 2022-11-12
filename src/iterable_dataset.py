import torch
import torchvision.io
from torch.utils.data import DataLoader, IterableDataset, Dataset
import os
import numpy as np

import cv2


#class UAVDTSeq(IterableDataset):

def collate_fn(batch):
    image_list, boxes_list = [], []
    for image, boxes in batch:
        image_list.append(image)
        boxes_list.append(boxes)
    return image_list, boxes_list

class Images(Dataset):
    def __init__(self,path,seq):
        self.path = path
        self.seq_path = os.path.join(self.path,'UAV-benchmark-M',seq)
        self.img_list = os.listdir(self.seq_path)
        self.img_list.sort()

        annotation_path = os.path.join(self.path,'UAV-benchmark-MOTD_v1.0','GT',seq+'_gt.txt')
        self.annotations = np.loadtxt(annotation_path, delimiter=',')

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, item):
        img_path = os.path.join(self.seq_path,self.img_list[item])
        img = torchvision.io.read_image(img_path)

        anots_ith_frame = self.annotations[self.annotations[:, 0] == item+1]
        bboxes = [
            anots_ith_frame[:, 2],
            anots_ith_frame[:, 3],
            anots_ith_frame[:, 2] + anots_ith_frame[:, 4],
            anots_ith_frame[:, 3] + anots_ith_frame[:, 5]
        ]
        bboxes = torch.tensor(bboxes).T
        return img, bboxes


if __name__=='__main__':
    TEST = 'Images'

    if TEST == 'Images':
        dataset = Images('/home/jatin/HDD2TB/datasets/UAVDT','M0101')
        dataloader = DataLoader(dataset,batch_size=64,collate_fn=collate_fn)
        for batch,(X,y) in enumerate(dataloader):
            for i in range(len(X)):
                img = X[i].squeeze().permute(1, 2, 0).numpy().copy()
                boxes = y[i]
                for box in boxes:
                    box = box.to(torch.int).numpy()
                    x1,y1,x2,y2 = box
                    cv2.rectangle(img,(x1,y1),(x2,y2),(1,0,0),2)
                    #cv2.rectangle(img, box, (1, 0, 0), 2)
                cv2.imshow('seq',img)
                cv2.waitKey(100)

    if TEST == 'Sequence':
        dataset = Sequence('/home/jatin/HDD2TB/datasets/UAVDT')
        dataloader = DataLoader(dataset,batch_size=1)
        for seq, X_loader in enumerate(dataloader):
            for i, X in enumerate(dataloader):
                for img in X:
                    cv2.imshow(str(seq), img.squeeze().permute(1, 2, 0).numpy())
                    cv2.waitKey(50)
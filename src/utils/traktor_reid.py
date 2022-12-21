import os
import random

import torchreid
#from torchreid.data import ImageDataset

import argparse
argsparser = argparse.ArgumentParser(description='Just passing some parameters')
argsparser.add_argument('--data', help='path to real data', default=None)
argsparser.add_argument('--syndata', help='path to synthetic data', default=None)
argsparser.add_argument('--model', help='path to model weights', default=None)
argsparser.add_argument('--mode', help='how to use real and synthetic datasets', default=None)
args = argsparser.parse_args()


class UAVDTReIdTrain(torchreid.data.ImageDataset):
    #dataset_dir = '/home/jatin/HDD2TB/datasets/UAVDT_reid'
    dataset_dir = args.data
    def __init__(self, root='', **kwargs):
        train = []
        query = []
        gallery = []

        #Train
        seqs = ['M0101', 'M0201', 'M0202', 'M0204', 'M0206', 'M0207', 'M0210', 'M0301', 'M0401', 'M0402', 'M0501',
                'M0603', 'M0604', 'M0605', 'M0702', 'M0703', 'M0704', 'M0901', 'M0902', 'M1002', 'M1003', 'M1005',
                'M1006', 'M1008', 'M1102', 'M1201', 'M1202', 'M1304', 'M1305', 'M1306']
        seqs.sort()

        objind = 0
        for seq in seqs:
            objs = os.listdir(os.path.join(self.dataset_dir,seq))
            objs.sort()
            for obj in objs:
                imgs = os.listdir(os.path.join(self.dataset_dir,seq,obj))
                imgs.sort()

                if len(imgs)>2:
                    q = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[q])
                    entry = [img_path, objind, seqs.index(seq)+100]
                    query.append(entry)
                    imgs.pop(q)

                    g = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[g])
                    entry = [img_path, objind, seqs.index(seq)]
                    gallery.append(entry)
                    imgs.pop(g)

                for img in imgs:
                    img_path = os.path.join(self.dataset_dir,seq,obj,img)
                    entry = [img_path, objind, seqs.index(seq)]
                    train.append(entry)
                objind += 1
        super(UAVDTReIdTrain, self).__init__(train, query, gallery, **kwargs)

class UAVDTReIdTrain5a(torchreid.data.ImageDataset):
    #dataset_dir = '/home/jatin/HDD2TB/datasets/UAVDT_reid'
    dataset_dir = args.data
    def __init__(self, root='', mode='train', **kwargs):
        train = []
        query = []
        gallery = []

        #Train
        print('Skipping some real sequences to balance the total number of samples')
        seqs = ['M0101', 'M0201', 'M0202', 'M0204', 'M0206',
                #'M0207', 'M0210', 'M0301', 'M0401', 'M0402',
                #'M0501', 'M0603', 'M0604', 'M0605', 'M0702',
                #'M0703', 'M0704', 'M0901', 'M0902', 'M1002',
                #'M1003', 'M1005', 'M1006', 'M1008', 'M1102',
                #'M1201', 'M1202', 'M1304', 'M1305', 'M1306'
                ]
        seqs.sort()

        objind = 0
        for seq in seqs:
            objs = os.listdir(os.path.join(self.dataset_dir,seq))
            objs.sort()
            for obj in objs:
                imgs = os.listdir(os.path.join(self.dataset_dir,seq,obj))
                imgs.sort()

                if len(imgs)>2:
                    q = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[q])
                    entry = [img_path, objind, seqs.index(seq)+100]
                    query.append(entry)
                    imgs.pop(q)

                    g = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[g])
                    entry = [img_path, objind, seqs.index(seq)]
                    gallery.append(entry)
                    imgs.pop(g)

                for img in imgs:
                    img_path = os.path.join(self.dataset_dir,seq,obj,img)
                    entry = [img_path, objind, seqs.index(seq)]
                    train.append(entry)
                objind += 1
        super(UAVDTReIdTrain5a, self).__init__(train, query, gallery, **kwargs)

class UAVDTReIdTrain5b(torchreid.data.ImageDataset):
    #dataset_dir = '/home/jatin/HDD2TB/datasets/UAVDT_reid'
    dataset_dir = args.data
    def __init__(self, root='', mode='train', **kwargs):
        train = []
        query = []
        gallery = []

        #Train
        print('Skipping some real sequences to balance the total number of samples')
        seqs = [#'M0101', 'M0201', 'M0202', 'M0204', 'M0206',
                'M0207', 'M0210', 'M0301', 'M0401', 'M0402',
                #'M0501', 'M0603', 'M0604', 'M0605', 'M0702',
                #'M0703', 'M0704', 'M0901', 'M0902', 'M1002',
                #'M1003', 'M1005', 'M1006', 'M1008', 'M1102',
                #'M1201', 'M1202', 'M1304', 'M1305', 'M1306'
                ]
        seqs.sort()

        objind = 0
        for seq in seqs:
            objs = os.listdir(os.path.join(self.dataset_dir,seq))
            objs.sort()
            for obj in objs:
                imgs = os.listdir(os.path.join(self.dataset_dir,seq,obj))
                imgs.sort()

                if len(imgs)>2:
                    q = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[q])
                    entry = [img_path, objind, seqs.index(seq)+100]
                    query.append(entry)
                    imgs.pop(q)

                    g = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[g])
                    entry = [img_path, objind, seqs.index(seq)]
                    gallery.append(entry)
                    imgs.pop(g)

                for img in imgs:
                    img_path = os.path.join(self.dataset_dir,seq,obj,img)
                    entry = [img_path, objind, seqs.index(seq)] 
                    train.append(entry)
                objind += 1
        super(UAVDTReIdTrain5b, self).__init__(train, query, gallery, **kwargs)

class UAVDTReIdTrain5c(torchreid.data.ImageDataset):
    #dataset_dir = '/home/jatin/HDD2TB/datasets/UAVDT_reid'
    dataset_dir = args.data
    def __init__(self, root='', mode='train', **kwargs):
        train = []
        query = []
        gallery = []

        #Train
        print('Skipping some real sequences to balance the total number of samples')
        seqs = [#'M0101', 'M0201', 'M0202', 'M0204', 'M0206',
                #'M0207', 'M0210', 'M0301', 'M0401', 'M0402',
                'M0501', 'M0603', 'M0604', 'M0605', 'M0702',
                #'M0703', 'M0704', 'M0901', 'M0902', 'M1002',
                #'M1003', 'M1005', 'M1006', 'M1008', 'M1102',
                #'M1201', 'M1202', 'M1304', 'M1305', 'M1306'
                ]
        seqs.sort()

        objind = 0
        for seq in seqs:
            objs = os.listdir(os.path.join(self.dataset_dir,seq))
            objs.sort()
            for obj in objs:
                imgs = os.listdir(os.path.join(self.dataset_dir,seq,obj))
                imgs.sort()

                if len(imgs)>2:
                    q = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[q])
                    entry = [img_path, objind, seqs.index(seq)+100]
                    query.append(entry)
                    imgs.pop(q)

                    g = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[g])
                    entry = [img_path, objind, seqs.index(seq)]
                    gallery.append(entry)
                    imgs.pop(g)

                for img in imgs:
                    img_path = os.path.join(self.dataset_dir,seq,obj,img)
                    entry = [img_path, objind, seqs.index(seq)] 
                    train.append(entry)
                objind += 1
        super(UAVDTReIdTrain5c, self).__init__(train, query, gallery, **kwargs)

class UAVDTReIdTrain5d(torchreid.data.ImageDataset):
    #dataset_dir = '/home/jatin/HDD2TB/datasets/UAVDT_reid'
    dataset_dir = args.data
    def __init__(self, root='', mode='train', **kwargs):
        train = []
        query = []
        gallery = []

        #Train
        print('Skipping some real sequences to balance the total number of samples')
        seqs = [#'M0101', 'M0201', 'M0202', 'M0204', 'M0206',
                #'M0207', 'M0210', 'M0301', 'M0401', 'M0402',
                #'M0501', 'M0603', 'M0604', 'M0605', 'M0702',
                'M0703', 'M0704', 'M0901', 'M0902', 'M1002',
                #'M1003', 'M1005', 'M1006', 'M1008', 'M1102',
                #'M1201', 'M1202', 'M1304', 'M1305', 'M1306'
                ]
        seqs.sort()

        objind = 0
        for seq in seqs:
            objs = os.listdir(os.path.join(self.dataset_dir,seq))
            objs.sort()
            for obj in objs:
                imgs = os.listdir(os.path.join(self.dataset_dir,seq,obj))
                imgs.sort()

                if len(imgs)>2:
                    q = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[q])
                    entry = [img_path, objind, seqs.index(seq)+100]
                    query.append(entry)
                    imgs.pop(q)

                    g = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[g])
                    entry = [img_path, objind, seqs.index(seq)]
                    gallery.append(entry)
                    imgs.pop(g)

                for img in imgs:
                    img_path = os.path.join(self.dataset_dir,seq,obj,img)
                    entry = [img_path, objind, seqs.index(seq)] 
                    train.append(entry)
                objind += 1
        super(UAVDTReIdTrain5d, self).__init__(train, query, gallery, **kwargs)

class UAVDTReIdTrain5e(torchreid.data.ImageDataset):
    #dataset_dir = '/home/jatin/HDD2TB/datasets/UAVDT_reid'
    dataset_dir = args.data
    def __init__(self, root='', mode='train', **kwargs):
        train = []
        query = []
        gallery = []

        #Train
        print('Skipping some real sequences to balance the total number of samples')
        seqs = [#'M0101', 'M0201', 'M0202', 'M0204', 'M0206',
                #'M0207', 'M0210', 'M0301', 'M0401', 'M0402',
                #'M0501', 'M0603', 'M0604', 'M0605', 'M0702',
                #'M0703', 'M0704', 'M0901', 'M0902', 'M1002',
                'M1003', 'M1005', 'M1006', 'M1008', 'M1102',
                #'M1201', 'M1202', 'M1304', 'M1305', 'M1306'
                ]
        seqs.sort()

        objind = 0
        for seq in seqs:
            objs = os.listdir(os.path.join(self.dataset_dir,seq))
            objs.sort()
            for obj in objs:
                imgs = os.listdir(os.path.join(self.dataset_dir,seq,obj))
                imgs.sort()

                if len(imgs)>2:
                    q = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[q])
                    entry = [img_path, objind, seqs.index(seq)+100]
                    query.append(entry)
                    imgs.pop(q)

                    g = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[g])
                    entry = [img_path, objind, seqs.index(seq)]
                    gallery.append(entry)
                    imgs.pop(g)

                for img in imgs:
                    img_path = os.path.join(self.dataset_dir,seq,obj,img)
                    entry = [img_path, objind, seqs.index(seq)] 
                    train.append(entry)
                objind += 1
        super(UAVDTReIdTrain5e, self).__init__(train, query, gallery, **kwargs)

class UAVDTReIdTrain5f(torchreid.data.ImageDataset):
    #dataset_dir = '/home/jatin/HDD2TB/datasets/UAVDT_reid'
    dataset_dir = args.data
    def __init__(self, root='', mode='train', **kwargs):
        train = []
        query = []
        gallery = []

        #Train
        print('Skipping some real sequences to balance the total number of samples')
        seqs = [#'M0101', 'M0201', 'M0202', 'M0204', 'M0206',
                #'M0207', 'M0210', 'M0301', 'M0401', 'M0402',
                #'M0501', 'M0603', 'M0604', 'M0605', 'M0702',
                #'M0703', 'M0704', 'M0901', 'M0902', 'M1002',
                #'M1003', 'M1005', 'M1006', 'M1008', 'M1102',
                'M1201', 'M1202', 'M1304', 'M1305', 'M1306'
                ]
        seqs.sort()

        objind = 0
        for seq in seqs:
            objs = os.listdir(os.path.join(self.dataset_dir,seq))
            objs.sort()
            for obj in objs:
                imgs = os.listdir(os.path.join(self.dataset_dir,seq,obj))
                imgs.sort()

                if len(imgs)>2:
                    q = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[q])
                    entry = [img_path, objind, seqs.index(seq)+100]
                    query.append(entry)
                    imgs.pop(q)

                    g = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[g])
                    entry = [img_path, objind, seqs.index(seq)]
                    gallery.append(entry)
                    imgs.pop(g)

                for img in imgs:
                    img_path = os.path.join(self.dataset_dir,seq,obj,img)
                    entry = [img_path, objind, seqs.index(seq)] 
                    train.append(entry)
                objind += 1
        super(UAVDTReIdTrain5f, self).__init__(train, query, gallery, **kwargs)

class UAVDTReIdTrain10a(torchreid.data.ImageDataset):
    #dataset_dir = '/home/jatin/HDD2TB/datasets/UAVDT_reid'
    dataset_dir = args.data
    def __init__(self, root='', mode='train', **kwargs):
        train = []
        query = []
        gallery = []

        #Train
        print('Skipping some real sequences to balance the total number of samples')
        seqs = ['M0101', 'M0201', 'M0202', 'M0204', 'M0206',
                'M0207', 'M0210', 'M0301', 'M0401', 'M0402',
                #'M0501', 'M0603', 'M0604', 'M0605', 'M0702',
                #'M0703', 'M0704', 'M0901', 'M0902', 'M1002',
                #'M1003', 'M1005', 'M1006', 'M1008', 'M1102',
                #'M1201', 'M1202', 'M1304', 'M1305', 'M1306'
                ]
        seqs.sort()

        objind = 0
        for seq in seqs:
            objs = os.listdir(os.path.join(self.dataset_dir,seq))
            objs.sort()
            for obj in objs:
                imgs = os.listdir(os.path.join(self.dataset_dir,seq,obj))
                imgs.sort()

                if len(imgs)>2:
                    q = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[q])
                    entry = [img_path, objind, seqs.index(seq)+100]
                    query.append(entry)
                    imgs.pop(q)

                    g = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[g])
                    entry = [img_path, objind, seqs.index(seq)]
                    gallery.append(entry)
                    imgs.pop(g)

                for img in imgs:
                    img_path = os.path.join(self.dataset_dir,seq,obj,img)
                    entry = [img_path, objind, seqs.index(seq)]
                    train.append(entry)
                objind += 1
        super(UAVDTReIdTrain10a, self).__init__(train, query, gallery, **kwargs)

class UAVDTReIdTrain10b(torchreid.data.ImageDataset):
    #dataset_dir = '/home/jatin/HDD2TB/datasets/UAVDT_reid'
    dataset_dir = args.data
    def __init__(self, root='', mode='train', **kwargs):
        train = []
        query = []
        gallery = []

        #Train
        print('Skipping some real sequences to balance the total number of samples')
        seqs = [#'M0101', 'M0201', 'M0202', 'M0204', 'M0206',
                #'M0207', 'M0210', 'M0301', 'M0401', 'M0402',
                'M0501', 'M0603', 'M0604', 'M0605', 'M0702',
                'M0703', 'M0704', 'M0901', 'M0902', 'M1002',
                #'M1003', 'M1005', 'M1006', 'M1008', 'M1102',
                #'M1201', 'M1202', 'M1304', 'M1305', 'M1306'
                ]
        seqs.sort()

        objind = 0
        for seq in seqs:
            objs = os.listdir(os.path.join(self.dataset_dir,seq))
            objs.sort()
            for obj in objs:
                imgs = os.listdir(os.path.join(self.dataset_dir,seq,obj))
                imgs.sort()

                if len(imgs)>2:
                    q = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[q])
                    entry = [img_path, objind, seqs.index(seq)+100]
                    query.append(entry)
                    imgs.pop(q)

                    g = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[g])
                    entry = [img_path, objind, seqs.index(seq)]
                    gallery.append(entry)
                    imgs.pop(g)

                for img in imgs:
                    img_path = os.path.join(self.dataset_dir,seq,obj,img)
                    entry = [img_path, objind, seqs.index(seq)]
                    train.append(entry)
                objind += 1
        super(UAVDTReIdTrain10b, self).__init__(train, query, gallery, **kwargs)
        
class UAVDTReIdTrain10c(torchreid.data.ImageDataset):
    #dataset_dir = '/home/jatin/HDD2TB/datasets/UAVDT_reid'
    dataset_dir = args.data
    def __init__(self, root='', mode='train', **kwargs):
        train = []
        query = []
        gallery = []

        #Train
        print('Skipping some real sequences to balance the total number of samples')
        seqs = [#'M0101', 'M0201', 'M0202', 'M0204', 'M0206',
                #'M0207', 'M0210', 'M0301', 'M0401', 'M0402',
                #'M0501', 'M0603', 'M0604', 'M0605', 'M0702',
                #'M0703', 'M0704', 'M0901', 'M0902', 'M1002',
                'M1003', 'M1005', 'M1006', 'M1008', 'M1102',
                'M1201', 'M1202', 'M1304', 'M1305', 'M1306'
                ]
        seqs.sort()

        objind = 0
        for seq in seqs:
            objs = os.listdir(os.path.join(self.dataset_dir,seq))
            objs.sort()
            for obj in objs:
                imgs = os.listdir(os.path.join(self.dataset_dir,seq,obj))
                imgs.sort()

                if len(imgs)>2:
                    q = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[q])
                    entry = [img_path, objind, seqs.index(seq)+100]
                    query.append(entry)
                    imgs.pop(q)

                    g = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[g])
                    entry = [img_path, objind, seqs.index(seq)]
                    gallery.append(entry)
                    imgs.pop(g)

                for img in imgs:
                    img_path = os.path.join(self.dataset_dir,seq,obj,img)
                    entry = [img_path, objind, seqs.index(seq)]
                    train.append(entry)
                objind += 1
        super(UAVDTReIdTrain10c, self).__init__(train, query, gallery, **kwargs)

class UAVDTReIdTrain15a(torchreid.data.ImageDataset):
    #dataset_dir = '/home/jatin/HDD2TB/datasets/UAVDT_reid'
    dataset_dir = args.data
    def __init__(self, root='', mode='train', **kwargs):
        train = []
        query = []
        gallery = []

        #Train
        print('Skipping some real sequences to balance the total number of samples')
        seqs = ['M0101', 'M0201', 'M0202', 'M0204', 'M0206',
                # 'M0207', 'M0210', 'M0301', 'M0401', 'M0402',
                # 'M0501', 'M0603', 'M0604', 'M0605', 'M0702',
                # 'M0703', 'M0704', 'M0901', 'M0902', 'M1002',
                'M1003', 'M1005', 'M1006', 'M1008', 'M1102',
                'M1201', 'M1202', 'M1304', 'M1305', 'M1306']
        seqs.sort()

        objind = 0
        for seq in seqs:
            objs = os.listdir(os.path.join(self.dataset_dir,seq))
            objs.sort()
            for obj in objs:
                imgs = os.listdir(os.path.join(self.dataset_dir,seq,obj))
                imgs.sort()

                if len(imgs)>2:
                    q = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[q])
                    entry = [img_path, objind, seqs.index(seq)+100]
                    query.append(entry)
                    imgs.pop(q)

                    g = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[g])
                    entry = [img_path, objind, seqs.index(seq)]
                    gallery.append(entry)
                    imgs.pop(g)

                for img in imgs:
                    img_path = os.path.join(self.dataset_dir,seq,obj,img)
                    entry = [img_path, objind, seqs.index(seq)]
                    train.append(entry)
                objind += 1
        super(UAVDTReIdTrain15a, self).__init__(train, query, gallery, **kwargs)

class UAVDTReIdTrain15b(torchreid.data.ImageDataset):
    #dataset_dir = '/home/jatin/HDD2TB/datasets/UAVDT_reid'
    dataset_dir = args.data
    def __init__(self, root='', mode='train', **kwargs):
        train = []
        query = []
        gallery = []

        #Train
        print('Skipping some real sequences to balance the total number of samples')
        seqs = [#'M0101', 'M0201', 'M0202', 'M0204', 'M0206',
                'M0207', 'M0210', 'M0301', 'M0401', 'M0402',
                'M0501', 'M0603', 'M0604', 'M0605', 'M0702',
                'M0703', 'M0704', 'M0901', 'M0902', 'M1002',
                #'M1003', 'M1005', 'M1006', 'M1008', 'M1102',
                #'M1201', 'M1202', 'M1304', 'M1305', 'M1306'
                ]
        seqs.sort()

        objind = 0
        for seq in seqs:
            objs = os.listdir(os.path.join(self.dataset_dir,seq))
            objs.sort()
            for obj in objs:
                imgs = os.listdir(os.path.join(self.dataset_dir,seq,obj))
                imgs.sort()

                if len(imgs)>2:
                    q = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[q])
                    entry = [img_path, objind, seqs.index(seq)+100]
                    query.append(entry)
                    imgs.pop(q)

                    g = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[g])
                    entry = [img_path, objind, seqs.index(seq)]
                    gallery.append(entry)
                    imgs.pop(g)

                for img in imgs:
                    img_path = os.path.join(self.dataset_dir,seq,obj,img)
                    entry = [img_path, objind, seqs.index(seq)]
                    train.append(entry)
                objind += 1
        super(UAVDTReIdTrain15b, self).__init__(train, query, gallery, **kwargs)

class UAVDTReIdTrain20a(torchreid.data.ImageDataset):
    #dataset_dir = '/home/jatin/HDD2TB/datasets/UAVDT_reid'
    dataset_dir = args.data
    def __init__(self, root='', mode='train', **kwargs):
        train = []
        query = []
        gallery = []

        #Train
        print('Skipping some real sequences to balance the total number of samples')
        seqs = ['M0101', 'M0201', 'M0202', 'M0204', 'M0206',
                'M0207', 'M0210', 'M0301', 'M0401', 'M0402',
                'M0501', 'M0603', 'M0604', 'M0605', 'M0702',
                'M0703', 'M0704', 'M0901', 'M0902', 'M1002',
                #'M1003', 'M1005', 'M1006', 'M1008', 'M1102',
                #'M1201', 'M1202', 'M1304', 'M1305', 'M1306'
                ]
        seqs.sort()

        objind = 0
        for seq in seqs:
            objs = os.listdir(os.path.join(self.dataset_dir,seq))
            objs.sort()
            for obj in objs:
                imgs = os.listdir(os.path.join(self.dataset_dir,seq,obj))
                imgs.sort()

                if len(imgs)>2:
                    q = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[q])
                    entry = [img_path, objind, seqs.index(seq)+100]
                    query.append(entry)
                    imgs.pop(q)

                    g = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[g])
                    entry = [img_path, objind, seqs.index(seq)]
                    gallery.append(entry)
                    imgs.pop(g)

                for img in imgs:
                    img_path = os.path.join(self.dataset_dir,seq,obj,img)
                    entry = [img_path, objind, seqs.index(seq)]
                    train.append(entry)
                objind += 1
        super(UAVDTReIdTrain20a, self).__init__(train, query, gallery, **kwargs)

class UAVDTReIdTrain20b(torchreid.data.ImageDataset):
    #dataset_dir = '/home/jatin/HDD2TB/datasets/UAVDT_reid'
    dataset_dir = args.data
    def __init__(self, root='', mode='train', **kwargs):
        train = []
        query = []
        gallery = []

        #Train
        print('Skipping some real sequences to balance the total number of samples')
        seqs = ['M0101', 'M0201', 'M0202', 'M0204', 'M0206',
                'M0207', 'M0210', 'M0301', 'M0401', 'M0402',
                #'M0501', 'M0603', 'M0604', 'M0605', 'M0702',
                #'M0703', 'M0704', 'M0901', 'M0902', 'M1002',
                'M1003', 'M1005', 'M1006', 'M1008', 'M1102',
                'M1201', 'M1202', 'M1304', 'M1305', 'M1306'
                ]
        seqs.sort()

        objind = 0
        for seq in seqs:
            objs = os.listdir(os.path.join(self.dataset_dir,seq))
            objs.sort()
            for obj in objs:
                imgs = os.listdir(os.path.join(self.dataset_dir,seq,obj))
                imgs.sort()

                if len(imgs)>2:
                    q = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[q])
                    entry = [img_path, objind, seqs.index(seq)+100]
                    query.append(entry)
                    imgs.pop(q)

                    g = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[g])
                    entry = [img_path, objind, seqs.index(seq)]
                    gallery.append(entry)
                    imgs.pop(g)

                for img in imgs:
                    img_path = os.path.join(self.dataset_dir,seq,obj,img)
                    entry = [img_path, objind, seqs.index(seq)]
                    train.append(entry)
                objind += 1
        super(UAVDTReIdTrain20b, self).__init__(train, query, gallery, **kwargs)

class UAVDTReIdTrain20c(torchreid.data.ImageDataset):
    #dataset_dir = '/home/jatin/HDD2TB/datasets/UAVDT_reid'
    dataset_dir = args.data
    def __init__(self, root='', mode='train', **kwargs):
        train = []
        query = []
        gallery = []

        #Train
        print('Skipping some real sequences to balance the total number of samples')
        seqs = [#'M0101', 'M0201', 'M0202', 'M0204', 'M0206',
                #'M0207', 'M0210', 'M0301', 'M0401', 'M0402',
                'M0501', 'M0603', 'M0604', 'M0605', 'M0702',
                'M0703', 'M0704', 'M0901', 'M0902', 'M1002',
                'M1003', 'M1005', 'M1006', 'M1008', 'M1102',
                'M1201', 'M1202', 'M1304', 'M1305', 'M1306'
                ]
        seqs.sort()

        objind = 0
        for seq in seqs:
            objs = os.listdir(os.path.join(self.dataset_dir,seq))
            objs.sort()
            for obj in objs:
                imgs = os.listdir(os.path.join(self.dataset_dir,seq,obj))
                imgs.sort()

                if len(imgs)>2:
                    q = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[q])
                    entry = [img_path, objind, seqs.index(seq)+100]
                    query.append(entry)
                    imgs.pop(q)

                    g = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[g])
                    entry = [img_path, objind, seqs.index(seq)]
                    gallery.append(entry)
                    imgs.pop(g)

                for img in imgs:
                    img_path = os.path.join(self.dataset_dir,seq,obj,img)
                    entry = [img_path, objind, seqs.index(seq)]
                    train.append(entry)
                objind += 1
        super(UAVDTReIdTrain20c, self).__init__(train, query, gallery, **kwargs)

class UAVDTReIdTrain25a(torchreid.data.ImageDataset):
    #dataset_dir = '/home/jatin/HDD2TB/datasets/UAVDT_reid'
    dataset_dir = args.data
    def __init__(self, root='', mode='train', **kwargs):
        train = []
        query = []
        gallery = []

        #Train
        print('Skipping some real sequences to balance the total number of samples')
        seqs = ['M0101', 'M0201', 'M0202', 'M0204', 'M0206',
                'M0207', 'M0210', 'M0301', 'M0401', 'M0402',
                'M0501', 'M0603', 'M0604', 'M0605', 'M0702',
                'M0703', 'M0704', 'M0901', 'M0902', 'M1002',
                'M1003', 'M1005', 'M1006', 'M1008', 'M1102',
                #'M1201', 'M1202', 'M1304', 'M1305', 'M1306'
                ]
        seqs.sort()

        objind = 0
        for seq in seqs:
            objs = os.listdir(os.path.join(self.dataset_dir,seq))
            objs.sort()
            for obj in objs:
                imgs = os.listdir(os.path.join(self.dataset_dir,seq,obj))
                imgs.sort()

                if len(imgs)>2:
                    q = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[q])
                    entry = [img_path, objind, seqs.index(seq)+100]
                    query.append(entry)
                    imgs.pop(q)

                    g = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[g])
                    entry = [img_path, objind, seqs.index(seq)]
                    gallery.append(entry)
                    imgs.pop(g)

                for img in imgs:
                    img_path = os.path.join(self.dataset_dir,seq,obj,img)
                    entry = [img_path, objind, seqs.index(seq)]
                    train.append(entry)
                objind += 1
        super(UAVDTReIdTrain25a, self).__init__(train, query, gallery, **kwargs)

class UAVDTReIdTrain25b(torchreid.data.ImageDataset):
    #dataset_dir = '/home/jatin/HDD2TB/datasets/UAVDT_reid'
    dataset_dir = args.data
    def __init__(self, root='', mode='train', **kwargs):
        train = []
        query = []
        gallery = []

        #Train
        print('Skipping some real sequences to balance the total number of samples')
        seqs = ['M0101', 'M0201', 'M0202', 'M0204', 'M0206',
                'M0207', 'M0210', 'M0301', 'M0401', 'M0402',
                'M0501', 'M0603', 'M0604', 'M0605', 'M0702',
                'M0703', 'M0704', 'M0901', 'M0902', 'M1002',
                #'M1003', 'M1005', 'M1006', 'M1008', 'M1102',
                'M1201', 'M1202', 'M1304', 'M1305', 'M1306'
                ]
        seqs.sort()

        objind = 0
        for seq in seqs:
            objs = os.listdir(os.path.join(self.dataset_dir,seq))
            objs.sort()
            for obj in objs:
                imgs = os.listdir(os.path.join(self.dataset_dir,seq,obj))
                imgs.sort()

                if len(imgs)>2:
                    q = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[q])
                    entry = [img_path, objind, seqs.index(seq)+100]
                    query.append(entry)
                    imgs.pop(q)

                    g = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[g])
                    entry = [img_path, objind, seqs.index(seq)]
                    gallery.append(entry)
                    imgs.pop(g)

                for img in imgs:
                    img_path = os.path.join(self.dataset_dir,seq,obj,img)
                    entry = [img_path, objind, seqs.index(seq)]
                    train.append(entry)
                objind += 1
        super(UAVDTReIdTrain25b, self).__init__(train, query, gallery, **kwargs)

class UAVDTReIdTrain25c(torchreid.data.ImageDataset):
    #dataset_dir = '/home/jatin/HDD2TB/datasets/UAVDT_reid'
    dataset_dir = args.data
    def __init__(self, root='', mode='train', **kwargs):
        train = []
        query = []
        gallery = []

        #Train
        print('Skipping some real sequences to balance the total number of samples')
        seqs = ['M0101', 'M0201', 'M0202', 'M0204', 'M0206',
                'M0207', 'M0210', 'M0301', 'M0401', 'M0402',
                'M0501', 'M0603', 'M0604', 'M0605', 'M0702',
                #'M0703', 'M0704', 'M0901', 'M0902', 'M1002',
                'M1003', 'M1005', 'M1006', 'M1008', 'M1102',
                'M1201', 'M1202', 'M1304', 'M1305', 'M1306'
                ]
        seqs.sort()

        objind = 0
        for seq in seqs:
            objs = os.listdir(os.path.join(self.dataset_dir,seq))
            objs.sort()
            for obj in objs:
                imgs = os.listdir(os.path.join(self.dataset_dir,seq,obj))
                imgs.sort()

                if len(imgs)>2:
                    q = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[q])
                    entry = [img_path, objind, seqs.index(seq)+100]
                    query.append(entry)
                    imgs.pop(q)

                    g = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[g])
                    entry = [img_path, objind, seqs.index(seq)]
                    gallery.append(entry)
                    imgs.pop(g)

                for img in imgs:
                    img_path = os.path.join(self.dataset_dir,seq,obj,img)
                    entry = [img_path, objind, seqs.index(seq)]
                    train.append(entry)
                objind += 1
        super(UAVDTReIdTrain25c, self).__init__(train, query, gallery, **kwargs)

class UAVDTReIdTrain25d(torchreid.data.ImageDataset):
    #dataset_dir = '/home/jatin/HDD2TB/datasets/UAVDT_reid'
    dataset_dir = args.data
    def __init__(self, root='', mode='train', **kwargs):
        train = []
        query = []
        gallery = []

        #Train
        print('Skipping some real sequences to balance the total number of samples')
        seqs = ['M0101', 'M0201', 'M0202', 'M0204', 'M0206',
                'M0207', 'M0210', 'M0301', 'M0401', 'M0402',
                #'M0501', 'M0603', 'M0604', 'M0605', 'M0702',
                'M0703', 'M0704', 'M0901', 'M0902', 'M1002',
                'M1003', 'M1005', 'M1006', 'M1008', 'M1102',
                'M1201', 'M1202', 'M1304', 'M1305', 'M1306'
                ]
        seqs.sort()

        objind = 0
        for seq in seqs:
            objs = os.listdir(os.path.join(self.dataset_dir,seq))
            objs.sort()
            for obj in objs:
                imgs = os.listdir(os.path.join(self.dataset_dir,seq,obj))
                imgs.sort()

                if len(imgs)>2:
                    q = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[q])
                    entry = [img_path, objind, seqs.index(seq)+100]
                    query.append(entry)
                    imgs.pop(q)

                    g = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[g])
                    entry = [img_path, objind, seqs.index(seq)]
                    gallery.append(entry)
                    imgs.pop(g)

                for img in imgs:
                    img_path = os.path.join(self.dataset_dir,seq,obj,img)
                    entry = [img_path, objind, seqs.index(seq)]
                    train.append(entry)
                objind += 1
        super(UAVDTReIdTrain25d, self).__init__(train, query, gallery, **kwargs)
        
class UAVDTReIdTrain25e(torchreid.data.ImageDataset):
    #dataset_dir = '/home/jatin/HDD2TB/datasets/UAVDT_reid'
    dataset_dir = args.data
    def __init__(self, root='', mode='train', **kwargs):
        train = []
        query = []
        gallery = []

        #Train
        print('Skipping some real sequences to balance the total number of samples')
        seqs = ['M0101', 'M0201', 'M0202', 'M0204', 'M0206',
                #'M0207', 'M0210', 'M0301', 'M0401', 'M0402',
                'M0501', 'M0603', 'M0604', 'M0605', 'M0702',
                'M0703', 'M0704', 'M0901', 'M0902', 'M1002',
                'M1003', 'M1005', 'M1006', 'M1008', 'M1102',
                'M1201', 'M1202', 'M1304', 'M1305', 'M1306'
                ]
        seqs.sort()

        objind = 0
        for seq in seqs:
            objs = os.listdir(os.path.join(self.dataset_dir,seq))
            objs.sort()
            for obj in objs:
                imgs = os.listdir(os.path.join(self.dataset_dir,seq,obj))
                imgs.sort()

                if len(imgs)>2:
                    q = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[q])
                    entry = [img_path, objind, seqs.index(seq)+100]
                    query.append(entry)
                    imgs.pop(q)

                    g = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[g])
                    entry = [img_path, objind, seqs.index(seq)]
                    gallery.append(entry)
                    imgs.pop(g)

                for img in imgs:
                    img_path = os.path.join(self.dataset_dir,seq,obj,img)
                    entry = [img_path, objind, seqs.index(seq)]
                    train.append(entry)
                objind += 1
        super(UAVDTReIdTrain25e, self).__init__(train, query, gallery, **kwargs)
        
class UAVDTReIdTrain25f(torchreid.data.ImageDataset):
    #dataset_dir = '/home/jatin/HDD2TB/datasets/UAVDT_reid'
    dataset_dir = args.data
    def __init__(self, root='', mode='train', **kwargs):
        train = []
        query = []
        gallery = []

        #Train
        print('Skipping some real sequences to balance the total number of samples')
        seqs = [#'M0101', 'M0201', 'M0202', 'M0204', 'M0206',
                'M0207', 'M0210', 'M0301', 'M0401', 'M0402',
                'M0501', 'M0603', 'M0604', 'M0605', 'M0702',
                'M0703', 'M0704', 'M0901', 'M0902', 'M1002',
                'M1003', 'M1005', 'M1006', 'M1008', 'M1102',
                'M1201', 'M1202', 'M1304', 'M1305', 'M1306'
                ]
        seqs.sort()

        objind = 0
        for seq in seqs:
            objs = os.listdir(os.path.join(self.dataset_dir,seq))
            objs.sort()
            for obj in objs:
                imgs = os.listdir(os.path.join(self.dataset_dir,seq,obj))
                imgs.sort()

                if len(imgs)>2:
                    q = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[q])
                    entry = [img_path, objind, seqs.index(seq)+100]
                    query.append(entry)
                    imgs.pop(q)

                    g = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[g])
                    entry = [img_path, objind, seqs.index(seq)]
                    gallery.append(entry)
                    imgs.pop(g)

                for img in imgs:
                    img_path = os.path.join(self.dataset_dir,seq,obj,img)
                    entry = [img_path, objind, seqs.index(seq)]
                    train.append(entry)
                objind += 1
        super(UAVDTReIdTrain25f, self).__init__(train, query, gallery, **kwargs)


class VisDroneTrain(torchreid.data.ImageDataset):
    # dataset_dir = '/home/jatin/HDD2TB/datasets/UAVDT_reid'
    dataset_dir = args.data

    def __init__(self, root='', **kwargs):
        train = []
        query = []
        gallery = []

        # Train
        seqs = ['uav0000124_00944_v', 'uav0000126_00001_v', 'uav0000140_01590_v', 'uav0000143_02250_v',
                'uav0000145_00000_v',
                'uav0000218_00001_v', 'uav0000222_03150_v', 'uav0000239_03720_v', 'uav0000239_12336_v',
                'uav0000243_00001_v',
                'uav0000244_01440_v', 'uav0000248_00001_v', 'uav0000263_03289_v', 'uav0000264_02760_v',
                'uav0000270_00001_v',
                'uav0000273_00001_v', 'uav0000278_00001_v', 'uav0000279_00001_v', 'uav0000281_00460_v',
                'uav0000289_00001_v',
                'uav0000289_06922_v', 'uav0000295_02300_v', 'uav0000307_00000_v', 'uav0000315_00000_v',
                'uav0000323_01173_v',  # 'uav0000300_00000_v', excluded for less cars to other ratio
                'uav0000326_01035_v', 'uav0000342_04692_v', 'uav0000352_05980_v', 'uav0000361_02323_v',
                'uav0000366_00001_v'
                ]
        seqs.sort()

        objind = 0
        for seq in seqs:
            objs = os.listdir(os.path.join(self.dataset_dir, seq))
            objs.sort()
            for obj in objs:
                imgs = os.listdir(os.path.join(self.dataset_dir, seq, obj))
                imgs.sort()

                if len(imgs) > 2:
                    q = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[q])
                    entry = [img_path, objind, seqs.index(seq) + 100]
                    query.append(entry)
                    imgs.pop(q)

                    g = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[g])
                    entry = [img_path, objind, seqs.index(seq)]
                    gallery.append(entry)
                    imgs.pop(g)

                for img in imgs:
                    img_path = os.path.join(self.dataset_dir, seq, obj, img)
                    entry = [img_path, objind, seqs.index(seq)]
                    train.append(entry)
                objind += 1
        super(VisDroneTrain, self).__init__(train, query, gallery, **kwargs)


class VisDroneTrain5a(torchreid.data.ImageDataset):
    # dataset_dir = '/home/jatin/HDD2TB/datasets/UAVDT_reid'
    dataset_dir = args.data

    def __init__(self, root='', mode='train', **kwargs):
        train = []
        query = []
        gallery = []

        # Train
        print('Skipping some real sequences to balance the total number of samples')
        seqs = ['uav0000124_00944_v', 'uav0000126_00001_v', 'uav0000140_01590_v', 'uav0000143_02250_v',
                'uav0000145_00000_v',
                # 'uav0000218_00001_v', 'uav0000222_03150_v', 'uav0000239_03720_v', 'uav0000239_12336_v', 'uav0000243_00001_v',
                # 'uav0000244_01440_v', 'uav0000248_00001_v', 'uav0000263_03289_v', 'uav0000264_02760_v', 'uav0000270_00001_v',
                # 'uav0000273_00001_v', 'uav0000278_00001_v', 'uav0000279_00001_v', 'uav0000281_00460_v', 'uav0000289_00001_v',
                # 'uav0000289_06922_v', 'uav0000295_02300_v', 'uav0000307_00000_v', 'uav0000315_00000_v', 'uav0000323_01173_v', #'uav0000300_00000_v', excluded for less cars to other ratio
                # 'uav0000326_01035_v', 'uav0000342_04692_v', 'uav0000352_05980_v', 'uav0000361_02323_v', 'uav0000366_00001_v'
                ]
        seqs.sort()

        objind = 0
        for seq in seqs:
            objs = os.listdir(os.path.join(self.dataset_dir, seq))
            objs.sort()
            for obj in objs:
                imgs = os.listdir(os.path.join(self.dataset_dir, seq, obj))
                imgs.sort()

                if len(imgs) > 2:
                    q = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[q])
                    entry = [img_path, objind, seqs.index(seq) + 100]
                    query.append(entry)
                    imgs.pop(q)

                    g = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[g])
                    entry = [img_path, objind, seqs.index(seq)]
                    gallery.append(entry)
                    imgs.pop(g)

                for img in imgs:
                    img_path = os.path.join(self.dataset_dir, seq, obj, img)
                    entry = [img_path, objind, seqs.index(seq)]
                    train.append(entry)
                objind += 1
        super(VisDroneTrain5a, self).__init__(train, query, gallery, **kwargs)


class VisDroneTrain5b(torchreid.data.ImageDataset):
    # dataset_dir = '/home/jatin/HDD2TB/datasets/UAVDT_reid'
    dataset_dir = args.data

    def __init__(self, root='', mode='train', **kwargs):
        train = []
        query = []
        gallery = []

        # Train
        print('Skipping some real sequences to balance the total number of samples')
        seqs = [
            # 'uav0000124_00944_v', 'uav0000126_00001_v', 'uav0000140_01590_v', 'uav0000143_02250_v', 'uav0000145_00000_v',
            'uav0000218_00001_v', 'uav0000222_03150_v', 'uav0000239_03720_v', 'uav0000239_12336_v',
            'uav0000243_00001_v',
            # 'uav0000244_01440_v', 'uav0000248_00001_v', 'uav0000263_03289_v', 'uav0000264_02760_v', 'uav0000270_00001_v',
            # 'uav0000273_00001_v', 'uav0000278_00001_v', 'uav0000279_00001_v', 'uav0000281_00460_v', 'uav0000289_00001_v',
            # 'uav0000289_06922_v', 'uav0000295_02300_v', 'uav0000307_00000_v', 'uav0000315_00000_v', 'uav0000323_01173_v', #'uav0000300_00000_v', excluded for less cars to other ratio
            # 'uav0000326_01035_v', 'uav0000342_04692_v', 'uav0000352_05980_v', 'uav0000361_02323_v', 'uav0000366_00001_v'
        ]
        seqs.sort()

        objind = 0
        for seq in seqs:
            objs = os.listdir(os.path.join(self.dataset_dir, seq))
            objs.sort()
            for obj in objs:
                imgs = os.listdir(os.path.join(self.dataset_dir, seq, obj))
                imgs.sort()

                if len(imgs) > 2:
                    q = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[q])
                    entry = [img_path, objind, seqs.index(seq) + 100]
                    query.append(entry)
                    imgs.pop(q)

                    g = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[g])
                    entry = [img_path, objind, seqs.index(seq)]
                    gallery.append(entry)
                    imgs.pop(g)

                for img in imgs:
                    img_path = os.path.join(self.dataset_dir, seq, obj, img)
                    entry = [img_path, objind, seqs.index(seq)]
                    train.append(entry)
                objind += 1
        super(VisDroneTrain5b, self).__init__(train, query, gallery, **kwargs)


class VisDroneTrain5c(torchreid.data.ImageDataset):
    # dataset_dir = '/home/jatin/HDD2TB/datasets/UAVDT_reid'
    dataset_dir = args.data

    def __init__(self, root='', mode='train', **kwargs):
        train = []
        query = []
        gallery = []

        # Train
        print('Skipping some real sequences to balance the total number of samples')
        seqs = [
            # 'uav0000124_00944_v', 'uav0000126_00001_v', 'uav0000140_01590_v', 'uav0000143_02250_v', 'uav0000145_00000_v',
            # 'uav0000218_00001_v', 'uav0000222_03150_v', 'uav0000239_03720_v', 'uav0000239_12336_v', 'uav0000243_00001_v',
            'uav0000244_01440_v', 'uav0000248_00001_v', 'uav0000263_03289_v', 'uav0000264_02760_v',
            'uav0000270_00001_v',
            # 'uav0000273_00001_v', 'uav0000278_00001_v', 'uav0000279_00001_v', 'uav0000281_00460_v', 'uav0000289_00001_v',
            # 'uav0000289_06922_v', 'uav0000295_02300_v', 'uav0000307_00000_v', 'uav0000315_00000_v', 'uav0000323_01173_v', #'uav0000300_00000_v', excluded for less cars to other ratio
            # 'uav0000326_01035_v', 'uav0000342_04692_v', 'uav0000352_05980_v', 'uav0000361_02323_v', 'uav0000366_00001_v'
        ]
        seqs.sort()

        objind = 0
        for seq in seqs:
            objs = os.listdir(os.path.join(self.dataset_dir, seq))
            objs.sort()
            for obj in objs:
                imgs = os.listdir(os.path.join(self.dataset_dir, seq, obj))
                imgs.sort()

                if len(imgs) > 2:
                    q = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[q])
                    entry = [img_path, objind, seqs.index(seq) + 100]
                    query.append(entry)
                    imgs.pop(q)

                    g = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[g])
                    entry = [img_path, objind, seqs.index(seq)]
                    gallery.append(entry)
                    imgs.pop(g)

                for img in imgs:
                    img_path = os.path.join(self.dataset_dir, seq, obj, img)
                    entry = [img_path, objind, seqs.index(seq)]
                    train.append(entry)
                objind += 1
        super(VisDroneTrain5c, self).__init__(train, query, gallery, **kwargs)


class VisDroneTrain5d(torchreid.data.ImageDataset):
    # dataset_dir = '/home/jatin/HDD2TB/datasets/UAVDT_reid'
    dataset_dir = args.data

    def __init__(self, root='', mode='train', **kwargs):
        train = []
        query = []
        gallery = []

        # Train
        print('Skipping some real sequences to balance the total number of samples')
        seqs = [
            # 'uav0000124_00944_v', 'uav0000126_00001_v', 'uav0000140_01590_v', 'uav0000143_02250_v', 'uav0000145_00000_v',
            # 'uav0000218_00001_v', 'uav0000222_03150_v', 'uav0000239_03720_v', 'uav0000239_12336_v', 'uav0000243_00001_v',
            # 'uav0000244_01440_v', 'uav0000248_00001_v', 'uav0000263_03289_v', 'uav0000264_02760_v', 'uav0000270_00001_v',
            'uav0000273_00001_v', 'uav0000278_00001_v', 'uav0000279_00001_v', 'uav0000281_00460_v',
            'uav0000289_00001_v',
            # 'uav0000289_06922_v', 'uav0000295_02300_v', 'uav0000307_00000_v', 'uav0000315_00000_v', 'uav0000323_01173_v', #'uav0000300_00000_v', excluded for less cars to other ratio
            # 'uav0000326_01035_v', 'uav0000342_04692_v', 'uav0000352_05980_v', 'uav0000361_02323_v', 'uav0000366_00001_v'
        ]
        seqs.sort()

        objind = 0
        for seq in seqs:
            objs = os.listdir(os.path.join(self.dataset_dir, seq))
            objs.sort()
            for obj in objs:
                imgs = os.listdir(os.path.join(self.dataset_dir, seq, obj))
                imgs.sort()

                if len(imgs) > 2:
                    q = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[q])
                    entry = [img_path, objind, seqs.index(seq) + 100]
                    query.append(entry)
                    imgs.pop(q)

                    g = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[g])
                    entry = [img_path, objind, seqs.index(seq)]
                    gallery.append(entry)
                    imgs.pop(g)

                for img in imgs:
                    img_path = os.path.join(self.dataset_dir, seq, obj, img)
                    entry = [img_path, objind, seqs.index(seq)]
                    train.append(entry)
                objind += 1
        super(VisDroneTrain5d, self).__init__(train, query, gallery, **kwargs)


class VisDroneTrain5e(torchreid.data.ImageDataset):
    # dataset_dir = '/home/jatin/HDD2TB/datasets/UAVDT_reid'
    dataset_dir = args.data

    def __init__(self, root='', mode='train', **kwargs):
        train = []
        query = []
        gallery = []

        # Train
        print('Skipping some real sequences to balance the total number of samples')
        seqs = [
            # 'uav0000124_00944_v', 'uav0000126_00001_v', 'uav0000140_01590_v', 'uav0000143_02250_v', 'uav0000145_00000_v',
            # 'uav0000218_00001_v', 'uav0000222_03150_v', 'uav0000239_03720_v', 'uav0000239_12336_v', 'uav0000243_00001_v',
            # 'uav0000244_01440_v', 'uav0000248_00001_v', 'uav0000263_03289_v', 'uav0000264_02760_v', 'uav0000270_00001_v',
            # 'uav0000273_00001_v', 'uav0000278_00001_v', 'uav0000279_00001_v', 'uav0000281_00460_v', 'uav0000289_00001_v',
            'uav0000289_06922_v', 'uav0000295_02300_v', 'uav0000307_00000_v', 'uav0000315_00000_v',
            'uav0000323_01173_v',  # 'uav0000300_00000_v', excluded for less cars to other ratio
            # 'uav0000326_01035_v', 'uav0000342_04692_v', 'uav0000352_05980_v', 'uav0000361_02323_v', 'uav0000366_00001_v'
        ]
        seqs.sort()

        objind = 0
        for seq in seqs:
            objs = os.listdir(os.path.join(self.dataset_dir, seq))
            objs.sort()
            for obj in objs:
                imgs = os.listdir(os.path.join(self.dataset_dir, seq, obj))
                imgs.sort()

                if len(imgs) > 2:
                    q = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[q])
                    entry = [img_path, objind, seqs.index(seq) + 100]
                    query.append(entry)
                    imgs.pop(q)

                    g = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[g])
                    entry = [img_path, objind, seqs.index(seq)]
                    gallery.append(entry)
                    imgs.pop(g)

                for img in imgs:
                    img_path = os.path.join(self.dataset_dir, seq, obj, img)
                    entry = [img_path, objind, seqs.index(seq)]
                    train.append(entry)
                objind += 1
        super(VisDroneTrain5e, self).__init__(train, query, gallery, **kwargs)


class VisDroneTrain5f(torchreid.data.ImageDataset):
    # dataset_dir = '/home/jatin/HDD2TB/datasets/UAVDT_reid'
    dataset_dir = args.data

    def __init__(self, root='', mode='train', **kwargs):
        train = []
        query = []
        gallery = []

        # Train
        print('Skipping some real sequences to balance the total number of samples')
        seqs = [
            # 'uav0000124_00944_v', 'uav0000126_00001_v', 'uav0000140_01590_v', 'uav0000143_02250_v', 'uav0000145_00000_v',
            # 'uav0000218_00001_v', 'uav0000222_03150_v', 'uav0000239_03720_v', 'uav0000239_12336_v', 'uav0000243_00001_v',
            # 'uav0000244_01440_v', 'uav0000248_00001_v', 'uav0000263_03289_v', 'uav0000264_02760_v', 'uav0000270_00001_v',
            # 'uav0000273_00001_v', 'uav0000278_00001_v', 'uav0000279_00001_v', 'uav0000281_00460_v', 'uav0000289_00001_v',
            # 'uav0000289_06922_v', 'uav0000295_02300_v', 'uav0000307_00000_v', 'uav0000315_00000_v', 'uav0000323_01173_v', #'uav0000300_00000_v', excluded for less cars to other ratio
            'uav0000326_01035_v', 'uav0000342_04692_v', 'uav0000352_05980_v', 'uav0000361_02323_v',
            'uav0000366_00001_v'
        ]
        seqs.sort()

        objind = 0
        for seq in seqs:
            objs = os.listdir(os.path.join(self.dataset_dir, seq))
            objs.sort()
            for obj in objs:
                imgs = os.listdir(os.path.join(self.dataset_dir, seq, obj))
                imgs.sort()

                if len(imgs) > 2:
                    q = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[q])
                    entry = [img_path, objind, seqs.index(seq) + 100]
                    query.append(entry)
                    imgs.pop(q)

                    g = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[g])
                    entry = [img_path, objind, seqs.index(seq)]
                    gallery.append(entry)
                    imgs.pop(g)

                for img in imgs:
                    img_path = os.path.join(self.dataset_dir, seq, obj, img)
                    entry = [img_path, objind, seqs.index(seq)]
                    train.append(entry)
                objind += 1
        super(VisDroneTrain5f, self).__init__(train, query, gallery, **kwargs)


class VisDroneTrain10a(torchreid.data.ImageDataset):
    # dataset_dir = '/home/jatin/HDD2TB/datasets/UAVDT_reid'
    dataset_dir = args.data

    def __init__(self, root='', mode='train', **kwargs):
        train = []
        query = []
        gallery = []

        # Train
        print('Skipping some real sequences to balance the total number of samples')
        seqs = ['uav0000124_00944_v', 'uav0000126_00001_v', 'uav0000140_01590_v', 'uav0000143_02250_v',
                'uav0000145_00000_v',
                'uav0000218_00001_v', 'uav0000222_03150_v', 'uav0000239_03720_v', 'uav0000239_12336_v',
                'uav0000243_00001_v',
                # 'uav0000244_01440_v', 'uav0000248_00001_v', 'uav0000263_03289_v', 'uav0000264_02760_v', 'uav0000270_00001_v',
                # 'uav0000273_00001_v', 'uav0000278_00001_v', 'uav0000279_00001_v', 'uav0000281_00460_v', 'uav0000289_00001_v',
                # 'uav0000289_06922_v', 'uav0000295_02300_v', 'uav0000307_00000_v', 'uav0000315_00000_v', 'uav0000323_01173_v', #'uav0000300_00000_v', excluded for less cars to other ratio
                # 'uav0000326_01035_v', 'uav0000342_04692_v', 'uav0000352_05980_v', 'uav0000361_02323_v', 'uav0000366_00001_v'
                ]
        seqs.sort()

        objind = 0
        for seq in seqs:
            objs = os.listdir(os.path.join(self.dataset_dir, seq))
            objs.sort()
            for obj in objs:
                imgs = os.listdir(os.path.join(self.dataset_dir, seq, obj))
                imgs.sort()

                if len(imgs) > 2:
                    q = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[q])
                    entry = [img_path, objind, seqs.index(seq) + 100]
                    query.append(entry)
                    imgs.pop(q)

                    g = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[g])
                    entry = [img_path, objind, seqs.index(seq)]
                    gallery.append(entry)
                    imgs.pop(g)

                for img in imgs:
                    img_path = os.path.join(self.dataset_dir, seq, obj, img)
                    entry = [img_path, objind, seqs.index(seq)]
                    train.append(entry)
                objind += 1
        super(VisDroneTrain10a, self).__init__(train, query, gallery, **kwargs)


class VisDroneTrain10b(torchreid.data.ImageDataset):
    # dataset_dir = '/home/jatin/HDD2TB/datasets/UAVDT_reid'
    dataset_dir = args.data

    def __init__(self, root='', mode='train', **kwargs):
        train = []
        query = []
        gallery = []

        # Train
        print('Skipping some real sequences to balance the total number of samples')
        seqs = [
            # 'uav0000124_00944_v', 'uav0000126_00001_v', 'uav0000140_01590_v', 'uav0000143_02250_v', 'uav0000145_00000_v',
            # 'uav0000218_00001_v', 'uav0000222_03150_v', 'uav0000239_03720_v', 'uav0000239_12336_v', 'uav0000243_00001_v',
            'uav0000244_01440_v', 'uav0000248_00001_v', 'uav0000263_03289_v', 'uav0000264_02760_v',
            'uav0000270_00001_v',
            'uav0000273_00001_v', 'uav0000278_00001_v', 'uav0000279_00001_v', 'uav0000281_00460_v',
            'uav0000289_00001_v',
            # 'uav0000289_06922_v', 'uav0000295_02300_v', 'uav0000307_00000_v', 'uav0000315_00000_v', 'uav0000323_01173_v', #'uav0000300_00000_v', excluded for less cars to other ratio
            # 'uav0000326_01035_v', 'uav0000342_04692_v', 'uav0000352_05980_v', 'uav0000361_02323_v', 'uav0000366_00001_v'
        ]
        seqs.sort()

        objind = 0
        for seq in seqs:
            objs = os.listdir(os.path.join(self.dataset_dir, seq))
            objs.sort()
            for obj in objs:
                imgs = os.listdir(os.path.join(self.dataset_dir, seq, obj))
                imgs.sort()

                if len(imgs) > 2:
                    q = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[q])
                    entry = [img_path, objind, seqs.index(seq) + 100]
                    query.append(entry)
                    imgs.pop(q)

                    g = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[g])
                    entry = [img_path, objind, seqs.index(seq)]
                    gallery.append(entry)
                    imgs.pop(g)

                for img in imgs:
                    img_path = os.path.join(self.dataset_dir, seq, obj, img)
                    entry = [img_path, objind, seqs.index(seq)]
                    train.append(entry)
                objind += 1
        super(VisDroneTrain10b, self).__init__(train, query, gallery, **kwargs)


class VisDroneTrain10c(torchreid.data.ImageDataset):
    # dataset_dir = '/home/jatin/HDD2TB/datasets/UAVDT_reid'
    dataset_dir = args.data

    def __init__(self, root='', mode='train', **kwargs):
        train = []
        query = []
        gallery = []

        # Train
        print('Skipping some real sequences to balance the total number of samples')
        seqs = [
            # 'uav0000124_00944_v', 'uav0000126_00001_v', 'uav0000140_01590_v', 'uav0000143_02250_v', 'uav0000145_00000_v',
            # 'uav0000218_00001_v', 'uav0000222_03150_v', 'uav0000239_03720_v', 'uav0000239_12336_v', 'uav0000243_00001_v',
            # 'uav0000244_01440_v', 'uav0000248_00001_v', 'uav0000263_03289_v', 'uav0000264_02760_v', 'uav0000270_00001_v',
            # 'uav0000273_00001_v', 'uav0000278_00001_v', 'uav0000279_00001_v', 'uav0000281_00460_v', 'uav0000289_00001_v',
            'uav0000289_06922_v', 'uav0000295_02300_v', 'uav0000307_00000_v', 'uav0000315_00000_v',
            'uav0000323_01173_v',  # 'uav0000300_00000_v', excluded for less cars to other ratio
            'uav0000326_01035_v', 'uav0000342_04692_v', 'uav0000352_05980_v', 'uav0000361_02323_v',
            'uav0000366_00001_v'
        ]
        seqs.sort()

        objind = 0
        for seq in seqs:
            objs = os.listdir(os.path.join(self.dataset_dir, seq))
            objs.sort()
            for obj in objs:
                imgs = os.listdir(os.path.join(self.dataset_dir, seq, obj))
                imgs.sort()

                if len(imgs) > 2:
                    q = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[q])
                    entry = [img_path, objind, seqs.index(seq) + 100]
                    query.append(entry)
                    imgs.pop(q)

                    g = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[g])
                    entry = [img_path, objind, seqs.index(seq)]
                    gallery.append(entry)
                    imgs.pop(g)

                for img in imgs:
                    img_path = os.path.join(self.dataset_dir, seq, obj, img)
                    entry = [img_path, objind, seqs.index(seq)]
                    train.append(entry)
                objind += 1
        super(VisDroneTrain10c, self).__init__(train, query, gallery, **kwargs)


class VisDroneTrain15a(torchreid.data.ImageDataset):
    # dataset_dir = '/home/jatin/HDD2TB/datasets/UAVDT_reid'
    dataset_dir = args.data

    def __init__(self, root='', mode='train', **kwargs):
        train = []
        query = []
        gallery = []

        # Train
        print('Skipping some real sequences to balance the total number of samples')
        seqs = ['uav0000124_00944_v', 'uav0000126_00001_v', 'uav0000140_01590_v', 'uav0000143_02250_v',
                'uav0000145_00000_v',
                # 'uav0000218_00001_v', 'uav0000222_03150_v', 'uav0000239_03720_v', 'uav0000239_12336_v', 'uav0000243_00001_v',
                # 'uav0000244_01440_v', 'uav0000248_00001_v', 'uav0000263_03289_v', 'uav0000264_02760_v', 'uav0000270_00001_v',
                # 'uav0000273_00001_v', 'uav0000278_00001_v', 'uav0000279_00001_v', 'uav0000281_00460_v', 'uav0000289_00001_v',
                'uav0000289_06922_v', 'uav0000295_02300_v', 'uav0000307_00000_v', 'uav0000315_00000_v',
                'uav0000323_01173_v',  # 'uav0000300_00000_v', excluded for less cars to other ratio
                'uav0000326_01035_v', 'uav0000342_04692_v', 'uav0000352_05980_v', 'uav0000361_02323_v',
                'uav0000366_00001_v']
        seqs.sort()

        objind = 0
        for seq in seqs:
            objs = os.listdir(os.path.join(self.dataset_dir, seq))
            objs.sort()
            for obj in objs:
                imgs = os.listdir(os.path.join(self.dataset_dir, seq, obj))
                imgs.sort()

                if len(imgs) > 2:
                    q = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[q])
                    entry = [img_path, objind, seqs.index(seq) + 100]
                    query.append(entry)
                    imgs.pop(q)

                    g = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[g])
                    entry = [img_path, objind, seqs.index(seq)]
                    gallery.append(entry)
                    imgs.pop(g)

                for img in imgs:
                    img_path = os.path.join(self.dataset_dir, seq, obj, img)
                    entry = [img_path, objind, seqs.index(seq)]
                    train.append(entry)
                objind += 1
        super(VisDroneTrain15a, self).__init__(train, query, gallery, **kwargs)


class VisDroneTrain15b(torchreid.data.ImageDataset):
    # dataset_dir = '/home/jatin/HDD2TB/datasets/UAVDT_reid'
    dataset_dir = args.data

    def __init__(self, root='', mode='train', **kwargs):
        train = []
        query = []
        gallery = []

        # Train
        print('Skipping some real sequences to balance the total number of samples')
        seqs = [
            # 'uav0000124_00944_v', 'uav0000126_00001_v', 'uav0000140_01590_v', 'uav0000143_02250_v', 'uav0000145_00000_v',
            'uav0000218_00001_v', 'uav0000222_03150_v', 'uav0000239_03720_v', 'uav0000239_12336_v',
            'uav0000243_00001_v',
            'uav0000244_01440_v', 'uav0000248_00001_v', 'uav0000263_03289_v', 'uav0000264_02760_v',
            'uav0000270_00001_v',
            'uav0000273_00001_v', 'uav0000278_00001_v', 'uav0000279_00001_v', 'uav0000281_00460_v',
            'uav0000289_00001_v',
            # 'uav0000289_06922_v', 'uav0000295_02300_v', 'uav0000307_00000_v', 'uav0000315_00000_v', 'uav0000323_01173_v', #'uav0000300_00000_v', excluded for less cars to other ratio
            # 'uav0000326_01035_v', 'uav0000342_04692_v', 'uav0000352_05980_v', 'uav0000361_02323_v', 'uav0000366_00001_v'
        ]
        seqs.sort()

        objind = 0
        for seq in seqs:
            objs = os.listdir(os.path.join(self.dataset_dir, seq))
            objs.sort()
            for obj in objs:
                imgs = os.listdir(os.path.join(self.dataset_dir, seq, obj))
                imgs.sort()

                if len(imgs) > 2:
                    q = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[q])
                    entry = [img_path, objind, seqs.index(seq) + 100]
                    query.append(entry)
                    imgs.pop(q)

                    g = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[g])
                    entry = [img_path, objind, seqs.index(seq)]
                    gallery.append(entry)
                    imgs.pop(g)

                for img in imgs:
                    img_path = os.path.join(self.dataset_dir, seq, obj, img)
                    entry = [img_path, objind, seqs.index(seq)]
                    train.append(entry)
                objind += 1
        super(VisDroneTrain15b, self).__init__(train, query, gallery, **kwargs)


class VisDroneTrain20a(torchreid.data.ImageDataset):
    # dataset_dir = '/home/jatin/HDD2TB/datasets/UAVDT_reid'
    dataset_dir = args.data

    def __init__(self, root='', mode='train', **kwargs):
        train = []
        query = []
        gallery = []

        # Train
        print('Skipping some real sequences to balance the total number of samples')
        seqs = ['uav0000124_00944_v', 'uav0000126_00001_v', 'uav0000140_01590_v', 'uav0000143_02250_v',
                'uav0000145_00000_v',
                'uav0000218_00001_v', 'uav0000222_03150_v', 'uav0000239_03720_v', 'uav0000239_12336_v',
                'uav0000243_00001_v',
                'uav0000244_01440_v', 'uav0000248_00001_v', 'uav0000263_03289_v', 'uav0000264_02760_v',
                'uav0000270_00001_v',
                'uav0000273_00001_v', 'uav0000278_00001_v', 'uav0000279_00001_v', 'uav0000281_00460_v',
                'uav0000289_00001_v',
                # 'uav0000289_06922_v', 'uav0000295_02300_v', 'uav0000307_00000_v', 'uav0000315_00000_v', 'uav0000323_01173_v', #'uav0000300_00000_v', excluded for less cars to other ratio
                # 'uav0000326_01035_v', 'uav0000342_04692_v', 'uav0000352_05980_v', 'uav0000361_02323_v', 'uav0000366_00001_v'
                ]
        seqs.sort()

        objind = 0
        for seq in seqs:
            objs = os.listdir(os.path.join(self.dataset_dir, seq))
            objs.sort()
            for obj in objs:
                imgs = os.listdir(os.path.join(self.dataset_dir, seq, obj))
                imgs.sort()

                if len(imgs) > 2:
                    q = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[q])
                    entry = [img_path, objind, seqs.index(seq) + 100]
                    query.append(entry)
                    imgs.pop(q)

                    g = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[g])
                    entry = [img_path, objind, seqs.index(seq)]
                    gallery.append(entry)
                    imgs.pop(g)

                for img in imgs:
                    img_path = os.path.join(self.dataset_dir, seq, obj, img)
                    entry = [img_path, objind, seqs.index(seq)]
                    train.append(entry)
                objind += 1
        super(VisDroneTrain20a, self).__init__(train, query, gallery, **kwargs)


class VisDroneTrain20b(torchreid.data.ImageDataset):
    # dataset_dir = '/home/jatin/HDD2TB/datasets/UAVDT_reid'
    dataset_dir = args.data

    def __init__(self, root='', mode='train', **kwargs):
        train = []
        query = []
        gallery = []

        # Train
        print('Skipping some real sequences to balance the total number of samples')
        seqs = ['uav0000124_00944_v', 'uav0000126_00001_v', 'uav0000140_01590_v', 'uav0000143_02250_v',
                'uav0000145_00000_v',
                'uav0000218_00001_v', 'uav0000222_03150_v', 'uav0000239_03720_v', 'uav0000239_12336_v',
                'uav0000243_00001_v',
                # 'uav0000244_01440_v', 'uav0000248_00001_v', 'uav0000263_03289_v', 'uav0000264_02760_v', 'uav0000270_00001_v',
                # 'uav0000273_00001_v', 'uav0000278_00001_v', 'uav0000279_00001_v', 'uav0000281_00460_v', 'uav0000289_00001_v',
                'uav0000289_06922_v', 'uav0000295_02300_v', 'uav0000307_00000_v', 'uav0000315_00000_v',
                'uav0000323_01173_v',  # 'uav0000300_00000_v', excluded for less cars to other ratio
                'uav0000326_01035_v', 'uav0000342_04692_v', 'uav0000352_05980_v', 'uav0000361_02323_v',
                'uav0000366_00001_v'
                ]
        seqs.sort()

        objind = 0
        for seq in seqs:
            objs = os.listdir(os.path.join(self.dataset_dir, seq))
            objs.sort()
            for obj in objs:
                imgs = os.listdir(os.path.join(self.dataset_dir, seq, obj))
                imgs.sort()

                if len(imgs) > 2:
                    q = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[q])
                    entry = [img_path, objind, seqs.index(seq) + 100]
                    query.append(entry)
                    imgs.pop(q)

                    g = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[g])
                    entry = [img_path, objind, seqs.index(seq)]
                    gallery.append(entry)
                    imgs.pop(g)

                for img in imgs:
                    img_path = os.path.join(self.dataset_dir, seq, obj, img)
                    entry = [img_path, objind, seqs.index(seq)]
                    train.append(entry)
                objind += 1
        super(VisDroneTrain20b, self).__init__(train, query, gallery, **kwargs)


class VisDroneTrain20c(torchreid.data.ImageDataset):
    # dataset_dir = '/home/jatin/HDD2TB/datasets/UAVDT_reid'
    dataset_dir = args.data

    def __init__(self, root='', mode='train', **kwargs):
        train = []
        query = []
        gallery = []

        # Train
        print('Skipping some real sequences to balance the total number of samples')
        seqs = [
            # 'uav0000124_00944_v', 'uav0000126_00001_v', 'uav0000140_01590_v', 'uav0000143_02250_v', 'uav0000145_00000_v',
            # 'uav0000218_00001_v', 'uav0000222_03150_v', 'uav0000239_03720_v', 'uav0000239_12336_v', 'uav0000243_00001_v',
            'uav0000244_01440_v', 'uav0000248_00001_v', 'uav0000263_03289_v', 'uav0000264_02760_v',
            'uav0000270_00001_v',
            'uav0000273_00001_v', 'uav0000278_00001_v', 'uav0000279_00001_v', 'uav0000281_00460_v',
            'uav0000289_00001_v',
            'uav0000289_06922_v', 'uav0000295_02300_v', 'uav0000307_00000_v', 'uav0000315_00000_v',
            'uav0000323_01173_v',  # 'uav0000300_00000_v', excluded for less cars to other ratio
            'uav0000326_01035_v', 'uav0000342_04692_v', 'uav0000352_05980_v', 'uav0000361_02323_v',
            'uav0000366_00001_v'
        ]
        seqs.sort()

        objind = 0
        for seq in seqs:
            objs = os.listdir(os.path.join(self.dataset_dir, seq))
            objs.sort()
            for obj in objs:
                imgs = os.listdir(os.path.join(self.dataset_dir, seq, obj))
                imgs.sort()

                if len(imgs) > 2:
                    q = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[q])
                    entry = [img_path, objind, seqs.index(seq) + 100]
                    query.append(entry)
                    imgs.pop(q)

                    g = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[g])
                    entry = [img_path, objind, seqs.index(seq)]
                    gallery.append(entry)
                    imgs.pop(g)

                for img in imgs:
                    img_path = os.path.join(self.dataset_dir, seq, obj, img)
                    entry = [img_path, objind, seqs.index(seq)]
                    train.append(entry)
                objind += 1
        super(VisDroneTrain20c, self).__init__(train, query, gallery, **kwargs)


class VisDroneTrain25a(torchreid.data.ImageDataset):
    # dataset_dir = '/home/jatin/HDD2TB/datasets/UAVDT_reid'
    dataset_dir = args.data

    def __init__(self, root='', mode='train', **kwargs):
        train = []
        query = []
        gallery = []

        # Train
        print('Skipping some real sequences to balance the total number of samples')
        seqs = ['uav0000124_00944_v', 'uav0000126_00001_v', 'uav0000140_01590_v', 'uav0000143_02250_v',
                'uav0000145_00000_v',
                'uav0000218_00001_v', 'uav0000222_03150_v', 'uav0000239_03720_v', 'uav0000239_12336_v',
                'uav0000243_00001_v',
                'uav0000244_01440_v', 'uav0000248_00001_v', 'uav0000263_03289_v', 'uav0000264_02760_v',
                'uav0000270_00001_v',
                'uav0000273_00001_v', 'uav0000278_00001_v', 'uav0000279_00001_v', 'uav0000281_00460_v',
                'uav0000289_00001_v',
                'uav0000289_06922_v', 'uav0000295_02300_v', 'uav0000307_00000_v', 'uav0000315_00000_v',
                'uav0000323_01173_v',  # 'uav0000300_00000_v', excluded for less cars to other ratio
                # 'uav0000326_01035_v', 'uav0000342_04692_v', 'uav0000352_05980_v', 'uav0000361_02323_v', 'uav0000366_00001_v'
                ]
        seqs.sort()

        objind = 0
        for seq in seqs:
            objs = os.listdir(os.path.join(self.dataset_dir, seq))
            objs.sort()
            for obj in objs:
                imgs = os.listdir(os.path.join(self.dataset_dir, seq, obj))
                imgs.sort()

                if len(imgs) > 2:
                    q = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[q])
                    entry = [img_path, objind, seqs.index(seq) + 100]
                    query.append(entry)
                    imgs.pop(q)

                    g = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[g])
                    entry = [img_path, objind, seqs.index(seq)]
                    gallery.append(entry)
                    imgs.pop(g)

                for img in imgs:
                    img_path = os.path.join(self.dataset_dir, seq, obj, img)
                    entry = [img_path, objind, seqs.index(seq)]
                    train.append(entry)
                objind += 1
        super(VisDroneTrain25a, self).__init__(train, query, gallery, **kwargs)


class VisDroneTrain25b(torchreid.data.ImageDataset):
    # dataset_dir = '/home/jatin/HDD2TB/datasets/UAVDT_reid'
    dataset_dir = args.data

    def __init__(self, root='', mode='train', **kwargs):
        train = []
        query = []
        gallery = []

        # Train
        print('Skipping some real sequences to balance the total number of samples')
        seqs = ['uav0000124_00944_v', 'uav0000126_00001_v', 'uav0000140_01590_v', 'uav0000143_02250_v',
                'uav0000145_00000_v',
                'uav0000218_00001_v', 'uav0000222_03150_v', 'uav0000239_03720_v', 'uav0000239_12336_v',
                'uav0000243_00001_v',
                'uav0000244_01440_v', 'uav0000248_00001_v', 'uav0000263_03289_v', 'uav0000264_02760_v',
                'uav0000270_00001_v',
                'uav0000273_00001_v', 'uav0000278_00001_v', 'uav0000279_00001_v', 'uav0000281_00460_v',
                'uav0000289_00001_v',
                # 'uav0000289_06922_v', 'uav0000295_02300_v', 'uav0000307_00000_v', 'uav0000315_00000_v', 'uav0000323_01173_v', #'uav0000300_00000_v', excluded for less cars to other ratio
                'uav0000326_01035_v', 'uav0000342_04692_v', 'uav0000352_05980_v', 'uav0000361_02323_v',
                'uav0000366_00001_v'
                ]
        seqs.sort()

        objind = 0
        for seq in seqs:
            objs = os.listdir(os.path.join(self.dataset_dir, seq))
            objs.sort()
            for obj in objs:
                imgs = os.listdir(os.path.join(self.dataset_dir, seq, obj))
                imgs.sort()

                if len(imgs) > 2:
                    q = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[q])
                    entry = [img_path, objind, seqs.index(seq) + 100]
                    query.append(entry)
                    imgs.pop(q)

                    g = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[g])
                    entry = [img_path, objind, seqs.index(seq)]
                    gallery.append(entry)
                    imgs.pop(g)

                for img in imgs:
                    img_path = os.path.join(self.dataset_dir, seq, obj, img)
                    entry = [img_path, objind, seqs.index(seq)]
                    train.append(entry)
                objind += 1
        super(VisDroneTrain25b, self).__init__(train, query, gallery, **kwargs)


class VisDroneTrain25c(torchreid.data.ImageDataset):
    # dataset_dir = '/home/jatin/HDD2TB/datasets/UAVDT_reid'
    dataset_dir = args.data

    def __init__(self, root='', mode='train', **kwargs):
        train = []
        query = []
        gallery = []

        # Train
        print('Skipping some real sequences to balance the total number of samples')
        seqs = ['uav0000124_00944_v', 'uav0000126_00001_v', 'uav0000140_01590_v', 'uav0000143_02250_v',
                'uav0000145_00000_v',
                'uav0000218_00001_v', 'uav0000222_03150_v', 'uav0000239_03720_v', 'uav0000239_12336_v',
                'uav0000243_00001_v',
                'uav0000244_01440_v', 'uav0000248_00001_v', 'uav0000263_03289_v', 'uav0000264_02760_v',
                'uav0000270_00001_v',
                # 'uav0000273_00001_v', 'uav0000278_00001_v', 'uav0000279_00001_v', 'uav0000281_00460_v', 'uav0000289_00001_v',
                'uav0000289_06922_v', 'uav0000295_02300_v', 'uav0000307_00000_v', 'uav0000315_00000_v',
                'uav0000323_01173_v',  # 'uav0000300_00000_v', excluded for less cars to other ratio
                'uav0000326_01035_v', 'uav0000342_04692_v', 'uav0000352_05980_v', 'uav0000361_02323_v',
                'uav0000366_00001_v'
                ]
        seqs.sort()

        objind = 0
        for seq in seqs:
            objs = os.listdir(os.path.join(self.dataset_dir, seq))
            objs.sort()
            for obj in objs:
                imgs = os.listdir(os.path.join(self.dataset_dir, seq, obj))
                imgs.sort()

                if len(imgs) > 2:
                    q = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[q])
                    entry = [img_path, objind, seqs.index(seq) + 100]
                    query.append(entry)
                    imgs.pop(q)

                    g = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[g])
                    entry = [img_path, objind, seqs.index(seq)]
                    gallery.append(entry)
                    imgs.pop(g)

                for img in imgs:
                    img_path = os.path.join(self.dataset_dir, seq, obj, img)
                    entry = [img_path, objind, seqs.index(seq)]
                    train.append(entry)
                objind += 1
        super(VisDroneTrain25c, self).__init__(train, query, gallery, **kwargs)


class VisDroneTrain25d(torchreid.data.ImageDataset):
    # dataset_dir = '/home/jatin/HDD2TB/datasets/UAVDT_reid'
    dataset_dir = args.data

    def __init__(self, root='', mode='train', **kwargs):
        train = []
        query = []
        gallery = []

        # Train
        print('Skipping some real sequences to balance the total number of samples')
        seqs = ['uav0000124_00944_v', 'uav0000126_00001_v', 'uav0000140_01590_v', 'uav0000143_02250_v',
                'uav0000145_00000_v',
                'uav0000218_00001_v', 'uav0000222_03150_v', 'uav0000239_03720_v', 'uav0000239_12336_v',
                'uav0000243_00001_v',
                # 'uav0000244_01440_v', 'uav0000248_00001_v', 'uav0000263_03289_v', 'uav0000264_02760_v', 'uav0000270_00001_v',
                'uav0000273_00001_v', 'uav0000278_00001_v', 'uav0000279_00001_v', 'uav0000281_00460_v',
                'uav0000289_00001_v',
                'uav0000289_06922_v', 'uav0000295_02300_v', 'uav0000307_00000_v', 'uav0000315_00000_v',
                'uav0000323_01173_v',  # 'uav0000300_00000_v', excluded for less cars to other ratio
                'uav0000326_01035_v', 'uav0000342_04692_v', 'uav0000352_05980_v', 'uav0000361_02323_v',
                'uav0000366_00001_v'
                ]
        seqs.sort()

        objind = 0
        for seq in seqs:
            objs = os.listdir(os.path.join(self.dataset_dir, seq))
            objs.sort()
            for obj in objs:
                imgs = os.listdir(os.path.join(self.dataset_dir, seq, obj))
                imgs.sort()

                if len(imgs) > 2:
                    q = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[q])
                    entry = [img_path, objind, seqs.index(seq) + 100]
                    query.append(entry)
                    imgs.pop(q)

                    g = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[g])
                    entry = [img_path, objind, seqs.index(seq)]
                    gallery.append(entry)
                    imgs.pop(g)

                for img in imgs:
                    img_path = os.path.join(self.dataset_dir, seq, obj, img)
                    entry = [img_path, objind, seqs.index(seq)]
                    train.append(entry)
                objind += 1
        super(VisDroneTrain25d, self).__init__(train, query, gallery, **kwargs)


class VisDroneTrain25e(torchreid.data.ImageDataset):
    # dataset_dir = '/home/jatin/HDD2TB/datasets/UAVDT_reid'
    dataset_dir = args.data

    def __init__(self, root='', mode='train', **kwargs):
        train = []
        query = []
        gallery = []

        # Train
        print('Skipping some real sequences to balance the total number of samples')
        seqs = ['uav0000124_00944_v', 'uav0000126_00001_v', 'uav0000140_01590_v', 'uav0000143_02250_v',
                'uav0000145_00000_v',
                # 'uav0000218_00001_v', 'uav0000222_03150_v', 'uav0000239_03720_v', 'uav0000239_12336_v', 'uav0000243_00001_v',
                'uav0000244_01440_v', 'uav0000248_00001_v', 'uav0000263_03289_v', 'uav0000264_02760_v',
                'uav0000270_00001_v',
                'uav0000273_00001_v', 'uav0000278_00001_v', 'uav0000279_00001_v', 'uav0000281_00460_v',
                'uav0000289_00001_v',
                'uav0000289_06922_v', 'uav0000295_02300_v', 'uav0000307_00000_v', 'uav0000315_00000_v',
                'uav0000323_01173_v',  # 'uav0000300_00000_v', excluded for less cars to other ratio
                'uav0000326_01035_v', 'uav0000342_04692_v', 'uav0000352_05980_v', 'uav0000361_02323_v',
                'uav0000366_00001_v'
                ]
        seqs.sort()

        objind = 0
        for seq in seqs:
            objs = os.listdir(os.path.join(self.dataset_dir, seq))
            objs.sort()
            for obj in objs:
                imgs = os.listdir(os.path.join(self.dataset_dir, seq, obj))
                imgs.sort()

                if len(imgs) > 2:
                    q = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[q])
                    entry = [img_path, objind, seqs.index(seq) + 100]
                    query.append(entry)
                    imgs.pop(q)

                    g = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[g])
                    entry = [img_path, objind, seqs.index(seq)]
                    gallery.append(entry)
                    imgs.pop(g)

                for img in imgs:
                    img_path = os.path.join(self.dataset_dir, seq, obj, img)
                    entry = [img_path, objind, seqs.index(seq)]
                    train.append(entry)
                objind += 1
        super(VisDroneTrain25e, self).__init__(train, query, gallery, **kwargs)


class VisDroneTrain25f(torchreid.data.ImageDataset):
    # dataset_dir = '/home/jatin/HDD2TB/datasets/UAVDT_reid'
    dataset_dir = args.data

    def __init__(self, root='', mode='train', **kwargs):
        train = []
        query = []
        gallery = []

        # Train
        print('Skipping some real sequences to balance the total number of samples')
        seqs = [
            # 'uav0000124_00944_v', 'uav0000126_00001_v', 'uav0000140_01590_v', 'uav0000143_02250_v', 'uav0000145_00000_v',
            'uav0000218_00001_v', 'uav0000222_03150_v', 'uav0000239_03720_v', 'uav0000239_12336_v',
            'uav0000243_00001_v',
            'uav0000244_01440_v', 'uav0000248_00001_v', 'uav0000263_03289_v', 'uav0000264_02760_v',
            'uav0000270_00001_v',
            'uav0000273_00001_v', 'uav0000278_00001_v', 'uav0000279_00001_v', 'uav0000281_00460_v',
            'uav0000289_00001_v',
            'uav0000289_06922_v', 'uav0000295_02300_v', 'uav0000307_00000_v', 'uav0000315_00000_v',
            'uav0000323_01173_v',  # 'uav0000300_00000_v', excluded for less cars to other ratio
            'uav0000326_01035_v', 'uav0000342_04692_v', 'uav0000352_05980_v', 'uav0000361_02323_v',
            'uav0000366_00001_v'
        ]
        seqs.sort()

        objind = 0
        for seq in seqs:
            objs = os.listdir(os.path.join(self.dataset_dir, seq))
            objs.sort()
            for obj in objs:
                imgs = os.listdir(os.path.join(self.dataset_dir, seq, obj))
                imgs.sort()

                if len(imgs) > 2:
                    q = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[q])
                    entry = [img_path, objind, seqs.index(seq) + 100]
                    query.append(entry)
                    imgs.pop(q)

                    g = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[g])
                    entry = [img_path, objind, seqs.index(seq)]
                    gallery.append(entry)
                    imgs.pop(g)

                for img in imgs:
                    img_path = os.path.join(self.dataset_dir, seq, obj, img)
                    entry = [img_path, objind, seqs.index(seq)]
                    train.append(entry)
                objind += 1
        super(VisDroneTrain25f, self).__init__(train, query, gallery, **kwargs)

class AirSimReIdTrain5a(torchreid.data.ImageDataset):
    #dataset_dir = '/home/jatin/HDD2TB/datasets/Synthetic/AirSim-UAV-Synth/reid-data'
    dataset_dir = args.syndata
    def __init__(self, root='', **kwargs):
        train = []
        query = []
        gallery = []

        #Train
        #seqs = os.listdir(self.dataset_dir)
        seqs = ['0', '1', '2', '3', '4',
                #'5', '6', '7', '8', '9',
                #'10', '11', '12', '13', '14'
                ]
        seqs.sort()

        objind = 0
        for seq in seqs:
            objs = os.listdir(os.path.join(self.dataset_dir,seq))
            objs.sort()
            for obj in objs:
                imgs = os.listdir(os.path.join(self.dataset_dir,seq,obj))
                imgs.sort()

                if len(imgs)>2:
                    q = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[q])
                    entry = [img_path, objind, seqs.index(seq)+100]
                    query.append(entry)
                    imgs.pop(q)

                    g = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[g])
                    entry = [img_path, objind, seqs.index(seq)]
                    gallery.append(entry)
                    imgs.pop(g)

                for img in imgs:
                    img_path = os.path.join(self.dataset_dir,seq,obj,img)
                    entry = [img_path, objind, seqs.index(seq)]
                    train.append(entry)
                objind += 1
        super(AirSimReIdTrain5a, self).__init__(train, query, gallery, **kwargs)

class AirSimReIdTrain5b(torchreid.data.ImageDataset):
    #dataset_dir = '/home/jatin/HDD2TB/datasets/Synthetic/AirSim-UAV-Synth/reid-data'
    dataset_dir = args.syndata
    def __init__(self, root='', **kwargs):
        train = []
        query = []
        gallery = []

        #Train
        #seqs = os.listdir(self.dataset_dir)
        seqs = [#'0', '1', '2', '3', '4',
                '5', '6', '7', '8', '9',
                #'10', '11', '12', '13', '14'
                ]
        seqs.sort()

        objind = 0
        for seq in seqs:
            objs = os.listdir(os.path.join(self.dataset_dir,seq))
            objs.sort()
            for obj in objs:
                imgs = os.listdir(os.path.join(self.dataset_dir,seq,obj))
                imgs.sort()

                if len(imgs)>2:
                    q = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[q])
                    entry = [img_path, objind, seqs.index(seq)+100]
                    query.append(entry)
                    imgs.pop(q)

                    g = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[g])
                    entry = [img_path, objind, seqs.index(seq)]
                    gallery.append(entry)
                    imgs.pop(g)

                for img in imgs:
                    img_path = os.path.join(self.dataset_dir,seq,obj,img)
                    entry = [img_path, objind, seqs.index(seq)]
                    train.append(entry)
                objind += 1
        super(AirSimReIdTrain5b, self).__init__(train, query, gallery, **kwargs)

class AirSimReIdTrain5c(torchreid.data.ImageDataset):
    #dataset_dir = '/home/jatin/HDD2TB/datasets/Synthetic/AirSim-UAV-Synth/reid-data'
    dataset_dir = args.syndata
    def __init__(self, root='', **kwargs):
        train = []
        query = []
        gallery = []

        #Train
        #seqs = os.listdir(self.dataset_dir)
        seqs = [#'0', '1', '2', '3', '4',
                #'5', '6', '7', '8', '9',
                '10', '11', '12', '13', '14'
                ]
        seqs.sort()

        objind = 0
        for seq in seqs:
            objs = os.listdir(os.path.join(self.dataset_dir,seq))
            objs.sort()
            for obj in objs:
                imgs = os.listdir(os.path.join(self.dataset_dir,seq,obj))
                imgs.sort()

                if len(imgs)>2:
                    q = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[q])
                    entry = [img_path, objind, seqs.index(seq)+100]
                    query.append(entry)
                    imgs.pop(q)

                    g = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[g])
                    entry = [img_path, objind, seqs.index(seq)]
                    gallery.append(entry)
                    imgs.pop(g)

                for img in imgs:
                    img_path = os.path.join(self.dataset_dir,seq,obj,img)
                    entry = [img_path, objind, seqs.index(seq)]
                    train.append(entry)
                objind += 1
        super(AirSimReIdTrain5c, self).__init__(train, query, gallery, **kwargs)

class AirSimReIdTrain10a(torchreid.data.ImageDataset):
    #dataset_dir = '/home/jatin/HDD2TB/datasets/Synthetic/AirSim-UAV-Synth/reid-data'
    dataset_dir = args.syndata
    def __init__(self, root='', **kwargs):
        train = []
        query = []
        gallery = []

        #Train
        #seqs = os.listdir(self.dataset_dir)
        seqs = ['0', '1', '2', '3', '4',
                '5', '6', '7', '8', '9',
                #'10', '11', '12', '13', '14'
                ]
        seqs.sort()

        objind = 0
        for seq in seqs:
            objs = os.listdir(os.path.join(self.dataset_dir,seq))
            objs.sort()
            for obj in objs:
                imgs = os.listdir(os.path.join(self.dataset_dir,seq,obj))
                imgs.sort()

                if len(imgs)>2:
                    q = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[q])
                    entry = [img_path, objind, seqs.index(seq)+100]
                    query.append(entry)
                    imgs.pop(q)

                    g = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[g])
                    entry = [img_path, objind, seqs.index(seq)]
                    gallery.append(entry)
                    imgs.pop(g)

                for img in imgs:
                    img_path = os.path.join(self.dataset_dir,seq,obj,img)
                    entry = [img_path, objind, seqs.index(seq)]
                    train.append(entry)
                objind += 1
        super(AirSimReIdTrain10a, self).__init__(train, query, gallery, **kwargs)

class AirSimReIdTrain10b(torchreid.data.ImageDataset):
    #dataset_dir = '/home/jatin/HDD2TB/datasets/Synthetic/AirSim-UAV-Synth/reid-data'
    dataset_dir = args.syndata
    def __init__(self, root='', **kwargs):
        train = []
        query = []
        gallery = []

        #Train
        #seqs = os.listdir(self.dataset_dir)
        seqs = ['0', '1', '2', '3', '4',
                #'5', '6', '7', '8', '9',
                '10', '11', '12', '13', '14'
                ]
        seqs.sort()

        objind = 0
        for seq in seqs:
            objs = os.listdir(os.path.join(self.dataset_dir,seq))
            objs.sort()
            for obj in objs:
                imgs = os.listdir(os.path.join(self.dataset_dir,seq,obj))
                imgs.sort()

                if len(imgs)>2:
                    q = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[q])
                    entry = [img_path, objind, seqs.index(seq)+100]
                    query.append(entry)
                    imgs.pop(q)

                    g = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[g])
                    entry = [img_path, objind, seqs.index(seq)]
                    gallery.append(entry)
                    imgs.pop(g)

                for img in imgs:
                    img_path = os.path.join(self.dataset_dir,seq,obj,img)
                    entry = [img_path, objind, seqs.index(seq)]
                    train.append(entry)
                objind += 1
        super(AirSimReIdTrain10b, self).__init__(train, query, gallery, **kwargs)

class AirSimReIdTrain10c(torchreid.data.ImageDataset):
    #dataset_dir = '/home/jatin/HDD2TB/datasets/Synthetic/AirSim-UAV-Synth/reid-data'
    dataset_dir = args.syndata
    def __init__(self, root='', **kwargs):
        train = []
        query = []
        gallery = []

        #Train
        #seqs = os.listdir(self.dataset_dir)
        seqs = [#'0', '1', '2', '3', '4',
                '5', '6', '7', '8', '9',
                '10', '11', '12', '13', '14'
                ]
        seqs.sort()

        objind = 0
        for seq in seqs:
            objs = os.listdir(os.path.join(self.dataset_dir,seq))
            objs.sort()
            for obj in objs:
                imgs = os.listdir(os.path.join(self.dataset_dir,seq,obj))
                imgs.sort()

                if len(imgs)>2:
                    q = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[q])
                    entry = [img_path, objind, seqs.index(seq)+100]
                    query.append(entry)
                    imgs.pop(q)

                    g = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[g])
                    entry = [img_path, objind, seqs.index(seq)]
                    gallery.append(entry)
                    imgs.pop(g)

                for img in imgs:
                    img_path = os.path.join(self.dataset_dir,seq,obj,img)
                    entry = [img_path, objind, seqs.index(seq)]
                    train.append(entry)
                objind += 1
        super(AirSimReIdTrain10c, self).__init__(train, query, gallery, **kwargs)

class AirSimReIdTrain(torchreid.data.ImageDataset):
    #dataset_dir = '/home/jatin/HDD2TB/datasets/Synthetic/AirSim-UAV-Synth/reid-data'
    dataset_dir = args.syndata
    def __init__(self, root='', **kwargs):
        train = []
        query = []
        gallery = []

        #Train
        #seqs = os.listdir(self.dataset_dir)
        seqs = ['0', '1', '2', '3', '4',
                '5', '6', '7', '8', '9',
                '10', '11', '12', '13', '14'
                ]
        seqs.sort()

        objind = 0
        for seq in seqs:
            objs = os.listdir(os.path.join(self.dataset_dir,seq))
            objs.sort()
            for obj in objs:
                imgs = os.listdir(os.path.join(self.dataset_dir,seq,obj))
                imgs.sort()

                if len(imgs)>2:
                    q = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[q])
                    entry = [img_path, objind, seqs.index(seq)+100]
                    query.append(entry)
                    imgs.pop(q)

                    g = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[g])
                    entry = [img_path, objind, seqs.index(seq)]
                    gallery.append(entry)
                    imgs.pop(g)

                for img in imgs:
                    img_path = os.path.join(self.dataset_dir,seq,obj,img)
                    entry = [img_path, objind, seqs.index(seq)]
                    train.append(entry)
                objind += 1
        super(AirSimReIdTrain, self).__init__(train, query, gallery, **kwargs)

class AirSimReIdTrain20(torchreid.data.ImageDataset):
    #dataset_dir = '/home/jatin/HDD2TB/datasets/Synthetic/AirSim-UAV-Synth/reid-data'
    dataset_dir = args.syndata
    def __init__(self, root='', **kwargs):
        train = []
        query = []
        gallery = []

        #Train
        #seqs = os.listdir(self.dataset_dir)
        seqs = ['0', '1', '2', '3', '4',
                '5', '6', '7', '8', '9',
                '10', '11', '12', '13', '14',
                '15', '16', '17', '18', '19'
                ]
        seqs.sort()

        objind = 0
        for seq in seqs:
            objs = os.listdir(os.path.join(self.dataset_dir,seq))
            objs.sort()
            for obj in objs:
                imgs = os.listdir(os.path.join(self.dataset_dir,seq,obj))
                imgs.sort()

                if len(imgs)>2:
                    q = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[q])
                    entry = [img_path, objind, seqs.index(seq)+100]
                    query.append(entry)
                    imgs.pop(q)

                    g = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[g])
                    entry = [img_path, objind, seqs.index(seq)]
                    gallery.append(entry)
                    imgs.pop(g)

                for img in imgs:
                    img_path = os.path.join(self.dataset_dir,seq,obj,img)
                    entry = [img_path, objind, seqs.index(seq)]
                    train.append(entry)
                objind += 1
        super(AirSimReIdTrain20, self).__init__(train, query, gallery, **kwargs)

class AirSimReIdTrain25(torchreid.data.ImageDataset):
    #dataset_dir = '/home/jatin/HDD2TB/datasets/Synthetic/AirSim-UAV-Synth/reid-data'
    dataset_dir = args.syndata
    def __init__(self, root='', **kwargs):
        train = []
        query = []
        gallery = []

        #Train
        #seqs = os.listdir(self.dataset_dir)
        seqs = ['0', '1', '2', '3', '4',
                '5', '6', '7', '8', '9',
                '10', '11', '12', '13', '14',
                '15', '16', '17', '18', '19',
                '20', '21', '22', '23', '24'
                ]
        seqs.sort()

        objind = 0
        for seq in seqs:
            objs = os.listdir(os.path.join(self.dataset_dir,seq))
            objs.sort()
            for obj in objs:
                imgs = os.listdir(os.path.join(self.dataset_dir,seq,obj))
                imgs.sort()

                if len(imgs)>2:
                    q = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[q])
                    entry = [img_path, objind, seqs.index(seq)+100]
                    query.append(entry)
                    imgs.pop(q)

                    g = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[g])
                    entry = [img_path, objind, seqs.index(seq)]
                    gallery.append(entry)
                    imgs.pop(g)

                for img in imgs:
                    img_path = os.path.join(self.dataset_dir,seq,obj,img)
                    entry = [img_path, objind, seqs.index(seq)]
                    train.append(entry)
                objind += 1
        super(AirSimReIdTrain25, self).__init__(train, query, gallery, **kwargs)

class UAVDTReIdTest(torchreid.data.ImageDataset):
    #dataset_dir = '/home/jatin/HDD2TB/datasets/UAVDT_reid'
    dataset_dir = args.data

    def __init__(self, root='', **kwargs):
        train = []
        query = []
        gallery = []

        # Test
        seqs = ['M0203', 'M0205', 'M0208', 'M0209', 'M0403', 'M0601', 'M0602', 'M0606', 'M0701', 'M0801',
                'M0802', 'M1001', 'M1004', 'M1007', 'M1009', 'M1101', 'M1301', 'M1302', 'M1303', 'M1401']
        seqs.sort()

        objind = 0
        for seq in seqs:
            objs = os.listdir(os.path.join(self.dataset_dir, seq))
            objs.sort()
            for obj in objs:
                imgs = os.listdir(os.path.join(self.dataset_dir, seq, obj))
                imgs.sort()

                if len(imgs) > 2:
                    random.seed(0)
                    q = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[q])
                    entry = [img_path, objind, seqs.index(seq) + 100]
                    query.append(entry)
                    imgs.pop(q)

                    g = random.randint(0, len(imgs) - 1)
                    img_path = os.path.join(self.dataset_dir, seq, obj, imgs[g])
                    entry = [img_path, objind, seqs.index(seq)]
                    gallery.append(entry)
                    imgs.pop(g)

                for img in imgs:
                    img_path = os.path.join(self.dataset_dir, seq, obj, img)
                    entry = [img_path, objind, seqs.index(seq)]
                    train.append(entry)
                objind += 1
        super(UAVDTReIdTest, self).__init__(train, query, gallery, **kwargs)


if __name__=='__main__':
    #import argparse

    #argsparser = argparse.ArgumentParser(description='Just passing some parameters')
    #argsparser.add_argument('--data', help='path to real data', default=None)
    #argsparser.add_argument('--syndata', help='path to synthetic data', default=None)
    #argsparser.add_argument('--model', help='path to model weights', default=None)
    #argsparser.add_argument('--mode', help='how to use real and synthetic datasets', default=None)
    #args = argsparser.parse_args()

    torchreid.data.register_image_dataset('UAVDT_reid_train', UAVDTReIdTrain)
    torchreid.data.register_image_dataset('UAVDT_reid_train5a', UAVDTReIdTrain5a)
    torchreid.data.register_image_dataset('UAVDT_reid_train5b', UAVDTReIdTrain5b)
    torchreid.data.register_image_dataset('UAVDT_reid_train5c', UAVDTReIdTrain5c)
    torchreid.data.register_image_dataset('UAVDT_reid_train5d', UAVDTReIdTrain5d)
    torchreid.data.register_image_dataset('UAVDT_reid_train5e', UAVDTReIdTrain5e)
    torchreid.data.register_image_dataset('UAVDT_reid_train5f', UAVDTReIdTrain5f)
    torchreid.data.register_image_dataset('UAVDT_reid_train10a', UAVDTReIdTrain10a)
    torchreid.data.register_image_dataset('UAVDT_reid_train10b', UAVDTReIdTrain10b)
    torchreid.data.register_image_dataset('UAVDT_reid_train10c', UAVDTReIdTrain10c)
    torchreid.data.register_image_dataset('UAVDT_reid_train15a', UAVDTReIdTrain15a)
    torchreid.data.register_image_dataset('UAVDT_reid_train15b', UAVDTReIdTrain15b)
    torchreid.data.register_image_dataset('UAVDT_reid_train20a', UAVDTReIdTrain20a)
    torchreid.data.register_image_dataset('UAVDT_reid_train20b', UAVDTReIdTrain20b)
    torchreid.data.register_image_dataset('UAVDT_reid_train20c', UAVDTReIdTrain20c)
    torchreid.data.register_image_dataset('UAVDT_reid_train25a', UAVDTReIdTrain25a)
    torchreid.data.register_image_dataset('UAVDT_reid_train25b', UAVDTReIdTrain25b)
    torchreid.data.register_image_dataset('UAVDT_reid_train25c', UAVDTReIdTrain25c)
    torchreid.data.register_image_dataset('UAVDT_reid_train25d', UAVDTReIdTrain25d)
    torchreid.data.register_image_dataset('UAVDT_reid_train25e', UAVDTReIdTrain25e)
    torchreid.data.register_image_dataset('UAVDT_reid_train25f', UAVDTReIdTrain25f)

    torchreid.data.register_image_dataset('VisDrone_reid_train', VisDroneTrain)
    torchreid.data.register_image_dataset('VisDrone_reid_train5a', VisDroneTrain5a)
    torchreid.data.register_image_dataset('VisDrone_reid_train5b', VisDroneTrain5b)
    torchreid.data.register_image_dataset('VisDrone_reid_train5c', VisDroneTrain5c)
    torchreid.data.register_image_dataset('VisDrone_reid_train5d', VisDroneTrain5d)
    torchreid.data.register_image_dataset('VisDrone_reid_train5e', VisDroneTrain5e)
    torchreid.data.register_image_dataset('VisDrone_reid_train5f', VisDroneTrain5f)
    torchreid.data.register_image_dataset('VisDrone_reid_train10a', VisDroneTrain10a)
    torchreid.data.register_image_dataset('VisDrone_reid_train10b', VisDroneTrain10b)
    torchreid.data.register_image_dataset('VisDrone_reid_train10c', VisDroneTrain10c)
    torchreid.data.register_image_dataset('VisDrone_reid_train15a', VisDroneTrain15a)
    torchreid.data.register_image_dataset('VisDrone_reid_train15b', VisDroneTrain15b)
    torchreid.data.register_image_dataset('VisDrone_reid_train20a', VisDroneTrain20a)
    torchreid.data.register_image_dataset('VisDrone_reid_train20b', VisDroneTrain20b)
    torchreid.data.register_image_dataset('VisDrone_reid_train20c', VisDroneTrain20c)
    torchreid.data.register_image_dataset('VisDrone_reid_train25a', VisDroneTrain25a)
    torchreid.data.register_image_dataset('VisDrone_reid_train25b', VisDroneTrain25b)
    torchreid.data.register_image_dataset('VisDrone_reid_train25c', VisDroneTrain25c)
    torchreid.data.register_image_dataset('VisDrone_reid_train25d', VisDroneTrain25d)
    torchreid.data.register_image_dataset('VisDrone_reid_train25e', VisDroneTrain25e)
    torchreid.data.register_image_dataset('VisDrone_reid_train25f', VisDroneTrain25f)

    torchreid.data.register_image_dataset('UAVDT_reid_test', UAVDTReIdTest)

    torchreid.data.register_image_dataset('AirSim_reid_train', AirSimReIdTrain)
    torchreid.data.register_image_dataset('AirSim_reid_train5a', AirSimReIdTrain5a)
    torchreid.data.register_image_dataset('AirSim_reid_train5b', AirSimReIdTrain5b)
    torchreid.data.register_image_dataset('AirSim_reid_train5c', AirSimReIdTrain5c)
    torchreid.data.register_image_dataset('AirSim_reid_train10a', AirSimReIdTrain10a)
    torchreid.data.register_image_dataset('AirSim_reid_train10b', AirSimReIdTrain10b)
    torchreid.data.register_image_dataset('AirSim_reid_train10c', AirSimReIdTrain10c)
    torchreid.data.register_image_dataset('AirSim_reid_train20', AirSimReIdTrain20)
    torchreid.data.register_image_dataset('AirSim_reid_train25', AirSimReIdTrain25)

    if args.dataset_class == 'UAVDT':
        if args.mode == 'R30':
            sources = ['UAVDT_reid_train']
        elif args.mode == 'R30S15':
            sources = ['UAVDT_reid_train','AirSim_reid_train']
        elif args.mode == 'R15aS15':
            sources = ['UAVDT_reid_train15a','AirSim_reid_train']
        elif args.mode == 'R15bS15':
            sources = ['UAVDT_reid_train15b','AirSim_reid_train']
        elif args.mode == 'test':
            sources = ['UAVDT_reid_test']
        elif args.mode == 'R10aS20':
            sources = ['UAVDT_reid_train10a','AirSim_reid_train20']
        elif args.mode == 'R10bS20':
            sources = ['UAVDT_reid_train10b','AirSim_reid_train20']
        elif args.mode == 'R10cS20':
            sources = ['UAVDT_reid_train10c','AirSim_reid_train20']
        elif args.mode == 'R20aS10a':
            sources = ['UAVDT_reid_train20a','AirSim_reid_train10a']
        elif args.mode == 'R20bS10b':
            sources = ['UAVDT_reid_train20b','AirSim_reid_train10b']
        elif args.mode == 'R20cS10c':
            sources = ['UAVDT_reid_train20c','AirSim_reid_train10c']
        elif args.mode == 'R25aS5a':
            sources = ['UAVDT_reid_train25a','AirSim_reid_train5a']
        elif args.mode == 'R25bS5b':
            sources = ['UAVDT_reid_train25b','AirSim_reid_train5b']
        elif args.mode == 'R25cS5c':
            sources = ['UAVDT_reid_train25c','AirSim_reid_train5c']
        elif args.mode == 'R25dS5a':
            sources = ['UAVDT_reid_train25d','AirSim_reid_train5a']
        elif args.mode == 'R25eS5b':
            sources = ['UAVDT_reid_train25e','AirSim_reid_train5b']
        elif args.mode == 'R25fS5c':
            sources = ['UAVDT_reid_train25f','AirSim_reid_train5c']
        elif args.mode == 'R5aS25':
            sources = ['UAVDT_reid_train5a','AirSim_reid_train25']
        elif args.mode == 'R5bS25':
            sources = ['UAVDT_reid_train5b','AirSim_reid_train25']
        elif args.mode == 'R5cS25':
            sources = ['UAVDT_reid_train5c','AirSim_reid_train25']
        elif args.mode == 'R5dS25':
            sources = ['UAVDT_reid_train5d','AirSim_reid_train25']
        elif args.mode == 'R5eS25':
            sources = ['UAVDT_reid_train5e','AirSim_reid_train25']
        elif args.mode == 'R5fS25':
            sources = ['UAVDT_reid_train5f','AirSim_reid_train25']
    elif args.dataset_class == 'VisDrone':
        if args.mode == 'R30':
            sources = ['VisDrone_reid_train']
        elif args.mode == 'R30S15':
            sources = ['VisDrone_reid_train', 'AirSim_reid_train']
        elif args.mode == 'R15aS15':
            sources = ['VisDrone_reid_train15a', 'AirSim_reid_train']
        elif args.mode == 'R15bS15':
            sources = ['VisDrone_reid_train15b', 'AirSim_reid_train']
        elif args.mode == 'test':
            sources = ['VisDrone_reid_test']
        elif args.mode == 'R10aS20':
            sources = ['VisDrone_reid_train10a', 'AirSim_reid_train20']
        elif args.mode == 'R10bS20':
            sources = ['VisDrone_reid_train10b', 'AirSim_reid_train20']
        elif args.mode == 'R10cS20':
            sources = ['VisDrone_reid_train10c', 'AirSim_reid_train20']
        elif args.mode == 'R20aS10a':
            sources = ['VisDrone_reid_train20a', 'AirSim_reid_train10a']
        elif args.mode == 'R20bS10b':
            sources = ['VisDrone_reid_train20b', 'AirSim_reid_train10b']
        elif args.mode == 'R20cS10c':
            sources = ['VisDrone_reid_train20c', 'AirSim_reid_train10c']
        elif args.mode == 'R25aS5a':
            sources = ['VisDrone_reid_train25a', 'AirSim_reid_train5a']
        elif args.mode == 'R25bS5b':
            sources = ['VisDrone_reid_train25b', 'AirSim_reid_train5b']
        elif args.mode == 'R25cS5c':
            sources = ['VisDrone_reid_train25c', 'AirSim_reid_train5c']
        elif args.mode == 'R25dS5a':
            sources = ['VisDrone_reid_train25d', 'AirSim_reid_train5a']
        elif args.mode == 'R25eS5b':
            sources = ['VisDrone_reid_train25e', 'AirSim_reid_train5b']
        elif args.mode == 'R25fS5c':
            sources = ['VisDrone_reid_train25f', 'AirSim_reid_train5c']
        elif args.mode == 'R5aS25':
            sources = ['VisDrone_reid_train5a', 'AirSim_reid_train25']
        elif args.mode == 'R5bS25':
            sources = ['VisDrone_reid_train5b', 'AirSim_reid_train25']
        elif args.mode == 'R5cS25':
            sources = ['VisDrone_reid_train5c', 'AirSim_reid_train25']
        elif args.mode == 'R5dS25':
            sources = ['VisDrone_reid_train5d', 'AirSim_reid_train25']
        elif args.mode == 'R5eS25':
            sources = ['VisDrone_reid_train5e', 'AirSim_reid_train25']
        elif args.mode == 'R5fS25':
            sources = ['VisDrone_reid_train5f', 'AirSim_reid_train25']
        
        datamanager = torchreid.data.ImageDataManager(
        root='reid-data',
        #sources=['UAVDT_reid_train','AirSim_reid_train'],
        sources=sources,
        targets='UAVDT_reid_test'
    )

    print(f'Building model: ')
    model = torchreid.models.build_model(
        name='resnet50_fc512',
        num_classes=datamanager.num_train_pids,
        loss='softmax',
        pretrained=True,
    )
    model.cuda()

    optimizer = torchreid.optim.build_optimizer(
        model,
        optim='amsgrad',
        lr=0.0001,
        staged_lr=True,
        new_layers=['fc', 'classifier']
    )

    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer,
        lr_scheduler='single_step',
        stepsize=10
    )

    if args.model is not None:
        print(f'Resuming: {args.model}')
        start_epoch = torchreid.utils.resume_from_checkpoint(args.model, model, optimizer)
    else:
        start_epoch = 0

    engine = torchreid.engine.ImageSoftmaxEngine(datamanager,model,optimizer,scheduler)
    engine.run(max_epoch=10,save_dir='models/tracktor_reid_'+args.mode,print_freq=200,start_epoch=start_epoch)

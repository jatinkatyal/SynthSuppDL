import os
import shutil
from distutils.dir_util import copy_tree

run = 'gt'
if run=='gt':
    all_path = '/home/jatin/HDD2TB/tracking analysis/TrackEval/data/gt/mot_challenge/UAVDT/UAVDT-all'
    train_path = '/home/jatin/HDD2TB/tracking analysis/TrackEval/data/gt/mot_challenge/UAVDT/UAVDT-train'
    test_path = '/home/jatin/HDD2TB/tracking analysis/TrackEval/data/gt/mot_challenge/UAVDT/UAVDT-test'
    train_seqs = ['M0101', 'M0201', 'M0202', 'M0204', 'M0206',
                  'M0207', 'M0210', 'M0301', 'M0401', 'M0402',
                  'M0501', 'M0603', 'M0604', 'M0605', 'M0702',
                  'M0703', 'M0704', 'M0901', 'M0902', 'M1002',
                  'M1003', 'M1005', 'M1006', 'M1008', 'M1102',
                  'M1201', 'M1202', 'M1304', 'M1305', 'M1306']
    test_seqs = ['M0203', 'M0205', 'M0208', 'M0209', 'M0403', 'M0601', 'M0602', 'M0606', 'M0701', 'M0801',
                 'M0802', 'M1001', 'M1004', 'M1007', 'M1009', 'M1101', 'M1301', 'M1302', 'M1303', 'M1401']

    if not os.path.exists(train_path):
        os.mkdir(train_path)
    if not os.path.exists(test_path):
        os.mkdir(test_path)

    for s in train_seqs:
        copy_tree(os.path.join(all_path, s), os.path.join(train_path, s))

    for s in test_seqs:
        copy_tree(os.path.join(all_path, s), os.path.join(test_path, s))

if run=='exp':
    all_path = '/home/jatin/HDD2TB/tracking analysis/TrackEval/data/trackers/mot_challenge/UAVDT-all'
    train_path = '/home/jatin/HDD2TB/tracking analysis/TrackEval/data/trackers/mot_challenge/UAVDT-train'
    test_path = '/home/jatin/HDD2TB/tracking analysis/TrackEval/data/trackers/mot_challenge/UAVDT-test'
    train_seqs = ['M0101', 'M0201', 'M0202', 'M0204', 'M0206',
                 'M0207', 'M0210', 'M0301', 'M0401', 'M0402',
                 'M0501', 'M0603', 'M0604', 'M0605', 'M0702',
                 'M0703', 'M0704', 'M0901', 'M0902', 'M1002',
                 'M1003', 'M1005', 'M1006', 'M1008', 'M1102',
                 'M1201', 'M1202', 'M1304', 'M1305', 'M1306']
    test_seqs = ['M0203', 'M0205', 'M0208', 'M0209', 'M0403', 'M0601', 'M0602', 'M0606', 'M0701', 'M0801',
                'M0802', 'M1001', 'M1004', 'M1007', 'M1009', 'M1101', 'M1301', 'M1302', 'M1303', 'M1401']

    if not os.path.exists(train_path):
        os.mkdir(train_path)
    if not os.path.exists(test_path):
        os.mkdir(test_path)

    experiments = os.listdir(all_path)
    for e in experiments:
        os.mkdir(os.path.join(train_path, e))
        os.mkdir(os.path.join(train_path, e, 'data'))
        for s in train_seqs:
            shutil.copyfile(os.path.join(all_path, e, 'data',s+'.txt'),os.path.join(train_path, e, 'data',s+'.txt'))


        os.mkdir(os.path.join(test_path, e))
        os.mkdir(os.path.join(test_path, e, 'data'))
        for s in test_seqs:
            shutil.copyfile(os.path.join(all_path, e, 'data',s+'.txt'),os.path.join(test_path, e, 'data',s+'.txt'))
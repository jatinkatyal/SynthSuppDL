import cv2
import os
#import numpy as np
import pandas as pd

dataset = '/home/jatin/InternalHDD/Work/computerVision/Tracking/tracking_analysis/TransTrack/mot/train'
gt = '/home/jatin/InternalHDD/Work/computerVision/Tracking/tracking_analysis/TransTrack/mot/annotations'
results = '/home/jatin/InternalHDD/Work/computerVision/Tracking/tracking_analysis/TransTrack/output/MOT17-train/transformer_r3as18a/data'

MOT_offsets = [300,525,418,262,327,450,375]

seqs = os.listdir(os.path.join(results))
seqs.sort()
for seq in seqs:
    #res = np.loadtxt(os.path.join(results,seq),delimiter=',')
    res = pd.read_csv(os.path.join(results,seq),delimiter=',',names=['frame','objid','x','y','w','h','score','x3','y3','z3'])
    frames = res['frame'].unique()
    frames.sort()

    for f in frames:
        img_pth = os.path.join(dataset,seq[:-4],'img1',str(format(f+MOT_offsets[seqs.index(seq)],'0>6')+'.jpg'))
        img = cv2.imread(img_pth)

        dets = res[res['frame']==f][['x','y','w','h']].to_numpy(dtype=int)
        for det in dets:
            x, y, w, h = det
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

        cv2.imshow('detections',img)
        cv2.waitKey(30)

    #exit(0)

"""
    dataset = Images(args.dataset,seq,args.dataset_type,transform)
    dataloader = DataLoader(dataset,batch_size=1)
    if args.results:
        with open(os.path.join(args.results,seq+'.json'),'rb') as resf:
            res = pickle.load(resf)
    for batch,(X,y) in enumerate(dataloader):
        frame = X.squeeze().permute([1,2,0]).numpy().copy()
        annot = y.squeeze()
        #GT
        for bbox in annot:
            x1,y1,x2,y2 = bbox
            x1, y1, x2, y2 = int(x1),int(y1), int(x2), int(y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 1, 0), 2)
        # Predicted _resfile
        if args.results:
            for oid in res.keys():
                if batch in res[oid].keys():
                    x1, y1, x2, y2, _ = res[oid][batch]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    # if _ > 0.2:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (1, 0, 0), 2)
                    cv2.putText(frame,str(oid),(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),4,cv2.LINE_AA)
        # RT detection
        if args.objdet:
            preds = det_model(X.cuda().float())
            for bbox in preds[0]['boxes']:
                x1, y1, x2, y2, = bbox
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 1), 2)
        cv2.imshow(seq,frame)
        cv2.waitKey(50)

'''
--dataset="/home/jatin/HDD2TB/datasets/VisDrone/VisDrone2019-MOT-test-dev"
--dataset_type="VisDrone"
--results="/home/jatin/HDD2TB/tracking analysis/results/results-VisDrone/results/tracktor/VisDrone/R10aS20-det_n_R10aS20-reid"
--objdet="/home/jatin/HDD2TB/tracking analysis models/R10aS20_epoch_30.model"
'''
'''
--dataset="/home/jatin/HDD2TB/datasets/UAVDT"
--dataset_type="UAVDT"
--results="/home/jatin/HDD2TB/tracking analysis/results/UAV15_n_synth15-det_N_UAV-reid"
--objdet="/home/jatin/HDD2TB/tracking analysis models/R10aS20_epoch_30.model"
'''"""
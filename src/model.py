import torch
from torch import nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class Tracktor(nn.Module):
    def __init__(self):
        super(Tracktor, self).__init__()
        self.backbone = fasterrcnn_resnet50_fpn(pretrained=True)
        in_fts = self.backbone.roi_heads.box_predictor.cls_score.in_features
        self.backbone.roi_heads.box_predictor = FastRCNNPredictor(in_fts, 2)
        self.backbone.roi_heads.nms_thresh = 0.3

    def forward(self, X, targets):
        if self.training:
            detections = self.backbone(X,targets)
        else:
            detections = self.backbone(X)
        return detections

if __name__ == '__main__':
    import argparse
    argsparser = argparse.ArgumentParser(description='Just passing some parameters')
    argsparser.add_argument('--dataset',
                            default='/home/jatin/InternalHDD/Work/computerVision/Datasets/UAVDT',
                            help='path to dataset')
    args = argsparser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Tracktor().to(device)#.cpu()
    model.eval()

    from datasets import UAVDTObjDet, plot, collate_fn
    from torch.utils.data import DataLoader
    dataset = UAVDTObjDet(args.dataset)

    loader = DataLoader(dataset,batch_size=4,collate_fn=collate_fn)

    for i,(imgs,targets) in enumerate(loader):
        out = model(imgs,targets)
        break
    print(out)

    for i in range(0,len(imgs)):
        plot(imgs[i],out[i]['boxes'].detach().cpu())

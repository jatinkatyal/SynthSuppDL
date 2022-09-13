#!/bin/bash

files=$(ls)

# Tracktor
if [[ "$files" == *"tracking_wo_bnw"* ]]; then
    echo "tracking_wo_bnw: already present, no download required"
else
    echo "tracking_wo_bnw: download required, cloning now"
    git clone https://github.com/phil-bergmann/tracking_wo_bnw.git
    cp tracking_wo_bnw/src/obj_det/utils.py src/utils/utils.py
    cp tracking_wo_bnw/src/obj_det/engine.py src/utils/engine.py
    cp tracking_wo_bnw/src/obj_det/transforms.py src/utils/transforms.py
    cp tracking_wo_bnw/src/obj_det/coco_utils.py src/utils/coco_utils.py
    cp tracking_wo_bnw/src/obj_det/coco_eval.py src/utils/coco_eval.py
fi

# deep-person-reid
if [[ "$files" == *"deep-person-reid"* ]]; then
    echo "deep-person-reid: already present, no download required"
else
    echo "deep-person-reid: download required, cloning now"
    git clone https://github.com/KaiyangZhou/deep-person-reid.git
    cd deep-person-reid
    pip install -r requirements.txt
    python setup.py develop
    cd ..
fi

# CenterTrack
if [[ "$files" == *"CenterTrack"* ]]; then
    echo "CenterTrack: already present, no download required"
else
    echo "CenterTrack: download required, cloning now"
    git clone https://github.com/xingyizhou/CenterTrack.git
fi
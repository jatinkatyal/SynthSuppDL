#!/bin/bash

files=$(ls)
# TrackEval
if [[ "$files" == *"TrackEval"* ]]; then
    echo "TrackEval: already present, no download required"
else
    echo "TrackEval: download required, cloning now"
    git clone https://github.com/JonathonLuiten/TrackEval.git
fi


# Tracktor
if [[ "$files" == *"tracking_wo_bnw"* ]]; then
    echo "tracking_wo_bnw: already present, no download required"
else
    echo "tracking_wo_bnw: download required, cloning now"
    git clone https://github.com/phil-bergmann/tracking_wo_bnw.git
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


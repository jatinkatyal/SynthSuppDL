#!/usr/bin/env bash


GROUNDTRUTH=mot/train
RESULTS=val/tracks
GT_TYPE=_val_half
THRESHOLD=-1

#python3 track_tools/eval_motchallenge.py \
#--groundtruths ${GROUNDTRUTH} \
#--tests ${RESULTS} \
#--gt_type ${GT_TYPE} \
#--eval_official \
#--score_threshold ${THRESHOLD}

# python3 track_tools/eval_motchallenge.py --groundtruths ${GROUNDTRUTH}  --tests ${RESULTS}  --gt_type ${GT_TYPE}  --eval_official  --score_threshold ${THRESHOLD}

python3 track_tools/eval_motchallenge.py --groundtruths "mot/train"  --tests "val/tracks" --gt_type val_half --eval_official --score_threshold -1

python3 track_tools/eval_motchallenge.py --groundtruths mot/train  --tests output/MOT17-train/transformer_r3as18a/data --gt_type val_half --eval_official --score_threshold -1
python3 track_tools/eval_motchallenge.py --groundtruths mot/train  --tests output/MOT17-train/transformer_r3bs18b/data --gt_type val_half --eval_official --score_threshold -1
python3 track_tools/eval_motchallenge.py --groundtruths mot/train  --tests output/MOT17-train/transformer_r3cs18c/data --gt_type val_half --eval_official --score_threshold -1
python3 track_tools/eval_motchallenge.py --groundtruths mot/train  --tests output/MOT17-train/transformer_r3ds18d/data --gt_type val_half --eval_official --score_threshold -1
python3 track_tools/eval_motchallenge.py --groundtruths mot/train  --tests output/MOT17-train/transformer_r3es18e/data --gt_type val_half --eval_official --score_threshold -1
python3 track_tools/eval_motchallenge.py --groundtruths mot/train  --tests output/MOT17-train/transformer_r3gs18g/data --gt_type val_half --eval_official --score_threshold -1
python3 track_tools/eval_motchallenge.py --groundtruths mot/train  --tests output/MOT17-train/transformer_r7as14a/data --gt_type val_half --eval_official --score_threshold -1
python3 track_tools/eval_motchallenge.py --groundtruths mot/train  --tests output/MOT17-train/transformer_r7bs14b/data --gt_type val_half --eval_official --score_threshold -1
python3 track_tools/eval_motchallenge.py --groundtruths mot/train  --tests output/MOT17-train/transformer_r7cs14c/data --gt_type val_half --eval_official --score_threshold -1
python3 track_tools/eval_motchallenge.py --groundtruths mot/train  --tests output/MOT17-train/transformer_r14as7a/data --gt_type val_half --eval_official --score_threshold -1
python3 track_tools/eval_motchallenge.py --groundtruths mot/train  --tests output/MOT17-train/transformer_r14bs7b/data --gt_type val_half --eval_official --score_threshold -1
python3 track_tools/eval_motchallenge.py --groundtruths mot/train  --tests output/MOT17-train/transformer_r14cs7c/data --gt_type val_half --eval_official --score_threshold -1
python3 track_tools/eval_motchallenge.py --groundtruths mot/train  --tests output/MOT17-train/transformer_r18as3a/data --gt_type val_half --eval_official --score_threshold -1
python3 track_tools/eval_motchallenge.py --groundtruths mot/train  --tests output/MOT17-train/transformer_r18bs3b/data --gt_type val_half --eval_official --score_threshold -1
python3 track_tools/eval_motchallenge.py --groundtruths mot/train  --tests output/MOT17-train/transformer_r18cs3c/data --gt_type val_half --eval_official --score_threshold -1
python3 track_tools/eval_motchallenge.py --groundtruths mot/train  --tests output/MOT17-train/transformer_r18ds3d/data --gt_type val_half --eval_official --score_threshold -1
python3 track_tools/eval_motchallenge.py --groundtruths mot/train  --tests output/MOT17-train/transformer_r18es3e/data --gt_type val_half --eval_official --score_threshold -1
python3 track_tools/eval_motchallenge.py --groundtruths mot/train  --tests output/MOT17-train/transformer_r18fs3f/data --gt_type val_half --eval_official --score_threshold -1
python3 track_tools/eval_motchallenge.py --groundtruths mot/train  --tests output/MOT17-train/transformer_r18gs3g/data --gt_type val_half --eval_official --score_threshold -1
python3 track_tools/eval_motchallenge.py --groundtruths mot/train  --tests output/MOT17-train/transformer_r21s0 --gt_type val_half --eval_official --score_threshold -1
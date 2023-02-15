#!/bin/bash
#SBATCH --time=23:59:00
#SBATCH --account=def-cpoullis
#SBATCH --mem=16000M
#SBATCH --gres=gpu:1
#SBATCH --output="/home/jatin/scratch/slurm-%j.out"
#SBATCH --mail-user=jatinkatyal96@gmail.com
#SBATCH --mail-type=ALL


# module load python/3.9 cuda cudnn scipy-stack opencv
module load cuda cudnn gcc/9.3 opencv python/3.9

python /home/jatin/work/tracking_analysis/src/utils/fidscores.py --source=/home/jatin/projects/def-cpoullis/jatin/MOT-Challenge/MOT17/train --target=/home/jatin/projects/def-cpoullis/jatin/MOT-Challenge/Synth/frames/ --type="MOT" --dims=64 --output=/home/jatin/results/fid
exit

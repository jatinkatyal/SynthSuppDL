#!/bin/bash
#SBATCH --time=23:59:00
#SBATCH --account=def-cpoullis
#SBATCH --mem=14000M
#SBATCH --gres=gpu:1
#SBATCH --output="/home/jatin/scratch/slurm-%j.out"
#SBATCH --mail-user=jatinkatyal96@gmail.com
#SBATCH --mail-type=ALL


# module load python/3.9 cuda cudnn scipy-stack opencv
module load cuda cudnn gcc/9.3 opencv python/3.9

# source /home/jatin/ENV/bin/activate
python /home/jatin/work/tracking_analysis/src/utils/traktor_objdet.py --dataset=/home/jatin/projects/def-cpoullis/jatin/MOT-Challenge/MOT17 --synth_dataset=/home/jatin/projects/def-cpoullis/jatin/MOT-Challenge/Synth --model_prefix=R18dS3d --mode="train-18d" --synthmode="train-3d" --dataset_class=MOT17 --model=/home/jatin/models/R18dS3d_epoch_20.model

exit

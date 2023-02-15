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

unzip -q /home/jatin/datasets/UAVDT/UAV-benchmark-M.zip -d $SLURM_TMPDIR/UAVDT/
unzip -q /home/jatin/datasets/airsim/AirSim-UAV-Synth.zip -d $SLURM_TMPDIR/airsim
unzip -q datasets/airsim/additional-synth-sequences.zip -d $SLURM_TMPDIR/airsim/data/Sequences/
unzip -q datasets/airsim/additional-synth-sequences2.zip -d $SLURM_TMPDIR/airsim/data/Sequences/

python /home/jatin/work/tracking_analysis/src/utils/fidscores.py --source=$SLURM_TMPDIR/UAVDT/UAV-benchmark-M --target=$SLURM_TMPDIR/airsim/data/Sequences --type="UAVDT" --dims=768 --output=/home/jatin/results/fid
exit

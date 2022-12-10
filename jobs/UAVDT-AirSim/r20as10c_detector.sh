#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --account=def-cpoullis
#SBATCH --mem=14000M
#SBATCH --gres=gpu:1

#SBATCH --mail-user=jatinkatyal96@gmail.com
#SBATCH --mail-type=ALL


# module load python/3.9 cuda cudnn scipy-stack opencv
module load cuda cudnn gcc/9.3 opencv python/3.9

unzip -q /home/jatin/datasets/UAVDT/UAV-benchmark-M.zip -d $SLURM_TMPDIR/UAVDT/
cp -r /home/jatin/datasets/UAVDT/UAV-benchmark-MOTD_v1.0 $SLURM_TMPDIR/UAVDT/UAV-benchmark-MOTD_v1.0
unzip -q /home/jatin/datasets/AirSim-UAV-Synth.zip -d $SLURM_TMPDIR/airsim

# source /home/jatin/ENV/bin/activate
python /home/jatin/work/tracking_analysis/src/utils/traktor_objdet.py --dataset=$SLURM_TMPDIR/UAVDT/ --synth_dataset=$SLURM_TMPDIR/airsim/data/ --model_prefix=R20aS10c --mode="train20-a" --synthmode="10c" --model=/home/jatin/projects/def-cpoullis/jatin/tracktor_UAVDT/R20aS10c_epoch_24.model

exit

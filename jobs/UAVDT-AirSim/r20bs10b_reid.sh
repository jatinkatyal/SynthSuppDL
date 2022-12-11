#!/bin/bash
#SBATCH --time=15:00:00
#SBATCH --account=def-cpoullis
#SBATCH --mem=10000M
#SBATCH --gres=gpu:1
#SBATCH --output="/home/jatin/scratch/slurm-%j.out"
#SBATCH --mail-user=jatinkatyal96@gmail.com
#SBATCH --mail-type=ALL


# module load python/3.9 cuda cudnn scipy-stack opencv
module load cuda cudnn gcc/9.3 opencv python/3.9

unzip -q /home/jatin/datasets/UAVDT/UAVDT_reid.zip -d $SLURM_TMPDIR/
unzip -q /home/jatin/datasets/airsim/airsim-reid-data.zip -d $SLURM_TMPDIR/AirSim-reid-data/

# cp -r /home/jatin/datasets/AirSim-UAV-Synth/reid-data $SLURM_TMPDIR/airsim_reid
# source /home/jatin/ENV/bin/activate

python /home/jatin/work/tracking_analysis/src/utils/traktor_reid.py --data=$SLURM_TMPDIR/UAVDT_reid --syndata=$SLURM_TMPDIR/AirSim-reid-data/reid-data --mode=R20bS10b
exit

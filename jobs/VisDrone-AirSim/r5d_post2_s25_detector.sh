#!/bin/bash
#SBATCH --time=15:59:00
#SBATCH --account=def-cpoullis
#SBATCH --mem=14000M
#SBATCH --gres=gpu:1
#SBATCH --output="/home/jatin/scratch/slurm-%j.out"
#SBATCH --mail-user=jatinkatyal96@gmail.com
#SBATCH --mail-type=ALL


# module load python/3.9 cuda cudnn scipy-stack opencv
module load cuda cudnn gcc/9.3 opencv python/3.9

#unzip -q /home/jatin/datasets/UAVDT/UAV-benchmark-M.zip -d $SLURM_TMPDIR/UAVDT/
#cp -r /home/jatin/datasets/UAVDT/UAV-benchmark-MOTD_v1.0 $SLURM_TMPDIR/UAVDT/UAV-benchmark-MOTD_v1.0
unzip -q /home/jatin/datasets/VisDrone/Visdrone-post_2.zip -d $SLURM_TMPDIR/VisDrone
unzip -q /home/jatin/datasets/airsim/AirSim-UAV-Synth.zip -d $SLURM_TMPDIR/airsim
unzip -q datasets/airsim/additional-synth-sequences.zip -d $SLURM_TMPDIR/airsim/data/Sequences/
unzip -q datasets/airsim/additional-synth-annots.zip -d $SLURM_TMPDIR/airsim/data/Annotations/
unzip -q datasets/airsim/additional-synth-sequences2.zip -d $SLURM_TMPDIR/airsim/data/Sequences/
unzip -q datasets/airsim/additional-synth-annots2.zip -d $SLURM_TMPDIR/airsim/data/Annotations/

# source /home/jatin/ENV/bin/activate
python /home/jatin/work/tracking_analysis/src/utils/traktor_objdet.py --dataset=$SLURM_TMPDIR/VisDrone/VisDrone2019-MOT-train/ --synth_dataset=$SLURM_TMPDIR/airsim/data/ --model_prefix=R5d_post2_S25 --mode="train5-d" --synthmode="25" --dataset_class=VisDrone

exit

#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --account=def-cpoullis
#SBATCH --mem=10000M
#SBATCH --gres=gpu:1
#SBATCH --output="/home/jatin/scratch/slurm-%j.out"
#SBATCH --mail-user=jatinkatyal96@gmail.com
#SBATCH --mail-type=ALL

module load cuda cudnn gcc/9.3 opencv python/3.9

unzip -q /home/jatin/datasets/UAVDT/UAV-benchmark-M.zip -d $SLURM_TMPDIR/UAVDT
cp -r /home/jatin/datasets/UAVDT/UAV-benchmark-MOTD_v1.0 $SLURM_TMPDIR/UAVDT/

python /home/jatin/work/tracking_analysis/src/tracktor.py --det_model=/home/jatin/models/R10bS20_epoch_30.model --reid_model=/home/jatin/models/tracktor_reid_R10bS20/model/model.pth.tar-10 --config=/home/jatin/work/tracking_analysis/cfgs/tracktor.yaml --dataset=$SLURM_TMPDIR/UAVDT --output=/home/jatin/results/tracktor/UAVDT/R10bS20-det_n_R10bS20-reid

exit

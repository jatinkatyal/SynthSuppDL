#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --account=def-cpoullis
#SBATCH --mem=10000M
#SBATCH --gres=gpu:1
#SBATCH --output="/home/jatin/scratch/slurm-%j.out"
#SBATCH --mail-user=jatinkatyal96@gmail.com
#SBATCH --mail-type=ALL

module load cuda cudnn gcc/9.3 opencv python/3.9

unzip -q /home/jatin/datasets/VisDrone/VisDrone2019-MOT-test-dev.zip -d $SLURM_TMPDIR/

python /home/jatin/work/tracking_analysis/src/tracktor.py --det_model=/home/jatin/models/R5bS25_epoch_30.model --reid_model=/home/jatin/models/tracktor_reid_R5bS25/model/model.pth.tar-10 --config=/home/jatin/work/tracking_analysis/cfgs/tracktor.yaml --dataset=$SLURM_TMPDIR/VisDrone2019-MOT-test-dev --output=/home/jatin/results/tracktor/VisDrone/R5bS25-det_n_R5bS25-reid  --dataset_type='VisDrone'

exit

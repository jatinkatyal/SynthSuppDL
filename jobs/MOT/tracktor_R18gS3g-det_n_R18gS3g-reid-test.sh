#!/bin/bash
#SBATCH --time=0:5:00
#SBATCH --account=def-cpoullis
#SBATCH --mem=10000M
#SBATCH --gres=gpu:1
#SBATCH --output="/home/jatin/scratch/slurm-%j.out"
#SBATCH --mail-user=jatinkatyal96@gmail.com
#SBATCH --mail-type=ALL

module load cuda cudnn gcc/9.3 opencv python/3.9

#unzip -q /home/jatin/datasets/VisDrone/VisDrone2019-MOT-test-dev.zip -d $SLURM_TMPDIR/

python /home/jatin/work/tracking_analysis/src/tracktor.py --det_model=/home/jatin/models/R18gS3g_epoch_30.model --reid_model=/home/jatin/models/tracktor_reid_R18gS3g/model/model.pth.tar-10 --config=/home/jatin/work/tracking_analysis/cfgs/tracktor.yaml --dataset=/home/jatin/projects/def-cpoullis/jatin/MOT-Challenge/MOT17 --output=/home/jatin/results/tracktor/MOT/R18gS3g-det_n_R18gS3g-reid  --dataset_type="MOT-train"

exit

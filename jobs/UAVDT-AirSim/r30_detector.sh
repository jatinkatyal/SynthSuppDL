#!/bin/bash
#SBATCH --time=1:30:00
#SBATCH --account=def-cpoullis
#SBATCH --mem=16000M
#SBATCH --gres=gpu:1

#SBATCH --mail-user=jatinkatyal96@gmail.com
#SBATCH --mail-type=ALL

#module load python/3.10 cuda cudnn scipy-stack
module load cuda cudnn gcc/9.3 opencv python/3.9


#source /home/jatin/ENV/bin/activate
python /home/jatin/work/tracking_analysis/src/utils/traktor_objdet.py --dataset=/home/jatin/datasets/UAVDT/ --model_prefix=R30_tracktor_det --mode=train --model=/home/jatin/models/R30_tracktor_det_epoch_28.model

exit

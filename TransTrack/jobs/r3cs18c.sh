#!/bin/bash
#SBATCH --time=15:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --account=def-cpoullis
#SBATCH --mem=32000M
#SBATCH --gres=gpu:4
#SBATCH --output="/home/jatin/scratch/slurm-%j.out"
#SBATCH --mail-user=jatinkatyal96@gmail.com
#SBATCH --mail-type=ALL

module load opencv python/3.7 cuda/11.4 cudnn
source /home/jatin/transtrack/bin/activate

cd /home/jatin/work/tracking_analysis/TransTrack
python3 -m torch.distributed.launch --nproc_per_node=4 --use_env main_track.py  --output_dir ./mixsets/r3cs18c --dataset_file r3cs18c --coco_path mixsets/r3cs18c --batch_size 4 --resume output_crowdhuman/crowdhuman_final.pth --with_box_refine --num_queries 500  --epochs 20 --lr_drop 10
